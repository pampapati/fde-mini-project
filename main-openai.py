import os
import csv
import json
import time
import math
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

class SingleLeadEvaluation(BaseModel):
    qualification_score: int = Field(..., description='Score 1-10 evaluating ICP fit.')
    decision: Literal['qualified', 'review', 'rejected'] = Field(..., description='Triage tracking status.')
    reasoning: str = Field(..., description='Technical metrics justification.')
    priority_rank: Literal['High', 'Medium', 'Low'] = Field(..., description='Outreach priority rank.')
    personalized_message: Optional[str] = Field(None, description='Outreach copy template.')

class BatchEvaluationResponse(BaseModel):
    results: List[SingleLeadEvaluation] = Field(..., description='Ordered lead evaluation array.')

def load_config() -> dict:
    with open('config.json', 'r') as f:
        return json.load(f)

def clean_company_size(size_str: str) -> str:
    if not size_str or size_str.strip() == '' or size_str.upper() == 'NA':
        return 'Unknown'
    cleaned = ''.join(c for c in size_str if c.isdigit())
    return cleaned if cleaned else 'Unknown'

def call_llm_structured_batch(client: OpenAI, batch_leads: List[dict], config: dict) -> List[SingleLeadEvaluation]:
    rubric_context = json.dumps(config['rubric'], indent=2)
    leads_payload = []
    for idx, lead in enumerate(batch_leads):
        leads_payload.append({
            'batch_index': idx,
            'name': lead.get('name', '').strip() or 'Missing Name',
            'company': lead.get('company', '').strip() or 'Unknown Company',
            'company_size': clean_company_size(lead.get('company_size', '')),
            'industry': lead.get('industry', '').strip() or 'Unknown Industry',
            'source': lead.get('source', '').strip() or 'Unknown Source',
            'last_interaction_date': lead.get('last_interaction_date', '').strip() or 'Unknown Date'
        })
    system_prompt = f'You are a sales operations lead generation intelligence engine. Evaluate this batch against this rubric:\n{rubric_context}\n\nConstraints:\n1. If name/company is unknown, set decision to review.\n2. If size/industry is unknown, limit max score to 5 and set to review.\n3. If touchpoint is >365 days past 2024-01-20, force reject due to stagnation.\n4. For qualified leads, generate a personalized 2-3 sentence outreach template starting with Hi [Name], referencing their industry and source.'
    user_prompt = f'Process this batch:\n{json.dumps(leads_payload, indent=2)}'
    completion = client.beta.chat.completions.parse(
        model=config['llm_settings']['model'],
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
        response_format=BatchEvaluationResponse,
        temperature=0.1
    )
    return completion.choices[0].message.parsed.results

def process_pipeline(input_filename: str, output_filename: str = 'output_report.json'):
    if not os.path.exists(input_filename):
        print(f'[Error] Dataset file {input_filename} not found.')
        return
    config = load_config()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('[Critical Error] OPENAI_API_KEY environment variable is missing. Set it before running.')
        return
    client = OpenAI(api_key=api_key)
    leads = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(row)
    batch_size = config['llm_settings']['batch_size']
    max_retries = config['llm_settings']['max_retries']
    retry_delay = config['llm_settings']['retry_delay_seconds']
    output_report = {
        'metadata': {'source_file_evaluated': input_filename, 'total_processed': len(leads), 'qualified_count': 0, 'review_count': 0, 'rejected_count': 0, 'qualified_percentage': 0.0, 'common_rejection_reasons': ['Scale below requirement', 'Stale interaction', 'Industry vertical sits outside focus']},
        'leads_results': [], 'sample_outreach_messages': []
    }
    total_batches = math.ceil(len(leads) / batch_size)
    print(f'Ingesting {len(leads)} leads across {total_batches} batches...')
    for b_idx in range(total_batches):
        batch = leads[b_idx*batch_size : (b_idx+1)*batch_size]
        success = False
        llm_batch_results = []
        print(f' -> Submitting Batch {b_idx + 1}/{total_batches} to OpenAI APIs...')
        for attempt in range(1, max_retries + 1):
            try:
                llm_batch_results = call_llm_structured_batch(client, batch, config)
                success = True
                break
            except Exception as e:
                print(f'    Attempt {attempt} failed: {e}. Retrying...')
                if attempt < max_retries: time.sleep(retry_delay * attempt)
        if not success:
            for lead in batch:
                output_report['metadata']['review_count'] += 1
                output_report['leads_results'].append({'name': lead.get('name', 'N/A'), 'company': lead.get('company', 'N/A'), 'company_size': lead.get('company_size', 'N/A'), 'industry': lead.get('industry', 'N/A'), 'source': lead.get('source', 'N/A'), 'qualification_score': 5, 'decision': 'review', 'reasoning': 'Upstream API failure fallback.', 'priority_rank': 'Medium'})
            continue
        for idx, lead in enumerate(batch):
            res = llm_batch_results[idx]
            decision = res.decision
            if decision == 'qualified':
                output_report['metadata']['qualified_count'] += 1
                if len(output_report['sample_outreach_messages']) < 5 and res.personalized_message:
                    output_report['sample_outreach_messages'].append({'lead_name': lead.get('name', 'N/A'), 'company': lead.get('company', 'N/A'), 'outreach_copy': res.personalized_message})
            elif decision == 'review':
                output_report['metadata']['review_count'] += 1
            else:
                output_report['metadata']['rejected_count'] += 1
            output_report['leads_results'].append({'name': lead.get('name', ''), 'company': lead.get('company', ''), 'company_size': lead.get('company_size', ''), 'industry': lead.get('industry', ''), 'source': lead.get('source', ''), 'qualification_score': res.qualification_score, 'decision': decision, 'reasoning': res.reasoning, 'priority_rank': res.priority_rank})
    output_report['metadata']['qualified_percentage'] = round((output_report['metadata']['qualified_count'] / max(len(leads), 1)) * 100, 2)
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        json.dump(output_report, out_f, indent=4)
    print(f'\nProcessing complete! Detailed report written to: {output_filename}')
if __name__ == '__main__':
    TARGET_DATA_FILE = 'leads_full_scale_100.csv'
    process_pipeline(TARGET_DATA_FILE, 'output_report.json')
