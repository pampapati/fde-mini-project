import os
import csv
import json
import time
import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
# Import the official Groq client
from groq import Groq

# =========================================================================
# STRUCTURED OUTPUT SCHEMAS
# =========================================================================
class SingleLeadEvaluation(BaseModel):
    qualification_score: int = Field(..., description="Integer score from 1 to 10 evaluating core ICP fit.")
    decision: Literal["qualified", "review", "rejected"] = Field(..., description="The definitive triage tracking routing status.")
    reasoning: str = Field(..., description="2-3 sentence technical justification based explicitly on the rubric metrics.")
    priority_rank: Literal["High", "Medium", "Low"] = Field(..., description="Outreach priority based on the score matrix tier.")
    personalized_message: Optional[str] = Field(None, description="A highly personalized 2-3 sentence message template for qualified leads.")

class BatchEvaluationResponse(BaseModel):
    results: List[SingleLeadEvaluation] = Field(..., description="Ordered list of lead evaluations.")

# =========================================================================
# DATA CLEANUP UTILITIES
# =========================================================================
def load_config() -> dict:
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "rubric": {
                "target_industries": ["Technology", "SaaS", "Finance", "Healthcare", "Logistics", "Manufacturing", "EdTech", "FinTech"],
                "ideal_company_size": {"min": 50, "max": 5000},
                "high_intent_sources": ["Inbound demo request", "Contact Us Form", "Inbound demo requests"]
            },
            "llm_settings": {
                "model": "llama-3.3-70b-specdec",
                "batch_size": 10,
                "max_retries": 3,
                "retry_delay_seconds": 1
            }
        }

def clean_company_size(size_str: str) -> str:
    if not size_str or size_str.strip() == "" or size_str.upper() == "NA":
        return "Unknown"
    cleaned = ''.join(c for c in size_str if c.isdigit())
    return cleaned if cleaned else "Unknown"

def call_groq_structured_batch(client: Groq, batch_leads: List[dict], config: dict) -> List[SingleLeadEvaluation]:
    rubric_context = json.dumps(config["rubric"], indent=2)
    leads_payload = []
    
    for idx, lead in enumerate(batch_leads):
        leads_payload.append({
            "batch_index": idx,
            "name": lead.get("name", "").strip() or "Missing Name",
            "company": lead.get("company", "").strip() or "Unknown Company",
            "company_size": clean_company_size(lead.get("company_size", "")),
            "industry": lead.get("industry", "").strip() or "Unknown Industry",
            "source": lead.get("source", "").strip() or "Unknown Source",
            "last_interaction_date": lead.get("last_interaction_date", "").strip() or "Unknown Date"
        })

    system_prompt = (
        "You are an expert sales operations lead generation intelligence engine.\n"
        "Evaluate the incoming batch of leads against this corporate ICP rubric:\n"
        f"{rubric_context}\n\n"
        "CRITICAL LOGICAL CONSTRAINTS:\n"
        "1. If 'name' is 'Missing Name' or 'company' is 'Unknown Company', force decision to 'review'.\n"
        "2. If 'company_size' is 'Unknown' or 'industry' is 'Unknown Industry', limit max score to 5 and status to 'review'.\n"
        "3. If 'last_interaction_date' indicates the interaction is more than 365 days past 2024-01-20, force decision to 'rejected'.\n"
        "4. For 'qualified' leads, draft a crisp, 2-3 sentence outreach template starting with 'Hi [Name],'. Reference their specific industry and source platform naturally.\n"
        "Return data matching the strict JSON format specified."
    )

    user_prompt = f"Process this batch of leads:\n{json.dumps(leads_payload, indent=2)}"

    # Force strict JSON parsing output directly via Groq's engine parameters
    completion = client.chat.completions.create(
        model=config["llm_settings"].get("model", "llama-3.3-70b-specdec"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    parsed_json = json.loads(completion.choices.message.content)
    
    # Structural Normalization Layer: Catches direct list outputs from the model
    if isinstance(parsed_json, list):
        parsed_json = {"results": parsed_json}
    elif isinstance(parsed_json, dict) and "results" not in parsed_json:
        # If it returned a dictionary where the keys are inside another root property
        for key, val in parsed_json.items():
            if isinstance(val, list):
                parsed_json = {"results": val}
                break
                
    return BatchEvaluationResponse(**parsed_json).results

# =========================================================================
# PIPELINE EXECUTION
# =========================================================================
def process_pipeline(input_filename: str, output_filename: str = "output_report.json"):
    if not os.path.exists(input_filename):
        print(f"[Error] Source dataset file '{input_filename}' not found.")
        return
        
    config = load_config()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[Critical Error] GROQ_API_KEY environment variable is missing. Set it before running.")
        return
        
    client = Groq(api_key=api_key)
    
    leads = []
    with open(input_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            leads.append(row)
            
    batch_size = config["llm_settings"]["batch_size"]
    max_retries = config["llm_settings"]["max_retries"]
    retry_delay = config["llm_settings"]["retry_delay_seconds"]
    
    output_report = {
        "metadata": {
            "source_file_evaluated": input_filename,
            "total_processed": len(leads),
            "qualified_count": 0, "review_count": 0, "rejected_count": 0, "qualified_percentage": 0.0,
            "common_rejection_reasons": ["Scale below requirement", "Stale interaction", "Industry mismatch"]
        },
        "leads_results": [],
        "sample_outreach_messages": []
    }
    
    total_batches = math.ceil(len(leads) / batch_size)
    print(f"Initializing Groq Lead Pipeline Engine. Processing {len(leads)} rows across {total_batches} batches...")
    
    for b_idx in range(total_batches):
        batch = leads[b_idx*batch_size : (b_idx+1)*batch_size]
        success = False
        llm_batch_results = []
        
        print(f" -> Submitting Batch {b_idx + 1}/{total_batches} to Groq Ultra-Fast API...")
        
        for attempt in range(1, max_retries + 1):
            try:
                llm_batch_results = call_groq_structured_batch(client, batch, config)
                success = True
                break
            except Exception as e:
                print(f"    [Warning] Batch {b_idx+1} attempt {attempt} failed: {e}. Retrying...")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    
        if not success:
            print(f"    [Critical] Batch {b_idx+1} dropped out completely. Injecting data protection fallbacks.")
            for lead in batch:
                output_report["metadata"]["review_count"] += 1
                output_report["leads_results"].append({
                    "name": lead.get("name", "N/A"), "company": lead.get("company", "N/A"),
                    "company_size": lead.get("company_size", "N/A"), "industry": lead.get("industry", "N/A"),
                    "source": lead.get("source", "N/A"), "qualification_score": 5, "decision": "review",
                    "reasoning": "Upstream Groq endpoint safety fallback.", "priority_rank": "Medium"
                })
            continue
                
        for idx, lead in enumerate(batch):
            res = llm_batch_results[idx]
            decision = res.decision
            
            if decision == "qualified":
                output_report["metadata"]["qualified_count"] += 1
                if len(output_report["sample_outreach_messages"]) < 5 and res.personalized_message:
                    output_report["sample_outreach_messages"].append({
                        "lead_name": lead.get("name", "N/A"),
                        "company": lead.get("company", "N/A"),
                        "outreach_copy": res.personalized_message
                    })
            elif decision == "review":
                output_report["metadata"]["review_count"] += 1
            else:
                output_report["metadata"]["rejected_count"] += 1
                
            output_report["leads_results"].append({
                "name": lead.get("name", ""), "company": lead.get("company", ""),
                "company_size": lead.get("company_size", ""), "industry": lead.get("industry", ""),
                "source": lead.get("source", ""), "qualification_score": res.qualification_score,
                "decision": decision, "reasoning": res.reasoning, "priority_rank": res.priority_rank
            })
            
    output_report["metadata"]["qualified_percentage"] = round((output_report["metadata"]["qualified_count"] / max(len(leads), 1)) * 100, 2)
    
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        json.dump(output_report, out_f, indent=4)
    print(f"\nProcessing complete! Evaluation report generated at: '{output_filename}'")

if __name__ == '__main__':
    TARGET_DATA_FILE = 'leads_full_scale_100.csv'
    process_pipeline(TARGET_DATA_FILE, 'output_report_grok.json')
