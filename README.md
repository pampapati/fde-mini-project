### Lead Intelligence System (Gemini Core Edition)

### Project Overview

This system solves a common mid-market SaaS bottleneck: processing thousands of inbound leads manually while leaking high-value opportunities. The Lead Intelligence tool automates the ingestion of standard CRM lead datasets, evaluates profiles in batches against a strict qualification rubric, assigns prioritization triage states, and generates highly personalized outreach email templates for high-fit conversions. 

By offloading the initial triage layer to an LLM, the evaluation time drops from 8–12 minutes per lead down to a few seconds for an entire batch. 

### Strategic Qualification Rubric Matrix

The automated triage engine evaluates each inbound lead profile using a 1–10 point calculation layer across three critical parameters: 

* **Firmographic Scale (Headcount Sizing):** The ideal ICP focus target maps between 50 and 5,000 employees. Companies falling below 50 receive automated point discounts.
* **Vertical Integration Fit:** High-value target industries are prioritized (SaaS, Technology, Finance, Healthcare, Logistics, Manufacturing, EdTech, FinTech).
* **Channel Action Signals:** High-conversion sources (Inbound Demo Requests, Contact Us Forms) receive maximum scoring weight, while passive resource downloads run at a logical discount.

### Automated Routing Logic:

* **Score >= 7 (Qualified):** Added directly to high-priority outbound channels with a custom personalized email draft.
* **Score 4 - 6 (Review):** Borderline profiles or records with missing fields are safely flagged for human-in-the-loop audit tracking.
* **Score < 4 (Rejected):** Incompatible verticals or historical profiles inactive for >365 days are archived to clean target pipelines.

### Complete Quick-Start Guide

### Step 1: Initialize the Isolated Virtual Environment

Ensure you have uv installed on your workstation terminal environment. Run this command sequence to spin up a clean virtual environment and register your core dependencies: 

bash

# 1. Create a clean local virtual environment
uv venv

# 2. Install the production-grade pipeline libraries
uv pip install google-genai pydantic

Use code with caution.

### Step 2: Generate and Set Your Free Gemini API Key

1. Visit **https://aistudio.google.com/** in your web browser and sign in with a Google account.
2. Click the blue **"Get API key"** button on the top-left menu.
3. Click **"Create API key"** (either in a new or existing project) and copy the generated token string (starts with AIzaSy...).
4. Export your live authentication token directly into your active shell environment process:

**

****