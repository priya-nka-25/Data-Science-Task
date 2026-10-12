# Task 2: Document-Aware Multi-Turn Support Assistant

## 📌 Problem Statement
Using your own documents as an enterprise knowledge base, build a document-aware support assistant that maintains context across a multi-turn conversation. The assistant must satisfy four essential constraints:
1. **Remember what the user has already asked** within the session (multi-turn memory & cross-turn entity recall).
2. **Avoid repeating information** it has already provided in prior responses (anti-repetition filtering).
3. **Gracefully handle topic switches** (smooth conversational transitions when switching domains).
4. **Cite the specific document section** it is drawing from in each response (e.g. `[01_sla_and_support_tiers.md § Section 1.2: Response Time Commitments (SLA)]`).
5. **Demonstrate across a conversation of at least 10 turns**.

---

## 🏗️ Solution Workflow & Architecture

```
User Turn (Query)
       │
       ▼
┌──────────────────────────────────────┐
│       Topic Domain Classifier        │  (SLA vs Billing vs Security vs API vs Privacy)
└──────────────────┬───────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐
│  Topic Switch    │  │  Memory Recall   │
│  Detector        │  │  Engine          │
│  (Transition)    │  │  (Turn History)  │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Section-Level Hybrid Retriever      │◄─── Enterprise Knowledge Base
│  (Document + Exact Section Header)   │     (5 Markdown Policy Specs)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Anti-Repetition Synthesis Engine    │  (Suppresses facts already shared in session)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Exact Section Citation Formatter    │  (Attaches file name & section title)
└──────────────────┬───────────────────┘
                   │
                   ▼
        Final Response Delivered
```

### Knowledge Base Documents Included:
1. `01_sla_and_support_tiers.md`: Severity definitions, response time commitments, escalation matrix.
2. `02_billing_and_refund_policy.md`: Subscription billing cycles, 30-day money-back guarantee, pro-rated refunds, grace periods.
3. `03_security_and_compliance.md`: MFA, SSO (SAML 2.0 / OIDC), password complexity & 90-day rotation, SOC 2 Type II, AES-256 encryption.
4. `04_api_rate_limits_and_integration.md`: Token bucket rate limits, HTTP 429 status, exponential backoff webhook retries.
5. `05_data_retention_and_privacy.md`: GDPR/CCPA Data Subject Access Requests (14 days), 30-day purge timelines, audit log retention.

---

## 🚀 How to Run

### Option 1: 10-Turn Automated Benchmark & Verification Script
Execute the full 10-turn conversation demo:
```bash
python task2_support_assistant/run_10_turns_demo.py
```
This runs 10 consecutive turns verifying memory, non-repetition, topic transitions, and citations.

### Option 2: Interactive Streamlit Web Interface
Run the live web UI:
```bash
streamlit run task2_support_assistant/app.py
```
Open your browser at `http://localhost:8501`.
Features:
- Live multi-turn conversation
- Sidebar document inspector
- Scripted turn trigger buttons
- Automatic topic badges and source citations

