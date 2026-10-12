# Task 3: Intelligent Document Processing (IDP) Pipeline

## 📌 Problem Statement
You are provided with a set of 10 mixed documents spanning identity proofs and insurance proposal forms:
1. Aadhaar Card
2. PAN Card
3. Driving Licence
4. Passport
5. NACH / ECS Mandate (handwritten)
6. FATCA Annexure Form (handwritten)
7. Benefit Illustration Declaration (handwritten)
8. Moral Hazard Questionnaire (handwritten)
9. Multiple Policies Consent Form (handwritten)
10. Suitability Profiler Declaration (handwritten)

The objective is to build an IDP pipeline that:
1. **Classifies each document** by type.
2. **Extracts target fields** specified for each document type with particular attention to handwritten fields (IFSC, TIN, bank account numbers, dates, place names).
3. **Calculates confidence score per field** (not just per document).
4. **Flags fields below confidence threshold** (`0.85`) for human-in-the-loop review with explicit rationale.
5. **Evaluates field-level accuracy** against a ground truth answer key.

---

## 🏗️ Solution Workflow & Architecture

```
                          Document Input (10 Scanned Images)
                                          │
                                          ▼
                            [Document Type Classifier]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          [Path A: Printed Pipeline]                 [Path B: Handwritten Pipeline]
          - Identity Cards (Aadhaar, PAN, DL, MRZ)   - Onboarding Forms (NACH, FATCA, MHQ, etc.)
          - Deskew & Regex Syntax Parsing            - Grid-Box Cell Segmentation & Stroke Normalization
          - Confidence: 0.93 - 0.99                  - Bounding Anchor Grounding & Cursive Slant Filter
                                                     - Checkbox Tickmark Density Analysis
                                                     - Confidence: 0.78 - 0.94
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                             [Per-Field Confidence Scorer]
                                          │
                                          ▼
                             [STP Gate: Threshold = 0.85]
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                     [STP Pass (≥ 0.85)]       [Flagged for Review (< 0.85)]
                            │                           │
                            ▼                           ▼
                     extractions.json            flagging_report.json
```

---

## 📁 Deliverables Produced
- `results/extractions.json`: Full structured JSON containing extracted values, per-field confidence, and notes for all 10 documents.
- `results/flagging_report.json`: Human review triage report listing fields below `0.85` threshold with rationale.
- `results/technical_note.md`: In-depth analysis of handling handwritten vs printed text and observed failure cases.
- `ground_truth.json`: Verified benchmark answer key.

---

## 🚀 How to Run

### Option 1: Automated Pipeline & Ground Truth Evaluation CLI
Execute the complete end-to-end extraction and evaluation:
```bash
python task3_document_pipeline/run_pipeline.py
```
Outputs:
- 100% field-level accuracy verification across 44 fields
- Per-field confidence score breakdown
- Triage report for the 4 flagged fields

### Option 2: Interactive Streamlit IDP Dashboard
Run the visual inspection dashboard:
```bash
streamlit run task3_document_pipeline/app.py
```
Open your browser at `http://localhost:8501`.
Features:
- Side-by-side document image preview and extracted field values
- Dynamic threshold adjustment slider (see how flags change as threshold shifts)
- Ground Truth accuracy comparison table
- Technical note viewer

