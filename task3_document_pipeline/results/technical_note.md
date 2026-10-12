# Technical Note: Handling Handwritten Text vs Printed Text in Insurance IDP

## 1. Architectural Strategy: Dual-Path Processing Pipeline
In document ingestion workflows spanning identity proofs and onboarding forms, machine-printed documents (Aadhaar, PAN, DL, Passport) and handwritten declaration forms (NACH mandate, FATCA, MHQ, Suitability Profiler) present fundamentally different visual and semantic distributions. To maximize field-level accuracy and avoid degradation, our pipeline implements a **Dual-Path Architecture**:

```
                       Input Document
                             │
                             ▼
                  [Document Classifier]
                  (Layout & Keyword Analysis)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Path A: Printed Pipeline]        [Path B: Handwritten Pipeline]
   - Adaptive Contrast & Binarization  - Morphological Noise Filtering
   - Fixed Anchor Layout Alignment    - Bounding Grid Cell Segmentation
   - Regex & Checksum Validations     - Cursive Stroke Stroke Normalization
   - Confidence Baseline: 0.93 - 0.99 - VLM / Layout-Aware Token Extractor
                                      - Confidence Baseline: 0.78 - 0.94
            └────────────────┬────────────────┘
                             │
                             ▼
               [Per-Field Confidence Scorer]
                             │
                             ▼
               [STP Threshold Gate (0.85)]
              ┌──────────────┴──────────────┐
              ▼                             ▼
       [Automated STP Pass]         [Human-in-the-Loop Flag]
```

### Path A: Machine-Printed Processing
- **Pre-Processing**: High-contrast Otsu binarization and affine deskewing.
- **Pattern Matching**: Rigid regular expression conformance (e.g., PAN `[A-Z]{5}[0-9]{4}[A-Z]`, 12-digit Aadhaar UID, 2-line ICAO Doc 9303 MRZ parser).
- **High Base Confidence**: Printed characters follow standard typography fonts with minimal baseline variance, yielding confidence scores consistently above `0.93`.

### Path B: Handwritten Field Processing
- **Grid-Box Character Segmentation**: In documents like the NACH mandate (Bank A/C number `31004258912`) and FATCA (`BPQPD3051R`), individual characters are inscribed inside comb/box boundaries. The pipeline isolates individual cell bounding boxes to eliminate character run-on and overlap.
- **Cursive Stroke & Baseline Drift Correction**: Freeform fields (e.g. Proposer name "Ashok", Father's name "Arjun Das Kumar", and Nominee Relationship "Nephew") lack grid boxes. We apply morphological closing, stroke thickness normalization, and line slant angle rectification.
- **Contextual Anchor Grounding**: Rather than relying purely on full-page OCR, extraction targets are localized relative to pre-printed static labels (e.g., `5. Please confirm exact relationship** of nominee ----- [Nephew]`).
- **Checkbox Detection**: For forms like the Multiple Policies Consent form and NACH mandate frequency, specialized contour density analysis determines if a checkbox contains pen ink (check/cross) versus empty whitespace.

---

## 2. Confidence Threshold Selection & Rationale
- **Threshold Value Chosen**: `0.85` (85%).
- **Operational & Regulatory Rationale**:
  - In Indian banking and insurance straight-through processing (STP) under IRDAI and RBI guidelines, automated financial debits and policy issuances cannot tolerate false positives on key financial identifiers (IFSC, Account Number, Nominee Relationship).
  - Any field scoring below `0.85` indicates handwritten cursive ambiguity, non-standard date formatting, or high-underwriting-risk designations (e.g., non-immediate family nominee).
  - Routing items `< 0.85` to a Level-2 Human Review / Underwriter queue ensures 100% compliance without blocking clean straight-through applications.

---

## 3. Observed Failure Cases & Mitigations

| # | Observed Failure Mode | Root Cause | Implemented Mitigation |
|---|----------------------|------------|------------------------|
| 1 | **TIN / PAN Character Ambiguity (`BPQPD3051R`)** | In `Fatca.jpeg`, the handwritten uppercase character `Q` closely mimics `O`, and `D` resembles `0`. | Implemented tax-id syntactic validator checking character position rules (positions 1-5 letters, 6-9 digits, 10 letter) and flagged for review with score `0.83`. |
| 2 | **Cursive Non-Standard Nominee Relationship (`Nephew`)** | In `Moral.jpeg`, the word "Nephew" is written in cursive handwriting. Furthermore, "Nephew" is a non-standard beneficiary requiring insurable interest justification. | Lowered confidence to `0.78` and flagged for mandatory underwriter scrutiny. |
| 3 | **Truncated Date Formatting (`26/04/26`)** | In `suitability.jpeg`, the date is inscribed with a 2-digit year (`26`) rather than standard 4-digit (`2026`). | Normalizer expands `26` to `2026`, but assigns `0.81` confidence to flag formatting discrepancy. |
| 4 | **Agent Signature & SP Name Variance (`Ramesh Kumar`)** | In `suitability.jpeg`, the Agent/SP name is handwritten in freehand script adjacent to the signature stamp. | Scored at `0.82` and flagged for insurance agent code database cross-validation. |
| 5 | **Comb-Box Digit Segmentation Noise (`ECS.jpeg`)** | Thick black box boundaries in NACH bank account boxes can bleed into digits `1` or `0`. | Subtracted pre-printed box borders using morphological horizontal/vertical line kernel masks before character isolation. |

