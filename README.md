# Cynbit Project Intake Router

An AI/ML assessment proof of concept: route an unstructured project brief to a likely
Cynbit service line, show the terms that influenced the prediction, and identify discovery
details that are still missing. The application uses a small fictional dataset and makes
no claims about Cynbit's internal operations or customer data.

## Run locally

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app is also compatible with `python -m streamlit run app.py`. No API key, database,
or external model service is required.

## Opportunity report

### Public evidence and scope

Cynbit's public [services page](https://www.cynbit.in/services) describes autonomous AI
agents and workflow automation; software engineering and SaaS development; cloud
infrastructure and legacy migration; data-driven marketing, CRM automation, and funnel
optimization. The public site also links to [BitElevate](https://www.cynbit.in/bitelevate),
a technical learning product. These public offerings are the basis for the opportunity
hypotheses below. No internal funnel, delivery, support, or learner data was available or
assumed.

### Opportunity comparison

Directional scores are from 1 (low) to 5 (high), based on an outside-in assessment.
“Pilot speed” is scored higher when a useful, bounded pilot could be built sooner. Scores
are prioritization hypotheses, not measured company outcomes.

| Opportunity | Business value | Feasibility | Innovation | Pilot speed | Data access | Potential impact | Pilot thought |
|---|---:|---:|---:|---:|---:|---:|---|
| Project-intake routing and discovery | 4 | 5 | 3 | 5 | 4 | 4 | Start with synthetic briefs; pilot on approved, de-identified enquiries |
| RFP / proposal evidence assistant | 4 | 3 | 4 | 2 | 2 | 4 | Requires approved proposals, capability evidence, and source citations |
| Engineering QA and test-case copilot | 4 | 4 | 3 | 4 | 3 | 4 | Trial on a non-sensitive sample spec/repository; measure review quality |
| Cloud reliability / FinOps signals | 3 | 3 | 5 | 3 | 2 | 4 | Needs safe access to telemetry, infrastructure context, and cost history |
| BitElevate adaptive practice paths | 3 | 3 | 4 | 3 | 2 | 3 | Requires consented learning events and validated mastery signals |

### Recommended opportunity

**Project-intake routing and discovery assistant.** A first-pass classifier can map a
prospective brief to a publicly described capability and give a human a structured starting
point. It is a good assessment prototype because its scope is bounded, the output is
auditable, and a demonstration can run without client data. Potential business value—faster
and more consistent first responses—is a hypothesis to test, not a known Cynbit pain point.

## Technical approach

### Model and architecture

- **Interface:** Streamlit form and results view.
- **Classifier:** scikit-learn `Pipeline(TfidfVectorizer, LogisticRegression)` over five
  service-line labels. Unigrams and bigrams help recognize phrases; logistic regression
  keeps a compact, explainable baseline.
- **Data:** 60 fictional, labeled examples in `data/synthetic_intakes.csv`. They are
  deliberately small seed examples, not representative customer data.
- **Explanation:** Show positive TF-IDF/classifier feature contributions that occur in the
  submitted brief.
- **Discovery readiness:** A transparent weighted checklist for problem/users, outcome,
  stack, timeline, and success measure. It is not a feasibility or lead-quality score.
- **Question suggestions:** Curated, deterministic prompts per service line; no generative
  model or prompt is used in this MVP.
- **State and privacy:** No database, file write, third-party inference API, or application
  logging of the submitted brief. Inputs are processed in memory. Do not submit personal or
  confidential information; a hosted platform still processes traffic under its own terms.

The app shows model scores only as relative outputs from a synthetic set. They are not
calibrated confidence values, validated commercial qualification, or performance claims.
An ambiguous result is flagged for human review.

### Data pipeline

1. Read and validate synthetic `text` / `label` rows.
2. Fit TF-IDF and logistic regression in memory when the app starts.
3. Normalize and vectorize the submitted project brief.
4. Rank the five service-line labels and extract the strongest positive terms present in
   the brief.
5. Apply deterministic completeness checks and display route-specific discovery questions.

### Execution plan for a real pilot

1. Confirm the service taxonomy and question templates with Cynbit service owners.
2. Approve a consent, retention, access, and de-identification policy before using real
   enquiries.
3. Create a time-separated evaluation set and measure macro-F1, per-class recall, top-2
   routing accuracy, abstention rate, and performance by enquiry type.
4. Shadow-test recommendations with reviewers. Track correction rate and time to first
   qualified response against a documented baseline.
5. Pilot with a human approval step; monitor drift, false routing, and reviewer override.
   Consider CRM integration or grounded LLM drafting only after evidence supports it.

## Testing

```powershell
python -m unittest discover -s tests -v
```

Tests check representative routing for every class, ranking and score shape, explanation
terms, short-input validation, and completeness scoring.

## Limitations and responsible use

- Synthetic examples do not establish accuracy on real enquiries.
- Service categories overlap; route scores can be ambiguous and need human confirmation.
- The completeness score measures only whether discovery fields were filled.
- Do not use the tool to rank people, infer sensitive attributes, or make commercial
  commitments.
- Real-data use requires explicit authorization, minimization, privacy review, and an
  agreed retention policy.

## Sources

- Cynbit Technologies, [Services](https://www.cynbit.in/services), accessed 29 October 2026.
- Cynbit Technologies, [BitElevate](https://www.cynbit.in/bitelevate), accessed 29 October 2026.
