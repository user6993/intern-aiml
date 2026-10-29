from __future__ import annotations

import streamlit as st

from router import ROUTES, IntakeRouter, score_completeness


st.set_page_config(
    page_title="Cynbit | Project Intake Router",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e8f0f7;
        --muted: #9aabba;
        --panel: #111d2b;
        --line: #26384a;
        --accent: #55e0bc;
    }
    .stApp { background: #0a1420; color: var(--ink); }
    [data-testid="stSidebar"] { background: #0d1926; border-right: 1px solid var(--line); }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(120deg, #10243a 0%, #142d3c 60%, #10221f 100%);
        margin-bottom: 1.5rem;
    }
    .hero-kicker { color: var(--accent); font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { margin: .35rem 0; color: var(--ink); font-size: 2rem; }
    .hero p, .muted { color: var(--muted); }
    .route-card {
        padding: 1.1rem 1.25rem;
        border: 1px solid #28534e;
        border-radius: 14px;
        background: #10251f;
        margin: .5rem 0 1rem;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 1rem;
        border-radius: 12px;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #06231e;
        border: 0;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_router() -> IntakeRouter:
    return IntakeRouter.from_csv()


router = get_router()


def load_example() -> None:
    examples = {
        "AI operations": {
            "project_brief": (
                "Our support team handles repetitive order-status and return questions "
                "across email. We want an AI workflow that drafts grounded replies and "
                "routes exceptions to an agent, with an approval step."
            ),
            "desired_outcome": "Reduce manual handling while keeping a person in control.",
            "current_stack": "Shopify, shared inbox, and an internal order API",
            "timeline": "Pilot in 6-8 weeks",
            "success_metric": "Lower average handling time without increasing escalations",
        },
        "Cloud platform": {
            "project_brief": (
                "We need to modernize a monolithic web application, move workloads to "
                "the cloud, add CI/CD and improve reliability during seasonal traffic."
            ),
            "desired_outcome": "A secure, scalable cloud platform with safer releases.",
            "current_stack": "Django, PostgreSQL, virtual machines, manual deployments",
            "timeline": "Planning a phased migration this quarter",
            "success_metric": "Release frequency and service availability",
        },
        "Learning product": {
            "project_brief": (
                "Our coding learners need a guided path through cloud and AI topics. "
                "We want to suggest the next practical exercise based on their completed "
                "lessons and quiz results."
            ),
            "desired_outcome": "More relevant practice and a clear learning progression.",
            "current_stack": "Learning portal with course and quiz records",
            "timeline": "Build a small cohort pilot",
            "success_metric": "Lesson completion and exercise mastery",
        },
    }
    selected = st.session_state.get("example_choice")
    if selected in examples:
        for field, value in examples[selected].items():
            st.session_state[field] = value


def render_intake_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">AI/ML opportunity prototype · Cynbit</div>
          <h1>Turn a project brief into a useful first conversation.</h1>
          <p>Route a hypothetical enquiry to a relevant capability, see why it matched,
          and identify the discovery questions still unanswered.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Prototype only: the model is trained on synthetic examples, not Cynbit customer "
        "data. Its scores are not calibrated business probabilities."
    )

    sample_col, _ = st.columns([1, 2])
    with sample_col:
        st.selectbox(
            "Load a synthetic example",
            ["Choose an example", "AI operations", "Cloud platform", "Learning product"],
            key="example_choice",
            on_change=load_example,
        )

    with st.form("intake_form"):
        project_brief = st.text_area(
            "What are you trying to build or improve? *",
            key="project_brief",
            height=130,
            placeholder="Describe the current challenge, users, and desired solution.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            desired_outcome = st.text_input(
                "Desired outcome", key="desired_outcome", placeholder="What should change?"
            )
            current_stack = st.text_input(
                "Current tools or systems",
                key="current_stack",
                placeholder="Frameworks, platforms, data sources, or 'not sure'",
            )
        with col_b:
            timeline = st.text_input(
                "Target timeline", key="timeline", placeholder="For example, pilot this quarter"
            )
            success_metric = st.text_input(
                "How would you measure success?",
                key="success_metric",
                placeholder="For example, response time or completion rate",
            )
        submitted = st.form_submit_button("Analyze project brief", type="primary")

    if not submitted:
        st.info("Enter a brief or load a synthetic example, then run the analysis.")
        return

    try:
        result = router.predict(project_brief)
    except ValueError as exc:
        st.error(str(exc))
        return

    completeness, missing = score_completeness(
        {
            "project_brief": project_brief,
            "desired_outcome": desired_outcome,
            "current_stack": current_stack,
            "timeline": timeline,
            "success_metric": success_metric,
        }
    )
    route = ROUTES[result.label]
    top_score = result.scores[0][1]
    second_score = result.scores[1][1]
    needs_review = top_score < 0.45 or top_score - second_score < 0.10

    st.divider()
    result_col, completeness_col = st.columns([1.5, 1])
    with result_col:
        st.subheader("Suggested service route")
        status = "Human review recommended" if needs_review else "Suggested match"
        st.markdown(
            f"""
            <div class="route-card">
              <div class="hero-kicker">{status}</div>
              <h2>{route.name}</h2>
              <p>{route.summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Why this route:** {route.reason}")
        if result.evidence:
            st.markdown("**Relevant terms in the brief:** " + " · ".join(f"`{term}`" for term in result.evidence))
        else:
            st.caption("No single high-signal term was found; review the route manually.")

    with completeness_col:
        st.subheader("Discovery readiness")
        st.metric("Brief completeness", f"{completeness}%")
        st.progress(completeness / 100)
        st.caption("A checklist score, not a project feasibility estimate.")
        if missing:
            st.markdown("**Useful details to add next:**")
            for item in missing:
                st.markdown(f"- {item}")
        else:
            st.success("All five discovery fields have a response.")

    with st.expander("Compare route scores", expanded=False):
        st.caption(
            "Relative outputs from a small synthetic training set. They are not calibrated "
            "probabilities or a commercial qualification score."
        )
        st.bar_chart(
            {
                "Relative model score": {
                    ROUTES[label].name: score for label, score in result.scores
                }
            },
            horizontal=True,
        )

    st.subheader("Suggested discovery questions")
    for question in route.questions:
        st.markdown(f"- {question}")
    st.info("Treat this as a first-pass assistant. A person should confirm the route and scope.")


def render_opportunities_page() -> None:
    st.markdown("## Where AI/ML could create value")
    st.write(
        "These are outside-in hypotheses based on Cynbit's public service areas, not claims "
        "about its current internal processes. Scores are directional (1 = low, 5 = high); "
        "implementation time is scored higher when a small pilot could be delivered sooner."
    )
    rows = [
        {
            "Opportunity": "Project-intake routing & discovery",
            "Business value": 4,
            "Feasibility": 5,
            "Innovation": 3,
            "Pilot speed": 5,
            "Data access": 4,
            "Potential impact": 4,
        },
        {
            "Opportunity": "RFP / proposal evidence assistant",
            "Business value": 4,
            "Feasibility": 3,
            "Innovation": 4,
            "Pilot speed": 2,
            "Data access": 2,
            "Potential impact": 4,
        },
        {
            "Opportunity": "Engineering QA & test-case copilot",
            "Business value": 4,
            "Feasibility": 4,
            "Innovation": 3,
            "Pilot speed": 4,
            "Data access": 3,
            "Potential impact": 4,
        },
        {
            "Opportunity": "Cloud reliability / FinOps signals",
            "Business value": 3,
            "Feasibility": 3,
            "Innovation": 5,
            "Pilot speed": 3,
            "Data access": 2,
            "Potential impact": 4,
        },
        {
            "Opportunity": "BitElevate adaptive practice paths",
            "Business value": 3,
            "Feasibility": 3,
            "Innovation": 4,
            "Pilot speed": 3,
            "Data access": 2,
            "Potential impact": 3,
        },
    ]
    import pandas as pd

    frame = pd.DataFrame(rows).set_index("Opportunity")
    st.dataframe(frame, use_container_width=True)
    st.markdown(
        """
        **Selection — project-intake routing & discovery.** It is a bounded workflow that
        directly maps to Cynbit's publicly described AI automation, software, cloud, and
        data/CRM services. A useful first demo can be built without client data or an LLM;
        a real pilot would require consented, de-identified historical enquiries and human
        review. It can improve consistency at the first response while generating evidence
        about which route and discovery fields are actually useful.
        """
    )
    st.caption(
        "Directional scoring only. No internal conversion, workload, or revenue figures "
        "were available or assumed."
    )


def render_model_notes_page() -> None:
    st.markdown("## Technical approach & execution plan")
    st.markdown(
        """
        **Problem.** A services enquiry often arrives as unstructured prose. Help a human
        identify the likely capability, see evidence for the recommendation, and spot missing
        discovery information—without automating a commercial decision.

        **Model.** A scikit-learn pipeline uses word/phrase TF-IDF features and multinomial
        logistic regression to select among five service-line labels. It is intentionally
        interpretable and cheap to run locally. Per-brief weighted terms provide a simple
        explanation. The displayed class scores are only relative outputs from synthetic
        training examples; they are not production confidence estimates.

        **Data pipeline.** Curated, fictional enquiries are stored in
        `data/synthetic_intakes.csv` with a service-line label. The app loads the CSV,
        fits the pipeline in memory, vectorizes the submitted brief, and returns a ranked
        route. The separate completeness checklist and route-specific discovery questions
        are deterministic rules. The app has no database, external API, or application-level
        persistence. Do not enter real personal or confidential client information.

        **Prompt / LLM architecture.** No prompt or LLM is used in this MVP. A generative
        model would add cost and introduce unsupported claims without a curated knowledge
        base. A later grounded assistant could draft discovery questions only from approved
        service documentation, cite those sources, and require human approval.

        **Execution plan.**
        1. Validate the taxonomy and question checklist with Cynbit service owners.
        2. Establish consent, retention, access controls, and a de-identification policy.
        3. Run a small retrospective test on approved enquiries; split evaluation by time,
           measure macro-F1, per-class recall, top-2 routing accuracy, and abstention rate.
        4. Shadow-test recommendations with reviewers; track correction rate and time-to-
           first-qualified-response against a baseline.
        5. Pilot behind human approval, monitor drift and errors, and only then consider a
           CRM integration or an LLM for grounded drafting.

        **Guardrails.** The app does not score a lead's worth, infer sensitive traits, make
        commitments, or persist input. A low/ambiguous match is flagged for human review.
        The synthetic model is a functional proof of concept—not evidence of live business
        impact.
        """
    )


st.sidebar.markdown("### Cynbit · AI/ML assessment")
page = st.sidebar.radio(
    "Workspace",
    ["Analyze an intake", "Opportunity analysis", "Technical approach"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic-data demonstration · no external AI service")

if page == "Analyze an intake":
    render_intake_page()
elif page == "Opportunity analysis":
    render_opportunities_page()
else:
    render_model_notes_page()

