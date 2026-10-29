from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "synthetic_intakes.csv"


@dataclass(frozen=True)
class Route:
    name: str
    summary: str
    reason: str
    questions: tuple[str, ...]


ROUTES: dict[str, Route] = {
    "ai_automation": Route(
        name="AI automation & agents",
        summary="Explore an AI-assisted workflow with clear human oversight.",
        reason="The brief emphasizes repetitive work, AI assistance, agents, or workflow automation.",
        questions=(
            "Which steps are repetitive, and what exceptions must stay with a person?",
            "What approved data or knowledge may the workflow use?",
            "How will you measure quality, time saved, and escalation safety?",
        ),
    ),
    "software_platform": Route(
        name="Software engineering & SaaS platforms",
        summary="Clarify product scope, users, integrations, and the delivery shape.",
        reason="The brief points to building or extending an application, product, or platform.",
        questions=(
            "Who are the primary users and what is the most important user journey?",
            "Which integrations, roles, or non-functional requirements are essential?",
            "What is the smallest useful release and how will it be validated?",
        ),
    ),
    "cloud_devops": Route(
        name="Cloud, DevOps & modernization",
        summary="Assess architecture, deployment, reliability, and migration constraints.",
        reason="The brief highlights cloud infrastructure, migration, CI/CD, reliability, or scale.",
        questions=(
            "What are the current hosting, deployment, and operational constraints?",
            "Which availability, security, compliance, or data-residency requirements apply?",
            "What workloads and dependencies are in scope for the first phase?",
        ),
    ),
    "data_marketing": Route(
        name="Data, CRM & growth engineering",
        summary="Connect data, customer workflows, and measurable growth outcomes.",
        reason="The brief emphasizes analytics, CRM, marketing automation, SEO, or funnel performance.",
        questions=(
            "Which data sources and customer systems need to be connected?",
            "What is the baseline and target for the selected funnel or marketing metric?",
            "How should consent, attribution, and data quality be handled?",
        ),
    ),
    "learning_tech": Route(
        name="BitElevate & learning technology",
        summary="Shape a learner-centered experience with measurable skill progression.",
        reason="The brief describes learners, courses, assessments, practice, or education products.",
        questions=(
            "Who are the learners and what skill should they demonstrate?",
            "Which learning events can be used, with appropriate consent and retention?",
            "How will you measure mastery and avoid recommending an unsuitable next step?",
        ),
    ),
}


@dataclass(frozen=True)
class Prediction:
    label: str
    scores: tuple[tuple[str, float], ...]
    evidence: tuple[str, ...]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def score_completeness(intake: Mapping[str, str]) -> tuple[int, list[str]]:
    fields = (
        ("project_brief", "Problem and users"),
        ("desired_outcome", "Desired outcome"),
        ("current_stack", "Current tools or systems"),
        ("timeline", "Target timeline"),
        ("success_metric", "Success measure"),
    )
    weights = (40, 20, 15, 10, 15)
    score = 0
    missing: list[str] = []
    for (field, label), weight in zip(fields, weights):
        value = _clean_text(intake.get(field, ""))
        if value and (field != "project_brief" or len(value) >= 12):
            score += weight
        else:
            missing.append(label)
    return score, missing


class IntakeRouter:
    def __init__(self, model: Pipeline):
        self._model = model

    @classmethod
    def from_csv(cls, path: Path = DATA_PATH) -> "IntakeRouter":
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic training data not found: {path}")
        data = pd.read_csv(path)
        required = {"text", "label"}
        if not required.issubset(data.columns):
            raise ValueError(f"Training data must include columns: {sorted(required)}")
        if data[list(required)].isna().any().any():
            raise ValueError("Training data contains missing text or label values.")
        labels = set(data["label"].astype(str))
        if labels != set(ROUTES):
            raise ValueError(
                f"Training labels must exactly match route labels; found: {sorted(labels)}"
            )
        if any(not _clean_text(value) for value in data["text"].astype(str)):
            raise ValueError("Training data includes an empty example.")

        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=12000,
                        stop_words="english",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(data["text"].astype(str), data["label"].astype(str))
        return cls(model)

    def predict(self, text: str) -> Prediction:
        cleaned = _clean_text(text)
        if len(cleaned) < 12:
            raise ValueError("Please enter a project brief with at least 12 characters.")

        probabilities = self._model.predict_proba([cleaned])[0]
        classes = self._model.named_steps["classifier"].classes_
        ranked = sorted(
            ((str(label), float(score)) for label, score in zip(classes, probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label = ranked[0][0]
        evidence = self._evidence(cleaned, top_label)
        return Prediction(label=top_label, scores=tuple(ranked), evidence=evidence)

    def _evidence(self, text: str, label: str, limit: int = 5) -> tuple[str, ...]:
        vectorizer = self._model.named_steps["tfidf"]
        classifier = self._model.named_steps["classifier"]
        features = vectorizer.get_feature_names_out()
        row = vectorizer.transform([text]).tocsr()
        class_index = list(classifier.classes_).index(label)
        weights = classifier.coef_[class_index]
        contributions = [
            (features[column], float(value * weights[column]))
            for column, value in zip(row.indices, row.data)
            if value * weights[column] > 0
            and re.search(
                rf"(?<!\w){re.escape(features[column])}(?!\w)",
                _clean_text(text).lower(),
            )
        ]
        contributions.sort(key=lambda item: item[1], reverse=True)
        return tuple(term for term, _ in contributions[:limit])
