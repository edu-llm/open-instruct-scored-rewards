"""Registered constructs and deterministic scoring helpers.

The registry deliberately distinguishes an observable tutor opportunity from a
learner action or learning outcome. Only the three pilot constructs are valid
probe labels; the wider taxonomy lives in ``METRIC_TAXONOMY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvidenceLevel(str, Enum):
    TUTOR_OPPORTUNITY = "tutor_opportunity"
    LEARNER_ENACTMENT = "learner_enactment"
    LEARNING_OUTCOME = "learning_outcome"


class StudentState(str, Enum):
    NO_ATTEMPT = "no_attempt"
    ATTEMPT = "attempt"
    REASONED_ATTEMPT = "reasoned_attempt"


class HelpLevel(str, Enum):
    QUESTION = "question"
    STRATEGY_HINT = "strategy_hint"
    WORKED_STEP = "worked_step"
    DIRECT_ANSWER = "direct_answer"


HELP_ORDINAL = {
    HelpLevel.QUESTION: 0,
    HelpLevel.STRATEGY_HINT: 1,
    HelpLevel.WORKED_STEP: 2,
    HelpLevel.DIRECT_ANSWER: 3,
}


@dataclass(frozen=True)
class Construct:
    key: str
    name: str
    level: EvidenceLevel
    unit_of_analysis: str
    positive_anchor: str
    negative_anchor: str
    required_fields: tuple[str, ...]
    evidence_limit: str

    def validate(self, record: dict[str, Any]) -> None:
        missing = [field for field in self.required_fields if record.get(field) in (None, "")]
        if missing:
            raise ValueError(f"{self.key}: missing required fields {missing}")


CONSTRUCTS: dict[str, Construct] = {
    "diagnostic_localization": Construct(
        key="diagnostic_localization",
        name="Diagnostic localization",
        level=EvidenceLevel.TUTOR_OPPORTUNITY,
        unit_of_analysis="learner attempt plus tutor response",
        positive_anchor="Names the first causal error or faulty proposition and its conflict with the reference.",
        negative_anchor="Generic feedback, final-answer-only diagnosis, invented intent, or downstream correction.",
        required_fields=("student_before", "reference", "candidate_action"),
        evidence_limit="A correct diagnosis is an instructional opportunity, not evidence that the learner repaired it.",
    ),
    "contingent_scaffolding": Construct(
        key="contingent_scaffolding",
        name="Contingent scaffolding",
        level=EvidenceLevel.TUTOR_OPPORTUNITY,
        unit_of_analysis="learner state plus prescribed and observed tutor help levels",
        positive_anchor="The response's help level matches the need demonstrated in this scenario.",
        negative_anchor="The response over-supports or under-supports this learner state.",
        required_fields=("student_state", "prescribed_action", "candidate_action"),
        evidence_limit="One turn measures matching, not escalation, fading, or learning.",
    ),
    "generative_elicitation": Construct(
        key="generative_elicitation",
        name="Generative elicitation",
        level=EvidenceLevel.TUTOR_OPPORTUNITY,
        unit_of_analysis="tutor response; learner enactment is scored on the next turn",
        positive_anchor="Requires retrieval, justification, self-explanation, comparison, or production of a next step.",
        negative_anchor="Rhetorical/check questions, copying, generic requests, or tutor-performed explanation.",
        required_fields=("candidate_action",),
        evidence_limit="The invitation does not establish that the learner engaged or learned.",
    ),
}


GENERATIVE_ACTIONS = {
    "retrieve",
    "justify",
    "self_explain",
    "compare",
    "produce_next_step",
    HelpLevel.QUESTION.value,
    HelpLevel.STRATEGY_HINT.value,
}

NON_GENERATIVE_ACTIONS = {
    "rhetorical_question",
    "comprehension_check",
    "generic_explain",
    "tutor_explanation",
    HelpLevel.WORKED_STEP.value,
    HelpLevel.DIRECT_ANSWER.value,
}


def coerce_student_state(value: str | StudentState) -> StudentState:
    try:
        return StudentState(value)
    except ValueError as exc:
        raise ValueError(f"unknown student_state {value!r}; expected {[s.value for s in StudentState]}") from exc


def coerce_help_level(value: str | HelpLevel) -> HelpLevel:
    try:
        return HelpLevel(value)
    except ValueError as exc:
        raise ValueError(f"unknown help level {value!r}; expected {[h.value for h in HelpLevel]}") from exc


def score_contingent_scaffolding(prescribed: str | HelpLevel, candidate: str | HelpLevel) -> int:
    """Binary scenario-conditioned match; lower help is not inherently better."""
    return int(coerce_help_level(prescribed) == coerce_help_level(candidate))


def scaffolding_distance(prescribed: str | HelpLevel, candidate: str | HelpLevel) -> int:
    """Signed help error: positive is over-support, negative is under-support."""
    target = HELP_ORDINAL[coerce_help_level(prescribed)]
    observed = HELP_ORDINAL[coerce_help_level(candidate)]
    return observed - target


def score_generative_elicitation(candidate_action: str) -> int:
    action = candidate_action.strip().lower()
    if action in GENERATIVE_ACTIONS:
        return 1
    if action in NON_GENERATIVE_ACTIONS:
        return 0
    raise ValueError(
        f"unknown generative action {candidate_action!r}; "
        f"expected one of {sorted(GENERATIVE_ACTIONS | NON_GENERATIVE_ACTIONS)}"
    )


def score_record(record: dict[str, Any]) -> int:
    """Return a deterministic binary label when the schema carries enough evidence.

    Diagnostic localization is never inferred from prose by keyword. It must be
    human- or dataset-annotated as ``diagnosis_correct`` and ``first_error_located``.
    """
    concept = str(record.get("concept", ""))
    construct = CONSTRUCTS.get(concept)
    if construct is None:
        raise ValueError(f"unknown concept {concept!r}; expected {sorted(CONSTRUCTS)}")
    construct.validate(record)

    if concept == "diagnostic_localization":
        if not isinstance(record.get("diagnosis_correct"), bool):
            raise ValueError("diagnostic_localization requires boolean diagnosis_correct")
        if not isinstance(record.get("first_error_located"), bool):
            raise ValueError("diagnostic_localization requires boolean first_error_located")
        return int(record["diagnosis_correct"] and record["first_error_located"])
    if concept == "contingent_scaffolding":
        return score_contingent_scaffolding(record["prescribed_action"], record["candidate_action"])
    return score_generative_elicitation(str(record["candidate_action"]))


def validate_label(record: dict[str, Any]) -> None:
    expected = score_record(record)
    supplied = record.get("label")
    if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {supplied!r}")
    if int(supplied) != expected:
        raise ValueError(f"{record.get('variant_id', '<unknown>')}: label={supplied} but deterministic score={expected}")
