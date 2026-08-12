"""The metrics, the scenarios that prescribe their target, and the reward built from both.

READ PLAN.md FIRST. The short version of why this file looks the way it does:

A reward that pays for withholding is maximised by saying nothing, and this project already has
the receipts - `pedagogy_rm`'s withholding-shaped dimensions had an effective rank of 2.9, their
joint optimum was a twelve-word question, and a trained policy found it. The successor design that
took "best K of N good qualities" failed the same way for the same reason: enumerating its maximum
gave a 56-way tie whose cheapest member was a fifteen-word template.

So the spine here is not a quality score at all. It is a MATCH between the help the tutor gave and
the help the scenario prescribed. The atlas singles this out as "the only construct whose optimum a
fixed policy cannot reach by style alone, because the target moves with the learner", and
`check_reward.py` confirms it: the best fixed rung scores 0.58 of 1.00 across our scenario mix, so
there is no template to collapse onto.

The one condition the atlas attaches is absolute: "Prescribed targets must come from the dataset. A
proxy target inferred by a model reintroduces the 18%-agreement problem with a machine-made answer
key." Hence SCENARIOS. We script the student's state, so the right help level is a property of the
generator, decided before any tutor turn exists. Never infer a target from a turn.
"""

from __future__ import annotations

import dataclasses

# Graesser's ladder, ordered from least to most tutor control. The ORDER is load-bearing: the
# reward gives partial credit for an adjacent rung, so a wrong order silently changes the metric.
RUNGS: tuple[str, ...] = ("pump", "hint", "prompt", "tell_step", "tell_answer")
# The shipped label rubric records assistance on three ordered levels, not all
# five named Graesser rungs. Keep that compression explicit at the reward
# boundary instead of pretending a 1-3 head can distinguish pump from hint.
TARGET_LEVEL: dict[str, float] = {
    "pump": 1.0,
    "hint": 1.0,
    "prompt": 2.0,
    "tell_step": 3.0,
    "tell_answer": 3.0,
}

RUNG_ANCHORS: dict[str, str] = {
    "pump": "Invites the student to continue with no new content. 'Go on', 'what next?', 'and so?'",
    "hint": "Points at where to look or what to consider, without naming the operation to perform.",
    "prompt": "Names the specific thing to do or account for, but leaves the student to do it.",
    "tell_step": "Carries out one step of the solution for the student, and shows the working.",
    "tell_answer": "States the final answer, or narrows the options until only one remains.",
}


@dataclasses.dataclass(frozen=True)
class Metric:
    """One thing a rater judges about one tutor turn.

    The atlas fields are carried through rather than summarised because they are what makes a
    label reproducible. `positive_anchor` and `hard_negative` go on screen in the labelling app;
    `says_only_negative` catches the failure where a turn talks ABOUT a construct without
    instantiating it, which is the single most common false positive in the atlas corpus.

    `eligibility` mirrors ontology.yaml. Anything `not_eligible_*` may be collected and reported
    but must never enter the reward - the first draft of this project put two such constructs in
    its reward and one in a gate, having cited the same file for a different column.
    """

    key: str
    question: str
    atlas: str | None
    eligibility: str
    positive_anchor: str
    hard_negative: str
    surface_confound: str
    length_coupling: str  # expected sign against log words: "-", "0", or "+"


# ---------------------------------------------------------------------------
# What the tutor supplies. Multi-label presence, scored as a CAPPED COUNT.
#
# A cap rather than "best 2 of 6" because an argmax over noisy continuous scores is biased upward:
# at a probe RMSE of 0.4 the best-2-of-6 adds +0.77 of pure noise, which swallows 23% of the gap
# between a good turn and a bad one. Thresholding first and counting second measures 0.000 bias in
# the same simulation. The cap still encodes the real finding - Chi et al. 2001, that explaining
# and eliciting are substitutes rather than addends - without paying for the selection.
# ---------------------------------------------------------------------------
CONTRIBUTIONS: tuple[Metric, ...] = (
    Metric(
        key="diagnoses_the_error",
        question="Does the turn say what is WRONG with the student's step, rather than only that something is wrong?",
        atlas="elaborated_feedback_specificity",
        eligibility="eligible_with_guard",
        positive_anchor="'You subtracted 3 from both sides, but the 3 is multiplied, not added.' Names the error.",
        hard_negative="'That's not quite right, try again.' A verdict with no account of what went wrong.",
        surface_confound="the word 'because' appearing anywhere",
        length_coupling="+",
    ),
    Metric(
        key="applies_principle_here",
        question="Does the turn name the relevant rule or relation AND say how it applies to this problem?",
        atlas="elaborated_feedback_specificity",
        eligibility="eligible_with_guard",
        positive_anchor="'Momentum is conserved in a collision, so the two masses after must sum to the 5 kg before.'",
        hard_negative="'Remember conservation of momentum.' Names the rule and leaves it unattached to the problem.",
        surface_confound="technical vocabulary density",
        length_coupling="+",
    ),
    Metric(
        key="models_one_step",
        question="Does the turn carry out one step of the solution in full, showing the working?",
        atlas="worked_example_provision",
        eligibility="eligible_with_guard",
        positive_anchor="'Dividing both sides by 4: 4x/4 = 12/4, so x = 3.' One step, worked.",
        hard_negative="'Then you divide by 4 and you get the answer.' Describes a step without performing it.",
        surface_confound="presence of an equals sign",
        length_coupling="+",
    ),
    Metric(
        key="offers_parallel_case",
        question="Does the turn give a second, simpler or contrasting case AND indicate what to compare?",
        atlas="analogical_case_comparison",
        eligibility="eligible_with_guard",
        positive_anchor="'If it were 2x = 6 you would divide by 2. What is different about 2x + 1 = 6?'",
        hard_negative="'This is like the last problem.' Asserts an analogy without giving the case or the mapping.",
        surface_confound="the words 'like' or 'imagine'",
        length_coupling="+",
    ),
    Metric(
        key="names_the_target",
        question="Does the turn say what the student is trying to establish RIGHT NOW - the sub-goal, not the action?",
        atlas="subgoal_labeled_structure",
        eligibility="eligible_with_guard",
        positive_anchor="'We are trying to get x on its own on one side. That is the goal of this step.'",
        hard_negative="'Now do the next bit.' Names an action with no goal attached.",
        surface_confound="the phrase 'we want'",
        length_coupling="+",
    ),
    Metric(
        key="asks_recall_of_established",
        question="Does the turn ask the student to state from memory something the dialogue already established?",
        atlas="retrieval_practice_opportunity",
        eligibility="eligible_with_guard",
        positive_anchor="'Without scrolling up - what was the rule we used for the first one?'",
        hard_negative="'Remember, we balance the oxygens first.' Supplies the answer inside the same turn.",
        surface_confound="the word 'remember'",
        length_coupling="0",
    ),
)

# ---------------------------------------------------------------------------
# Anchoring and validity. Scored individually, not capped.
# ---------------------------------------------------------------------------
ANCHORING: tuple[Metric, ...] = (
    Metric(
        key="locates_student_object",
        question="Does the turn point at a specific number, step, operation or word in what the student wrote?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="'Your second line says 2x = 10.' Quotes the student's own object.",
        hard_negative="'Check your algebra.' Could follow any wrong answer to any problem.",
        surface_confound="quotation marks",
        length_coupling="0",
    ),
    Metric(
        key="verdict_with_referent",
        question="Does the turn say whether a NAMED step of the student's is right or wrong?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="'Your first line is right; the sign in the second one is not.'",
        hard_negative="'Great work!' Approval with nothing named, and nothing verified.",
        surface_confound="exclamation marks",
        length_coupling="+",
    ),
    Metric(
        key="demand_is_specific",
        question="If the turn asks the student to do something, does it name a particular quantity, step or claim?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="'What does the 3 represent in your equation?' Names the object of the demand.",
        hard_negative="'Why?' or 'Can you explain?' The student must work out what is being asked.",
        surface_confound="question marks",
        length_coupling="0",
    ),
)

# ---------------------------------------------------------------------------
# Penalties. SUBTRACTIVE, never multiplicative gates.
#
# The first draft multiplied by three gates and it failed twice over. Gates are absences, and an
# absence gets harder to hold as a turn grows, so multiplying by them made expected reward PEAK at
# 30 words while quality was still climbing at 100 - rebuilding the brevity pressure the design
# existed to remove. They also zeroed 14.3% of samples, which in a GRPO group of eight tripled the
# group standard deviation and left only 29% of groups with clean signal. Subtracting costs the
# same behaviour and neither distorts the length gradient nor destroys the group.
# ---------------------------------------------------------------------------
PENALTIES: tuple[Metric, ...] = (
    Metric(
        key="reference_conflict",
        question="Does anything the turn asserts conflict with the reference solution shown to you?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="Turn says the answer is 7; the reference says 5. Conflict.",
        hard_negative="Turn asserts nothing the reference can check. Not a conflict - leave unmarked.",
        surface_confound="length, strongly - more words means more checkable claims",
        length_coupling="-",
    ),
    Metric(
        key="dialogue_conflict",
        question="Does the turn contradict or ignore something already settled earlier in the dialogue?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="Re-asks a question the student already answered correctly two turns ago.",
        hard_negative="Repeats a point for emphasis without contradicting it.",
        surface_confound="dialogue position - later turns have more to contradict",
        length_coupling="-",
    ),
    Metric(
        key="not_single_focus",
        question="Is the student being asked to attend to more than one thing?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="Corrects the sign error, explains the rule, and asks about units. Three things.",
        hard_negative="One correction stated in two sentences. Still one thing.",
        surface_confound="sentence count",
        length_coupling="-",
    ),
    Metric(
        key="empty_praise",
        question="Does the turn praise the student or their ability without naming anything specific they did?",
        atlas=None,
        eligibility="eligible_with_guard",
        positive_anchor="'Great job, you're really good at this!' Praises the person, names nothing.",
        hard_negative="'Good - factoring first was the right call.' Praise with a referent. Not empty.",
        surface_confound="second-person address",
        length_coupling="-",
    ),
)

# ---------------------------------------------------------------------------
# Collected and reported. NEVER optimised. Each line is a specific prohibition
# from ontology.yaml, and the first draft of this plan violated three of them.
# ---------------------------------------------------------------------------
MEASURED_ONLY: tuple[Metric, ...] = (
    Metric(
        key="answer_withheld",
        question="Does the turn stop short of stating the final answer?",
        atlas="answer_withholding_level",
        eligibility="not_eligible_degenerate",
        positive_anchor="Asks a question and stops. Answer not stated.",
        hard_negative="'So x = 5.' Stated.",
        surface_confound="turn length, negatively",
        length_coupling="-",
    ),
    Metric(
        key="icap_class",
        question="Which ICAP class does the requested activity fall in: passive, active, constructive, interactive?",
        atlas="icap_engagement_class",
        eligibility="not_eligible_unreliable",
        positive_anchor="Report as four counts per arm. Never average, never regress on it.",
        hard_negative="Treating it as an interval scale, which is what a reward would silently do.",
        surface_confound="imperative mood",
        length_coupling="0",
    ),
)

ALL: tuple[Metric, ...] = (*CONTRIBUTIONS, *ANCHORING, *PENALTIES, *MEASURED_ONLY)
BY_KEY: dict[str, Metric] = {m.key: m for m in ALL}

SCORED: tuple[str, ...] = tuple(m.key for m in (*CONTRIBUTIONS, *ANCHORING, *PENALTIES))


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A scripted student state, and the help level it prescribes.

    `target` IS THE ANSWER KEY and it is set here, before any tutor turn exists. It must never be
    inferred from a turn: tutors pick the same next action in 18% of cases, and a tutor agrees with
    outside raters about their own move at kappa 0.34, so an inferred target is a machine-made
    answer key with a coin-flip inside it. Prescribed, the same judgement reaches kappa 0.55-0.70.

    `check` is what a rater or judge is asked in order to confirm the student MODEL actually
    occupied the state. A generation whose student drifted is discarded rather than relabelled,
    because a scenario the student did not occupy has the wrong key attached.
    """

    key: str
    student_state: str
    target: str
    rationale: str
    check: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="lost",
        student_state="Has produced no attempt and says they do not know where to start.",
        target="tell_step",
        rationale="No foothold to build on. Withholding here is the inversion contingency exists to catch: "
        "a learner with nothing to work from needs more explicit support, not less.",
        check="Did the student produce no mathematical content and express not knowing how to begin?",
    ),
    Scenario(
        key="slip",
        student_state="Has the correct method but has made an arithmetic error in one step.",
        target="hint",
        rationale="The method is sound, so the cheapest sufficient help is pointing at where to look. "
        "Explaining the method would be redundant with what the student has already shown.",
        check="Did the student show a correct approach containing exactly one computational mistake?",
    ),
    Scenario(
        key="confident_wrong",
        student_state="Has produced a complete method that is wrong in principle, and states it confidently.",
        target="prompt",
        rationale="A hint will be absorbed into the wrong frame. The student must be brought to test "
        "their own method against something, which is what naming the thing to account for does.",
        check="Did the student give a complete but principally incorrect method without hedging?",
    ),
    Scenario(
        key="partial",
        student_state="Has taken a correct first step and then stalled without continuing.",
        target="hint",
        rationale="Progress exists and has stopped. Where to look next is the missing piece, not the method.",
        check="Did the student complete at least one correct step and then stop?",
    ),
    Scenario(
        key="reasoned_attempt",
        student_state="Has shown correct method and reasoning, with a small gap remaining.",
        target="pump",
        rationale="The student is carrying the work. Anything more than an invitation to continue takes "
        "over reasoning they have demonstrated they can do.",
        check="Did the student show both a correct answer or step AND the reasoning behind it?",
    ),
    Scenario(
        key="asks_outright",
        student_state="Asks for the answer directly without attempting the problem.",
        target="prompt",
        rationale="Not lost - unwilling. Naming a specific first thing to do is the smallest move that "
        "returns the work without refusing to help.",
        check="Did the student request the answer or a solution without showing any attempt?",
    ),
)

SCENARIO_BY_KEY: dict[str, Scenario] = {s.key: s for s in SCENARIOS}

# Weights. Set so no single term dominates the way pedagogy_rm's length term did - measured at
# 59.6% / 22.7% / 6.5% / 6.5% / 6.2% of reward variance over the scenario mix in check_reward.py.
# Contingency carries the most on purpose: it is the spine, and it is the only term a fixed policy
# cannot satisfy by style.
WEIGHTS: dict[str, float] = {
    "contingency": 2.0,
    "contributions": 1.5,
    "locates_student_object": 0.5,
    "verdict_with_referent": 0.5,
    "demand_is_specific": 0.5,
}
PENALTY_WEIGHTS: dict[str, float] = {
    "reference_conflict": 1.0,
    "dialogue_conflict": 0.5,
    "not_single_focus": 0.5,
    "empty_praise": 0.5,
}
CONTRIBUTION_CAP = 2


def contingency(assistance_level: str, target: str) -> float:
    """How well the help given matches the help the scenario prescribed.

    Partial credit for an adjacent rung, because the ladder is ordinal and a neighbouring rung is a
    near miss rather than a different kind of error. Nothing beyond adjacent scores at all.
    """
    if assistance_level not in RUNGS:
        raise ValueError(f"{assistance_level!r} is not a rung; have {RUNGS}")
    if target not in RUNGS:
        raise ValueError(f"{target!r} is not a rung; have {RUNGS}")
    gap = abs(RUNGS.index(assistance_level) - RUNGS.index(target))
    return {0: 1.0, 1: 0.5}.get(gap, 0.0)


def reward(labels: dict[str, float], target: str, length_term: float = 0.0) -> float:
    """The scalar a policy is trained against.

    `labels` holds `assistance_level` plus any of SCORED. Missing keys count as absent, which is
    what a sparse one-metric-at-a-time labelling pass produces.
    """
    total = WEIGHTS["contingency"] * contingency(str(labels["assistance_level"]), target)

    present = sum(1 for m in CONTRIBUTIONS if labels.get(m.key))
    total += WEIGHTS["contributions"] * min(present, CONTRIBUTION_CAP) / CONTRIBUTION_CAP

    for key in ("locates_student_object", "verdict_with_referent", "demand_is_specific"):
        total += WEIGHTS[key] * float(bool(labels.get(key)))

    for key, weight in PENALTY_WEIGHTS.items():
        total -= weight * float(bool(labels.get(key)))

    return total + length_term


def probe_reward(labels: dict[str, float], target: str, length_term: float = 0.0) -> tuple[float, dict[str, float]]:
    """Compose continuous 1-3 probe predictions into the deployed reward.

    Agent labels use 1=absent/low, 2=partial/medium, 3=clear/high. Converting
    them with ``bool(value)`` would make even an explicit absence positive.
    Continuous presence preserves useful ranking inside a GRPO group while
    keeping every term within the bounds used by the reward design.
    """
    if target not in TARGET_LEVEL:
        raise ValueError(f"{target!r} is not a target level; have {sorted(TARGET_LEVEL)}")
    if "assistance_level" not in labels:
        raise ValueError("the deployed reward requires an assistance_level head")

    def presence(key: str) -> float:
        value = float(labels.get(key, 1.0))
        return min(1.0, max(0.0, (value - 1.0) / 2.0))

    actual = min(3.0, max(1.0, float(labels["assistance_level"])))
    target_value = TARGET_LEVEL[target]
    contingency_score = max(0.0, 1.0 - 0.5 * abs(actual - target_value))

    contribution_mass = sum(presence(metric.key) for metric in CONTRIBUTIONS)
    terms = {
        "contingency": WEIGHTS["contingency"] * contingency_score,
        "contributions": WEIGHTS["contributions"] * min(contribution_mass, CONTRIBUTION_CAP) / CONTRIBUTION_CAP,
        "locates_student_object": WEIGHTS["locates_student_object"] * presence("locates_student_object"),
        "verdict_with_referent": WEIGHTS["verdict_with_referent"] * presence("verdict_with_referent"),
        "demand_is_specific": WEIGHTS["demand_is_specific"] * presence("demand_is_specific"),
        "reference_conflict": -PENALTY_WEIGHTS["reference_conflict"] * presence("reference_conflict"),
        "dialogue_conflict": -PENALTY_WEIGHTS["dialogue_conflict"] * presence("dialogue_conflict"),
        "not_single_focus": -PENALTY_WEIGHTS["not_single_focus"] * presence("not_single_focus"),
        "empty_praise": -PENALTY_WEIGHTS["empty_praise"] * presence("empty_praise"),
        "length": float(length_term),
    }
    return float(sum(terms.values())), terms
