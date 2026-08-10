"""The pool LABELING_GUIDE.md specifies, built from `ontology.yaml` and checked against it.

    # the design only: 184 prompts, no model, no GPU
    python -m projects.learning_science_semantic_atlas.build_corpus \
        --ontology projects/learning_science_semantic_atlas/ontology.yaml \
        --out projects/learning_science_semantic_atlas/data/corpus

    # generate the artifact texts, then assemble and audit
    python -m projects.learning_science_semantic_atlas.build_corpus --mode all \
        --out data/corpus --model allenai/OLMoE-1B-7B-0924-Instruct

TWO MANIFESTS, AND THEY ARE DISJOINT. The 24-item calibration set exists so that
the few-shot examples an agent is calibrated on are not the units it is later
checked against (LABELING_GUIDE.md §2.4). That disjointness is enforced rather
than intended: the two blocks are salted separately, so no id can collide, and
`audit` compares their normalised texts and fails on an overlap. The predecessor
round had to report "NO CLEAN UNITS: every unit you labelled was used to
calibrate the agents", and the only way to not repeat that is to make the overlap
a build failure.

THE COMPOSITION IS FIXED IN ADVANCE, so it is checked rather than reported.
LABELING_GUIDE.md §2.1-§2.2 fix 120 targeted items (24 constructs x 5 roles), 18
debunked-control items (6 x 3), 22 open-pool items, and an artifact-type budget
of 70 conversational / 30 worked solution / 25 explanation / 20 practice set / 15
schedule. Those totals are exact, they interact, and they are not satisfiable by
sampling: four constructs are schedule-level and their designed positives may
only live in the bottom 35 rows, three constructs accept conversational artifacts
and nothing else, and every item's type has to be one its construct declares in
`output_types`. `_solve_format_budget` therefore searches the assignment
completely and fails with the remaining capacity named, instead of drifting to
"about 70" and calling it balance. `assessment_item` is the one declared output
type round 1 does not sample at all, so the audit asserts a count of zero for it
rather than letting the omission read as coverage.

THE 2x2 IS BUILT ON PURPOSE. Every construct gets an item that enacts it without
naming it, and an item that names it without enacting it - the says-only trap that
`schema.NamesEnacts` exists to keep separable. The fourth cell,
`names_and_enacts`, is the one that is easy to leave out and load-bearing:
without it, naming is perfectly anti-correlated with enacting across the pool and
a probe scores well by detecting topic and inverting. Half the canonical
positives therefore name their principle as well as enacting it, and the audit
checks the contingency table has no empty cell and no constant row or column.

THE HARD NEGATIVE IS THE SIBLING, WRITTEN WELL. A negative that is simply a worse
artifact teaches a probe one global quality direction, which is the failure the
README's matched-sibling design is against. So the `sibling_hard_negative` item
for a construct is a competent instance of its `sibling_confusable`, written on
the same scenario as that construct's canonical positive, sharing the surface
form its `hard_negative` field describes and failing its defining requirement.
Same stem, same intended quality, one construct swapped: `audit` requires 24 of
these pairs and checks each shares a lineage with its positive. The pair's
artifact type also has to be one the sibling accepts, or the rater marks
`not_applicable_wrong_artifact` and discriminates nothing - which is why the
budget is solved twice, strictly first.

CANDIDATE SETS ARE PART OF THE POOL, NOT OF THE PROMPT. Judging 24 constructs on
160 items is 3,840 decisions and would be skimmed, so §2.3 puts six constructs in
front of each designed item and all 24 in front of each open-pool item. The floor
that makes the resulting statistics interpretable - every construct in at least 45
of the 160 candidate sets - is not automatic, because eleven of the 24 are nobody's
`sibling_confusable` and reach it only through filler allocation. It is allocated
greedily on the running count and then checked.

One thing §2.3 does not say and this builder has to: a candidate set of
{target, sibling(target), fillers} gives the target away whenever exactly one
candidate's sibling is also in the set, and the sibling graph is public. Fillers
therefore prefer to add a second edge when appearance counts tie, and the manifest
reports how many items remain uniquely recoverable rather than claiming the
problem away.

WITHHELDING IS BY DIRECTORY, NOT BY DISCIPLINE. `Artifact.as_corpus_row` is an
allowlist, so a blinded row cannot carry a field somebody forgot to strip, and
everything else - the plan, the generation prompts, the raw generations, the key,
the retest mapping - is written under `withheld/`. The generation prompt is the
sharpest of those: it states the target construct, the role, the anchor to
instantiate and the boundary condition to violate. A model that echoes its brief
would put all of that into the corpus, so `sanitize_generation` refuses any
generation containing a construct key, a role name, an underscored withheld field
name or a phrase from the brief, and the refusal is counted in the manifest
instead of being quietly cleaned up. `audit` then re-scans every row that reached
the corpus.

OPEN-POOL PROMPTS CARRY NO CONSTRUCT AT ALL. Their whole job is to estimate base
rates (§2.3), and a base rate measured on artifacts that were asked for a
construct is not a base rate. The audit checks their prompts mention no construct
key and no display name.

NO MODEL IS REQUIRED TO CHECK THE DESIGN. `--mode plan` writes the plan, the
prompts and every count-and-balance check without loading weights, and
`--mode all --no-model` assembles a scaffolded corpus - artifact texts built from
the ontology's own anchors, each marked - so the labelling and packet code can be
exercised end to end before a GPU is asked for. Scaffolded rows are recorded as
`scaffold` in the key's `source_model`, because a placeholder that cannot be told
from a generation is worse than no placeholder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

try:
    from . import schema
except ImportError:  # pragma: no cover - direct `python build_corpus.py` execution
    import schema

MANIFEST_SCHEMA = "semantic-atlas/corpus-manifest-v1"
PLAN_SCHEMA = "semantic-atlas/corpus-plan-v1"
PROMPT_SCHEMA = "semantic-atlas/generation-prompt-v1"
GENERATION_SCHEMA = "semantic-atlas/generation-v1"

DEFAULT_SALT = "atlas-v1"
DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
SCAFFOLD_MODEL = "scaffold"
SCAFFOLD_MARK = "[scaffold]"

ROUND1_BLOCK = "round1"
CALIBRATION_BLOCK = "calibration"

TARGETED_POOL = "targeted"
CONTROL_POOL = "debunked_control"
OPEN_POOL = "open_pool"

# The concept a row carries when there is no designed target. It is not a
# construct key and must not become one; `audit` checks that.
OPEN_POOL_CONCEPT = "open_pool"

# LABELING_GUIDE.md §2.1 and §2.4.
N_CONSTRUCTS = 24
ROLES_PER_CONSTRUCT: tuple[schema.ItemRole, ...] = (
    schema.ItemRole.POSITIVE_CANONICAL,
    schema.ItemRole.POSITIVE_ATYPICAL,
    schema.ItemRole.POSITIVE_PARTIAL,
    schema.ItemRole.SIBLING_HARD_NEGATIVE,
    schema.ItemRole.SAYS_ONLY,
)
ITEMS_PER_CONSTRUCT = len(ROLES_PER_CONSTRUCT)
ITEMS_PER_CONTROL = 3
TARGETED_TOTAL = N_CONSTRUCTS * ITEMS_PER_CONSTRUCT
OPEN_POOL_TOTAL = 22
ROUND1_TOTAL = 160
CALIBRATION_TOTAL = N_CONSTRUCTS

# LABELING_GUIDE.md §2.3: six candidates on a designed item, the full checklist
# on an open-pool item, and every construct in at least 45 of the 160.
CANDIDATE_SET_SIZE = 6
APPEARANCE_FLOOR = 45

# LABELING_GUIDE.md §4.
RETEST_STRATA: Mapping[str, int] = {TARGETED_POOL: 24, CONTROL_POOL: 8, OPEN_POOL: 8}
RETEST_TOTAL = sum(RETEST_STRATA.values())

_F = schema.OutputFormat

# LABELING_GUIDE.md §2.2. The two-format groups are the guide's own rows: it
# budgets `tutor_turn`/`dialogue_episode` together and `study_schedule`/
# `curriculum_plan` together, because the choice inside each pair is a rendering
# decision and the choice between pairs is what a schedule-level construct cannot
# make.
FORMAT_GROUPS: tuple[tuple[str, tuple[schema.OutputFormat, ...], int], ...] = (
    ("conversational", (_F.TUTOR_TURN, _F.DIALOGUE_EPISODE), 70),
    ("worked_solution", (_F.WORKED_SOLUTION,), 30),
    ("explanation_text", (_F.EXPLANATION_TEXT,), 25),
    ("practice_item_set", (_F.PRACTICE_ITEM_SET,), 20),
    ("schedule", (_F.STUDY_SCHEDULE, _F.CURRICULUM_PLAN), 15),
)
GROUP_NAMES: tuple[str, ...] = tuple(name for name, _, _ in FORMAT_GROUPS)
GROUP_CAPACITY: tuple[int, ...] = tuple(capacity for _, _, capacity in FORMAT_GROUPS)
GROUP_OF_FORMAT: Mapping[schema.OutputFormat, int] = {
    fmt: index for index, (_, formats, _) in enumerate(FORMAT_GROUPS) for fmt in formats
}

# The one output type round 1 does not sample. Nine constructs accept it and all
# nine accept something in the table above, so no construct loses its items.
UNSAMPLED_FORMAT = _F.ASSESSMENT_ITEM

# LABELING_GUIDE.md §2.2: a single turn cannot contain a schedule, so these four
# constructs are confined to the budget's bottom 35 rows.
SCHEDULE_LEVEL_CONSTRUCTS: tuple[str, ...] = (
    "spaced_practice_schedule",
    "successive_relearning_criterion",
    "interleaved_discrimination_practice",
    "guidance_fading_sequence",
)
BOTTOM_35_GROUPS: tuple[str, ...] = ("practice_item_set", "schedule")

# Constructs whose content is a change across turns. Given the conversational
# group they take `dialogue_episode`; a single turn cannot show contingency or
# fading and an item that pretends otherwise tests vocabulary.
MULTI_TURN_CONSTRUCTS: frozenset[str] = frozenset(
    {
        "contingent_help_calibration",
        "guidance_fading_sequence",
        "problem_solving_before_instruction",
        "elaborated_feedback_specificity",
    }
)


# --------------------------------------------------------------------- scenarios
#
# Four stems per domain. A stem is subject matter plus a reference solution plus
# what the learner said, and it is deliberately reused across constructs: topic
# is a balance axis, so the same content appearing under different constructs is
# what keeps a construct from being confounded with a subject.


@dataclass(frozen=True, slots=True)
class Stem:
    domain: schema.Domain
    topic: str
    target: str
    problem: str
    reference: str
    misconception: str
    misstep: str
    worked_step: str


_D = schema.Domain

STEMS: tuple[Stem, ...] = (
    Stem(
        _D.ALGEBRA,
        "two-step linear equations",
        "isolating a variable by undoing operations in reverse order",
        "Solve 3x + 7 = 22 for x.",
        "Subtract 7 from both sides to get 3x = 15, then divide both sides by 3, so x = 5.",
        "I got x = 9.67, because I divided 22 by 3 first and then took away 7.",
        "dividing by the coefficient before removing the constant term",
        "Subtract 7 from both sides: 3x = 15.",
    ),
    Stem(
        _D.ALGEBRA,
        "systems of linear equations by substitution",
        "expressing one variable in terms of the other before substituting",
        "Solve the system y = 2x - 1 and 3x + y = 9.",
        "Substituting y = 2x - 1 into 3x + y = 9 gives 5x - 1 = 9, so x = 2 and y = 3.",
        "I added the two equations and got 3x + 2y = 8, and now I am stuck.",
        "adding equations that are not aligned in the same variables",
        "Replace y in the second equation with 2x - 1.",
    ),
    Stem(
        _D.ALGEBRA,
        "factoring quadratic trinomials",
        "finding the factor pair of the constant term that sums to the linear coefficient",
        "Factor x^2 - 5x + 6.",
        "The pair -2 and -3 multiplies to 6 and sums to -5, so x^2 - 5x + 6 = (x - 2)(x - 3).",
        "I wrote (x + 2)(x + 3), because 2 times 3 is 6.",
        "ignoring the sign of the linear coefficient when choosing the factor pair",
        "List the factor pairs of 6 and test which pair sums to -5.",
    ),
    Stem(
        _D.ALGEBRA,
        "slope from two points",
        "computing rise over run with the two points taken in the same order",
        "Find the slope of the line through (2, 5) and (6, 13).",
        "Slope = (13 - 5) / (6 - 2) = 8 / 4 = 2.",
        "I did (6 - 2) / (13 - 5) and got one half.",
        "inverting rise and run",
        "Write the difference of the y-values over the difference of the x-values, in that order.",
    ),
    Stem(
        _D.CHEMISTRY,
        "balancing combustion equations",
        "conserving the atoms of each element across the arrow",
        "Balance C3H8 + O2 -> CO2 + H2O.",
        "C3H8 + 5 O2 -> 3 CO2 + 4 H2O. Carbon and hydrogen are balanced first and oxygen last, "
        "because oxygen appears in both products.",
        "I put a 3 in front of CO2 and a 4 in front of H2O, and then the oxygens would not come out even.",
        "balancing oxygen before the elements that appear in only one product",
        "Balance carbon first: 3 CO2 on the right.",
    ),
    Stem(
        _D.CHEMISTRY,
        "limiting reagent",
        "comparing amounts divided by stoichiometric coefficients rather than raw amounts",
        "2.0 mol of H2 reacts with 1.5 mol of O2 to form water. Which reactant limits the yield?",
        "2 H2 + O2 -> 2 H2O needs 2 mol of H2 per mol of O2, so 2.0 mol of H2 pairs with only 1.0 mol "
        "of O2: H2 runs out first and O2 is in excess.",
        "O2 must be limiting, because there is less of it.",
        "comparing raw amounts instead of amounts divided by coefficients",
        "Divide each amount by its coefficient: H2 gives 1.0, O2 gives 1.5.",
    ),
    Stem(
        _D.CHEMISTRY,
        "pH of a strong acid solution",
        "reading pH as the negative base-ten logarithm of the hydronium concentration",
        "Find the pH of 0.001 M HCl.",
        "HCl dissociates fully, so the hydronium concentration is 1e-3 M and the pH is 3.",
        "Is the pH 0.001?",
        "reporting the concentration instead of its negative logarithm",
        "Take the negative base-ten logarithm of 1e-3.",
    ),
    Stem(
        _D.CHEMISTRY,
        "ionic versus covalent bonding",
        "predicting bond type from the electronegativity difference",
        "Predict the bond type in NaCl and in CO.",
        "Sodium and chlorine differ by about 2.1 in electronegativity, which is ionic; carbon and oxygen "
        "differ by about 1.0, which is polar covalent.",
        "Both are covalent, because in both cases the atoms share electrons.",
        "treating every bond as sharing regardless of the electronegativity difference",
        "Look up each element's electronegativity and subtract.",
    ),
    Stem(
        _D.CELL_BIOLOGY,
        "osmosis across a semipermeable membrane",
        "predicting the direction water moves from the solute gradient",
        "A cell whose interior is 2% salt is placed in a 6% salt solution. What happens to the cell?",
        "Water leaves the cell toward the higher external solute concentration, so the cell shrinks.",
        "The salt moves into the cell to even things out, so the cell swells.",
        "moving solute rather than water across a membrane permeable only to water",
        "Compare the solute concentration inside the cell with the one outside it.",
    ),
    Stem(
        _D.CELL_BIOLOGY,
        "mitosis versus meiosis",
        "tracking chromosome number and genetic identity across the divisions",
        "A diploid cell with 8 chromosomes divides. Give the products of mitosis and of meiosis.",
        "Mitosis yields two diploid cells of 8 chromosomes each; meiosis yields four haploid cells of "
        "4 chromosomes each.",
        "Both give four cells, they just look different afterwards.",
        "assigning meiosis's product count to mitosis",
        "Count the chromosomes after the first division in each pathway.",
    ),
    Stem(
        _D.CELL_BIOLOGY,
        "enzyme saturation",
        "explaining a plateau in reaction rate by occupancy of the active sites",
        "Why does the reaction rate stop rising as substrate concentration increases?",
        "Every active site is occupied, so the rate is set by enzyme concentration and turnover rather "
        "than by the supply of substrate.",
        "The enzyme gets tired, so the reaction slows down.",
        "attributing the plateau to enzyme fatigue rather than to saturated active sites",
        "Ask what runs out first as the substrate concentration rises.",
    ),
    Stem(
        _D.CELL_BIOLOGY,
        "ATP yield in cellular respiration",
        "locating where most ATP is produced and by what mechanism",
        "Where in respiration is most ATP made, and how?",
        "Oxidative phosphorylation at the inner mitochondrial membrane makes most of it, driven by the "
        "proton gradient; glycolysis and the citric-acid cycle contribute a small direct amount.",
        "Glycolysis makes the most, because it comes first.",
        "equating the first pathway with the largest yield",
        "Compare the direct ATP from glycolysis with the ATP from the electron transport chain.",
    ),
    Stem(
        _D.PROGRAMMING,
        "loop bounds and off-by-one errors",
        "choosing bounds that visit every index exactly once",
        "Why does `for i in range(len(a) - 1)` skip the last element of `a`?",
        "range(n) yields 0 through n-1, so range(len(a) - 1) stops one index early; range(len(a)) "
        "visits every index.",
        "I subtract one because the indexes start at zero.",
        "subtracting one from the count as well as starting from zero",
        "Write out the indexes range produces for a list of length three.",
    ),
    Stem(
        _D.PROGRAMMING,
        "mutable default arguments",
        "recognising that a default value is created once, when the function is defined",
        "Why does `def add(x, bucket=[]): bucket.append(x); return bucket` accumulate across calls?",
        "The default list is created once at definition time and shared by every call. Use bucket=None "
        "and create a new list in the body.",
        "Each call should start with an empty list, so I do not see the problem.",
        "assuming default arguments are re-evaluated on every call",
        "Call the function twice and print the result each time.",
    ),
    Stem(
        _D.PROGRAMMING,
        "recursion base cases",
        "identifying the case that stops the recursive descent",
        "Why does `def f(n): return n * f(n - 1)` recurse forever?",
        "There is no base case. Adding `if n <= 1: return 1` terminates the descent.",
        "I thought Python stops on its own when n reaches zero.",
        "omitting the terminating case",
        "Trace f(2) by hand and note what f(0) calls.",
    ),
    Stem(
        _D.PROGRAMMING,
        "membership tests in lists and sets",
        "comparing the average cost of a hash lookup with a linear scan",
        "A membership test over 100000 records in a list is slow. What should change, and why?",
        "A list membership test scans linearly; a set lookup hashes the key and averages constant time, "
        "so build a set from the records first.",
        "I sorted the list first, but it is still slow.",
        "assuming that sorting speeds up a linear membership scan",
        "Convert the records to a set and repeat the membership test.",
    ),
    Stem(
        _D.STATISTICS,
        "interpreting a p-value",
        "stating which probability a p-value reports and what it is conditioned on",
        "A test returns p = 0.03. What may and may not be concluded?",
        "It is the probability of data at least this extreme if the null hypothesis were true. It is "
        "not the probability that the null is true, and it is not a measure of effect size.",
        "So there is a 3% chance that the null hypothesis is true.",
        "inverting the conditional probability",
        "Name the quantity the probability is conditioned on.",
    ),
    Stem(
        _D.STATISTICS,
        "what a confidence level describes",
        "attaching the coverage claim to the procedure rather than to one interval",
        "A 95% confidence interval for a mean is [4.1, 6.3]. What does the 95% describe?",
        "Ninety-five percent of intervals built this way across repeated samples contain the true mean. "
        "This particular interval either contains it or does not.",
        "There is a 95% chance the true mean is between 4.1 and 6.3.",
        "assigning a probability to a fixed interval instead of to the procedure",
        "Say which quantity varies across repeated samples.",
    ),
    Stem(
        _D.STATISTICS,
        "confounded associations",
        "identifying a third variable that moves both quantities",
        "Ice-cream sales correlate with drowning deaths. What explains the association?",
        "Temperature raises both. The association is confounded and neither variable causes the other.",
        "So buying ice cream is dangerous in summer.",
        "reading an association as a causal effect",
        "Look for a variable that moves both quantities at once.",
    ),
    Stem(
        _D.STATISTICS,
        "mean and median under skew",
        "choosing a measure of centre that resists a single extreme value",
        "Salaries are 30k, 32k, 35k, 36k and 400k. Which measure of centre summarises them better?",
        "The median of 35k resists the outlier; the mean of about 106.6k sits above every value but one.",
        "The mean is always the right average.",
        "using the mean on a heavily skewed sample",
        "Compute both and compare each with the bulk of the data.",
    ),
    Stem(
        _D.HISTORY,
        "causes of the French Revolution",
        "separating long-run structural causes from triggering events",
        "Sort the causes of the French Revolution into structural causes and triggers.",
        "Fiscal crisis, an inflexible estate structure and rising bread prices are structural; the "
        "calling of the Estates-General and the fall of the Bastille are triggers.",
        "It happened because the king was unpopular.",
        "collapsing long-run structural pressure into one proximate cause",
        "Sort each factor by how long it had been building.",
    ),
    Stem(
        _D.HISTORY,
        "reading a primary source for interest",
        "attributing a claim to its author's position and intended audience",
        "An 1858 plantation owner's letter calls his workers content. How should that claim be treated?",
        "As evidence of what the author wanted his correspondent to believe. Independent corroboration "
        "is required before it is treated as a description of the workers.",
        "It is a primary source, so it is reliable.",
        "treating proximity to events as accuracy",
        "Name the author's interest in the claim before weighing it.",
    ),
    Stem(
        _D.HISTORY,
        "industrialisation and urban growth",
        "linking mechanised production to where labour had to move",
        "Explain why British cities grew so fast between 1780 and 1850.",
        "Mechanised textile production concentrated work in mills sited near coal and water power, "
        "drawing rural labour in faster than housing and sanitation followed.",
        "People just liked cities better than farms.",
        "explaining migration by preference rather than by the location of work",
        "Say where the new work was, and why it was there.",
    ),
    Stem(
        _D.HISTORY,
        "periodising the Cold War",
        "justifying a period boundary with an event and a stated criterion",
        "Defend a start date for the Cold War.",
        "A defensible answer names an event, states the criterion that makes it a boundary, and says "
        "what an earlier or later date would capture instead.",
        "It started in 1945, when the war ended.",
        "asserting a date without a criterion that makes it a boundary",
        "State the criterion before naming the year.",
    ),
)

STEMS_BY_DOMAIN: Mapping[schema.Domain, tuple[Stem, ...]] = {
    domain: tuple(stem for stem in STEMS if stem.domain is domain) for domain in schema.Domain
}
DOMAIN_ORDER: tuple[schema.Domain, ...] = tuple(schema.Domain)

# What the scenario DECLARES about the learner, rendered as the learner's own
# words so that the declared state and the visible turn cannot disagree.
# LearnerState is a required input to `expertise_reversal_adaptation`, which
# LABELING_GUIDE.md §3.1 makes unlabelable when expertise is inferred from tone.
LEARNER_TURNS: Mapping[schema.LearnerState, str] = {
    schema.LearnerState.NO_ATTEMPT_YET: "I have not tried this one yet.",
    schema.LearnerState.NOVICE_DECLARED: "I am new to this topic and I have not seen a problem like this before.",
    schema.LearnerState.PARTIAL_ATTEMPT: "I got as far as this and then stopped: {worked_step}",
    schema.LearnerState.MISCONCEPTION_STATED: "{misconception}",
    schema.LearnerState.ADVANCED_DECLARED: (
        "I have finished twelve problems of this type unaided; I want to check this one."
    ),
}

CONVERSATIONAL_STATES: tuple[schema.LearnerState, ...] = tuple(schema.LearnerState)
AUTHORING_STATES: tuple[schema.LearnerState, ...] = (
    schema.LearnerState.NO_ATTEMPT_YET,
    schema.LearnerState.NOVICE_DECLARED,
    schema.LearnerState.ADVANCED_DECLARED,
)

REQUIRED_LEARNER_STATE: Mapping[str, schema.LearnerState] = {
    "expertise_reversal_adaptation": schema.LearnerState.ADVANCED_DECLARED,
    "problem_solving_before_instruction": schema.LearnerState.NO_ATTEMPT_YET,
    "contingent_help_calibration": schema.LearnerState.PARTIAL_ATTEMPT,
    "answer_withholding_level": schema.LearnerState.PARTIAL_ATTEMPT,
    "elaborated_feedback_specificity": schema.LearnerState.MISCONCEPTION_STATED,
    "attributional_feedback_framing": schema.LearnerState.MISCONCEPTION_STATED,
    "belonging_affirmation_framing": schema.LearnerState.MISCONCEPTION_STATED,
    "worked_example_provision": schema.LearnerState.NOVICE_DECLARED,
}

# The rater-visible instruction the artifact answers. It never names the target
# construct: what makes an item hard is that the brief could be answered many
# ways, and a task that says which way is not a discrimination test.
FORMAT_TASKS: Mapping[schema.OutputFormat, str] = {
    _F.TUTOR_TURN: "The learner is working on this problem: {problem} Write the tutor's next single turn.",
    _F.DIALOGUE_EPISODE: (
        "The learner is working on this problem: {problem} Write the next three or four turns of the exchange."
    ),
    _F.EXPLANATION_TEXT: (
        "Write a prose explanation for this learner, aimed at {target}, with no learner turn to answer. "
        "The problem under discussion is: {problem}"
    ),
    _F.WORKED_SOLUTION: "Produce a step-by-step solution to this problem: {problem}",
    _F.PRACTICE_ITEM_SET: (
        "Produce a set of five practice items aimed at {target}, in the order the learner will meet them."
    ),
    _F.STUDY_SCHEDULE: "Produce a dated study plan for this learner, aimed at {target}.",
    _F.CURRICULUM_PLAN: "Produce a three-session plan for this class, aimed at {target}.",
}

FORMAT_BRIEFS: Mapping[schema.OutputFormat, str] = {
    _F.TUTOR_TURN: "Write exactly one tutor turn, addressed to the learner.",
    _F.DIALOGUE_EPISODE: "Write three or four turns, alternating tutor and learner, starting with the tutor.",
    _F.EXPLANATION_TEXT: "Write a short explanation in prose. There is no learner turn to answer.",
    _F.WORKED_SOLUTION: "Write a step-by-step solution.",
    _F.PRACTICE_ITEM_SET: "Write five practice items, numbered, in the order the learner meets them.",
    _F.STUDY_SCHEDULE: "Write a study plan whose sessions are dated by calendar day.",
    _F.CURRICULUM_PLAN: "Write a plan covering three numbered sessions.",
}

FORMAT_MAX_TOKENS: Mapping[schema.OutputFormat, int] = {
    _F.TUTOR_TURN: 256,
    _F.DIALOGUE_EPISODE: 512,
    _F.EXPLANATION_TEXT: 448,
    _F.WORKED_SOLUTION: 448,
    _F.PRACTICE_ITEM_SET: 448,
    _F.STUDY_SCHEDULE: 384,
    _F.CURRICULUM_PLAN: 512,
}

FORMAT_PROMPT_MODES: Mapping[schema.OutputFormat, tuple[schema.PromptMode, ...]] = {
    _F.TUTOR_TURN: (schema.PromptMode.TUTOR_ROLEPLAY, schema.PromptMode.FEEDBACK_REQUEST),
    _F.DIALOGUE_EPISODE: (schema.PromptMode.TUTOR_ROLEPLAY,),
    _F.EXPLANATION_TEXT: (schema.PromptMode.DIRECT_INSTRUCTION_REQUEST, schema.PromptMode.LESSON_AUTHORING),
    _F.WORKED_SOLUTION: (schema.PromptMode.DIRECT_INSTRUCTION_REQUEST, schema.PromptMode.FEEDBACK_REQUEST),
    _F.PRACTICE_ITEM_SET: (schema.PromptMode.LESSON_AUTHORING, schema.PromptMode.DIRECT_INSTRUCTION_REQUEST),
    _F.STUDY_SCHEDULE: (schema.PromptMode.LESSON_AUTHORING,),
    _F.CURRICULUM_PLAN: (schema.PromptMode.LESSON_AUTHORING,),
}

PROMPT_MODE_FRAMING: Mapping[schema.PromptMode, str] = {
    schema.PromptMode.DIRECT_INSTRUCTION_REQUEST: "You are producing instructional material to a written brief.",
    schema.PromptMode.TUTOR_ROLEPLAY: "You are the tutor in a one-to-one session with the learner described below.",
    schema.PromptMode.LESSON_AUTHORING: "You are authoring course materials ahead of the sessions they cover.",
    schema.PromptMode.FEEDBACK_REQUEST: "You are responding to work the learner has already submitted.",
}

# One clause per construct whose `required_context` names something a scenario has
# to DECLARE. Without these, LABELING_GUIDE.md §3.1 requires the rater to mark
# `not_applicable_missing_context`, and an item that arrives inapplicable is an
# item that measured nothing. The clause states the context and never the answer:
# `successive_relearning_criterion` gets per-term correctness recorded, not a
# criterion, because supplying the criterion would enact the construct in the
# brief.
CONTEXT_CLAUSES: Mapping[str, str] = {
    "spaced_practice_schedule": (
        "Scheduling context: the assessment is on calendar day 30, and occasions are dated by calendar "
        "day rather than counted in conversational turns."
    ),
    "successive_relearning_criterion": (
        "Scheduling context: the assessment is on calendar day 30, occasions are dated by calendar day, "
        "and each occasion records for every term whether the learner recalled it correctly."
    ),
    "interleaved_discrimination_practice": (
        "Category context: this unit also covers the two neighbouring procedures that learners routinely "
        "confuse with {target}, and the order the learner meets items in is fixed as written."
    ),
    "guidance_fading_sequence": (
        "Sequence context: there are at least three ordered positions available, and the reference "
        "solution above lists every step, so how much of it is supplied at each position is checkable."
    ),
    "pretest_prequestion": (
        "Ordering context: any instruction that answers a question asked here arrives later in this same "
        "artifact."
    ),
    "problem_solving_before_instruction": (
        "Curriculum context: {target} has not been taught to this grade-11 class yet, and the instruction "
        "that follows this artifact is the first teaching of it."
    ),
    "contingent_help_calibration": (
        "House rule for this scenario: when the learner has produced a partial attempt, the prescribed "
        "help level is one hint aimed at the first error and nothing beyond it."
    ),
    "expertise_reversal_adaptation": (
        "Declared learner expertise: advanced. This is a declared field of the scenario, not an inference "
        "from the learner's vocabulary or tone."
    ),
    "teach_back_elicitation": (
        "Audience context: before studying, the learner was told they would have to explain this to a "
        "classmate who missed the lesson."
    ),
    "confidence_calibration_elicitation": (
        "Scoring context: every item here is scored, so a judgment recorded before the answer is revealed "
        "can be compared with the outcome."
    ),
    "analogical_case_comparison": (
        "Second-case context: a structurally similar case is available for comparison, namely {second_case}."
    ),
    "icap_engagement_class": (
        "Setting context: a partner is available to work with, and the expected output format is stated in "
        "the request above."
    ),
    "self_explanation_prompt": (
        "Material context: the reference solution above is in front of the learner while they work."
    ),
    "elaborated_feedback_specificity": (
        "Marking context: the reference solution above is the standard to check the learner's work "
        "against, and is not to be re-derived."
    ),
    "answer_withholding_level": (
        "Marking context: the reference solution above is the complete answer, and what the learner has "
        "produced so far is quoted above it."
    ),
    "completion_problem_scaffold": (
        "Marking context: the reference solution above establishes which step of the work is which."
    ),
    "subgoal_labeled_structure": (
        "Marking context: the reference solution above establishes where the functional boundaries of the "
        "work fall."
    ),
}

# LABELING_GUIDE.md §2.3: a control is only a control against the constructs it
# is put in front of, so each one faces the constructs its `expected_label`
# names, and otherwise the warmth- and framing-adjacent constructs a drifting
# instrument would reward.
DRIFT_ADJACENT_CONSTRUCTS: tuple[str, ...] = (
    "attributional_feedback_framing",
    "autonomy_supportive_framing",
    "belonging_affirmation_framing",
    "utility_value_relevance_prompt",
    "elaborated_feedback_specificity",
    "metacognitive_regulation_prompt",
)

_QUOTED = re.compile(r"'([^']+)'")
_WORD = re.compile(r"[a-z][a-z\-]+")
_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(?P<body>.*?)\n?```\s*\Z", re.DOTALL)

# Phrases that only reach an artifact by way of the brief that asked for it, or by
# way of the protocol around it. Every one is checked as a substring, so each has
# to be a phrase no artifact would contain on its own: `boundary condition` and
# `sibling` were both in this list and both had to come out, because a loop-bounds
# artifact discusses boundary conditions and a genetics one discusses siblings.
BRIEF_PHRASES: tuple[str, ...] = (
    "do for the subject above",
    "target knowledge:",
    "reference solution, for your use",
    "output rules",
    "reply with the artifact",
    "the artifact must",
    "positive anchor",
    "hard negative",
    "says-only",
    "designed target",
    "the brief",
    "your brief",
    "as an ai",
)

# Checked only against the start of the text, because that is where a preamble is
# and because the substring forms are ordinary tutor language. "Here is a hint"
# is a tutor turn; "Here is the artifact you asked for" is a model narrating.
PREAMBLE_STARTS: tuple[str, ...] = (
    "here is the artifact",
    "here's the artifact",
    "here is the tutor",
    "here is my",
    "here's my",
    "below is the",
    "as requested",
    "sure,",
    "sure!",
    "certainly",
    "of course,",
)

MIN_ARTIFACT_CHARS = 40


def _leak_tokens(ontology: schema.Ontology) -> tuple[str, ...]:
    """Identifier-shaped strings that may never appear in a corpus row.

    Underscored only. `WITHHELD_FIELDS` also holds the bare words `names`,
    `enacts` and `split`, which occur in ordinary English and would make this
    check fire on artifacts that leaked nothing.
    """
    tokens = {construct.key for construct in ontology}
    tokens |= {control.key for control in ontology.debunked_controls}
    tokens |= {role.value for role in schema.ItemRole}
    tokens |= {name for name in schema.WITHHELD_FIELDS if "_" in name}
    return tuple(sorted(tokens))


def trigger_words(construct: schema.Construct) -> tuple[str, ...]:
    """The construct's own giveaway vocabulary, taken from `surface_confounds`.

    An atypical positive has to be a genuine instance with none of these
    (LABELING_GUIDE.md §2.1), which is the item that separates the construct from
    the words that co-occur with it. Where the ontology quotes no words, the
    display name's content words stand in.
    """
    words: set[str] = set()
    for entry in construct.surface_confounds:
        for quoted in _QUOTED.findall(entry):
            word = quoted.strip().lower()
            if len(word) >= 3:
                words.add(word)
    if not words:
        words = {word for word in _WORD.findall(construct.display_name.lower()) if len(word) >= 4}
    return tuple(sorted(words))


def _despecify(text: str, ontology: schema.Ontology) -> str:
    """Ontology prose with its cross-references spelled out.

    Several fields point at other constructs by key - `boundary_conditions` for
    `worked_example_provision` says "see expertise_reversal_adaptation" - and a
    key is exactly what may not reach a corpus row. Scaffold text is built from
    this prose, so the reference becomes the construct's display name and the row
    stays clean.
    """
    for construct in ontology:
        if construct.key in text:
            text = text.replace(construct.key, construct.display_name.lower())
    return text


_PRESCRIPTIVE = ("must", "requires", "require", "matters", "part of", "should", "not monotone")


def _boundary(construct: schema.Construct, position: int) -> str:
    """A boundary condition a partial positive can actually fall short of.

    Two entries are filtered out for the same reason. `escalation:` entries record
    what data would be needed to score the construct at a higher claim level, so
    an artifact cannot fail one. And most of the rest are moderator statements
    about effect size - "benefits grow with the retention interval" - which an
    artifact also cannot fail; the prescriptive ones are preferred where a
    construct has any, because those name something a writer can leave out.
    """
    conditions = [entry for entry in construct.boundary_conditions if not entry.lower().startswith("escalation")]
    conditions = conditions or list(construct.boundary_conditions)
    prescriptive = [entry for entry in conditions if any(word in entry.lower() for word in _PRESCRIPTIVE)]
    pool = prescriptive or conditions
    return pool[position % len(pool)]


# ------------------------------------------------------------------- plan types


@dataclass(frozen=True, slots=True)
class PlanItem:
    """One designed item: everything except the artifact text a model has yet to write.

    Split from `schema.Artifact` on purpose. An `Artifact` requires text and is
    the record a corpus row and a key row are built from; a `PlanItem` is what
    exists before generation and what a resumed job matches against. Holding the
    brief here, rather than deriving it at generation time, is what makes
    `prompt_sha` a stable identity: if the plan changes, the digest changes and a
    stale generations file is rejected instead of silently re-used.
    """

    unit_id: str
    lineage_id: str
    block: str
    pool: str
    concept: str
    role: schema.ItemRole
    enactment: schema.NamesEnacts
    scenario: schema.Scenario
    candidates: tuple[str, ...]
    system_prompt: str
    user_prompt: str
    max_new_tokens: int
    expected_fidelity: int | None = None
    enacted_construct: str | None = None
    named_construct: str | None = None
    sibling_expected_fidelity: int | None = None
    forbidden_words: tuple[str, ...] = ()
    context_clause: str = ""
    scaffold: str = ""

    @property
    def output_format(self) -> schema.OutputFormat:
        return self.scenario.output_format

    @property
    def prompt_sha(self) -> str:
        return prompt_digest(self.system_prompt, self.user_prompt)

    def artifact(self, text: str) -> schema.Artifact:
        return schema.Artifact(
            artifact_id=self.unit_id,
            lineage_id=self.lineage_id,
            concept=self.concept,
            role=self.role,
            enactment=self.enactment,
            text=text,
            scenario=self.scenario,
            enacted_construct=self.enacted_construct,
            named_construct=self.named_construct,
            expected_fidelity=self.expected_fidelity,
        )

    def plan_row(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "unit_id": self.unit_id,
            "item_id": self.lineage_id,
            "block": self.block,
            "pool": self.pool,
            "concept": self.concept,
            "item_role": self.role.value,
            "candidates": list(self.candidates),
            "expected_fidelity": self.expected_fidelity,
            "enacted_construct": self.enacted_construct,
            "named_construct": self.named_construct,
            "sibling_expected_fidelity": self.sibling_expected_fidelity,
            "forbidden_words": list(self.forbidden_words),
            "prompt_sha": self.prompt_sha,
            "max_new_tokens": self.max_new_tokens,
            **self.enactment.as_dict(),
            **self.scenario.as_dict(),
        }

    def prompt_row(self) -> dict[str, object]:
        return {
            "schema": PROMPT_SCHEMA,
            "unit_id": self.unit_id,
            "block": self.block,
            "pool": self.pool,
            "concept": self.concept,
            "item_role": self.role.value,
            "output_format": self.output_format.value,
            "prompt_sha": self.prompt_sha,
            "max_new_tokens": self.max_new_tokens,
            "system": self.system_prompt,
            "user": self.user_prompt,
        }

    def item_row(self, text: str) -> dict[str, object]:
        """The fully blinded rater view: the artifact, its stem, and its candidate set.

        `Artifact.as_corpus_row` carries `concept`, which for a targeted item is
        the designed target. That is harmless where the reader is
        `labeling.build_packet`, since a packet asks about one construct and every
        option in it shares that value and the judge view never shows it. It is
        not harmless in front of a person, so the human-facing file is assembled
        separately and lists the six candidates of LABELING_GUIDE.md §2.3 in
        alphabetical order instead - candidate order is randomised per rater at
        presentation, and a file that ordered them target-first would answer the
        question it was written to ask.
        """
        return {
            "unit_id": self.unit_id,
            "item_id": self.lineage_id,
            "candidates": list(self.candidates),
            "candidate_action": text,
            "question": self.scenario.task,
            "reference": self.scenario.reference,
            "student_before": self.scenario.student_before,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """Both manifests, their candidate sets and the retest block, ready to generate."""

    construct_version: str
    salt: str
    round1: tuple[PlanItem, ...]
    calibration: tuple[PlanItem, ...]
    retest: tuple[Mapping[str, object], ...]
    appearances: Mapping[str, int]
    format_solve: str = "strict"

    @property
    def items(self) -> tuple[PlanItem, ...]:
        return self.round1 + self.calibration

    def by_unit_id(self) -> dict[str, PlanItem]:
        return {item.unit_id: item for item in self.items}

    def targeted(self, block: str = ROUND1_BLOCK) -> tuple[PlanItem, ...]:
        source = self.round1 if block == ROUND1_BLOCK else self.calibration
        return tuple(item for item in source if item.pool == TARGETED_POOL)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class Audit:
    checks: tuple[Check, ...]
    stats: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.ok)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
            "stats": dict(self.stats),
        }


@dataclass(frozen=True, slots=True)
class Rejection:
    unit_id: str
    reasons: tuple[str, ...]
    text: str

    def as_dict(self) -> dict[str, object]:
        return {"unit_id": self.unit_id, "reasons": list(self.reasons), "text": self.text}


@dataclass(frozen=True, slots=True)
class Assembly:
    """Artifacts, their provenance and everything that did not make it."""

    artifacts: Mapping[str, schema.Artifact]
    provenance: Mapping[str, Mapping[str, object]]
    rejections: tuple[Rejection, ...]
    missing: tuple[str, ...]
    scaffolded: tuple[str, ...]


# ------------------------------------------------------------- format budgeting


@dataclass(frozen=True, slots=True)
class _FormatUnit:
    """One indivisible format decision and the weights it spends.

    A construct's canonical positive, its partial positive, its sibling hard
    negative and its says-only item share one scenario, so they share one output
    type and one decision of weight 4. Its atypical positive is a second
    scenario, and takes a different type wherever the construct allows one.
    """

    name: str
    weights: tuple[int, ...]
    options: tuple[tuple[int, ...], ...]
    # Groups this unit would rather spend its first slot on. A preference and not a
    # constraint: the budget is exact and has to be met, and an unsatisfiable
    # preference should cost an item its ideal artifact type rather than fail the
    # build. `_solve_format_budget` tries these first at every node.
    prefer_first: tuple[int, ...] = ()


def _solve_format_budget(
    units: Sequence[_FormatUnit], capacity: Sequence[int], flexible: int, *, node_budget: int = 400_000
) -> dict[str, tuple[int, ...]]:
    """Assign every unit an output-type group so the guide's budget is met exactly.

    A complete depth-first search, not a heuristic. The budget rows are exact
    counts that interact with `output_types`, so an allocator that gets close
    produces a pool whose artifact-type mix is not the one the round claims. When
    no assignment exists the failure names the group and the shortfall, which is
    the information needed to change the budget or the ontology.
    """
    if sum(capacity) != sum(sum(unit.weights) for unit in units) + flexible:
        raise ValueError(
            f"format budget {list(capacity)} sums to {sum(capacity)}, but the units and the flexible "
            f"pool need {sum(sum(u.weights) for u in units) + flexible}"
        )

    order = sorted(range(len(units)), key=lambda index: (len(units[index].options), units[index].name))
    n_groups = len(capacity)
    reach = [
        [max(sum(w for w, g in zip(unit.weights, option) if g == group) for option in unit.options) for group in range(n_groups)]
        for unit in units
    ]
    suffix = [[0] * n_groups for _ in range(len(order) + 1)]
    for position in range(len(order) - 1, -1, -1):
        unit_index = order[position]
        for group in range(n_groups):
            suffix[position][group] = suffix[position + 1][group] + reach[unit_index][group]

    chosen: dict[str, tuple[int, ...]] = {}
    visits = 0

    def search(position: int, remaining: tuple[int, ...]) -> bool:
        nonlocal visits
        visits += 1
        if visits > node_budget:  # pragma: no cover - the real ontology solves in a few hundred nodes
            raise ValueError(f"format budget search exceeded {node_budget} nodes; the constraints are too tight")
        if position == len(order):
            return all(left >= 0 for left in remaining) and sum(remaining) == flexible
        for group in range(n_groups):
            if remaining[group] > flexible + suffix[position][group]:
                return False
        unit = units[order[position]]
        prefer = unit.prefer_first
        options = sorted(
            unit.options,
            key=lambda option: (bool(prefer) and option[0] not in prefer, -sum(remaining[g] for g in option), option),
        )
        for option in options:
            spend = list(remaining)
            for weight, group in zip(unit.weights, option):
                spend[group] -= weight
            if any(left < 0 for left in spend):
                continue
            chosen[unit.name] = option
            if search(position + 1, tuple(spend)):
                return True
            del chosen[unit.name]
        return False

    if not search(0, tuple(capacity)):
        raise ValueError(
            "no artifact-type assignment satisfies LABELING_GUIDE.md 2.2 against this ontology's "
            f"output_types; capacity was {dict(zip(GROUP_NAMES, capacity))}"
        )
    return chosen


def _targeted_groups(construct: schema.Construct) -> tuple[int, ...]:
    """Groups a construct's designed items may use, in the guide's own order."""
    allowed = sorted(
        {
            GROUP_OF_FORMAT[fmt]
            for fmt in construct.output_types
            if fmt is not UNSAMPLED_FORMAT and fmt in GROUP_OF_FORMAT
        }
    )
    if construct.key in SCHEDULE_LEVEL_CONSTRUCTS:
        bottom = {GROUP_NAMES.index(name) for name in BOTTOM_35_GROUPS}
        allowed = [group for group in allowed if group in bottom]
    if not allowed:
        raise schema.OntologyError(
            f"{construct.key}: every declared output type is outside the round-1 artifact-type table"
        )
    return tuple(allowed)


def _assign_formats(ontology: schema.Ontology) -> tuple[dict[str, tuple[int, ...]], str]:
    """Output-type groups for every lineage, and which of the two solves produced them.

    The strict solve additionally requires the scenario carrying each construct's
    canonical positive and its sibling hard negative to sit on a type the sibling
    also accepts, so that LABELING_GUIDE.md §2.3's claim - that all 120 targeted
    items are discrimination tests - holds rather than being asserted. On this
    ontology that is satisfiable inside the budget. It is attempted first and, if a
    future ontology puts the two in conflict, the relaxed solve runs instead and
    the audit fails naming the pairs it could not place.
    """
    units = _format_units(ontology)
    strict = tuple(
        replace(unit, options=tuple(option for option in unit.options if option[0] in unit.prefer_first) or unit.options)
        if unit.prefer_first
        else unit
        for unit in units
    )
    try:
        return _solve_format_budget(strict, GROUP_CAPACITY, OPEN_POOL_TOTAL), "strict"
    except ValueError:
        return _solve_format_budget(units, GROUP_CAPACITY, OPEN_POOL_TOTAL), "relaxed"


def _shared_groups(ontology: schema.Ontology, construct: schema.Construct) -> tuple[int, ...]:
    """Groups whose type both the construct and its `sibling_confusable` accept.

    LABELING_GUIDE.md §2.3 says every targeted item is a discrimination test
    because the target and its confusable are both in the candidate set. That only
    holds where the confusable can apply to the artifact at all: on a type outside
    its `output_types` a rater marks `not_applicable_wrong_artifact`, which is a
    real judgment and not a discrimination. Twelve constructs share only the
    conversational group with their sibling, so requiring this everywhere would
    push half the pool into tutor turns and confound construct with type. It is
    therefore a preference on the scenario that carries the pair.
    """
    sibling = ontology.sibling_of(construct.key)
    allowed = set(_targeted_groups(construct))
    return tuple(
        sorted(
            {
                GROUP_OF_FORMAT[fmt]
                for fmt in construct.output_types
                if fmt is not UNSAMPLED_FORMAT and fmt in GROUP_OF_FORMAT and sibling.accepts(fmt)
            }
            & allowed
        )
    )


def _format_units(ontology: schema.Ontology) -> tuple[_FormatUnit, ...]:
    units: list[_FormatUnit] = []
    for construct in ontology:
        groups = _targeted_groups(construct)
        pairs = [(first, second) for first in groups for second in groups if first != second]
        # A construct with one usable group keeps it for both scenarios; the
        # atypical item then changes type inside the group instead.
        units.append(
            _FormatUnit(
                f"targeted:{construct.key}",
                (4, 1),
                tuple(pairs) or ((groups[0], groups[0]),),
                prefer_first=_shared_groups(ontology, construct),
            )
        )
    for control in ontology.debunked_controls:
        triples = tuple(
            (first, second, third)
            for first in range(len(GROUP_NAMES))
            for second in range(first + 1, len(GROUP_NAMES))
            for third in range(second + 1, len(GROUP_NAMES))
        )
        # Three items in three different types: a control that only ever appears
        # as a tutor turn cannot detect drift anywhere else.
        units.append(_FormatUnit(f"control:{control.key}", (1, 1, 1), triples))
    return tuple(units)


def _pick_format(
    group: int, construct: schema.Construct | None, *, alternate: bool, avoid: Sequence[schema.OutputFormat] = ()
) -> schema.OutputFormat:
    """The concrete type inside a budget group.

    `alternate` asks for the group's second type, which is how the atypical
    positive of a construct with only one usable group still lands on an unusual
    artifact type, and how both members of a two-format group get used. `avoid` is
    for the debunked controls, which skip `study_schedule`: only two constructs
    accept that type, so a control written as one could not be put in front of six
    constructs that might mistake it for something.
    """
    usable = [fmt for fmt in FORMAT_GROUPS[group][1] if construct is None or construct.accepts(fmt)]
    formats = [fmt for fmt in usable if fmt not in avoid] or usable
    if not formats:  # pragma: no cover - _targeted_groups filters this out first
        raise schema.OntologyError(f"group {GROUP_NAMES[group]!r} has no type {construct.key if construct else '?'} accepts")
    if len(formats) == 1:
        return formats[0]
    prefers_second = construct is not None and construct.key in MULTI_TURN_CONSTRUCTS
    return formats[1] if (alternate != prefers_second) else formats[0]


# --------------------------------------------------------------- candidate sets


def _control_candidates(
    ontology: schema.Ontology,
    control: schema.DebunkedControl,
    appearances: Mapping[str, int],
    output_format: schema.OutputFormat,
) -> tuple[str, ...]:
    """The six constructs one control item is judged against.

    LABELING_GUIDE.md §2.3: a control has no `sibling_confusable` to pair with, so
    its six are the constructs it is most likely to be mistaken for - the ones its
    `expected_label` names where it names any, and otherwise the warmth- and
    framing-adjacent constructs a drifting instrument would reward.

    Every one has to accept this item's artifact type. A control shown to a
    construct that cannot apply to it collects `not_applicable_wrong_artifact` and
    detects no drift, which is the one thing a control is for.
    """
    eligible = [construct.key for construct in ontology if construct.accepts(output_format)]
    named = [key for key in eligible if key in f"{control.expected_label} {control.why}"]
    chosen: list[str] = list(dict.fromkeys(named))
    for key in DRIFT_ADJACENT_CONSTRUCTS:
        if len(chosen) >= CANDIDATE_SET_SIZE:
            break
        if key not in chosen and key in eligible:
            chosen.append(key)
    remaining = sorted((key for key in eligible if key not in chosen), key=lambda key: (appearances.get(key, 0), key))
    chosen.extend(remaining[: CANDIDATE_SET_SIZE - len(chosen)])
    if len(chosen) < CANDIDATE_SET_SIZE:
        raise schema.OntologyError(
            f"{control.key}: only {len(chosen)} constructs accept {output_format.value}, and a control needs "
            f"{CANDIDATE_SET_SIZE} that could be mistaken for it"
        )
    return tuple(sorted(chosen[:CANDIDATE_SET_SIZE]))


def _assign_fillers(
    ontology: schema.Ontology, targeted: Sequence[tuple[str, str]], appearances: dict[str, int]
) -> dict[str, tuple[str, ...]]:
    """Four fillers per targeted item, allocated to the appearance floor first.

    LABELING_GUIDE.md §2.3 puts a floor of 45 candidate-set appearances on every
    construct and notes the shortfall is 319 against 480 filler slots, so the
    allocation is not tight - but it is also not automatic, because eleven
    constructs are nobody's `sibling_confusable` and reach the floor only here.
    Greedy on the current count is what keeps the minimum rising.

    The secondary key is a blinding repair the guide does not ask for. A
    candidate set of {target, sibling(target), fillers} identifies its target
    whenever exactly one candidate's sibling is also present, and sibling edges
    are public in `ontology.yaml`. Preferring a filler that adds a second edge
    costs nothing when appearance counts tie, and the manifest reports how many
    items remain uniquely recoverable rather than claiming the problem away.
    """
    sets: dict[str, tuple[str, ...]] = {}
    for unit_id, concept in targeted:
        target = ontology.get(concept)
        base = [target.key, target.sibling_key]
        picked: list[str] = []
        for _ in range(CANDIDATE_SET_SIZE - len(base)):
            chosen_now = set(base + picked)
            siblings_of_chosen = {ontology.get(key).sibling_key for key in chosen_now}
            best: schema.Construct | None = None
            best_rank: tuple[int, bool, str] | None = None
            for candidate in ontology:
                if candidate.key in chosen_now or candidate.family is target.family:
                    continue
                decoy = candidate.sibling_key in chosen_now or candidate.key in siblings_of_chosen
                rank = (appearances[candidate.key], not decoy, candidate.key)
                if best_rank is None or rank < best_rank:
                    best, best_rank = candidate, rank
            if best is None:  # pragma: no cover - every family leaves 18 constructs eligible
                raise schema.OntologyError(f"{target.key}: no out-of-family filler left for {unit_id}")
            picked.append(best.key)
            appearances[best.key] += 1
        sets[unit_id] = tuple(sorted(base + picked))
    return sets


def _sibling_edges(ontology: schema.Ontology, keys: Iterable[str]) -> int:
    present = set(keys)
    return sum(1 for key in present if ontology.get(key).sibling_key in present)


# ----------------------------------------------------------------- plan builder


def _clause(construct: schema.Construct | None, stem: Stem, second_case: str) -> str:
    """The declared context field this construct cannot be labelled without."""
    if construct is None:
        return ""
    template = CONTEXT_CLAUSES.get(construct.key, "")
    return template.format(target=stem.target, topic=stem.topic, second_case=second_case) if template else ""


def _learner_state(construct: schema.Construct | None, fmt: schema.OutputFormat, counter: int) -> schema.LearnerState:
    if construct is not None and construct.key in REQUIRED_LEARNER_STATE:
        return REQUIRED_LEARNER_STATE[construct.key]
    pool = CONVERSATIONAL_STATES if fmt in (_F.TUTOR_TURN, _F.DIALOGUE_EPISODE, _F.WORKED_SOLUTION) else AUTHORING_STATES
    return pool[counter % len(pool)]


def _scenario(
    *,
    construct: schema.Construct | None,
    stem: Stem,
    clause: str,
    fmt: schema.OutputFormat,
    counter: int,
    index: int,
) -> schema.Scenario:
    state = _learner_state(construct, fmt, counter)
    modes = FORMAT_PROMPT_MODES[fmt]
    mode = modes[counter % len(modes)]
    task = " ".join(
        part
        for part in (
            f"Topic: {stem.topic}.",
            FORMAT_TASKS[fmt].format(problem=stem.problem, target=stem.target),
            clause,
        )
        if part
    )
    learner = LEARNER_TURNS[state].format(worked_step=stem.worked_step, misconception=stem.misconception)
    shows_error = state in (schema.LearnerState.MISCONCEPTION_STATED, schema.LearnerState.PARTIAL_ATTEMPT)
    return schema.Scenario(
        domain=stem.domain,
        output_format=fmt,
        learner_state=state,
        prompt_mode=mode,
        index=index,
        topic=stem.topic,
        target=stem.target,
        task=task,
        reference=stem.reference,
        student_before=learner,
        misstep=stem.misstep if shows_error else "",
        worked_step=stem.worked_step,
    )


def _directive(
    ontology: schema.Ontology,
    *,
    construct: schema.Construct | None,
    control: schema.DebunkedControl | None,
    role: schema.ItemRole,
    names: bool,
    exemplary: bool,
    boundary: str,
    forbidden: Sequence[str],
) -> str:
    if role is schema.ItemRole.OPEN_POOL:
        return "Answer the request above as a competent tutor would, with no further constraints."
    if role is schema.ItemRole.DEBUNKED_CONTROL:
        assert control is not None
        return (
            f"Answer the request above in a way that leans on one widely held claim: {control.display_name}. "
            "Assert it as though it were established, and let it drive the advice you give."
        )
    assert construct is not None
    parts: list[str] = []
    # Every anchor below is quoted from `ontology.yaml`, and every one of them is
    # written in some subject the ontology happened to pick. The scenario's subject
    # is fixed, so the anchor is handed over as a pattern to transpose rather than
    # as content to reuse - otherwise a chemistry anchor arrives inside an algebra
    # item and the artifact is about the wrong thing.
    transpose = "Do for the subject above what this example does"
    if role in (
        schema.ItemRole.POSITIVE_CANONICAL,
        schema.ItemRole.NAMED_AND_ENACTED,
        schema.ItemRole.POSITIVE_ATYPICAL,
    ):
        parts.append(f"{transpose}, unambiguously: {construct.positive_anchor}")
    if role is schema.ItemRole.POSITIVE_ATYPICAL:
        parts.append(
            "Use an unusual surface form for this kind of artifact, and use none of these words anywhere: "
            f"{', '.join(forbidden)}."
        )
    if role is schema.ItemRole.POSITIVE_PARTIAL:
        parts.append(f"{transpose}: {construct.positive_anchor}")
        parts.append(
            "Then make it a partial instance rather than a sound one: keep the defining feature, and omit "
            f"exactly one element it needs, so that this condition is not met: {boundary}"
        )
    if exemplary:
        parts.append(
            "Also respect, as far as one artifact can, the conditions attached to this move: "
            f"{' '.join(construct.boundary_conditions[:2])}"
        )
    if role is schema.ItemRole.SIBLING_HARD_NEGATIVE:
        sibling = ontology.sibling_of(construct.key)
        parts.append(f"{transpose}, and do it competently: {sibling.positive_anchor}")
        parts.append(
            "It must also share the surface form of a neighbouring move without performing it. That near "
            f"miss looks like this: {construct.hard_negative}"
        )
        parts.append(f"Keep this distinction on the right side: {construct.sibling_confusable.discriminator}")
    if role is schema.ItemRole.SAYS_ONLY:
        parts.append(
            "The artifact must discuss a teaching principle correctly while containing none of it. Do for "
            f"the subject above what this example does: {construct.says_only_negative}"
        )
    if names:
        parts.append("Name the principle you are following in the artifact, and say in one clause what it is.")
    elif role is not schema.ItemRole.POSITIVE_ATYPICAL:
        parts.append("Do not name, label or discuss any teaching principle in the artifact.")
    return " ".join(parts)


def _prompts(
    scenario: schema.Scenario, directive: str, *, has_learner_turn: bool
) -> tuple[str, str]:
    system = (
        f"{PROMPT_MODE_FRAMING[scenario.prompt_mode]} Write only the artifact that is asked for, in plain text."
    )
    lines = [
        f"Target knowledge: {scenario.target}.",
        f"Request: {scenario.task}",
        f"Reference solution, for your use and not to be pasted wholesale: {scenario.reference}",
    ]
    if has_learner_turn:
        lines.append(f"The learner has just written: \"{scenario.student_before}\"")
    if scenario.misstep:
        lines.append(f"Their first error is: {scenario.misstep}.")
    lines.append(f"Form: {FORMAT_BRIEFS[scenario.output_format]}")
    lines.append(f"Constraints: {directive}")
    lines.append(
        "Output rules: reply with the artifact text and nothing else. No title, no preamble, no closing "
        "comment, no markdown code fence, and no reference to these instructions."
    )
    return system, "\n".join(lines)


def _scaffold(
    ontology: schema.Ontology,
    *,
    construct: schema.Construct | None,
    control: schema.DebunkedControl | None,
    role: schema.ItemRole,
    scenario: schema.Scenario,
    boundary: str,
) -> str:
    """A deterministic stand-in, marked, built from the ontology's own anchors.

    It exists so the labelling and packet code can be exercised without a GPU. It
    is marked because a placeholder that cannot be distinguished from a
    generation will eventually be labelled as one, and every key row carrying one
    records `scaffold` as its source model. The stem index is in the text so that
    two placeholders written from the same anchor are still distinguishable, which
    is what the near-duplicate check needs to mean something.
    """
    head = f"{SCAFFOLD_MARK} stem {scenario.index}, {scenario.topic}: {scenario.target}."
    if role is schema.ItemRole.OPEN_POOL:
        body = f"{scenario.reference} {scenario.worked_step}"
    elif role is schema.ItemRole.DEBUNKED_CONTROL:
        body = f"{control.display_name}. Status in the ontology: {control.status}." if control else scenario.reference
    elif construct is None:  # pragma: no cover - only the two pool roles lack a construct
        body = scenario.reference
    elif role is schema.ItemRole.SIBLING_HARD_NEGATIVE:
        body = f"{ontology.sibling_of(construct.key).positive_anchor} {construct.hard_negative}"
    elif role is schema.ItemRole.SAYS_ONLY:
        body = construct.says_only_negative
    elif role is schema.ItemRole.POSITIVE_PARTIAL:
        body = f"{construct.positive_anchor} It falls short of this: {boundary}"
    elif role is schema.ItemRole.POSITIVE_ATYPICAL:
        body = f"{construct.positive_anchor} {scenario.worked_step}"
    else:
        body = construct.positive_anchor
    return _despecify(f"{head} {body}", ontology)


@dataclass
class _PoolBuilder:
    """The two counters a build shares, and the two things it makes with them.

    Both counters are state rather than arguments because both are global to the
    build. The stem rotation is what keeps a subject from being confounded with a
    construct - it hands out the next topic in a domain rather than the same one -
    and the lineage counter is part of `lineage_id`'s input, so it is also what
    guarantees two scenarios cannot collide. Neither can be computed locally, and
    both have to advance in a fixed order for the pool to be reproducible.
    """

    ontology: schema.Ontology
    salt: str
    stem_index: dict[schema.Domain, int] = field(default_factory=lambda: {domain: 0 for domain in schema.Domain})
    lineages: int = 0

    def next_stem(self, domain: schema.Domain) -> tuple[Stem, str]:
        """The domain's next stem, and the topic after it as a second case to compare with."""
        pool = STEMS_BY_DOMAIN[domain]
        index = self.stem_index[domain]
        self.stem_index[domain] = index + 1
        return pool[index % len(pool)], pool[(index + 1) % len(pool)].topic

    def lineage(
        self,
        *,
        block: str,
        construct: schema.Construct | None,
        group: int,
        alternate: bool,
        domain: schema.Domain,
        avoid: Sequence[schema.OutputFormat] = (),
    ) -> tuple[schema.Scenario, str, str]:
        """One scenario, its id, and the context clause its construct needs declared."""
        stem, second_case = self.next_stem(domain)
        fmt = _pick_format(group, construct, alternate=alternate, avoid=avoid)
        clause = _clause(construct, stem, second_case)
        scenario = _scenario(
            construct=construct, stem=stem, clause=clause, fmt=fmt, counter=self.lineages, index=self.lineages
        )
        concept = construct.key if construct is not None else OPEN_POOL_CONCEPT
        lineage = schema.lineage_id(
            concept,
            scenario.domain,
            fmt,
            scenario.learner_state,
            scenario.prompt_mode,
            scenario.index,
            salt=f"{self.salt}:{block}",
        )
        self.lineages += 1
        return scenario, lineage, clause

    def item(
        self,
        *,
        block: str,
        pool: str,
        concept: str,
        construct: schema.Construct | None,
        control: schema.DebunkedControl | None,
        role: schema.ItemRole,
        names: bool,
        enacts: bool,
        fidelity: int | None,
        scenario: schema.Scenario,
        lineage: str,
        clause: str,
        exemplary: bool = False,
        boundary: str = "",
    ) -> PlanItem:
        """One planned item: its brief, its scaffold, and the truth withheld about it."""
        atypical = role is schema.ItemRole.POSITIVE_ATYPICAL
        forbidden = trigger_words(construct) if (construct is not None and atypical) else ()
        directive = _directive(
            self.ontology,
            construct=construct,
            control=control,
            role=role,
            names=names,
            exemplary=exemplary,
            boundary=boundary,
            forbidden=forbidden,
        )
        has_learner_turn = scenario.output_format in (_F.TUTOR_TURN, _F.DIALOGUE_EPISODE, _F.WORKED_SOLUTION)
        system, user = _prompts(scenario, directive, has_learner_turn=has_learner_turn)
        enacted = None
        if enacts:
            enacted = concept
        elif role is schema.ItemRole.SIBLING_HARD_NEGATIVE and construct is not None:
            enacted = construct.sibling_key
        return PlanItem(
            unit_id=schema.artifact_id(lineage, role, salt=f"{self.salt}:{block}"),
            lineage_id=lineage,
            block=block,
            pool=pool,
            concept=concept,
            role=role,
            enactment=schema.NamesEnacts(names=names, enacts=enacts),
            scenario=scenario,
            candidates=(),
            system_prompt=system,
            user_prompt=user,
            max_new_tokens=FORMAT_MAX_TOKENS[scenario.output_format],
            expected_fidelity=fidelity if enacts else None,
            enacted_construct=enacted,
            named_construct=concept if names else None,
            sibling_expected_fidelity=2 if role is schema.ItemRole.SIBLING_HARD_NEGATIVE else None,
            forbidden_words=forbidden,
            context_clause=clause,
            scaffold=_scaffold(
                self.ontology,
                construct=construct,
                control=control,
                role=role,
                scenario=scenario,
                boundary=boundary,
            ),
        )


def _targeted_items(builder: _PoolBuilder, assignment: Mapping[str, tuple[int, ...]]) -> list[PlanItem]:
    """The 120 designed items: five roles per construct, over two scenarios.

    Four roles share the first scenario, which is what makes the canonical positive
    and the sibling hard negative a matched pair and the says-only item a matched
    control on the same content. The atypical positive gets its own scenario,
    because it is the item required to be in an unusual form.
    """
    items: list[PlanItem] = []
    for position, construct in enumerate(builder.ontology):
        group_a, group_b = assignment[f"targeted:{construct.key}"]
        scenario_a, lineage_a, clause_a = builder.lineage(
            block=ROUND1_BLOCK,
            construct=construct,
            group=group_a,
            alternate=False,
            domain=DOMAIN_ORDER[position % len(DOMAIN_ORDER)],
        )
        # Half the canonical positives name their principle as well as enacting it.
        # Without that cell, naming is perfectly anti-correlated with enacting and a
        # probe wins by reading topic and inverting.
        names_canonical = position % 2 == 1
        # Roughly one positive in six is written to satisfy the boundary conditions
        # too, which is the rate LABELING_GUIDE.md §3.3 expects a 3 to occur at.
        exemplary = position % 6 == 0
        shared = dict(
            block=ROUND1_BLOCK, pool=TARGETED_POOL, concept=construct.key, construct=construct, control=None
        )
        first = dict(scenario=scenario_a, lineage=lineage_a, clause=clause_a)
        items.append(
            builder.item(
                **shared,
                **first,
                role=schema.ItemRole.POSITIVE_CANONICAL,
                names=names_canonical,
                enacts=True,
                fidelity=3 if exemplary else 2,
                exemplary=exemplary,
            )
        )
        items.append(
            builder.item(
                **shared,
                **first,
                role=schema.ItemRole.POSITIVE_PARTIAL,
                names=False,
                enacts=True,
                fidelity=1,
                boundary=_boundary(construct, position),
            )
        )
        items.append(
            builder.item(
                **shared, **first, role=schema.ItemRole.SIBLING_HARD_NEGATIVE, names=False, enacts=False, fidelity=None
            )
        )
        items.append(
            builder.item(**shared, **first, role=schema.ItemRole.SAYS_ONLY, names=True, enacts=False, fidelity=None)
        )
        scenario_b, lineage_b, clause_b = builder.lineage(
            block=ROUND1_BLOCK,
            construct=construct,
            group=group_b,
            alternate=group_b == group_a,
            domain=DOMAIN_ORDER[(position + 3) % len(DOMAIN_ORDER)],
        )
        items.append(
            builder.item(
                **shared,
                role=schema.ItemRole.POSITIVE_ATYPICAL,
                names=False,
                enacts=True,
                fidelity=2,
                scenario=scenario_b,
                lineage=lineage_b,
                clause=clause_b,
            )
        )
    return items


def _control_items(
    builder: _PoolBuilder, assignment: Mapping[str, tuple[int, ...]], appearances: dict[str, int]
) -> tuple[list[PlanItem], dict[str, tuple[str, ...]]]:
    """The 18 debunked-control items, three per control and each in a different type.

    Their candidate sets are chosen here rather than with the rest, because they
    depend on the artifact type the budget gave the item: a control put in front of
    a construct that cannot apply to it collects an inapplicable judgment and
    detects nothing. `appearances` is updated as they are chosen, so the filler
    allocation that follows sees these appearances.
    """
    items: list[PlanItem] = []
    sets: dict[str, tuple[str, ...]] = {}
    for position, control in enumerate(builder.ontology.debunked_controls):
        for slot, group in enumerate(assignment[f"control:{control.key}"]):
            scenario, lineage, clause = builder.lineage(
                block=ROUND1_BLOCK,
                construct=None,
                group=group,
                alternate=slot % 2 == 1,
                domain=DOMAIN_ORDER[(position * ITEMS_PER_CONTROL + slot) % len(DOMAIN_ORDER)],
                avoid=(_F.STUDY_SCHEDULE,),
            )
            candidates = _control_candidates(builder.ontology, control, appearances, scenario.output_format)
            for key in candidates:
                appearances[key] += 1
            item = builder.item(
                block=ROUND1_BLOCK,
                pool=CONTROL_POOL,
                concept=control.key,
                construct=None,
                control=control,
                role=schema.ItemRole.DEBUNKED_CONTROL,
                names=False,
                enacts=False,
                fidelity=None,
                scenario=scenario,
                lineage=lineage,
                clause=clause,
            )
            items.append(item)
            sets[item.unit_id] = candidates
    return items, sets


def _open_pool_items(builder: _PoolBuilder, groups: Sequence[int]) -> list[PlanItem]:
    """The 22 natural artifacts, which have no designed target and no directive.

    They carry the whole 24-construct checklist and they are the only source of
    base rates, so their briefs mention no construct at all (LABELING_GUIDE.md
    §2.3). They also absorb whatever the artifact-type budget has left, which is
    how the exact totals close.
    """
    items: list[PlanItem] = []
    for position, group in enumerate(groups):
        scenario, lineage, clause = builder.lineage(
            block=ROUND1_BLOCK,
            construct=None,
            group=group,
            alternate=position % 2 == 1,
            domain=DOMAIN_ORDER[position % len(DOMAIN_ORDER)],
        )
        items.append(
            builder.item(
                block=ROUND1_BLOCK,
                pool=OPEN_POOL,
                concept=OPEN_POOL_CONCEPT,
                construct=None,
                control=None,
                role=schema.ItemRole.OPEN_POOL,
                names=False,
                enacts=False,
                fidelity=None,
                scenario=scenario,
                lineage=lineage,
                clause=clause,
            )
        )
    return items


def _calibration_items(builder: _PoolBuilder) -> list[PlanItem]:
    """The 24 calibration items of LABELING_GUIDE.md §2.4, one per construct.

    Disjoint from the 160 and never counted in a reliability statistic, so they are
    salted into their own block and cannot collide. The role rotates across
    constructs so the calibration session argues about every kind of item: a session
    made only of clean positives calibrates nobody on the says-only trap, and the
    two gates that rest on one item each are exactly the ones a rater needs to have
    argued about before meeting them.
    """
    roles = ROLES_PER_CONSTRUCT + (schema.ItemRole.NAMED_AND_ENACTED,)
    enacting = {
        schema.ItemRole.POSITIVE_CANONICAL,
        schema.ItemRole.POSITIVE_ATYPICAL,
        schema.ItemRole.POSITIVE_PARTIAL,
        schema.ItemRole.NAMED_AND_ENACTED,
    }
    items: list[PlanItem] = []
    for position, construct in enumerate(builder.ontology):
        role = roles[position % len(roles)]
        groups = _targeted_groups(construct)
        scenario, lineage, clause = builder.lineage(
            block=CALIBRATION_BLOCK,
            construct=construct,
            group=groups[position % len(groups)],
            alternate=position % 2 == 1,
            domain=DOMAIN_ORDER[(position * 5) % len(DOMAIN_ORDER)],
        )
        enacts = role in enacting
        fidelity = None
        if enacts:
            fidelity = 1 if role is schema.ItemRole.POSITIVE_PARTIAL else 2
        item = builder.item(
            block=CALIBRATION_BLOCK,
            pool=TARGETED_POOL,
            concept=construct.key,
            construct=construct,
            control=None,
            role=role,
            names=role in (schema.ItemRole.SAYS_ONLY, schema.ItemRole.NAMED_AND_ENACTED),
            enacts=enacts,
            fidelity=fidelity,
            scenario=scenario,
            lineage=lineage,
            clause=clause,
            boundary=_boundary(construct, position),
        )
        items.append(replace(item, candidates=(construct.key, construct.sibling_key)))
    return items


def build_plan(ontology: schema.Ontology, *, salt: str = DEFAULT_SALT) -> Plan:
    """Both manifests, deterministically, from the ontology and a salt.

    Nothing here draws from a random source. Ids come from `schema.stable_digest`,
    which is content-keyed rather than hash-salted per interpreter, so the same
    ontology and salt rebuild the same pool in a different process - which is what
    makes a preempted generation job resumable and a published corpus reproducible
    from its recorded salt.

    The order of the four blocks is load-bearing and not cosmetic. Lineage ids hash
    a counter, so building the controls before the targeted items would renumber
    every scenario after them; and the candidate-set allocation runs last on
    purpose, because the appearance floor it has to satisfy counts the appearances
    the other three blocks have already spent.
    """
    if len(ontology) != N_CONSTRUCTS:
        raise schema.OntologyError(
            f"the round-1 composition is fixed at {N_CONSTRUCTS} constructs x {ITEMS_PER_CONSTRUCT} items; "
            f"this ontology has {len(ontology)}"
        )
    n_controls = len(ontology.debunked_controls)
    if n_controls * ITEMS_PER_CONTROL + TARGETED_TOTAL + OPEN_POOL_TOTAL != ROUND1_TOTAL:
        raise schema.OntologyError(
            f"{n_controls} debunked controls x {ITEMS_PER_CONTROL} items do not close the "
            f"{ROUND1_TOTAL}-item round with {TARGETED_TOTAL} targeted and {OPEN_POOL_TOTAL} open-pool items"
        )

    assignment, solve = _assign_formats(ontology)
    spent = [0] * len(GROUP_NAMES)
    for name, option in assignment.items():
        weights = (4, 1) if name.startswith("targeted:") else (1, 1, 1)
        for weight, group in zip(weights, option):
            spent[group] += weight
    open_groups: list[int] = []
    for group, capacity in enumerate(GROUP_CAPACITY):
        open_groups.extend([group] * (capacity - spent[group]))
    if len(open_groups) != OPEN_POOL_TOTAL:  # pragma: no cover - the solver guarantees this
        raise ValueError(f"open pool needs {OPEN_POOL_TOTAL} slots, the budget left {len(open_groups)}")

    builder = _PoolBuilder(ontology=ontology, salt=salt)
    targeted = _targeted_items(builder, assignment)

    # Each construct appears as its own items' target, and in the candidate set of
    # every item whose target names it as a sibling_confusable.
    appearances = {key: 0 for key in ontology.keys}
    for item in targeted:
        appearances[item.concept] += 1
        appearances[ontology.get(item.concept).sibling_key] += 1

    controls, control_sets = _control_items(builder, assignment, appearances)
    open_items = _open_pool_items(builder, open_groups)
    for key in appearances:
        appearances[key] += len(open_items)  # the open pool carries the full checklist

    filler_sets = _assign_fillers(ontology, [(item.unit_id, item.concept) for item in targeted], appearances)
    # Alphabetical, like every other candidate set. The ontology's own order groups
    # constructs by family, and a checklist in that order tells a rater which
    # neighbours to compare against before they have read the artifact.
    checklist = tuple(sorted(ontology.keys))
    round1 = (
        tuple(replace(item, candidates=filler_sets[item.unit_id]) for item in targeted)
        + tuple(replace(item, candidates=control_sets[item.unit_id]) for item in controls)
        + tuple(replace(item, candidates=checklist) for item in open_items)
    )

    return Plan(
        construct_version=ontology.construct_version,
        salt=salt,
        round1=round1,
        calibration=tuple(_calibration_items(builder)),
        retest=_retest_plan(round1, salt=salt),
        appearances=dict(appearances),
        format_solve=solve,
    )


def _retest_plan(items: Sequence[PlanItem], *, salt: str) -> tuple[Mapping[str, object], ...]:
    """The 40 blinded retest judgments of LABELING_GUIDE.md §4.

    Re-salted ids, so an item cannot be matched to its original by lookup, and
    one targeted item per construct with the role rotating across constructs. The
    mapping lives under `withheld/`: it is the answer key to the question of which
    forty were repeated.
    """
    chosen: list[Mapping[str, object]] = []
    targeted = [item for item in items if item.pool == TARGETED_POOL]
    by_concept: dict[str, list[PlanItem]] = {}
    for item in targeted:
        by_concept.setdefault(item.concept, []).append(item)
    for position, concept in enumerate(sorted(by_concept)):
        pool = sorted(by_concept[concept], key=lambda item: ROLES_PER_CONSTRUCT.index(item.role))
        pick = pool[position % len(pool)]
        chosen.append({"stratum": TARGETED_POOL, "of_unit_id": pick.unit_id, "item_role": pick.role.value})
    for stratum, pool_name in ((CONTROL_POOL, CONTROL_POOL), (OPEN_POOL, OPEN_POOL)):
        pool = [item for item in items if item.pool == pool_name]
        wanted = RETEST_STRATA[stratum]
        stride = max(len(pool) // wanted, 1)
        for index in range(wanted):
            pick = pool[(index * stride) % len(pool)]
            chosen.append({"stratum": stratum, "of_unit_id": pick.unit_id, "item_role": pick.role.value})
    out = []
    for entry in chosen:
        retest_id = "rt" + schema.stable_digest(salt, "retest", str(entry["of_unit_id"]), size=7)
        out.append({**entry, "retest_id": retest_id})
    return tuple(out)


# --------------------------------------------------------------------- prompting


def prompt_digest(system: str, user: str) -> str:
    """The identity of a brief. A resumed job compares this, not the unit id alone."""
    payload = f"{system}\x1f{user}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def sanitize_generation(raw: str, item: PlanItem, leak_tokens: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """A generation, or the reasons it may not enter the corpus.

    Refusing rather than repairing, on the same reasoning as
    `labeling.parse_label`: a pipeline that strips a preamble hides the fact that
    the brief was not followed, and the refusal rate is the signal worth seeing
    early. The one exception is a wrapping code fence, which is a rendering
    artifact rather than content.
    """
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    reasons: list[str] = []
    if not text:
        reasons.append("empty")
    elif len(text) < MIN_ARTIFACT_CHARS:
        reasons.append(f"shorter than {MIN_ARTIFACT_CHARS} characters")
    lowered = text.lower()
    leaked = sorted({token for token in leak_tokens if token in lowered})
    if leaked:
        reasons.append(f"withheld identifier(s) in the text: {', '.join(leaked)}")
    echoed = sorted({phrase for phrase in BRIEF_PHRASES if phrase in lowered})
    if echoed:
        reasons.append(f"echoes the brief: {', '.join(echoed)}")
    opener = next((phrase for phrase in PREAMBLE_STARTS if lowered.startswith(phrase)), "")
    if opener:
        reasons.append(f"opens with a preamble: {opener!r}")
    if text.count("```") % 2:
        reasons.append("unbalanced code fence")
    if item.role is schema.ItemRole.POSITIVE_ATYPICAL and item.forbidden_words:
        hits = sorted({word for word in item.forbidden_words if word in lowered})
        if hits:
            reasons.append(f"trigger vocabulary an atypical positive must avoid: {', '.join(hits)}")
    return text, tuple(reasons)


# ------------------------------------------------------------- generation files


def read_generations(path: str | Path, *, repair: bool = True) -> tuple[dict[str, dict[str, object]], int]:
    """Generations by unit id, keeping the newest, tolerating one truncated tail line.

    A preempted job dies mid-write, so the last line of an append-only file can
    be half a JSON object. That one case is repaired; a broken line anywhere else
    is a corrupt file and raises, because silently skipping it would drop a
    generation and quietly change the corpus.
    """
    target = Path(path)
    if not target.exists():
        return {}, 0
    lines = target.read_text(encoding="utf-8").splitlines()
    rows: dict[str, dict[str, object]] = {}
    dropped = 0
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            if number != len(lines):
                raise ValueError(f"{target}:{number}: not JSON, and not the final line: {exc}") from exc
            dropped += 1
            continue
        if not isinstance(row, dict) or not row.get("unit_id"):
            raise ValueError(f"{target}:{number}: a generation row needs a unit_id")
        rows[str(row["unit_id"])] = row
    if dropped and repair:
        kept = [line for line in lines[:-1] if line.strip()]
        target.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return rows, dropped


def pending_units(plan: Plan, done: Mapping[str, Mapping[str, object]]) -> tuple[PlanItem, ...]:
    """Items still to generate: absent, stale, or previously refused."""
    out = []
    for item in plan.items:
        row = done.get(item.unit_id)
        if (
            row is None
            or row.get("prompt_sha") != item.prompt_sha
            or bool(row.get("refusal_reasons"))
            or not str(row.get("text") or "").strip()
        ):
            out.append(item)
    return tuple(out)


def generate(
    plan: Plan,
    *,
    out_path: str | Path,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    max_attempts: int = 3,
    limit: int = 0,
    leak_tokens: Sequence[str] = (),
    resume: bool = True,
) -> dict[str, object]:
    """Write one artifact per pending item, appending as it goes.

    Torch and transformers are imported here rather than at module scope so that
    the plan, the audit and the whole test suite run in an environment with
    neither. Each item's sampling seed is derived from the run seed and the unit
    id, so a requeued job regenerates a lost item to the same text: resume cannot
    change a corpus that was already partly published.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    done, dropped = read_generations(target) if resume else ({}, 0)
    if not resume and target.exists():
        target.unlink()
    todo = pending_units(plan, done)
    if limit > 0:
        todo = todo[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model)
    weights = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights.to(device)
    weights.eval()
    policy = "greedy" if temperature <= 0 else f"sample-t{temperature}-p{top_p}"

    written = 0
    refused = 0
    with target.open("a", encoding="utf-8") as handle:
        for item in todo:
            text = ""
            reasons: tuple[str, ...] = ("not attempted",)
            attempt = 0
            for attempt in range(max_attempts):
                item_seed = int(schema.stable_digest(str(seed), item.unit_id, str(attempt), size=4), 16)
                torch.manual_seed(item_seed)
                raw = _one_generation(
                    tokenizer,
                    weights,
                    item,
                    device=device,
                    temperature=temperature,
                    top_p=top_p,
                    torch_module=torch,
                )
                text, reasons = sanitize_generation(raw, item, leak_tokens)
                if not reasons:
                    break
            if reasons:
                refused += 1
            row = {
                "schema": GENERATION_SCHEMA,
                "unit_id": item.unit_id,
                "block": item.block,
                "prompt_sha": item.prompt_sha,
                "text": text,
                "source_model": model,
                "source_policy": policy,
                "temperature": temperature,
                "sample_index": attempt,
                "seed": seed,
                "refusal_reasons": list(reasons),
            }
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    return {
        "model": model,
        "policy": policy,
        "written": written,
        "already_present": len(done),
        "still_refused": refused,
        "repaired_tail_lines": dropped,
    }


def _one_generation(tokenizer, weights, item: PlanItem, *, device: str, temperature: float, top_p: float, torch_module):
    """One artifact. Only the new tokens are decoded, so the brief cannot come back with them."""
    messages = [{"role": "system", "content": item.system_prompt}, {"role": "user", "content": item.user_prompt}]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{item.system_prompt}\n\n{item.user_prompt}\n\n"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    sampling = {"do_sample": False}
    if temperature > 0:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p}
    with torch_module.no_grad():
        out = weights.generate(
            **encoded,
            max_new_tokens=item.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **sampling,
        )
    new_tokens = out[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------- assembly


def assemble(
    plan: Plan,
    generations: Mapping[str, Mapping[str, object]],
    *,
    leak_tokens: Sequence[str],
    scaffold_missing: bool = False,
) -> Assembly:
    """Turn the plan plus generations into artifacts, refusing what does not belong.

    A generation whose `prompt_sha` does not match the plan is fatal rather than
    ignored: it was written for a different brief, and pairing it with this plan's
    key would mislabel the item it describes.
    """
    artifacts: dict[str, schema.Artifact] = {}
    provenance: dict[str, Mapping[str, object]] = {}
    rejections: list[Rejection] = []
    missing: list[str] = []
    scaffolded: list[str] = []

    for item in plan.items:
        row = generations.get(item.unit_id)
        text = ""
        source: Mapping[str, object] = {}
        if row is not None:
            if row.get("prompt_sha") != item.prompt_sha:
                raise ValueError(
                    f"{item.unit_id}: the stored generation was written for prompt {row.get('prompt_sha')!r} "
                    f"and this plan's brief digests to {item.prompt_sha!r}. Move the generations file aside "
                    "or rebuild the plan with the salt it was generated under."
                )
            text, reasons = sanitize_generation(str(row.get("text") or ""), item, leak_tokens)
            if reasons:
                rejections.append(Rejection(item.unit_id, reasons, text[:400]))
                text = ""
            else:
                source = {
                    "source_model": row.get("source_model"),
                    "source_policy": row.get("source_policy"),
                    "temperature": row.get("temperature"),
                    "sample_index": row.get("sample_index"),
                    "seed": row.get("seed"),
                }
        if not text:
            if not scaffold_missing:
                missing.append(item.unit_id)
                continue
            text = item.scaffold
            source = {"source_model": SCAFFOLD_MODEL, "source_policy": "ontology-anchor", "temperature": 0.0}
            scaffolded.append(item.unit_id)
        artifacts[item.unit_id] = item.artifact(text)
        provenance[item.unit_id] = source

    if missing:
        raise ValueError(
            f"{len(missing)} of {len(plan.items)} items have no usable generation "
            f"({len(rejections)} were refused). Generate them, or pass --scaffold-missing to build a "
            f"marked placeholder corpus. First few: {missing[:5]}"
        )
    return Assembly(artifacts, provenance, tuple(rejections), tuple(missing), tuple(scaffolded))


# ------------------------------------------------------------------------ audit


def _counter(values: Iterable[object]) -> dict[str, int]:
    """Counts keyed by string, sorted, so a manifest diff is a content diff.

    Enum members are keyed by their value: `str()` on a `str`-mixin enum returns
    `Class.MEMBER` on this Python, which would put the class name in the manifest.
    """
    out: dict[str, int] = {}
    for value in values:
        key = value.value if isinstance(value, Enum) else str(value)
        out[str(key)] = out.get(str(key), 0) + 1
    return dict(sorted(out.items()))


@dataclass
class _Report:
    """Accumulator for the audit. A check fails the build; a stat only informs it."""

    checks: list[Check] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def check(self, name: str, ok: object, detail: str) -> None:
        self.checks.append(Check(name, bool(ok), detail))

    def stat(self, name: str, value: object) -> None:
        self.stats[name] = value


@dataclass(frozen=True, slots=True)
class _Pools:
    """Round 1 split into the three blocks of LABELING_GUIDE.md §2.1."""

    targeted: tuple[PlanItem, ...]
    controls: tuple[PlanItem, ...]
    open_pool: tuple[PlanItem, ...]

    @property
    def all(self) -> tuple[PlanItem, ...]:
        return self.targeted + self.controls + self.open_pool

    def by_concept(self) -> dict[str, list[PlanItem]]:
        grouped: dict[str, list[PlanItem]] = {}
        for item in self.targeted:
            grouped.setdefault(item.concept, []).append(item)
        return grouped


def _check_composition(out: _Report, ontology: schema.Ontology, pools: _Pools) -> None:
    """The three fixed block sizes and the five roles inside each construct."""
    round1 = pools.all
    controls_wanted = ROUND1_TOTAL - TARGETED_TOTAL - OPEN_POOL_TOTAL
    out.check("round1_total", len(round1) == ROUND1_TOTAL, f"{len(round1)} items, expected {ROUND1_TOTAL}")
    out.check(
        "block_composition",
        (len(pools.targeted), len(pools.controls), len(pools.open_pool))
        == (TARGETED_TOTAL, controls_wanted, OPEN_POOL_TOTAL),
        f"targeted={len(pools.targeted)} controls={len(pools.controls)} open_pool={len(pools.open_pool)}, "
        f"expected {TARGETED_TOTAL}/{controls_wanted}/{OPEN_POOL_TOTAL}",
    )

    wanted_roles = sorted(role.value for role in ROLES_PER_CONSTRUCT)
    by_concept = pools.by_concept()
    wrong_roles = {
        concept: sorted(item.role.value for item in items)
        for concept, items in by_concept.items()
        if sorted(item.role.value for item in items) != wanted_roles
    }
    out.check(
        "five_roles_per_construct",
        len(by_concept) == N_CONSTRUCTS and not wrong_roles,
        f"{len(by_concept)} constructs with items; role-set mismatches: {wrong_roles or 'none'}",
    )

    control_counts = _counter(item.concept for item in pools.controls)
    out.check(
        "three_items_per_control",
        len(control_counts) == len(ontology.debunked_controls) and set(control_counts.values()) == {ITEMS_PER_CONTROL},
        f"{control_counts}",
    )


def _check_artifact_types(out: _Report, ontology: schema.Ontology, plan: Plan, pools: _Pools) -> None:
    """The §2.2 budget, which is a table of exact counts and not a target."""
    formats = _counter(item.output_format for item in pools.all)
    group_counts = {name: sum(formats.get(fmt.value, 0) for fmt in group) for name, group, _ in FORMAT_GROUPS}
    out.stat("format_counts", formats)
    out.stat("format_group_counts", group_counts)
    out.check(
        "artifact_type_budget",
        all(group_counts[name] == capacity for name, _, capacity in FORMAT_GROUPS),
        f"{group_counts} against {dict(zip(GROUP_NAMES, GROUP_CAPACITY))}",
    )
    out.check(
        "assessment_item_unsampled",
        formats.get(UNSAMPLED_FORMAT.value, 0) == 0,
        f"{formats.get(UNSAMPLED_FORMAT.value, 0)} {UNSAMPLED_FORMAT.value} items; round 1 samples none",
    )
    out.check(
        "every_sampled_type_present",
        all(formats.get(fmt.value, 0) > 0 for fmt in schema.OutputFormat if fmt is not UNSAMPLED_FORMAT),
        f"{formats}",
    )
    bad_types = [
        item.unit_id
        for item in pools.targeted + plan.calibration
        if not ontology.get(item.concept).accepts(item.output_format)
    ]
    out.check("declared_output_types", not bad_types, f"{len(bad_types)} items on a type their construct rejects")
    stray_schedule = [
        item.unit_id
        for item in pools.targeted
        if item.concept in SCHEDULE_LEVEL_CONSTRUCTS
        and item.enactment.enacts
        and GROUP_NAMES[GROUP_OF_FORMAT[item.output_format]] not in BOTTOM_35_GROUPS
    ]
    out.check(
        "schedule_positives_in_bottom_35",
        not stray_schedule,
        f"{len(stray_schedule)} schedule-level positives outside {list(BOTTOM_35_GROUPS)}",
    )


def _check_enactment(out: _Report, ontology: schema.Ontology, plan: Plan, pools: _Pools) -> None:
    """The names/enacts 2x2, the two one-item gates, and the matched sibling pairs."""
    targeted = pools.targeted
    cells = _counter(item.enactment.cell for item in targeted)
    out.stat("enactment_cells", cells)
    out.check(
        "all_four_enactment_cells",
        all(cells.get(cell.value, 0) > 0 for cell in schema.EnactmentCell),
        f"{cells}",
    )
    naming = [item for item in targeted if item.enactment.names]
    enacting = [item for item in targeted if item.enactment.enacts]
    mixed = (
        0 < len(naming) < len(targeted)
        and 0 < len(enacting) < len(targeted)
        and any(item.enactment.names for item in enacting)
        and any(not item.enactment.names for item in enacting)
        and any(item.enactment.enacts for item in naming)
        and any(not item.enactment.enacts for item in naming)
    )
    out.check(
        "names_and_enacts_marginals_mixed",
        mixed,
        f"names={len(naming)}/{len(targeted)} enacts={len(enacting)}/{len(targeted)}; "
        "neither marginal may be constant within the other",
    )

    says_only = _counter(item.concept for item in targeted if item.role is schema.ItemRole.SAYS_ONLY)
    hard_negative = _counter(item.concept for item in targeted if item.role is schema.ItemRole.SIBLING_HARD_NEGATIVE)
    out.check(
        "one_says_only_and_one_hard_negative_each",
        set(says_only.values()) == {1}
        and set(hard_negative.values()) == {1}
        and len(says_only) == len(hard_negative) == N_CONSTRUCTS,
        f"says_only per construct {set(says_only.values())}, hard negative {set(hard_negative.values())}; "
        "gates G4 and G5 rest on one item each and their denominator is two judgments",
    )

    contrasts = 0
    broken: list[str] = []
    for concept, items in pools.by_concept().items():
        negative = next(item for item in items if item.role is schema.ItemRole.SIBLING_HARD_NEGATIVE)
        canonical = next(item for item in items if item.role is schema.ItemRole.POSITIVE_CANONICAL)
        matched = negative.lineage_id == canonical.lineage_id
        enacts_sibling = negative.enacted_construct == ontology.get(concept).sibling_key
        if matched and enacts_sibling and not negative.enactment.enacts:
            contrasts += 1
        else:
            broken.append(concept)
    out.stat("sibling_contrast_pairs", contrasts)
    out.check(
        "quality_matched_sibling_contrasts",
        contrasts == N_CONSTRUCTS,
        f"{contrasts} of {N_CONSTRUCTS} constructs have a same-lineage negative that enacts the sibling; "
        f"broken: {broken or 'none'}",
    )

    out.stat("format_solve", plan.format_solve)
    out.stat(
        "items_whose_sibling_can_apply",
        sum(1 for item in targeted if ontology.sibling_of(item.concept).accepts(item.output_format)),
    )
    unmatched = [
        item.concept
        for item in targeted
        if item.role is schema.ItemRole.SIBLING_HARD_NEGATIVE
        and not ontology.sibling_of(item.concept).accepts(item.output_format)
    ]
    out.check(
        "hard_negatives_are_discriminations",
        not unmatched,
        f"{len(unmatched)} hard negatives sit on a type their sibling_confusable cannot apply to, where a "
        f"rater marks not_applicable_wrong_artifact instead of discriminating: {unmatched or 'none'}",
    )

    fidelities = _counter(item.expected_fidelity for item in targeted if item.expected_fidelity is not None)
    out.stat("expected_fidelity", fidelities)
    out.check(
        "fidelity_spread",
        len(fidelities) >= 3,
        f"{fidelities}; the ordinal carries no information if every positive is designed to the same score",
    )


def _check_candidate_sets(out: _Report, ontology: schema.Ontology, pools: _Pools) -> None:
    """§2.3: six candidates on a designed item, the full checklist on an open-pool one,
    fillers out of family, and the appearance floor every §7 denominator rests on."""
    targeted, controls, openpool = pools.targeted, pools.controls, pools.open_pool
    sizes = {len(item.candidates) for item in targeted + controls}
    out.check(
        "candidate_set_size",
        sizes == {CANDIDATE_SET_SIZE} and all(len(item.candidates) == len(ontology) for item in openpool),
        f"designed items {sorted(sizes)}, open pool {sorted({len(i.candidates) for i in openpool})}",
    )
    missing_pair = [
        item.unit_id
        for item in targeted
        if item.concept not in item.candidates or ontology.get(item.concept).sibling_key not in item.candidates
    ]
    out.check(
        "target_and_sibling_in_candidates",
        not missing_pair,
        f"{len(missing_pair)} targeted items without both the target and its sibling_confusable",
    )
    same_family = [
        item.unit_id
        for item in targeted
        for key in item.candidates
        if key not in (item.concept, ontology.get(item.concept).sibling_key)
        and ontology.get(key).family is ontology.get(item.concept).family
    ]
    out.check("fillers_from_other_families", not same_family, f"{len(same_family)} fillers share the target's family")
    duplicated = [item.unit_id for item in pools.all if len(set(item.candidates)) != len(item.candidates)]
    out.check("candidates_unique", not duplicated, f"{len(duplicated)} candidate sets repeat a construct")

    inapplicable = [
        (item.concept, key)
        for item in controls
        for key in item.candidates
        if not ontology.get(key).accepts(item.output_format)
    ]
    out.check(
        "control_candidates_can_apply",
        not inapplicable,
        f"{len(inapplicable)} control judgments would be not_applicable_wrong_artifact, which detects no "
        f"drift: {inapplicable[:3]}",
    )

    realized = {key: 0 for key in ontology.keys}
    for item in pools.all:
        for key in item.candidates:
            realized[key] += 1
    out.stat("candidate_appearances", dict(sorted(realized.items())))
    out.stat("candidate_appearance_min", min(realized.values()))
    out.check(
        "appearance_floor",
        min(realized.values()) >= APPEARANCE_FLOOR,
        f"lowest appearance count {min(realized.values())} against a floor of {APPEARANCE_FLOOR}; "
        "it is the denominator for every statistic in LABELING_GUIDE.md 7",
    )
    out.stat(
        "uniquely_recoverable_targets",
        sum(1 for item in targeted if _sibling_edges(ontology, item.candidates) == 1),
    )


def _check_scenarios(out: _Report, plan: Plan, pools: _Pools) -> None:
    """The balance axes, and the declared context three constructs are unlabelable without."""
    domains = _counter(item.scenario.domain for item in pools.all)
    learner_states = _counter(item.scenario.learner_state for item in pools.all)
    prompt_modes = _counter(item.scenario.prompt_mode for item in pools.all)
    out.stat("domains", domains)
    out.stat("learner_states", learner_states)
    out.stat("prompt_modes", prompt_modes)
    counts = list(domains.values())
    out.check(
        "domain_balance",
        len(domains) == len(schema.Domain) and max(counts) - min(counts) <= 4,
        f"{domains}; a construct confounded with a subject is not a construct",
    )
    out.check("learner_states_all_declared", len(learner_states) == len(schema.LearnerState), f"{learner_states}")
    out.check("prompt_modes_all_used", len(prompt_modes) == len(schema.PromptMode), f"{prompt_modes}")

    clause_gaps = [
        item.unit_id
        for item in pools.targeted + plan.calibration
        if CONTEXT_CLAUSES.get(item.concept)
        and (not item.context_clause or item.context_clause not in item.scenario.task)
    ]
    out.check(
        "required_context_declared",
        not clause_gaps,
        f"{len(clause_gaps)} items whose construct needs a declared context field the stem does not state; "
        "an item that arrives not_applicable_missing_context measured nothing",
    )


def _check_calibration_and_ids(out: _Report, ontology: schema.Ontology, plan: Plan, pools: _Pools) -> None:
    """§2.4's disjoint calibration set, §4's retest block, and the ids that keep both blind."""
    out.check(
        "calibration_total", len(plan.calibration) == CALIBRATION_TOTAL, f"{len(plan.calibration)} calibration items"
    )
    concepts = _counter(item.concept for item in plan.calibration)
    out.check(
        "calibration_one_per_construct",
        len(concepts) == N_CONSTRUCTS and set(concepts.values()) == {1},
        f"{len(concepts)} constructs, counts {sorted(set(concepts.values()))}",
    )
    roles = _counter(item.role for item in plan.calibration)
    out.stat("calibration_roles", roles)
    out.check(
        "calibration_roles_span_the_pool",
        len(roles) == len(ROLES_PER_CONSTRUCT) + 1,
        f"{roles}; a calibration session of clean positives argues about nothing",
    )
    overlap = {item.unit_id for item in pools.all} & {item.unit_id for item in plan.calibration}
    lineages = {item.lineage_id for item in pools.all} & {item.lineage_id for item in plan.calibration}
    out.check(
        "calibration_disjoint_ids",
        not overlap and not lineages,
        f"{len(overlap)} shared unit ids, {len(lineages)} shared lineages",
    )

    all_units = [item.unit_id for item in plan.items]
    out.check(
        "unit_ids_unique", len(set(all_units)) == len(all_units), f"{len(all_units) - len(set(all_units))} duplicates"
    )
    out.check(
        "open_pool_concept_is_not_a_construct",
        OPEN_POOL_CONCEPT not in ontology
        and all(control.key not in ontology for control in ontology.debunked_controls),
        f"{OPEN_POOL_CONCEPT!r} must not collide with a construct key",
    )
    open_prompt_leaks = [
        item.unit_id
        for item in pools.open_pool
        if any(
            construct.key in item.user_prompt or construct.display_name in item.user_prompt for construct in ontology
        )
    ]
    out.check(
        "open_pool_prompts_name_no_construct",
        not open_prompt_leaks,
        f"{len(open_prompt_leaks)} open-pool briefs mention a construct; base rates measured on a "
        "construct-directed prompt are not base rates",
    )

    retest_counts = _counter(entry["stratum"] for entry in plan.retest)
    out.check(
        "retest_strata",
        len(plan.retest) == RETEST_TOTAL and retest_counts == dict(sorted(RETEST_STRATA.items())),
        f"{retest_counts}, expected {dict(sorted(RETEST_STRATA.items()))}",
    )
    retest_ids = [entry["retest_id"] for entry in plan.retest]
    out.check(
        "retest_ids_resalted",
        len(set(retest_ids)) == len(retest_ids) and not set(retest_ids) & set(all_units),
        "a retest id that matches its original can be looked up",
    )


def _check_artifacts(out: _Report, ontology: schema.Ontology, plan: Plan, assembly: Assembly) -> None:
    """What reached the corpus: its columns, its blinding, and whether it is 184 distinct texts.

    This repeats `sanitize_generation`'s scan on purpose. The sanitizer runs on the
    way in and can be bypassed - by a scaffold, by a hand-edited generations file,
    by a future caller passing no leak tokens - and the property that matters is
    the one true of the file that ships.
    """
    leak_tokens = _leak_tokens(ontology)
    corpus_rows = [artifact.as_corpus_row() for artifact in assembly.artifacts.values()]
    allowed = {"unit_id", "item_id", "concept", "candidate_action", "question", "reference", "student_before"}
    stray = sorted({column for row in corpus_rows for column in row} - allowed)
    out.check("corpus_columns_allowlisted", not stray, f"unexpected columns {stray}")
    leaking = {
        str(row["unit_id"])
        for row in corpus_rows
        for token in leak_tokens
        if token in str(row["candidate_action"]).lower() or token in str(row["question"]).lower()
    }
    out.check(
        "no_withheld_identifier_in_corpus",
        not leaking,
        f"{len(leaking)} rows carry a construct key, a role name or a withheld field name",
    )
    stems = [
        item.unit_id for item in plan.items if any(construct.key in item.scenario.task for construct in ontology)
    ]
    out.check("stems_name_no_construct", not stems, f"{len(stems)} rater-visible stems name a construct key")

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for unit_id, artifact in sorted(assembly.artifacts.items()):
        key = schema.normalize_text(artifact.text)
        if key in seen:
            duplicates.append(f"{unit_id}~{seen[key]}")
        else:
            seen[key] = unit_id
    out.check(
        "artifact_texts_distinct",
        not duplicates,
        f"{len(duplicates)} near-duplicate texts after normalisation: {duplicates[:3]}",
    )
    out.check(
        "every_item_has_an_artifact",
        len(assembly.artifacts) == len(plan.items),
        f"{len(assembly.artifacts)} artifacts for {len(plan.items)} planned items",
    )
    out.stat(
        "generation",
        {
            "status": "scaffolded" if assembly.scaffolded else "generated",
            "sources": _counter(str(entry.get("source_model")) for entry in assembly.provenance.values()),
            "scaffolded": len(assembly.scaffolded),
            "rejected": len(assembly.rejections),
            "rejection_reasons": _counter(reason for r in assembly.rejections for reason in r.reasons),
        },
    )


def audit(ontology: schema.Ontology, plan: Plan, assembly: Assembly | None = None) -> Audit:
    """Every fixed count and balance constraint the guide states, checked.

    Reported and checked are different things here. A count the guide fixes is a
    check and fails the build; a distribution the guide leaves open - which
    domains, how many candidate sets remain uniquely recoverable, how many
    generations were refused - is a statistic in the manifest, because a build
    that fails on a soft target teaches its operator to pass `--force`.

    Without an assembly this audits the design alone, which is what `--mode plan`
    reports and what a reviewer can check before any GPU time is spent.
    """
    out = _Report()
    pools = _Pools(
        targeted=tuple(item for item in plan.round1 if item.pool == TARGETED_POOL),
        controls=tuple(item for item in plan.round1 if item.pool == CONTROL_POOL),
        open_pool=tuple(item for item in plan.round1 if item.pool == OPEN_POOL),
    )
    _check_composition(out, ontology, pools)
    _check_artifact_types(out, ontology, plan, pools)
    _check_enactment(out, ontology, plan, pools)
    _check_candidate_sets(out, ontology, pools)
    _check_scenarios(out, plan, pools)
    _check_calibration_and_ids(out, ontology, plan, pools)
    if assembly is None:
        out.stat("generation", {"status": "not started"})
    else:
        _check_artifacts(out, ontology, plan, assembly)
    return Audit(tuple(out.checks), out.stats)


# ----------------------------------------------------------------------- output


CORPUS_FILE = "corpus.jsonl"
CALIBRATION_CORPUS_FILE = "calibration_corpus.jsonl"
ITEMS_FILE = "items.jsonl"
MANIFEST_FILE = "manifest.json"
WITHHELD_DIR = "withheld"
PLAN_FILE = "plan.jsonl"
PROMPTS_FILE = "prompts.jsonl"
GENERATIONS_FILE = "generations.jsonl"
KEY_FILE = "key.jsonl"
CALIBRATION_KEY_FILE = "calibration_key.jsonl"
CALIBRATION_SHOTS_FILE = "calibration_shots.jsonl"
REJECTIONS_FILE = "rejections.jsonl"
RETEST_FILE = "retest_plan.jsonl"


def write_plan_files(plan: Plan, out_dir: str | Path) -> dict[str, int]:
    """The design, all of it withheld.

    Until there are artifacts there is nothing a rater may see, and each of these
    three files names the answer for every item it covers.
    """
    withheld = Path(out_dir) / WITHHELD_DIR
    return {
        f"{WITHHELD_DIR}/{PLAN_FILE}": schema.write_jsonl(
            (item.plan_row() for item in plan.items), withheld / PLAN_FILE
        ),
        f"{WITHHELD_DIR}/{PROMPTS_FILE}": schema.write_jsonl(
            (item.prompt_row() for item in plan.items), withheld / PROMPTS_FILE
        ),
        f"{WITHHELD_DIR}/{RETEST_FILE}": schema.write_jsonl(plan.retest, withheld / RETEST_FILE),
    }


# The one-sentence rationale LABELING_GUIDE.md §6.1 asks to accompany each
# few-shot example. It is the design's reason, not a reading of the artifact.
SHOT_RATIONALES: Mapping[schema.ItemRole, str] = {
    schema.ItemRole.POSITIVE_CANONICAL: "Present: the artifact does the thing, without naming it.",
    schema.ItemRole.NAMED_AND_ENACTED: "Present: the artifact names the principle and also does it.",
    schema.ItemRole.POSITIVE_ATYPICAL: "Present: a genuine instance in an unusual form and without the usual words.",
    schema.ItemRole.POSITIVE_PARTIAL: "Present at low fidelity: the defining feature is there, a required element is not.",
    schema.ItemRole.SIBLING_HARD_NEGATIVE: "Absent: this is a competent instance of the neighbouring construct.",
    schema.ItemRole.SAYS_ONLY: "Absent: correct discussion of the principle containing none of it.",
}


def _shot_row(item: PlanItem, artifact: schema.Artifact) -> dict[str, object]:
    """One few-shot example: the blinded row, the design's gold label, and its reason.

    `quality` is deliberately null. The design entails presence and fidelity, and
    it does not entail what a person thinks of the artifact overall - two careful
    raters agreed on that 39% of the time (LABELING_GUIDE.md §3.4). Writing a
    constant 2 here would teach every agent to answer 2, which is exactly the
    constant column §6.1 records as having already happened once. So
    `labeling.read_exemplars` refuses this file until the calibration session of
    §10 step 4 supplies the column, and that refusal is the intended behaviour.
    """
    gold = schema.gold_label(artifact)
    return {
        **artifact.as_corpus_row(),
        "label": {
            "applicable": gold.applicable,
            "presence": gold.presence,
            "fidelity": gold.fidelity,
            "quality": gold.quality,
            "span": gold.span,
        },
        "rationale": SHOT_RATIONALES[item.role],
    }


def write_corpus_files(plan: Plan, assembly: Assembly, out_dir: str | Path) -> dict[str, int]:
    out = Path(out_dir)
    withheld = out / WITHHELD_DIR

    def rows(items: Sequence[PlanItem], render) -> list[dict[str, object]]:
        return [render(item, assembly.artifacts[item.unit_id]) for item in items if item.unit_id in assembly.artifacts]

    def key_row(item: PlanItem, artifact: schema.Artifact) -> dict[str, object]:
        return {
            **artifact.as_key_row(),
            "block": item.block,
            "pool": item.pool,
            "sibling_expected_fidelity": item.sibling_expected_fidelity,
            "prompt_sha": item.prompt_sha,
            **assembly.provenance.get(item.unit_id, {}),
        }

    shots = [
        _shot_row(item, assembly.artifacts[item.unit_id])
        for item in plan.calibration
        if item.unit_id in assembly.artifacts
    ]

    return {
        CORPUS_FILE: schema.write_jsonl(rows(plan.round1, lambda _i, a: a.as_corpus_row()), out / CORPUS_FILE),
        ITEMS_FILE: schema.write_jsonl(rows(plan.round1, lambda i, a: i.item_row(a.text)), out / ITEMS_FILE),
        CALIBRATION_CORPUS_FILE: schema.write_jsonl(
            rows(plan.calibration, lambda _i, a: a.as_corpus_row()), out / CALIBRATION_CORPUS_FILE
        ),
        f"{WITHHELD_DIR}/{KEY_FILE}": schema.write_jsonl(rows(plan.round1, key_row), withheld / KEY_FILE),
        f"{WITHHELD_DIR}/{CALIBRATION_KEY_FILE}": schema.write_jsonl(
            rows(plan.calibration, key_row), withheld / CALIBRATION_KEY_FILE
        ),
        f"{WITHHELD_DIR}/{CALIBRATION_SHOTS_FILE}": schema.write_jsonl(shots, withheld / CALIBRATION_SHOTS_FILE),
        f"{WITHHELD_DIR}/{REJECTIONS_FILE}": schema.write_jsonl(
            (rejection.as_dict() for rejection in assembly.rejections), withheld / REJECTIONS_FILE
        ),
    }


def existing_files(out_dir: str | Path) -> dict[str, int]:
    """What a previous stage recorded and is still on disk.

    `--mode generate` and `--mode assemble` are separate invocations, so the
    manifest each one writes would otherwise forget the files the stage before it
    produced. Entries whose file has since been removed are dropped rather than
    carried, because the manifest is a claim about the directory.
    """
    out = Path(out_dir)
    target = out / MANIFEST_FILE
    if not target.exists():
        return {}
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    recorded = blob.get("files") if isinstance(blob, Mapping) else None
    if not isinstance(recorded, Mapping):
        return {}
    return {name: int(count) for name, count in recorded.items() if (out / name).exists()}


def write_manifest(
    plan: Plan, report: Audit, out_dir: str | Path, *, ontology_path: str, files: Mapping[str, int]
) -> Path:
    target = Path(out_dir) / MANIFEST_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "construct_version": plan.construct_version,
        "guide_composition": {
            "round1_total": ROUND1_TOTAL,
            "targeted": TARGETED_TOTAL,
            "debunked_controls": ROUND1_TOTAL - TARGETED_TOTAL - OPEN_POOL_TOTAL,
            "open_pool": OPEN_POOL_TOTAL,
            "calibration": CALIBRATION_TOTAL,
            "retest": RETEST_TOTAL,
            "artifact_type_budget": dict(zip(GROUP_NAMES, GROUP_CAPACITY)),
            "candidate_set_size": CANDIDATE_SET_SIZE,
            "appearance_floor": APPEARANCE_FLOOR,
        },
        "ontology": ontology_path,
        "salt": plan.salt,
        "files": dict(sorted(files.items())),
        "audit": report.as_dict(),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# -------------------------------------------------------------------------- cli


def _stale_corpus(out_dir: str | Path, plan: Plan) -> int:
    """Rows in an existing corpus that this plan does not account for.

    A second salt in the same directory is the way to get a round-2 key beside a
    round-1 corpus, which is a mislabelling waiting to happen. The ids are digests
    and cannot be eyeballed, so the check is worth making rather than trusting.
    """
    target = Path(out_dir) / CORPUS_FILE
    if not target.exists():
        return 0
    planned = {item.unit_id for item in plan.round1}
    stale = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return -1
        if row.get("unit_id") not in planned:
            stale += 1
    return stale


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the calibration and round-1 manifests LABELING_GUIDE.md specifies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ontology", default=None, help="ontology.yaml; the packaged one by default")
    parser.add_argument("--out", required=True, help="output directory; withheld files land in out/withheld")
    parser.add_argument(
        "--mode",
        default="plan",
        choices=("plan", "generate", "assemble", "all"),
        help="plan and assemble need no model; generate loads one",
    )
    parser.add_argument("--salt", default=DEFAULT_SALT, help="id salt; a new salt is a new, non-colliding pool")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-model", action="store_true", help="with --mode all, scaffold instead of generating")
    parser.add_argument("--scaffold-missing", action="store_true", help="fill absent or refused generations")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3, help="resamples before a refusal is recorded")
    parser.add_argument("--limit", type=int, default=0, help="generate at most this many pending items")
    parser.add_argument("--no-resume", action="store_true", help="ignore and overwrite an existing generations file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ontology_path = Path(args.ontology) if args.ontology else schema.default_ontology_path()
    ontology = schema.load_ontology(ontology_path)
    plan = build_plan(ontology, salt=args.salt)
    out = Path(args.out)
    leak_tokens = _leak_tokens(ontology)
    files: dict[str, int] = existing_files(out)

    if args.mode in ("plan", "all"):
        files.update(write_plan_files(plan, out))
        print(
            f"planned {len(plan.round1)} round-1 items and {len(plan.calibration)} calibration items "
            f"from {ontology_path} at construct_version {plan.construct_version}"
        )
        stale = _stale_corpus(out, plan)
        if stale:
            print(
                f"WARNING: {out / CORPUS_FILE} holds {stale} row(s) this plan did not produce, so it was "
                f"built under a different ontology or salt. Assemble will overwrite it; move it aside first "
                "if you meant to keep it.",
                file=sys.stderr,
            )

    generations_path = out / WITHHELD_DIR / GENERATIONS_FILE
    if args.mode in ("generate", "all") and not args.no_model:
        report = generate(
            plan,
            out_path=generations_path,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_attempts=args.max_attempts,
            limit=args.limit,
            leak_tokens=leak_tokens,
            resume=not args.no_resume,
        )
        print(f"generated {report['written']} artifacts with {report['model']}; {report}")

    if args.mode in ("assemble", "all"):
        generations, dropped = read_generations(generations_path)
        if dropped:
            print(f"repaired {dropped} truncated line(s) at the end of {generations_path}")
        assembly = assemble(
            plan,
            generations,
            leak_tokens=leak_tokens,
            scaffold_missing=args.scaffold_missing or args.no_model,
        )
        files.update(write_corpus_files(plan, assembly, out))
        report = audit(ontology, plan, assembly)
    else:
        report = audit(ontology, plan)

    manifest = write_manifest(plan, report, out, ontology_path=str(ontology_path), files=files)
    print(f"wrote {manifest} with {len(report.checks)} checks")
    for failure in report.failures:
        print(f"FAILED {failure.name}: {failure.detail}", file=sys.stderr)
    if not report.ok:
        return 1
    scaffolded = report.stats.get("generation", {})
    if isinstance(scaffolded, Mapping) and scaffolded.get("scaffolded"):
        print(
            f"WARNING: {scaffolded['scaffolded']} of {len(plan.items)} artifacts are marked scaffolds, not "
            "model output. This corpus is for exercising the pipeline, not for labelling.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
