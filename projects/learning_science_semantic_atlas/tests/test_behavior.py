"""CPU tests for the behaviour batteries. No weights, no network, no GPU.

The ontology probes run against the real `ontology.yaml`, because the properties
worth checking are properties of that file: whether its sibling links cover every
construct, whether its `output_types` leave any construct with no inapplicable
artifact type to be crossed against, whether its boundary prose smuggles a
construct key into a prompt. A fixture would pass while the real file failed.

The corpus probes run against a small handwritten corpus instead, since what
needs checking there is the join and the pairing: that a sibling hard negative is
found although its key row says it enacts nothing, that a scenario's variants
stay together, and that two files written from different plans are refused.

Scoring is exercised with toy runners rather than skipped. Three of them stand in
for three failure modes a real report has to be able to tell apart: a model that
prefers one text whatever it is asked, a model that answers correctly, and a
model that follows the construct's vocabulary and so walks into the says-only
trap. Getting different numbers out of those three is the whole point of the
crossing, and none of it needs a 7B model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from projects.learning_science_semantic_atlas import behavior, schema

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CONSTRUCT = "retrieval_practice_opportunity"
SIBLING = "self_explanation_prompt"


@pytest.fixture(scope="module")
def ontology() -> schema.Ontology:
    return schema.load_ontology()


# --------------------------------------------------------------- toy runners


@dataclass
class ToyRunner:
    """A stand-in model. `rank` returns the mean per-token log-probability of an option.

    Length is the option's word count, so `logprob` and `mean_logprob` are both
    exactly controllable: a test that wants the two normalizations to disagree can
    ask for it rather than hope for it.
    """

    rank: Callable[[str, str], float]
    scored: list[tuple[str, str]] = field(default_factory=list)
    generated: list[tuple[str, int]] = field(default_factory=list)

    @property
    def description(self) -> dict[str, object]:
        return {"model": "toy-generator", "architecture": "dense", "n_experts": None}

    def score(self, system_prompt: str, prompt: str, options: Sequence[str]) -> list[behavior.TokenScore]:
        self.scored.append((system_prompt, prompt))
        scores = []
        for option in options:
            tokens = max(1, len(option.split()))
            scores.append(behavior.TokenScore(self.rank(prompt, option) * tokens, tokens))
        return scores

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        self.generated.append((prompt, seed))
        return f"A tutor turn written at seed {seed} with retrieval practice in it."


def oracle_runner(probes: Sequence[behavior.Probe], *, margin: float = 1.0) -> ToyRunner:
    """A model that always prefers the keyed option. The ceiling every statistic should show."""
    keyed = {(probe.prompt, probe.keyed_option.text) for probe in probes}
    return ToyRunner(lambda prompt, option: -1.0 + (margin if (prompt, option) in keyed else 0.0))


def shortest_option_runner() -> ToyRunner:
    """A model with one constant preference and no reading of the instruction.

    The option texts of a pair are identical across its two directions, so this
    wins one direction and loses the other and its paired shift is exactly zero.
    """
    return ToyRunner(lambda prompt, option: -len(option) / 1000.0)


def vocabulary_runner(terms: Sequence[str]) -> ToyRunner:
    """A model that follows the construct's words wherever they are.

    On the mismatch contrast it prefers the artifact that talks about the
    principle over the one that performs it, which is the says-only trap.
    """
    return ToyRunner(lambda prompt, option: -1.0 + sum(term in option.lower() for term in terms))


# ------------------------------------------------------------- a small corpus


ENACTING_TEXT = "Without looking back at your notes, write out the three steps and then check them."
SIBLING_TEXT = "Look at line three of the solution in front of you and explain why that step is allowed."
SAYS_ONLY_TEXT = "Research on retrieval practice shows that quizzing beats rereading, so try to self-quiz later."
PARTIAL_TEXT = "Write out the steps for this one, and the worked answer is right here above it for you."
SYSTEM = "You are the tutor in a one-to-one session with the learner described below."


def lineage_rows(
    lineage: str, concept: str = CONSTRUCT, sibling: str = SIBLING
) -> tuple[list[dict], list[dict], list[dict]]:
    """One scenario's four variants, shaped the way `build_corpus` writes them.

    The interesting row is the sibling hard negative: relative to its designed
    target it enacts nothing, and the construct it does enact is recorded only in
    `enacted_construct`.
    """
    variants = [
        ("positive_canonical", ENACTING_TEXT, concept, None, True, False),
        ("positive_partial", PARTIAL_TEXT, concept, None, True, False),
        ("sibling_hard_negative", SIBLING_TEXT, sibling, None, False, False),
        ("says_only", SAYS_ONLY_TEXT, None, concept, False, True),
    ]
    corpus, key, prompts = [], [], []
    for index, (role, text, enacted, named, enacts, names) in enumerate(variants):
        unit_id = f"{lineage}u{index}"
        digest = f"sha{lineage}{index}"
        corpus.append(
            {
                "unit_id": unit_id,
                "item_id": lineage,
                "concept": concept,
                "candidate_action": text,
                "question": "Topic: two-step equations. Write the tutor's next single turn.",
                "reference": "Subtract 7, then divide by 3.",
                "student_before": "I have not tried this one yet.",
            }
        )
        key.append(
            {
                "unit_id": unit_id,
                "item_id": lineage,
                "designed_target_construct": concept,
                "item_role": role,
                "enacted_construct": enacted,
                "named_construct": named,
                "enacts": enacts,
                "names": names,
                "expected_presence": enacts,
                "output_format": "tutor_turn",
                "pool": "targeted",
                "prompt_sha": digest,
                "source_model": "toy-generator",
            }
        )
        prompts.append(
            {
                "unit_id": unit_id,
                "prompt_sha": digest,
                "system": SYSTEM,
                "user": f"Write a {role} for {concept}.",
                "max_new_tokens": 128,
            }
        )
    return corpus, key, prompts


def write_corpus(root: Path, lineages: Sequence[str] = ("ln1", "ln2")) -> Path:
    corpus, key, prompts = [], [], []
    for lineage in lineages:
        rows = lineage_rows(lineage)
        corpus.extend(rows[0])
        key.extend(rows[1])
        prompts.extend(rows[2])
    schema.write_jsonl(corpus, root / behavior.build_corpus.CORPUS_FILE)
    withheld = root / behavior.build_corpus.WITHHELD_DIR
    schema.write_jsonl(key, withheld / behavior.build_corpus.KEY_FILE)
    schema.write_jsonl(prompts, withheld / behavior.build_corpus.PROMPTS_FILE)
    (root / behavior.build_corpus.MANIFEST_FILE).write_text(
        json.dumps({"construct_version": "atlas-v1.0.0", "salt": "toy"}), encoding="utf-8"
    )
    return root


@pytest.fixture
def bundle(tmp_path: Path) -> behavior.CorpusBundle:
    return behavior.load_bundle(write_corpus(tmp_path / "corpus"))


# --------------------------------------------------------- crossing invariants


def test_every_kind_is_crossed_by_two_directions_except_the_declared_control() -> None:
    for kind, crossing in behavior.CROSSING.items():
        expected = 1 if kind is behavior.ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL else 2
        assert len(crossing) == expected, kind
        assert len({tag for _, tag in crossing}) == len(crossing)


def test_a_pair_presents_the_same_two_texts_in_both_directions(ontology: schema.Ontology) -> None:
    """The one property the paired shift depends on, checked over every crossed pair."""
    probes = behavior.all_probes([behavior.declarative_battery(ontology), behavior.boundary_battery(ontology)])
    by_pair: dict[str, list[behavior.Probe]] = {}
    for probe in probes:
        by_pair.setdefault(probe.pair_id, []).append(probe)

    for pair_id, members in by_pair.items():
        kind = members[0].kind
        directions = [direction for direction, _ in behavior.CROSSING[kind]]
        assert [probe.direction for probe in members] == directions, pair_id
        first = [(option.option_id, option.text) for option in members[0].options]
        for probe in members[1:]:
            assert [(option.option_id, option.text) for option in probe.options] == first
            assert probe.prompt != members[0].prompt
        keyed = [probe.keyed_option.tag for probe in members]
        assert keyed == [tag for _, tag in behavior.CROSSING[kind]]


def test_option_order_puts_the_first_direction_s_answer_first(ontology: schema.Ontology) -> None:
    """`ProbeResult.margin` subtracts in `CROSSING` order, so this is what makes it signed."""
    for probe in behavior.declarative_battery(ontology).probes:
        first_direction, first_tag = behavior.CROSSING[probe.kind][0]
        assert probe.options[0].tag is first_tag
        assert probe.options[0].keyed is (probe.direction is first_direction)


def test_ids_reproduce_and_a_new_salt_is_a_new_non_colliding_set(ontology: schema.Ontology) -> None:
    first = behavior.declarative_battery(ontology)
    again = behavior.declarative_battery(ontology)
    other = behavior.declarative_battery(ontology, salt="behavior-v2")
    assert [probe.probe_id for probe in first.probes] == [probe.probe_id for probe in again.probes]
    assert [probe.probe_sha for probe in first.probes] == [probe.probe_sha for probe in again.probes]
    assert not {probe.probe_id for probe in first.probes} & {probe.probe_id for probe in other.probes}


def test_the_digest_covers_the_options_and_not_only_the_prompt(ontology: schema.Ontology) -> None:
    probe = behavior.declarative_battery(ontology).probes[0]
    edited = replace(
        probe,
        options=(replace(probe.options[0], text=probe.options[0].text + " and one more clause"),) + probe.options[1:],
    )
    assert edited.probe_sha != probe.probe_sha


def test_a_probe_that_cannot_be_scored_is_refused(ontology: schema.Ontology) -> None:
    probe = behavior.declarative_battery(ontology).probes[0]
    with pytest.raises(behavior.BehaviorError, match="same text"):
        replace(probe, options=(probe.options[0], replace(probe.options[1], text=probe.options[0].text)))
    with pytest.raises(behavior.BehaviorError, match="exactly one option must be keyed"):
        replace(probe, options=tuple(replace(option, keyed=True) for option in probe.options))
    with pytest.raises(behavior.BehaviorError, match="keys"):
        replace(probe, direction=behavior.Direction.SIBLING)


# ------------------------------------------------------------ ontology probes


def test_every_construct_is_probed_and_no_construct_is_its_own_sibling(ontology: schema.Ontology) -> None:
    pairs = behavior.sibling_pairs(ontology)
    covered = {pair.target for pair in pairs} | {pair.sibling for pair in pairs}
    assert covered == set(ontology.keys)
    assert all(pair.target != pair.sibling for pair in pairs)
    assert len({frozenset({pair.target, pair.sibling}) for pair in pairs}) == len(pairs)


def test_escalation_conditions_are_not_asked_about(ontology: schema.Ontology) -> None:
    """They say what a higher claim rung would need, so a model cannot be right about them."""
    escalation = [
        entry
        for construct in ontology
        for entry in construct.boundary_conditions
        if entry.lower().startswith("escalation")
    ]
    assert escalation, "the ontology used to carry escalation entries; this test is now vacuous"
    asked = {option.text for probe in behavior.boundary_battery(ontology).probes for option in probe.options}
    assert not asked & set(escalation)
    for construct in ontology:
        assert all(not entry.lower().startswith("escalation") for entry in behavior.moderator_conditions(construct))


def test_no_prompt_or_option_carries_a_construct_key(ontology: schema.Ontology) -> None:
    """An identifier is not prose, and here it would mark one option and not the other."""
    keys = [construct.key for construct in ontology]
    for probe in behavior.all_probes([behavior.declarative_battery(ontology), behavior.boundary_battery(ontology)]):
        haystack = " ".join([probe.prompt, *(option.text for option in probe.options)])
        assert not [key for key in keys if key in haystack], probe.probe_id


def test_the_format_probe_crosses_a_type_the_ontology_admits_against_one_it_excludes(
    ontology: schema.Ontology,
) -> None:
    probes = behavior.boundary_battery(ontology).of_kind(behavior.ProbeKind.FORMAT_APPLICABILITY)
    assert probes
    for probe in probes:
        construct = ontology.get(probe.construct)
        accepted = schema.OutputFormat(probe.metadata["accepted_format"])
        refused = schema.OutputFormat(probe.metadata["refused_format"])
        assert construct.accepts(accepted)
        assert not construct.accepts(refused)


def test_the_unconditional_control_is_uncrossed_and_says_so(ontology: schema.Ontology) -> None:
    probes = behavior.boundary_battery(ontology).of_kind(behavior.ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL)
    assert probes
    assert {probe.direction for probe in probes} == {behavior.Direction.TARGET}
    report = behavior.summarize_kind([], behavior.ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL)
    assert report["crossed"] is False
    assert "control" in report["claim_limit"]
    assert "paired_shift" not in report


# --------------------------------------------------------------- corpus intake


def test_a_key_and_a_brief_from_different_plans_are_refused(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "corpus")
    path = root / behavior.build_corpus.WITHHELD_DIR / behavior.build_corpus.PROMPTS_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["prompt_sha"] = "a-different-plan"
    schema.write_jsonl(rows, path)
    with pytest.raises(behavior.BehaviorError, match="written from different plans"):
        behavior.load_bundle(root)


def test_a_corpus_row_with_no_key_entry_is_refused(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "corpus")
    path = root / behavior.build_corpus.WITHHELD_DIR / behavior.build_corpus.KEY_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    schema.write_jsonl(rows[1:], path)
    with pytest.raises(behavior.BehaviorError, match="no key entry"):
        behavior.load_bundle(root)


def test_an_ordinal_dressed_as_a_boolean_is_refused(tmp_path: Path) -> None:
    """`enacts: 1` would read as true and quietly move an item into the wrong cell."""
    root = write_corpus(tmp_path / "corpus")
    path = root / behavior.build_corpus.WITHHELD_DIR / behavior.build_corpus.KEY_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["enacts"] = 1
    schema.write_jsonl(rows, path)
    with pytest.raises(behavior.BehaviorError, match="must be a JSON boolean"):
        behavior.load_bundle(root)


def test_the_sibling_hard_negative_is_found_although_its_key_says_it_enacts_nothing(
    bundle: behavior.CorpusBundle,
) -> None:
    """`enacts` is relative to the designed target; the construct performed is `enacted_construct`."""
    hard_negative = next(unit for unit in bundle.units.values() if unit.item_role == "sibling_hard_negative")
    assert hard_negative.enacts is False
    assert hard_negative.enacted_key == SIBLING
    canonical = next(unit for unit in bundle.units.values() if unit.item_role == "positive_canonical")
    assert canonical.enacted_key == CONSTRUCT
    says_only = next(unit for unit in bundle.units.values() if unit.item_role == "says_only")
    assert says_only.enacted_key is None


def test_enacted_pairs_stay_inside_one_scenario_and_reuse_its_system_prompt(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle
) -> None:
    battery = behavior.enacted_battery(ontology, bundle)
    assert battery.level is behavior.Level.ENACTED_OUTPUT
    assert len(battery.probes) == 8, "two scenarios, two contrasts each, two directions each"
    for probe in battery.probes:
        units = [bundle.units[unit_id] for unit_id in probe.source_units]
        assert len({unit.item_id for unit in units}) == 1
        assert probe.system_prompt == SYSTEM
        assert probe.metadata["system_prompts_agree"] is True

    strategy = battery.of_kind(behavior.ProbeKind.ENACTED_STRATEGY)
    assert {probe.construct for probe in strategy} == {CONSTRUCT}
    assert {probe.sibling for probe in strategy} == {SIBLING}
    texts = {option.tag: option.text for option in strategy[0].options}
    assert texts[behavior.OptionTag.ENACTS_TARGET] == ENACTING_TEXT
    assert texts[behavior.OptionTag.ENACTS_SIBLING] == SIBLING_TEXT

    mismatch = battery.of_kind(behavior.ProbeKind.NAMED_VERSUS_ENACTED)
    texts = {option.tag: option.text for option in mismatch[0].options}
    assert texts[behavior.OptionTag.NAMES_WITHOUT_ENACTING] == SAYS_ONLY_TEXT


def test_the_withheld_brief_is_never_used_as_a_probe_prompt(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle
) -> None:
    """It names its own item's target and role, so it would answer one direction only."""
    briefs = {unit.withheld_user_prompt for unit in bundle.units.values()}
    for probe in behavior.enacted_battery(ontology, bundle).probes:
        assert probe.prompt not in briefs
        assert "positive_canonical" not in probe.prompt and "says_only" not in probe.prompt


def test_labels_narrow_the_battery_and_a_missing_unit_is_refused(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    def label(unit_id: str, concept: str, presence: bool) -> dict[str, object]:
        return {
            "unit_id": unit_id,
            "item_id": "ln1",
            "concept": concept,
            "rater": "agent",
            "applicable": True,
            "applicability": "applicable",
            "presence": presence,
            "fidelity": 2 if presence else None,
            "span": "quoted move" if presence else None,
        }

    rows = [label(unit.unit_id, unit.concept, unit.enacts) for unit in bundle.units.values() if unit.item_id == "ln1"]
    rows[0] = {**rows[0], "presence": not rows[0]["presence"], "fidelity": None, "span": None}
    path = tmp_path / "labels.jsonl"
    schema.write_jsonl(rows, path)

    allowed = behavior.verified_units([path], bundle)
    assert allowed[rows[0]["unit_id"]] is False

    narrowed = behavior.enacted_battery(ontology, bundle, allowed_units=allowed)
    kept = {unit_id for probe in narrowed.probes for unit_id in probe.source_units}
    assert rows[0]["unit_id"] not in kept, "a unit whose label contradicts the design is dropped"
    assert {probe.metadata["item_id"] for probe in narrowed.probes} == {"ln1"}, "ln2 has no labels at all"
    audit = behavior.enacted_design_audit(narrowed.probes, ontology)
    assert audit["enacting_option_roles"] == {
        "enacted_strategy": {"positive_partial": 1},
        "named_versus_enacted": {"positive_partial": 1},
    }, "with the canonical positive dropped the pair falls back to the partial positive, and says so"

    schema.write_jsonl([{**rows[1], "unit_id": "nosuchunit"}], path)
    with pytest.raises(behavior.BehaviorError, match="which this corpus has not"):
        behavior.verified_units([path], bundle)


# ------------------------------------------------------------ scoring, resume


def score_all(probes: Sequence[behavior.Probe], runner: ToyRunner, path: Path, **kwargs) -> list[behavior.ProbeResult]:
    behavior.run_probes(probes, runner, out_path=path, **kwargs)
    responses, _ = behavior.read_responses(path)
    return behavior.join_responses(probes, responses)


def test_resume_skips_what_is_already_scored_and_rescores_a_changed_probe(
    ontology: schema.Ontology, tmp_path: Path
) -> None:
    probes = behavior.declarative_battery(ontology).probes[:6]
    runner = oracle_runner(probes)
    path = tmp_path / behavior.RESPONSES_FILE

    first = behavior.run_probes(probes, runner, out_path=path)
    assert first["scored"] == len(probes)
    assert first["still_pending"] == 0

    again = behavior.run_probes(probes, runner, out_path=path)
    assert again["scored"] == 0
    assert again["already_present"] == len(probes)

    edited = list(probes)
    edited[0] = replace(
        edited[0], prompt=edited[0].prompt + "\nAnd answer in one sentence.", probe_id=probes[0].probe_id
    )
    responses, _ = behavior.read_responses(path)
    assert behavior.pending_probes(edited, responses) == (edited[0],)

    behavior.run_probes(edited, runner, out_path=path)
    assert behavior.pending_probes(edited, behavior.read_responses(path)[0]) == ()


def test_no_resume_discards_the_file(ontology: schema.Ontology, tmp_path: Path) -> None:
    probes = behavior.declarative_battery(ontology).probes[:4]
    path = tmp_path / behavior.RESPONSES_FILE
    behavior.run_probes(probes, oracle_runner(probes), out_path=path)
    report = behavior.run_probes(probes, oracle_runner(probes), out_path=path, resume=False)
    assert report["already_present"] == 0
    assert report["scored"] == len(probes)


def test_a_truncated_final_line_is_repaired_and_earlier_corruption_is_not(
    ontology: schema.Ontology, tmp_path: Path
) -> None:
    """A preempted job dies mid-write; a broken line in the middle is a different problem."""
    probes = behavior.declarative_battery(ontology).probes[:3]
    path = tmp_path / behavior.RESPONSES_FILE
    behavior.run_probes(probes, oracle_runner(probes), out_path=path)
    lines = path.read_text().splitlines()

    path.write_text("\n".join(lines) + "\n" + lines[0][:20], encoding="utf-8")
    rows, dropped = behavior.read_responses(path)
    assert dropped == 1
    assert len(rows) == len(probes)
    assert len(path.read_text().splitlines()) == len(probes)

    path.write_text("\n".join([lines[0], lines[1][:20], lines[2]]) + "\n", encoding="utf-8")
    with pytest.raises(behavior.BehaviorError, match="not the final line"):
        behavior.read_responses(path)


def test_a_limit_leaves_the_rest_pending_rather_than_dropping_it(ontology: schema.Ontology, tmp_path: Path) -> None:
    probes = behavior.declarative_battery(ontology).probes[:6]
    report = behavior.run_probes(probes, oracle_runner(probes), out_path=tmp_path / "r.jsonl", limit=2)
    assert report["scored"] == 2
    assert report["still_pending"] == 4


def test_probe_seeds_follow_the_run_seed_and_are_stable() -> None:
    assert behavior.probe_seed(1701, "bpabc") == behavior.probe_seed(1701, "bpabc")
    assert behavior.probe_seed(1701, "bpabc") != behavior.probe_seed(1702, "bpabc")
    assert behavior.probe_seed(1701, "bpabc") != behavior.probe_seed(1701, "bpabd")


# ------------------------------------------------------- what the numbers mean


def enacted_results(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, runner: ToyRunner, tmp_path: Path
) -> tuple[behavior.Battery, list[behavior.ProbeResult]]:
    battery = behavior.enacted_battery(ontology, bundle)
    return battery, score_all(battery.probes, runner, tmp_path / behavior.RESPONSES_FILE)


def test_a_constant_option_preference_scores_a_coin_flip_and_shifts_nothing(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    """The failure the crossing exists to catch: right half the time, informative never."""
    battery, results = enacted_results(ontology, bundle, shortest_option_runner(), tmp_path)
    assert battery.probes
    report = behavior.summarize_kind(results, behavior.ProbeKind.ENACTED_STRATEGY, bootstrap_samples=200)
    shift = report["paired_shift"]
    assert report["keyed_option_rate"] == 0.5
    assert shift["mean"] == 0.0
    assert shift["reads_the_instruction"] is False
    assert shift["pairs_tied"] == shift["n_complete_pairs"]
    assert shift["sign_test_p"] is None, "every pair tied, so the sign test has nothing to test"


def test_a_model_that_answers_correctly_shifts_positive_in_both_directions(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    battery = behavior.enacted_battery(ontology, bundle)
    results = score_all(battery.probes, oracle_runner(battery.probes), tmp_path / behavior.RESPONSES_FILE)
    report = behavior.summarize_kind(results, behavior.ProbeKind.ENACTED_STRATEGY, bootstrap_samples=200)
    assert report["keyed_option_rate"] == 1.0
    assert report["by_direction"] == {"target": 1.0, "sibling": 1.0}
    assert report["paired_shift"]["mean"] > 0
    assert report["paired_shift"]["pairs_positive"] == report["paired_shift"]["n_complete_pairs"]
    assert report["paired_shift"]["reads_the_instruction"] is True


def test_the_says_only_trap_is_counted_when_the_vocabulary_wins(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    """A model keyed to `retrieval practice` prefers the artifact that discusses it."""
    runner = vocabulary_runner(["retrieval practice"])
    battery, results = enacted_results(ontology, bundle, runner, tmp_path)
    trap = behavior.says_only_trap(results)
    assert trap["n_decided"] == 2
    assert trap["says_only_trap_rate"] == 1.0

    report = behavior.summarize_kind(results, behavior.ProbeKind.NAMED_VERSUS_ENACTED, bootstrap_samples=200)
    assert report["by_direction"] == {"enact": 0.0, "describe": 1.0}
    assert report["paired_shift"]["mean"] == 0.0, "a constant vocabulary preference is not instruction reading"
    assert battery.of_kind(behavior.ProbeKind.NAMED_VERSUS_ENACTED)


def test_the_design_audit_counts_says_only_artifacts_that_carry_no_vocabulary(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle
) -> None:
    battery = behavior.enacted_battery(ontology, bundle)
    audit = behavior.enacted_design_audit(battery.probes, ontology)
    assert audit["n_pairs"] == 2
    assert audit["names_only_vocabulary_rate"] == 1.0
    assert audit["enacting_option_vocabulary_free_rate"] == 1.0
    assert audit["enacting_option_roles"] == {
        "enacted_strategy": {"positive_canonical": 2},
        "named_versus_enacted": {"positive_canonical": 2},
    }

    stripped = [
        replace(
            probe, options=(probe.options[0], replace(probe.options[1], text="Try the next one and see how it goes."))
        )
        if probe.kind is behavior.ProbeKind.NAMED_VERSUS_ENACTED
        else probe
        for probe in battery.probes
    ]
    weakened = behavior.enacted_design_audit(stripped, ontology)
    assert weakened["names_only_vocabulary_rate"] == 0.0


def test_a_tie_is_undecided_rather_than_a_win(ontology: schema.Ontology, bundle: behavior.CorpusBundle) -> None:
    battery = behavior.enacted_battery(ontology, bundle)
    probe = battery.of_kind(behavior.ProbeKind.ENACTED_STRATEGY)[0]
    tied = behavior.ProbeResult(probe, {option.option_id: behavior.TokenScore(-4.0, 2) for option in probe.options})
    assert tied.keyed_wins(behavior.Normalization.MEAN) is None
    report = behavior.summarize_kind([tied], behavior.ProbeKind.ENACTED_STRATEGY, bootstrap_samples=50)
    assert report["n_undecided"] == 1
    assert report["keyed_option_rate"] is None


def test_a_normalization_dependent_contrast_is_reported_as_one(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle
) -> None:
    """A short option can win on total log-probability and lose per token."""
    probe = behavior.enacted_battery(ontology, bundle).of_kind(behavior.ProbeKind.ENACTED_STRATEGY)[0]
    keyed, other = probe.options[0], probe.options[1]
    result = behavior.ProbeResult(
        probe, {keyed.option_id: behavior.TokenScore(-10.0, 10), other.option_id: behavior.TokenScore(-1.5, 1)}
    )
    assert result.keyed_wins(behavior.Normalization.MEAN) is True
    assert result.keyed_wins(behavior.Normalization.TOTAL) is False
    report = behavior.summarize_kind([result], behavior.ProbeKind.ENACTED_STRATEGY, bootstrap_samples=50)
    assert report["normalization_agreement"] == 0.0


def test_the_lexical_control_is_reported_beside_the_score(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    """The instruction shares words with the option that instantiates it, so it is measured."""
    battery, results = enacted_results(ontology, bundle, shortest_option_runner(), tmp_path)
    report = behavior.summarize_kind(results, behavior.ProbeKind.ENACTED_STRATEGY, bootstrap_samples=200)
    assert report["lexical_overlap_shift"]["mean"] is not None
    assert behavior.lexical_overlap("give the learner retrieval practice", ENACTING_TEXT) >= 0.0
    assert behavior.lexical_overlap("", ENACTING_TEXT) == 0.0
    assert battery.probes


def test_the_bootstrap_and_the_sign_test_are_reproducible_and_exact() -> None:
    values = [0.4, -0.1, 0.9, 0.2, 0.5, -0.3, 0.7]
    first = behavior.cluster_bootstrap_ci(values, samples=500, seed=7)
    assert first == behavior.cluster_bootstrap_ci(values, samples=500, seed=7)
    assert first != behavior.cluster_bootstrap_ci(values, samples=500, seed=8)
    assert first[1] <= first[0] <= first[2]
    assert behavior.cluster_bootstrap_ci([]) is None
    assert behavior.cluster_bootstrap_ci([0.3]) == (0.3, 0.3, 0.3)

    assert behavior.sign_test_p(5, 5) == pytest.approx(2 / 32)
    assert behavior.sign_test_p(3, 6) == pytest.approx(1.0)
    assert behavior.sign_test_p(0, 0) is None


# ---------------------------------------------------------------- the report


def full_report(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path, **kwargs
) -> dict[str, object]:
    batteries = behavior.build_batteries(ontology, bundle)
    probes = behavior.all_probes(batteries)
    results = score_all(probes, oracle_runner(probes), tmp_path / behavior.RESPONSES_FILE)
    return behavior.build_report(
        batteries,
        results,
        ontology,
        ontology_path="ontology.yaml",
        bundle=bundle,
        runner_description={"model": "toy-generator", "architecture": "dense"},
        bootstrap_samples=200,
        **kwargs,
    )


def test_the_report_records_agent_grounded_exploratory_calibration_and_no_human_labels(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    report = full_report(ontology, bundle, tmp_path)
    calibration = report["calibration"]
    assert calibration["label_source"] == behavior.LABEL_SOURCE
    assert calibration["status"] == "exploratory"
    assert calibration["human_labels_used"] is False
    assert calibration["confirmatory"] is False
    assert calibration["keying"] == "designed_unverified"
    assert calibration["inter_rater_reliability"] is None
    assert calibration["intra_rater_retest_reliability"] is None
    assert report["claim_limits"] and all(isinstance(limit, str) for limit in report["claim_limits"])


def test_the_report_answers_every_required_reporting_field_or_nulls_it(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    """An absent key reads as an oversight; a null reads as a limit, which is the truth here."""
    fields = full_report(ontology, bundle, tmp_path)["required_reporting_fields"]
    assert set(fields) == set(ontology.required_reporting_fields)
    assert fields["label_source"] == behavior.LABEL_SOURCE
    assert fields["inter_rater_reliability"] is None
    assert fields["reward_eligibility_status"] == "measurement_only"
    assert "learner_enactment" in fields["evidence_limit"]


def test_no_battery_claims_a_rung_it_cannot_reach(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    forbidden = {schema.EvidenceLevel.LEARNER_ENACTMENT.value, schema.EvidenceLevel.LEARNING_OUTCOME.value}
    assert {level.value for level in behavior.Level}.isdisjoint(forbidden)
    report = full_report(ontology, bundle, tmp_path)
    assert {battery["level"] for battery in report["batteries"].values()}.isdisjoint(forbidden)


def test_the_report_flags_scoring_a_model_on_its_own_generations(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    report = full_report(ontology, bundle, tmp_path)
    assert report["corpus"]["source_models"] == ["toy-generator"]
    assert report["corpus"]["scored_model_generated_the_corpus"] is True

    batteries = behavior.build_batteries(ontology, bundle)
    other = behavior.build_report(
        batteries,
        [],
        ontology,
        ontology_path="ontology.yaml",
        bundle=bundle,
        runner_description={"model": "allenai/OLMo-2-1124-7B-Instruct"},
    )
    assert other["corpus"]["scored_model_generated_the_corpus"] is False


def test_the_report_is_strict_json_with_no_nan_and_no_missing_battery(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    report = full_report(ontology, bundle, tmp_path)
    json.dumps(report, allow_nan=False, sort_keys=True)
    assert set(report["batteries"]) == set(behavior.BATTERY_KINDS)
    assert report["coverage"]["n_unanswered"] == 0
    assert not report["coverage"]["constructs_with_no_probe"]


def test_a_partly_scored_battery_still_reports_and_says_how_much_is_missing(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    batteries = behavior.build_batteries(ontology, bundle)
    probes = behavior.all_probes(batteries)
    path = tmp_path / behavior.RESPONSES_FILE
    behavior.run_probes(probes, oracle_runner(probes), out_path=path, limit=8)
    results = behavior.join_responses(probes, behavior.read_responses(path)[0])
    report = behavior.build_report(
        batteries,
        results,
        ontology,
        ontology_path="ontology.yaml",
        bundle=bundle,
        runner_description={"model": "toy-generator"},
        bootstrap_samples=100,
    )
    assert report["coverage"]["n_answered"] == 8
    assert report["coverage"]["n_unanswered"] == len(probes) - 8


def test_the_enacted_battery_needs_a_corpus_and_says_so(ontology: schema.Ontology) -> None:
    with pytest.raises(behavior.BehaviorError, match="needs a generated corpus"):
        behavior.build_batteries(ontology, None, names=["enacted_strategy"])
    with pytest.raises(behavior.BehaviorError, match="unknown batter"):
        behavior.build_batteries(ontology, None, names=["representation"])
    ontology_only = behavior.build_batteries(ontology, None, names=["declarative_knowledge"])
    assert [battery.name for battery in ontology_only] == ["declarative_knowledge"]


# ------------------------------------------------------------- the naming arm


def test_the_naming_arm_measures_naming_only_and_resumes(
    ontology: schema.Ontology, bundle: behavior.CorpusBundle, tmp_path: Path
) -> None:
    battery = behavior.enacted_battery(ontology, bundle)
    runner = ToyRunner(lambda prompt, option: -1.0)
    path = tmp_path / behavior.NAMING_FILE

    report = behavior.run_naming_arm(battery.probes, runner, ontology, out_path=path)
    assert report["generated"] == 2, "one generation per mismatch pair, in the enact direction only"
    assert report["naming_rate"] == 1.0, "the toy writes 'retrieval practice' into every turn"
    assert report["policy"] == "greedy"
    assert [seed for _, seed in runner.generated] == [
        behavior.probe_seed(behavior.DEFAULT_SEED, probe.probe_id)
        for probe in battery.probes
        if probe.kind is behavior.ProbeKind.NAMED_VERSUS_ENACTED and probe.direction is behavior.Direction.ENACT
    ]

    again = behavior.run_naming_arm(battery.probes, runner, ontology, out_path=path)
    assert again["generated"] == 0

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all("retrieval" in row["named_terms"] for row in rows)


# ------------------------------------------------- the real scoring arithmetic


VOCAB_SIZE = 64


class StubEncoding(dict):
    """What a tokenizer returns for `return_tensors='pt'`: a mapping that can be moved."""

    def to(self, _device):
        return self


class StubTokenizer:
    """Whitespace tokenization over a fixed vocabulary, which is enough to merge tokens.

    The property being tested is arithmetic, not language: whether `score`
    charges an option for its own tokens and no others, and whether it notices
    when the option's first token has merged with the prompt's last one. A
    whitespace tokenizer merges exactly when the rendered prompt does not end in
    whitespace, which is a real chat template's business and is what the two
    stubs below differ in.
    """

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, chat_template: str | None = None) -> None:
        self.chat_template = chat_template
        self.vocab: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                assert len(self.vocab) < VOCAB_SIZE, "widen VOCAB_SIZE"
                self.vocab[word] = len(self.vocab) + 1
            ids.append(self.vocab[word])
        return ids

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert not tokenize and add_generation_prompt
        body = " ".join(message["content"] for message in messages)
        return f"{body} {self.chat_template}"

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str | None = None):
        ids = self._ids(text)
        if return_tensors is None:
            return {"input_ids": ids}
        import torch  # noqa: PLC0415

        return StubEncoding(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        lookup = {value: key for key, value in self.vocab.items()}
        return " ".join(lookup.get(int(value), "?") for value in ids)


@dataclass
class StubModel:
    """Uniform next-token logits, except one token id that is boosted everywhere."""

    torch: object
    boosted: int = 1
    boost: float = 4.0
    new_tokens: tuple[int, ...] = (7, 8)

    def __call__(self, ids):
        logits = self.torch.zeros(ids.shape[0], ids.shape[1], VOCAB_SIZE)
        logits[..., self.boosted] = self.boost
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids=None, max_new_tokens: int = 0, **_kwargs):
        extra = self.torch.tensor([list(self.new_tokens[:max_new_tokens])], dtype=input_ids.dtype)
        return self.torch.cat([input_ids, extra], dim=1)

    def parameters(self):
        return iter([self.torch.zeros(1)])


def stub_runner(chat_template: str | None, **kwargs) -> behavior.HuggingFaceRunner:
    torch = pytest.importorskip("torch")
    tokenizer = StubTokenizer(chat_template)
    weights = StubModel(torch, **kwargs)
    return behavior.HuggingFaceRunner("stub", weights, tokenizer, torch.device("cpu"), torch)


def expected_logprob(option: str, boosted_word: str, *, boost: float = 4.0) -> float:
    partition = math.log(math.exp(boost) + VOCAB_SIZE - 1)
    words = option.split()
    return sum((boost if word == boosted_word else 0.0) - partition for word in words)


def test_the_runner_charges_an_option_for_its_own_tokens_and_no_others() -> None:
    runner = stub_runner("<assistant> ")
    boosted_word = "steps"
    runner.tokenizer.vocab[boosted_word] = runner.weights.boosted
    option = "write the three steps from memory"
    scores = runner.score("You are a tutor.", "Question: what next?", [option, "no"])

    assert scores[0].n_tokens == len(option.split())
    assert scores[0].clean_boundary is True
    assert scores[0].logprob == pytest.approx(expected_logprob(option, boosted_word))
    assert scores[0].mean_logprob == pytest.approx(scores[0].logprob / scores[0].n_tokens)
    assert scores[1].n_tokens == 1


def test_the_runner_reports_an_option_that_merged_into_the_prompt() -> None:
    """A template ending mid-token makes the two options slightly incomparable, so it is counted."""
    runner = stub_runner("<assistant>")
    scores = runner.score("You are a tutor.", "Question: what next?", ["Yes. It can.", "No."])
    assert [score.clean_boundary for score in scores] == [False, False]
    assert scores[0].n_tokens == 3, "the merged token is charged to the option that created it"


def test_the_runner_scores_without_a_chat_template_too() -> None:
    runner = stub_runner(None)
    scores = runner.score("You are a tutor.", "Question: what next?", ["a b c"])
    assert scores[0].n_tokens == 3
    assert scores[0].clean_boundary is True


def test_an_option_that_adds_no_tokens_is_refused() -> None:
    runner = stub_runner("<assistant> ")
    with pytest.raises(behavior.BehaviorError, match="adds no tokens"):
        runner.score("You are a tutor.", "Question: what next?", ["   "])


def test_the_runner_decodes_only_the_new_tokens_and_seeds_each_generation() -> None:
    runner = stub_runner("<assistant> ")
    first = runner.generate("You are a tutor.", "Write a turn.", max_new_tokens=2, seed=11)
    again = runner.generate("You are a tutor.", "Write a turn.", max_new_tokens=2, seed=11)
    sampled = runner.generate("You are a tutor.", "Write a turn.", max_new_tokens=2, seed=11, temperature=0.7)
    assert first == again == sampled, "the stub ignores the policy; what matters is that both are accepted"
    assert "Write" not in first, "only the continuation is decoded, so the prompt cannot come back with it"


def test_names_construct_uses_the_corpus_builder_s_vocabulary(ontology: schema.Ontology) -> None:
    construct = ontology.get(CONSTRUCT)
    assert behavior.names_construct(SAYS_ONLY_TEXT, construct)
    assert not behavior.names_construct("Write out the three steps from memory.", construct)
    assert set(behavior.names_construct(SAYS_ONLY_TEXT, construct)) <= set(
        behavior.build_corpus.trigger_words(construct)
    ) | {construct.display_name.lower()}


# ---------------------------------------------------------------------- cli


def test_dry_run_writes_the_plan_and_never_reaches_for_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("a dry run must not load a model")

    monkeypatch.setattr(behavior.HuggingFaceRunner, "load", refuse)
    corpus = write_corpus(tmp_path / "corpus")
    out = tmp_path / "out"
    assert behavior.main(["--corpus", str(corpus), "--out", str(out), "--dry-run"]) == 0

    report = json.loads((out / behavior.REPORT_FILE).read_text())
    assert report["dry_run"] is True
    assert report["model"] == {"model": behavior.DEFAULT_MODEL, "loaded": False}
    assert report["named_versus_enacted_summary"]["design_audit"]["n_pairs"] == 2

    probes = [json.loads(line) for line in (out / behavior.PROBES_FILE).read_text().splitlines()]
    assert probes and {row["schema"] for row in probes} == {behavior.PROBE_SCHEMA}
    assert not (out / behavior.RESPONSES_FILE).exists()


def test_the_cli_refuses_labels_without_a_corpus(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs --corpus"):
        behavior.main(["--out", str(tmp_path / "out"), "--labels", "labels.jsonl", "--dry-run"])


def test_the_cli_scores_resumes_and_writes_one_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = shortest_option_runner()
    monkeypatch.setattr(behavior.HuggingFaceRunner, "load", classmethod(lambda _cls, *a, **k: runner))
    argv = [
        "--corpus",
        str(write_corpus(tmp_path / "corpus")),
        "--out",
        str(tmp_path / "out"),
        "--bootstrap-samples",
        "50",
        "--generate",
    ]

    assert behavior.main(argv) == 0
    out = tmp_path / "out"
    scored = len(runner.scored)
    assert scored == len([json.loads(line) for line in (out / behavior.RESPONSES_FILE).read_text().splitlines()])

    report = json.loads((out / behavior.REPORT_FILE).read_text())
    assert report["dry_run"] is False
    assert report["coverage"]["n_unanswered"] == 0
    assert report["naming_arm"]["generated"] == 2
    assert "not evidence of enactment" in report["naming_arm"]["claim_limit"]
    assert report["enacted_strategy"]["kinds"]["enacted_strategy"]["keyed_option_rate"] == 0.5

    assert behavior.main(argv) == 0
    assert len(runner.scored) == scored, "a second run rescored nothing"
