"""CPU tests for the typed ontology and the corpus builder. No network, no weights, no GPU.

Most of these run against the real `ontology.yaml` rather than a fixture, because
that is the file the round is built from and the properties being checked are
properties of it: whether its `output_types` admit the guide's artifact-type
budget at all, whether the eleven constructs that are nobody's sibling can still
reach the candidate-set floor, whether its own prose smuggles a construct key
into a blinded row. A fixture would pass while the real file failed, which is the
wrong way round.

The invariant tests go the other way and use the smallest broken record that
still parses, since the failures worth catching are the ones that produce a
plausible number: a fidelity of 1 that came from a boolean, a presence of 0 that
was really "not applicable", a resumed job pairing this plan's key with the last
plan's generations.

The generation loop is covered with a stub tokenizer and a stub model rather than
skipped. Appending, resuming, and resampling a refused generation are where a
preempted job loses or duplicates work, and none of that needs weights to test.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from projects.learning_science_semantic_atlas import build_corpus, labeling, schema

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def ontology() -> schema.Ontology:
    return schema.load_ontology()


@pytest.fixture(scope="module")
def plan(ontology: schema.Ontology) -> build_corpus.Plan:
    return build_corpus.build_plan(ontology)


@pytest.fixture(scope="module")
def scaffolded(ontology: schema.Ontology, plan: build_corpus.Plan) -> build_corpus.Assembly:
    return build_corpus.assemble(
        plan, {}, leak_tokens=build_corpus._leak_tokens(ontology), scaffold_missing=True
    )


def read_shots(out_dir) -> list[labeling.Exemplar]:
    """The shots file with the quality column a human session would have filled in."""
    path = out_dir / build_corpus.WITHHELD_DIR / build_corpus.CALIBRATION_SHOTS_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("rationale", None)
        row["label"]["quality"] = 2
    completed = out_dir / "shots_with_quality.jsonl"
    completed.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return labeling.read_exemplars(str(completed))


def fake_generations(plan: build_corpus.Plan, *, text: str | None = None) -> dict[str, dict[str, object]]:
    """One compliant generation per planned item, distinct enough to pass the duplicate check."""
    return {
        item.unit_id: {
            "unit_id": item.unit_id,
            "prompt_sha": item.prompt_sha,
            "text": text or f"Model artifact {index} about {item.scenario.topic}. Here it goes, at some length.",
            "source_model": "test-model",
            "source_policy": "greedy",
            "temperature": 0.0,
            "sample_index": 0,
            "seed": 0,
        }
        for index, item in enumerate(plan.items)
    }


# --------------------------------------------------------------------- ontology


def test_real_ontology_loads_and_is_self_consistent(ontology: schema.Ontology) -> None:
    assert len(ontology) == build_corpus.N_CONSTRUCTS
    assert ontology.construct_version == "atlas-v1.0.0"
    assert len(ontology.debunked_controls) * build_corpus.ITEMS_PER_CONTROL == 18
    for construct in ontology:
        assert ontology.sibling_of(construct.key).key != construct.key
        assert construct.evidence_level is schema.EvidenceLevel.ENACTED_OUTPUT
        assert construct.output_types


def test_fallback_parser_matches_pyyaml() -> None:
    """The bundled parser is checked against the file it exists to read, not trusted."""
    text = schema.default_ontology_path().read_text(encoding="utf-8")
    if not schema.PYYAML_AVAILABLE:  # pragma: no cover - the CI image ships PyYAML
        pytest.skip("PyYAML is absent, so there is nothing to compare against")
    assert schema.parse_yaml(text, prefer_pyyaml=True) == schema.parse_yaml(text, prefer_pyyaml=False)
    assert schema.yaml_backend(prefer_pyyaml=False) == "bundled-subset"


def test_fallback_parser_names_what_it_does_not_implement() -> None:
    with pytest.raises(schema.YamlSubsetError, match="flow mappings"):
        schema.parse_yaml_subset("a: {b: 1}")
    with pytest.raises(schema.YamlSubsetError, match="anchors"):
        schema.parse_yaml_subset("a: &anchor 1")
    with pytest.raises(schema.YamlSubsetError, match="tab"):
        schema.parse_yaml_subset("a:\n\t- 1")


def test_vocabulary_drift_is_refused(ontology: schema.Ontology) -> None:
    """A value added to the YAML and not to the enum is unusable, so loading fails."""
    blob = schema.parse_yaml(schema.default_ontology_path().read_text(encoding="utf-8"))
    blob["output_types"]["podcast_episode"] = "not a thing this code knows"
    with pytest.raises(schema.OntologyError, match="output_types disagrees with OutputFormat"):
        schema.ontology_from_mapping(blob)


def test_declared_construct_count_must_match(ontology: schema.Ontology) -> None:
    blob = schema.parse_yaml(schema.default_ontology_path().read_text(encoding="utf-8"))
    blob["n_constructs"] = 25
    with pytest.raises(schema.OntologyError, match="n_constructs says 25"):
        schema.ontology_from_mapping(blob)


# ------------------------------------------------------------------ the 2x2 type


def test_names_and_enacts_keeps_the_two_apart() -> None:
    says_only = schema.NamesEnacts(names=True, enacts=False)
    assert says_only.cell is schema.EnactmentCell.NAMES_ONLY
    assert says_only.is_says_only
    assert says_only.expected_presence is False

    both = schema.NamesEnacts(names=True, enacts=True)
    assert both.cell is schema.EnactmentCell.NAMES_AND_ENACTS
    assert both.expected_presence is True
    assert not both.is_says_only

    assert schema.NamesEnacts(names=False, enacts=True).cell is schema.EnactmentCell.ENACTS_ONLY
    assert schema.NamesEnacts(names=False, enacts=False).cell is schema.EnactmentCell.NEITHER


def test_names_and_enacts_refuses_integers() -> None:
    with pytest.raises(schema.RecordError, match="names must be a boolean"):
        schema.NamesEnacts(names=1, enacts=False)


# -------------------------------------------------------------------- artifacts


def make_scenario(**overrides) -> schema.Scenario:
    base = dict(
        domain=schema.Domain.ALGEBRA,
        output_format=schema.OutputFormat.TUTOR_TURN,
        learner_state=schema.LearnerState.PARTIAL_ATTEMPT,
        prompt_mode=schema.PromptMode.TUTOR_ROLEPLAY,
        index=0,
        topic="two-step equations",
        target="isolating a variable",
        task="Write the next turn.",
        reference="Subtract 7, then divide by 3.",
        student_before="I divided first.",
    )
    return schema.Scenario(**{**base, **overrides})


def make_artifact(**overrides) -> schema.Artifact:
    base = dict(
        artifact_id="it1",
        lineage_id="ln1",
        concept="retrieval_practice_opportunity",
        role=schema.ItemRole.POSITIVE_CANONICAL,
        enactment=schema.NamesEnacts(names=False, enacts=True),
        text="Without looking back, write the three steps.",
        scenario=make_scenario(),
        expected_fidelity=2,
    )
    return schema.Artifact(**{**base, **overrides})


def test_artifact_rejects_a_fidelity_on_something_it_does_not_enact() -> None:
    with pytest.raises(schema.RecordError, match="enacts nothing"):
        make_artifact(enactment=schema.NamesEnacts(names=True, enacts=False), named_construct="x", expected_fidelity=1)


def test_artifact_requires_a_fidelity_when_it_enacts() -> None:
    with pytest.raises(schema.RecordError, match="must declare an expected_fidelity"):
        make_artifact(expected_fidelity=None)


def test_artifact_requires_a_named_construct_when_it_names() -> None:
    with pytest.raises(schema.RecordError, match="no named_construct is recorded"):
        make_artifact(enactment=schema.NamesEnacts(names=True, enacts=True))


def test_corpus_row_cannot_carry_a_withheld_field() -> None:
    row = make_artifact().as_corpus_row()
    assert set(row) == {"unit_id", "item_id", "concept", "candidate_action", "question", "reference", "student_before"}
    assert not set(row) & set(schema.WITHHELD_FIELDS)
    key = make_artifact().as_key_row(schema.Split.DEV)
    assert key["item_role"] == "positive_canonical"
    assert key["expected_presence"] is True
    assert key["split"] == "dev"


def test_gold_label_follows_enactment_and_never_naming() -> None:
    says_only = make_artifact(
        enactment=schema.NamesEnacts(names=True, enacts=False),
        named_construct="retrieval_practice_opportunity",
        expected_fidelity=None,
        text="Research shows quizzing beats rereading, so quiz yourself.",
    )
    label = schema.gold_label(says_only)
    assert label.presence is False
    assert label.fidelity is None
    assert schema.gold_label(make_artifact()).presence is True


def test_ids_are_content_keyed_and_salt_sensitive() -> None:
    args = (
        "retrieval_practice_opportunity",
        schema.Domain.ALGEBRA,
        schema.OutputFormat.TUTOR_TURN,
        schema.LearnerState.PARTIAL_ATTEMPT,
        schema.PromptMode.TUTOR_ROLEPLAY,
        0,
    )
    first = schema.lineage_id(*args)
    assert first == schema.lineage_id(*args)
    assert first != schema.lineage_id(*args, salt="round2")
    assert schema.artifact_id(first, schema.ItemRole.SAYS_ONLY) != schema.artifact_id(
        first, schema.ItemRole.POSITIVE_CANONICAL
    )
    assert "says_only" not in schema.artifact_id(first, schema.ItemRole.SAYS_ONLY)


# ----------------------------------------------------------------- label records


def test_label_record_null_rules() -> None:
    common = dict(artifact_id="it1", lineage_id="ln1", concept="c", rater="sophia")
    with pytest.raises(schema.RecordError, match="no span was quoted"):
        schema.LabelRecord(**common, applicable=True, applicability=schema.Applicability.APPLICABLE, presence=True, fidelity=2)
    with pytest.raises(schema.RecordError, match="fidelity must be null unless presence"):
        schema.LabelRecord(
            **common, applicable=True, applicability=schema.Applicability.APPLICABLE, presence=False, fidelity=0
        )
    with pytest.raises(schema.RecordError, match="must be null when applicable is false"):
        schema.LabelRecord(
            **common,
            applicable=False,
            applicability=schema.Applicability.NOT_APPLICABLE_MISSING_CONTEXT,
            presence=False,
        )
    with pytest.raises(schema.RecordError, match="the two may not disagree"):
        schema.LabelRecord(**common, applicable=True, applicability=schema.Applicability.NOT_APPLICABLE_WRONG_ARTIFACT)
    with pytest.raises(schema.RecordError, match="got the boolean"):
        schema.LabelRecord(
            **common,
            applicable=True,
            applicability=schema.Applicability.APPLICABLE,
            presence=True,
            fidelity=True,
            span="here",
        )


def test_label_record_rejects_unknown_fields() -> None:
    with pytest.raises(schema.RecordError, match="unknown field"):
        schema.LabelRecord.from_mapping(
            {"unit_id": "it1", "concept": "c", "applicable": True, "applicability": "applicable", "presence": False, "vibe": 3}
        )


# ------------------------------------------------------------ composition counts


def test_round1_matches_the_guide_composition(plan: build_corpus.Plan) -> None:
    assert len(plan.round1) == build_corpus.ROUND1_TOTAL == 160
    pools = {pool: [i for i in plan.round1 if i.pool == pool] for pool in {i.pool for i in plan.round1}}
    assert len(pools[build_corpus.TARGETED_POOL]) == 120
    assert len(pools[build_corpus.CONTROL_POOL]) == 18
    assert len(pools[build_corpus.OPEN_POOL]) == 22


def test_five_roles_per_construct(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    wanted = sorted(role.value for role in build_corpus.ROLES_PER_CONSTRUCT)
    for construct in ontology:
        roles = sorted(i.role.value for i in plan.targeted() if i.concept == construct.key)
        assert roles == wanted, construct.key


def test_artifact_type_budget_is_exact(plan: build_corpus.Plan) -> None:
    """LABELING_GUIDE.md 2.2 is a table of exact counts, not a target."""
    counts = {fmt: 0 for fmt in schema.OutputFormat}
    for item in plan.round1:
        counts[item.output_format] += 1
    for name, formats, capacity in build_corpus.FORMAT_GROUPS:
        assert sum(counts[fmt] for fmt in formats) == capacity, name
    assert counts[schema.OutputFormat.ASSESSMENT_ITEM] == 0
    assert all(counts[fmt] > 0 for fmt in schema.OutputFormat if fmt is not schema.OutputFormat.ASSESSMENT_ITEM)


def test_every_item_is_a_type_its_construct_declares(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    for item in plan.targeted() + plan.targeted(build_corpus.CALIBRATION_BLOCK):
        assert ontology.get(item.concept).accepts(item.output_format), item.unit_id


def test_schedule_level_positives_live_in_the_bottom_35(plan: build_corpus.Plan) -> None:
    bottom = {schema.OutputFormat.PRACTICE_ITEM_SET, schema.OutputFormat.STUDY_SCHEDULE, schema.OutputFormat.CURRICULUM_PLAN}
    for item in plan.targeted():
        if item.concept in build_corpus.SCHEDULE_LEVEL_CONSTRUCTS and item.enactment.enacts:
            assert item.output_format in bottom, item.unit_id


def test_format_budget_solver_reports_an_impossible_budget() -> None:
    units = (build_corpus._FormatUnit("only-group-0", (4,), ((0,),)),)
    with pytest.raises(ValueError, match="no artifact-type assignment"):
        build_corpus._solve_format_budget(units, (0, 4), 0)
    with pytest.raises(ValueError, match="sums to"):
        build_corpus._solve_format_budget(units, (9, 9), 0)


# ---------------------------------------------------------------------- the 2x2


def test_all_four_enactment_cells_are_built(plan: build_corpus.Plan) -> None:
    cells = {item.enactment.cell for item in plan.targeted()}
    assert cells == set(schema.EnactmentCell)


def test_neither_marginal_of_the_2x2_is_constant(plan: build_corpus.Plan) -> None:
    """Without this, a probe reaches the right answer by reading topic and inverting."""
    targeted = plan.targeted()
    enacting = [item.enactment.names for item in targeted if item.enactment.enacts]
    naming = [item.enactment.enacts for item in targeted if item.enactment.names]
    assert any(enacting) and not all(enacting)
    assert any(naming) and not all(naming)


def test_says_only_items_are_gold_absent(plan: build_corpus.Plan) -> None:
    says_only = [i for i in plan.targeted() if i.role is schema.ItemRole.SAYS_ONLY]
    assert len(says_only) == build_corpus.N_CONSTRUCTS
    for item in says_only:
        assert item.enactment.names and not item.enactment.enacts
        assert item.expected_fidelity is None
        assert item.named_construct == item.concept


def test_hard_negative_is_a_quality_matched_sibling(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    for construct in ontology:
        items = [i for i in plan.targeted() if i.concept == construct.key]
        negative = next(i for i in items if i.role is schema.ItemRole.SIBLING_HARD_NEGATIVE)
        canonical = next(i for i in items if i.role is schema.ItemRole.POSITIVE_CANONICAL)
        assert negative.lineage_id == canonical.lineage_id
        assert negative.scenario == canonical.scenario
        assert negative.enacted_construct == construct.sibling_key
        assert negative.enactment.expected_presence is False
        assert negative.sibling_expected_fidelity == 2


def test_every_hard_negative_is_a_real_discrimination(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    """A sibling that cannot apply to the artifact type is marked not-applicable, not discriminated."""
    assert plan.format_solve == "strict"
    for item in plan.targeted():
        if item.role is schema.ItemRole.SIBLING_HARD_NEGATIVE:
            assert ontology.sibling_of(item.concept).accepts(item.output_format), item.concept


def test_control_candidates_can_all_apply(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    """A control shown to a construct that cannot apply to it detects no drift."""
    controls = [i for i in plan.round1 if i.pool == build_corpus.CONTROL_POOL]
    assert len(controls) == 18
    for item in controls:
        for key in item.candidates:
            assert ontology.get(key).accepts(item.output_format), (item.concept, key, item.output_format)


def test_expected_fidelity_spans_the_ordinal(plan: build_corpus.Plan) -> None:
    scores = {i.expected_fidelity for i in plan.targeted() if i.expected_fidelity is not None}
    assert scores == {1, 2, 3}


# -------------------------------------------------------------- candidate sets


def test_candidate_sets_follow_2_3(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    for item in plan.round1:
        assert item.candidates == tuple(sorted(item.candidates))
        if item.pool == build_corpus.OPEN_POOL:
            assert set(item.candidates) == set(ontology.keys)
            continue
        assert len(item.candidates) == build_corpus.CANDIDATE_SET_SIZE
        assert len(set(item.candidates)) == build_corpus.CANDIDATE_SET_SIZE
        if item.pool != build_corpus.TARGETED_POOL:
            continue
        target = ontology.get(item.concept)
        assert item.concept in item.candidates
        assert target.sibling_key in item.candidates
        fillers = [k for k in item.candidates if k not in (target.key, target.sibling_key)]
        assert all(ontology.get(k).family is not target.family for k in fillers)


def test_every_construct_clears_the_appearance_floor(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    counts = {key: 0 for key in ontology.keys}
    for item in plan.round1:
        for key in item.candidates:
            counts[key] += 1
    assert min(counts.values()) >= build_corpus.APPEARANCE_FLOOR
    nobodys_sibling = {c.key for c in ontology} - {c.sibling_key for c in ontology}
    assert len(nobodys_sibling) == 11
    assert all(counts[key] >= build_corpus.APPEARANCE_FLOOR for key in nobodys_sibling)


def test_controls_face_the_constructs_they_could_be_mistaken_for(
    plan: build_corpus.Plan, ontology: schema.Ontology
) -> None:
    """The one control whose `expected_label` names a construct meets it wherever it can apply."""
    discovery = [i for i in plan.round1 if i.concept == "unguided_discovery_learning"]
    assert len(discovery) == build_corpus.ITEMS_PER_CONTROL
    named = ontology.get("problem_solving_before_instruction")
    reachable = [i for i in discovery if named.accepts(i.output_format)]
    assert reachable
    assert all(named.key in i.candidates for i in reachable)
    # The other two are written as types it cannot apply to, so they face the
    # warmth- and framing-adjacent constructs a drifting instrument would reward.
    for item in discovery:
        if item not in reachable:
            assert set(item.candidates) & set(build_corpus.DRIFT_ADJACENT_CONSTRUCTS)


# ------------------------------------------------------------------- calibration


def test_calibration_is_one_per_construct_and_disjoint(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    assert len(plan.calibration) == build_corpus.CALIBRATION_TOTAL == 24
    assert sorted(i.concept for i in plan.calibration) == sorted(ontology.keys)
    assert not {i.unit_id for i in plan.calibration} & {i.unit_id for i in plan.round1}
    assert not {i.lineage_id for i in plan.calibration} & {i.lineage_id for i in plan.round1}


def test_calibration_covers_every_role(plan: build_corpus.Plan) -> None:
    roles = {i.role for i in plan.calibration}
    assert roles == set(build_corpus.ROLES_PER_CONSTRUCT) | {schema.ItemRole.NAMED_AND_ENACTED}


def test_calibration_shots_carry_both_gold_answers(plan: build_corpus.Plan, scaffolded: build_corpus.Assembly) -> None:
    """`select_shots` balances few-shot examples across presence, so both must exist."""
    presence = {
        schema.gold_label(scaffolded.artifacts[item.unit_id]).presence for item in plan.calibration
    }
    assert presence == {True, False}


def test_shot_labels_leave_quality_for_the_calibration_session(
    tmp_path, plan: build_corpus.Plan, scaffolded: build_corpus.Assembly
) -> None:
    """A constant quality column in the shots is how agents learn to answer a constant."""
    build_corpus.write_corpus_files(plan, scaffolded, tmp_path)
    path = tmp_path / build_corpus.WITHHELD_DIR / build_corpus.CALIBRATION_SHOTS_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 24
    assert all(row["label"]["quality"] is None for row in rows)
    assert all(row["rationale"] for row in rows)
    with pytest.raises(labeling.LabelParseError, match="quality"):
        labeling.read_exemplars(str(path))
    assert len(read_shots(tmp_path)) == 24


def test_calibration_texts_do_not_recur_in_round_one(
    plan: build_corpus.Plan, scaffolded: build_corpus.Assembly
) -> None:
    round1 = {schema.normalize_text(scaffolded.artifacts[i.unit_id].text) for i in plan.round1}
    calibration = {schema.normalize_text(scaffolded.artifacts[i.unit_id].text) for i in plan.calibration}
    assert not round1 & calibration


# --------------------------------------------------------------- determinism


def test_the_plan_is_reproducible(ontology: schema.Ontology, plan: build_corpus.Plan) -> None:
    again = build_corpus.build_plan(ontology)
    assert [i.plan_row() for i in again.items] == [i.plan_row() for i in plan.items]
    assert [i.prompt_row() for i in again.items] == [i.prompt_row() for i in plan.items]


def test_a_new_salt_is_a_new_non_colliding_pool(ontology: schema.Ontology, plan: build_corpus.Plan) -> None:
    other = build_corpus.build_plan(ontology, salt="round2")
    assert not {i.unit_id for i in other.items} & {i.unit_id for i in plan.items}
    assert [i.prompt_sha for i in other.items] == [i.prompt_sha for i in plan.items]


def test_build_plan_refuses_an_ontology_of_the_wrong_size(ontology: schema.Ontology) -> None:
    smaller = schema.Ontology(
        schema_version=ontology.schema_version,
        construct_version=ontology.construct_version,
        status=ontology.status,
        constructs=ontology.constructs[:4],
        debunked_controls=ontology.debunked_controls,
        excluded_latent_constructs=(),
        required_reporting_fields=ontology.required_reporting_fields,
    )
    with pytest.raises(schema.OntologyError, match="composition is fixed"):
        build_corpus.build_plan(smaller)


# ----------------------------------------------------------------- prompt files


def test_every_item_has_exactly_one_brief(plan: build_corpus.Plan) -> None:
    rows = [item.prompt_row() for item in plan.items]
    assert len(rows) == 184
    assert len({row["unit_id"] for row in rows}) == 184
    assert all(row["system"] and row["user"] for row in rows)
    assert all(row["prompt_sha"] == build_corpus.prompt_digest(row["system"], row["user"]) for row in rows)


def test_open_pool_briefs_mention_no_construct(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    """A base rate measured on a construct-directed prompt is not a base rate."""
    for item in plan.round1:
        if item.pool != build_corpus.OPEN_POOL:
            continue
        for construct in ontology:
            assert construct.key not in item.user_prompt
            assert construct.display_name not in item.user_prompt


def test_context_dependent_constructs_get_their_declared_field(plan: build_corpus.Plan) -> None:
    for key in ("spaced_practice_schedule", "contingent_help_calibration", "expertise_reversal_adaptation"):
        items = [i for i in plan.targeted() if i.concept == key]
        assert items
        for item in items:
            assert item.context_clause
            assert item.context_clause in item.scenario.task


def test_atypical_positives_are_told_which_vocabulary_to_avoid(plan: build_corpus.Plan) -> None:
    atypical = [i for i in plan.targeted() if i.role is schema.ItemRole.POSITIVE_ATYPICAL]
    assert len(atypical) == build_corpus.N_CONSTRUCTS
    for item in atypical:
        assert item.forbidden_words
        assert all(word in item.user_prompt for word in item.forbidden_words)


# ------------------------------------------------------------------- sanitizing


def test_a_generation_that_echoes_the_brief_is_refused(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    tokens = build_corpus._leak_tokens(ontology)
    item = plan.round1[0]
    for bad, expected in (
        ("", "empty"),
        ("Too short.", "shorter than"),
        ("Here is the artifact you asked for, at a reasonable length for a tutor turn.", "preamble"),
        (
            "This turn instantiates retrieval_practice_opportunity and withholds the answer from the learner.",
            "withheld identifier",
        ),
        ("I followed the positive anchor and avoided the near miss, as your brief asked me to do.", "echoes"),
    ):
        _, reasons = build_corpus.sanitize_generation(bad, item, tokens)
        assert reasons, bad
        assert any(expected in reason for reason in reasons), (bad, reasons)


def test_ordinary_tutor_language_is_not_mistaken_for_an_echo(
    plan: build_corpus.Plan, ontology: schema.Ontology
) -> None:
    """The refusal list is checked as substrings, so it may not contain real words."""
    tokens = build_corpus._leak_tokens(ontology)
    for fine in (
        "Here is a hint: look at what happens to the last index when the loop stops one step early.",
        "Check the boundary condition of that loop before you run it again, and say what you expect.",
        "Two sibling cells come out of this division. How many chromosomes does each one carry?",
    ):
        _, reasons = build_corpus.sanitize_generation(fine, plan.round1[0], tokens)
        assert not reasons, (fine, reasons)


def test_a_wrapping_code_fence_is_a_rendering_artifact_not_a_refusal(
    plan: build_corpus.Plan, ontology: schema.Ontology
) -> None:
    tokens = build_corpus._leak_tokens(ontology)
    body = "Without looking back at your notes, write out the two steps you would take next."
    text, reasons = build_corpus.sanitize_generation(f"```\n{body}\n```", plan.round1[0], tokens)
    assert not reasons
    assert text == body


def test_balanced_internal_code_fences_are_allowed_but_truncated_ones_are_not(
    plan: build_corpus.Plan, ontology: schema.Ontology
) -> None:
    tokens = build_corpus._leak_tokens(ontology)
    body = "Try this example:\n```python\nprint('hello')\n```\nThen explain what the line displays."
    _, reasons = build_corpus.sanitize_generation(body, plan.round1[0], tokens)
    assert not reasons

    truncated = "Try this example:\n```python\nprint('hello')\nThen explain what the line displays."
    _, reasons = build_corpus.sanitize_generation(truncated, plan.round1[0], tokens)
    assert any("unbalanced code fence" in reason for reason in reasons)


def test_an_atypical_positive_using_trigger_vocabulary_is_refused(
    plan: build_corpus.Plan, ontology: schema.Ontology
) -> None:
    item = next(
        i
        for i in plan.targeted()
        if i.role is schema.ItemRole.POSITIVE_ATYPICAL and i.concept == "spaced_practice_schedule"
    )
    text = f"Day 1, day 8, day 20: {item.forbidden_words[0]} the same twelve terms before the day-30 quiz."
    _, reasons = build_corpus.sanitize_generation(text, item, build_corpus._leak_tokens(ontology))
    assert any("trigger vocabulary" in reason for reason in reasons)


# --------------------------------------------------------------------- assembly


def test_assembling_generations_fills_both_blocks(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    assembly = build_corpus.assemble(
        plan, fake_generations(plan), leak_tokens=build_corpus._leak_tokens(ontology)
    )
    assert len(assembly.artifacts) == 184
    assert not assembly.rejections
    assert not assembly.scaffolded
    assert build_corpus.audit(ontology, plan, assembly).ok


def test_a_missing_generation_stops_the_build(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    generations = fake_generations(plan)
    generations.pop(plan.round1[0].unit_id)
    with pytest.raises(ValueError, match="no usable generation"):
        build_corpus.assemble(plan, generations, leak_tokens=build_corpus._leak_tokens(ontology))


def test_a_refused_generation_is_recorded_and_scaffolded(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    generations = fake_generations(plan)
    victim = plan.round1[3].unit_id
    generations[victim]["text"] = "Here is the artifact, which follows the positive anchor in your brief."
    assembly = build_corpus.assemble(
        plan, generations, leak_tokens=build_corpus._leak_tokens(ontology), scaffold_missing=True
    )
    assert [r.unit_id for r in assembly.rejections] == [victim]
    assert assembly.scaffolded == (victim,)
    assert assembly.provenance[victim]["source_model"] == build_corpus.SCAFFOLD_MODEL
    assert build_corpus.SCAFFOLD_MARK in assembly.artifacts[victim].text


def test_a_generation_for_a_different_brief_is_fatal(plan: build_corpus.Plan, ontology: schema.Ontology) -> None:
    """A stale generations file would pair last plan's text with this plan's key."""
    generations = fake_generations(plan)
    generations[plan.round1[0].unit_id]["prompt_sha"] = "0" * 16
    with pytest.raises(ValueError, match="written for prompt"):
        build_corpus.assemble(plan, generations, leak_tokens=build_corpus._leak_tokens(ontology))


def test_scaffolds_are_marked_and_never_duplicated(plan: build_corpus.Plan, scaffolded: build_corpus.Assembly) -> None:
    texts = [artifact.text for artifact in scaffolded.artifacts.values()]
    assert all(text.startswith(build_corpus.SCAFFOLD_MARK) for text in texts)
    assert len({schema.normalize_text(text) for text in texts}) == len(texts)


# ------------------------------------------------------------------ resumption


def test_pending_units_skips_what_is_already_generated(plan: build_corpus.Plan) -> None:
    generations = fake_generations(plan)
    half = dict(list(generations.items())[:90])
    assert len(build_corpus.pending_units(plan, half)) == 184 - 90
    assert not build_corpus.pending_units(plan, generations)


def test_a_changed_brief_makes_an_item_pending_again(plan: build_corpus.Plan) -> None:
    generations = fake_generations(plan)
    generations[plan.round1[0].unit_id]["prompt_sha"] = "stale"
    pending = build_corpus.pending_units(plan, generations)
    assert [item.unit_id for item in pending] == [plan.round1[0].unit_id]


def test_a_refused_or_empty_generation_is_pending_again(plan: build_corpus.Plan) -> None:
    generations = fake_generations(plan)
    refused, empty = plan.round1[:2]
    generations[refused.unit_id]["refusal_reasons"] = ["echoes the brief"]
    generations[empty.unit_id]["text"] = ""

    assert {item.unit_id for item in build_corpus.pending_units(plan, generations)} == {
        refused.unit_id,
        empty.unit_id,
    }


def test_a_truncated_final_line_is_repaired_and_the_rest_kept(tmp_path, plan: build_corpus.Plan) -> None:
    """A preempted job dies mid-write, so the last line can be half an object."""
    path = tmp_path / "generations.jsonl"
    good = [json.dumps(row) for row in list(fake_generations(plan).values())[:3]]
    path.write_text("\n".join(good) + '\n{"unit_id": "half-writ', encoding="utf-8")
    rows, dropped = build_corpus.read_generations(path)
    assert dropped == 1
    assert len(rows) == 3
    again, dropped_again = build_corpus.read_generations(path)
    assert dropped_again == 0
    assert set(again) == set(rows)


def test_a_broken_line_in_the_middle_is_not_repaired(tmp_path, plan: build_corpus.Plan) -> None:
    path = tmp_path / "generations.jsonl"
    good = [json.dumps(row) for row in list(fake_generations(plan).values())[:3]]
    path.write_text("\n".join([good[0], "{oops", good[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the final line"):
        build_corpus.read_generations(path)


# ------------------------------------------------------- the generation loop


class _Shape:
    shape = (1, 4)


class _Encoding(dict):
    def to(self, _device):
        return self


class _StubTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, _prompt, return_tensors=None):
        return _Encoding(input_ids=_Shape())

    def decode(self, tokens, skip_special_tokens=True):
        return str(tokens[0])


class _StubModel:
    """Refuses once per item, then complies, so the retry path is exercised."""

    def __init__(self) -> None:
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            return [[0, 0, 0, 0, "Here is the artifact you asked for; it follows the brief above closely."]]
        return [[0, 0, 0, 0, f"A tutor turn, number {self.calls}, long enough to clear the minimum length."]]


def test_generate_appends_retries_and_resumes(tmp_path, monkeypatch, plan: build_corpus.Plan) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    stub = _StubModel()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _StubTokenizer()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **k: stub))

    path = tmp_path / "generations.jsonl"
    first = build_corpus.generate(plan, out_path=path, model="stub", limit=3, temperature=0.0, max_attempts=2)
    assert first["written"] == 3
    assert first["still_refused"] == 0  # the second attempt complies every time
    rows, _ = build_corpus.read_generations(path)
    assert len(rows) == 3
    assert all(row["source_model"] == "stub" and row["source_policy"] == "greedy" for row in rows.values())

    remaining_before = len(build_corpus.pending_units(plan, rows))
    second = build_corpus.generate(plan, out_path=path, model="stub", limit=2, temperature=0.0, max_attempts=2)
    assert second["already_present"] == 3
    rows_after, _ = build_corpus.read_generations(path)
    assert len(rows_after) == 5
    assert len(build_corpus.pending_units(plan, rows_after)) == remaining_before - 2


def test_generate_records_a_refusal_it_could_not_fix(tmp_path, monkeypatch, plan: build_corpus.Plan) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class _AlwaysEchoes(_StubModel):
        def generate(self, **_kwargs):
            return [[0, 0, 0, 0, "Here is the artifact, written against the positive anchor in your brief."]]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _StubTokenizer()))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **k: _AlwaysEchoes())
    )
    path = tmp_path / "generations.jsonl"
    report = build_corpus.generate(plan, out_path=path, model="stub", limit=1, temperature=0.0, max_attempts=2)
    assert report["still_refused"] == 1
    rows, _ = build_corpus.read_generations(path)
    assert next(iter(rows.values()))["refusal_reasons"]


# ---------------------------------------------------------------------- auditing


def test_audit_passes_on_the_real_ontology(ontology: schema.Ontology, plan: build_corpus.Plan) -> None:
    report = build_corpus.audit(ontology, plan)
    assert report.ok, [c.name for c in report.failures]
    assert len(report.checks) >= 25
    assert report.stats["candidate_appearance_min"] >= build_corpus.APPEARANCE_FLOOR
    assert report.stats["sibling_contrast_pairs"] == build_corpus.N_CONSTRUCTS


def test_audit_fails_when_the_composition_is_broken(ontology: schema.Ontology, plan: build_corpus.Plan) -> None:
    broken = replace(plan, round1=plan.round1[:-1])
    failed = {check.name for check in build_corpus.audit(ontology, broken).failures}
    assert "round1_total" in failed
    assert "artifact_type_budget" in failed


def test_audit_catches_a_leaked_identifier_in_a_corpus_row(
    ontology: schema.Ontology, plan: build_corpus.Plan
) -> None:
    """The audit re-scans what reached the corpus, so a bypassed sanitizer still fails."""
    assembly = build_corpus.assemble(plan, {}, leak_tokens=(), scaffold_missing=True)
    victim = plan.round1[0]
    leaked = dict(assembly.artifacts)
    leaked[victim.unit_id] = victim.artifact("This turn enacts retrieval_practice_opportunity, plainly and at length.")
    report = build_corpus.audit(ontology, plan, replace(assembly, artifacts=leaked))
    assert "no_withheld_identifier_in_corpus" in {check.name for check in report.failures}


# ---------------------------------------------------------- downstream contract


def test_the_corpus_is_readable_by_the_labeling_module(
    tmp_path, plan: build_corpus.Plan, scaffolded: build_corpus.Assembly
) -> None:
    """`labeling.build_packet` is the next stage, so its reader is the acceptance test."""
    build_corpus.write_corpus_files(plan, scaffolded, tmp_path)
    units = labeling.read_corpus(str(tmp_path / build_corpus.CORPUS_FILE))
    assert len(units) == 160
    shots = read_shots(tmp_path)
    assert len(shots) == 24

    spec = labeling.ConstructSpec(
        key="retrieval_practice_opportunity",
        question="Does the artifact require production from memory?",
        positive_anchor="A closed-source production request.",
        negative_anchor="A rhetorical check.",
    )
    packet = labeling.build_packet(units, spec, shots=shots, seed=7)
    assert packet.slots
    view = json.dumps(packet.judge_view())
    assert "positive_canonical" not in view
    assert "designed_target_construct" not in view


def test_variants_of_one_lineage_share_one_stem(plan: build_corpus.Plan) -> None:
    """`labeling._shared_context` refuses a corpus whose variants disagree about context."""
    by_lineage: dict[str, list[build_corpus.PlanItem]] = {}
    for item in plan.items:
        by_lineage.setdefault(item.lineage_id, []).append(item)
    assert any(len(group) > 1 for group in by_lineage.values())
    for group in by_lineage.values():
        assert len({(i.scenario.task, i.scenario.reference, i.scenario.student_before) for i in group}) == 1


# ---------------------------------------------------------------------- the cli


def test_cli_plan_only_writes_the_design(tmp_path) -> None:
    assert build_corpus.main(["--out", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / build_corpus.MANIFEST_FILE).read_text())
    assert manifest["audit"]["ok"]
    assert manifest["audit"]["stats"]["generation"] == {"status": "not started"}
    assert (tmp_path / build_corpus.WITHHELD_DIR / build_corpus.PROMPTS_FILE).exists()
    # Nothing blinded exists yet: until there are artifacts there is nothing a
    # rater may see, and every file the plan does write names its item's answer.
    assert not (tmp_path / build_corpus.CORPUS_FILE).exists()
    assert not (tmp_path / build_corpus.ITEMS_FILE).exists()
    assert all(name.startswith(build_corpus.WITHHELD_DIR) for name in manifest["files"])


def test_cli_no_model_run_is_complete_and_marked(tmp_path) -> None:
    assert build_corpus.main(["--out", str(tmp_path), "--mode", "all", "--no-model"]) == 0
    manifest = json.loads((tmp_path / build_corpus.MANIFEST_FILE).read_text())
    assert manifest["audit"]["ok"]
    assert manifest["files"][build_corpus.CORPUS_FILE] == 160
    assert manifest["files"][f"{build_corpus.WITHHELD_DIR}/{build_corpus.KEY_FILE}"] == 160
    assert manifest["files"][f"{build_corpus.WITHHELD_DIR}/{build_corpus.RETEST_FILE}"] == build_corpus.RETEST_TOTAL
    assert manifest["files"][build_corpus.ITEMS_FILE] == 160
    generation = manifest["audit"]["stats"]["generation"]
    assert generation["status"] == "scaffolded"
    assert generation["sources"] == {build_corpus.SCAFFOLD_MODEL: 184}

    items = [json.loads(line) for line in (tmp_path / build_corpus.ITEMS_FILE).read_text().splitlines()]
    assert all(row["candidate_action"] and row["candidates"] for row in items)
    assert all(row["candidates"] == sorted(row["candidates"]) for row in items)


def test_cli_output_is_byte_identical_across_runs(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    assert build_corpus.main(["--out", str(first), "--mode", "all", "--no-model"]) == 0
    assert build_corpus.main(["--out", str(second), "--mode", "all", "--no-model"]) == 0
    for name in (build_corpus.CORPUS_FILE, build_corpus.ITEMS_FILE, build_corpus.MANIFEST_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_blinded_files_carry_no_role_and_no_provenance(tmp_path) -> None:
    build_corpus.main(["--out", str(tmp_path), "--mode", "all", "--no-model"])
    for name in (build_corpus.CORPUS_FILE, build_corpus.ITEMS_FILE, build_corpus.CALIBRATION_CORPUS_FILE):
        rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        columns = {column for row in rows for column in row}
        assert not columns & {"item_role", "designed_target_construct", "source_model", "expected_presence", "names"}
        blob = json.dumps(rows)
        assert "positive_canonical" not in blob
        assert "sibling_hard_negative" not in blob


def test_a_second_salt_in_one_directory_is_reported(tmp_path, capsys) -> None:
    """A round-2 key beside a round-1 corpus is a mislabelling nobody could eyeball."""
    assert build_corpus.main(["--out", str(tmp_path), "--mode", "all", "--no-model"]) == 0
    capsys.readouterr()
    assert build_corpus.main(["--out", str(tmp_path), "--salt", "round2"]) == 0
    assert "did not produce" in capsys.readouterr().err
    manifest = json.loads((tmp_path / build_corpus.MANIFEST_FILE).read_text())
    assert manifest["salt"] == "round2"
    assert manifest["files"][build_corpus.CORPUS_FILE] == 160


def test_retest_plan_is_stratified_and_resalted(plan: build_corpus.Plan) -> None:
    assert len(plan.retest) == build_corpus.RETEST_TOTAL == 40
    counts: dict[str, int] = {}
    for entry in plan.retest:
        counts[str(entry["stratum"])] = counts.get(str(entry["stratum"]), 0) + 1
    assert counts == dict(build_corpus.RETEST_STRATA)
    targeted = [e for e in plan.retest if e["stratum"] == build_corpus.TARGETED_POOL]
    by_unit = {i.unit_id: i for i in plan.round1}
    assert len({by_unit[str(e["of_unit_id"])].concept for e in targeted}) == build_corpus.N_CONSTRUCTS
    assert len({str(e["item_role"]) for e in targeted}) > 1
    assert not {str(e["retest_id"]) for e in plan.retest} & set(by_unit)
