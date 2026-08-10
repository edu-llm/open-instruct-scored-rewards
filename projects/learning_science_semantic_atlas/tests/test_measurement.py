"""CPU tests for the labelling, agreement and PPI helpers. No network, no torch, toy data only.

The statistics are checked against values worked out by hand rather than against a
second implementation, so a test failing means the arithmetic changed and not that
two libraries disagree. The interesting cases are the ones where a plausible
implementation is wrong in a way that still produces a number: kappa on a rare
construct, a bootstrap that forgets its clusters, a rectifier that cancels itself.
"""

from __future__ import annotations

import json

import pytest
from projects.learning_science_semantic_atlas import agreement, ppi
from projects.learning_science_semantic_atlas.labeling import (
    FIDELITY,
    KEY_SCHEMA,
    ConstructSpec,
    Exemplar,
    Label,
    LabelParseError,
    Unit,
    build_packet,
    ingest_responses,
    parse_label,
    parse_response,
    read_key,
    select_shots,
    write_packet,
)

CONSTRUCT = "generative_elicitation"
OTHER = "contingent_scaffolding"

SPEC = ConstructSpec(
    key=CONSTRUCT,
    question="Does the turn require the learner to produce or retrieve something?",
    positive_anchor="Asks for retrieval, justification, or a next step.",
    negative_anchor="Rhetorical questions and tutor-performed explanation.",
    applicability_note="The turn is administrative and carries no instructional move.",
)


def make_units(n_items: int = 6, variants: int = 2, construct: str = CONSTRUCT) -> list[Unit]:
    """A toy corpus: ``variants`` candidate actions per item, sharing the item's stem."""
    return [
        Unit(
            unit_id=f"{construct[:4]}-i{item}-v{variant}",
            item_id=f"item{item}",
            construct=construct,
            text=f"candidate {item}.{variant}",
            context={"question": f"stem {item}", "reference": f"answer {item}"},
        )
        for item in range(n_items)
        for variant in range(variants)
    ]


def label_record(
    unit_id: str,
    rater: str,
    *,
    applicable: bool = True,
    presence: bool | None = True,
    fidelity: int | None = 2,
    quality: int | None = 2,
    item_id: str | None = None,
    construct: str = CONSTRUCT,
    quarantined: bool = False,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "item_id": item_id or f"item-of-{unit_id}",
        "construct": construct,
        "rater": rater,
        "applicable": applicable,
        "presence": presence if applicable else None,
        "fidelity": fidelity if applicable and presence else None,
        "quality": quality if applicable else None,
        "quarantined": quarantined,
    }


def binary_records(rater: str, values: list[int], construct: str = CONSTRUCT) -> list[dict[str, object]]:
    """One label per unit, ``presence`` following ``values``.

    Fidelity walks the scale on the present units rather than sitting on one
    value: a field where every rating is identical has no variance for a
    chance-corrected statistic to work with, and the gate reports that as
    unmeasurable rather than as agreement.
    """
    return [
        label_record(
            f"u{i}", rater, presence=bool(v), fidelity=i % 4 if v else None, item_id=f"item{i}", construct=construct
        )
        for i, v in enumerate(values)
    ]


# ------------------------------------------------------------------- packets


def test_packet_asks_about_one_construct_and_shows_nothing_else():
    units = make_units() + make_units(construct=OTHER)
    packet = build_packet(units, SPEC, seed=3)

    assert packet.construct == CONSTRUCT
    assert len(packet.slots) == 6
    text = json.dumps(packet.judge_view())
    assert SPEC.question in text
    assert OTHER not in text
    for slot in packet.slots:
        assert slot.item_id not in text  # items are addressed by opaque slot id only
        for _, unit_id, _ in slot.options:
            assert unit_id not in text


def test_option_order_is_shuffled_reproducibly_and_letters_cover_every_unit():
    units = make_units(n_items=8, variants=3)
    first = build_packet(units, SPEC, seed=11)
    again = build_packet(units, SPEC, seed=11)
    other_seed = build_packet(units, SPEC, seed=12)

    assert first.key() == again.key(), "the recorded seed must reproduce the blinding exactly"
    assert first.key() != other_seed.key(), "a different seed must reorder something"

    for slot in first.slots:
        letters = [letter for letter, _, _ in slot.options]
        assert letters == ["A", "B", "C"]
        assert len({unit_id for _, unit_id, _ in slot.options}) == 3

    # Every corpus unit appears exactly once across the packet, wherever it landed.
    placed = sorted(unit_id for slot in first.slots for _, unit_id, _ in slot.options)
    assert placed == sorted(unit.unit_id for unit in units)


def test_shuffle_is_not_a_fixed_permutation_across_items():
    """A per-item seed, not one shuffle reused - otherwise position still encodes source."""
    units = make_units(n_items=40, variants=3)
    orders = {
        tuple(unit_id for _, unit_id, _ in slot.options)[0][-2:] for slot in build_packet(units, SPEC, seed=5).slots
    }
    assert len(orders) > 1


def test_few_shot_items_are_quarantined_along_with_their_siblings():
    units = make_units(n_items=5, variants=2)
    shot = Exemplar(units[0], Label(applicable=True, presence=True, fidelity=3, quality=3, span="candidate 0.0"))
    packet = build_packet(units, SPEC, shots=[shot], seed=1)

    assert shot.unit.item_id in packet.quarantined_item_ids
    evaluated = {unit_id for slot in packet.slots for _, unit_id, _ in slot.options}
    assert shot.unit.unit_id not in evaluated
    # The sibling variant shares the stem and the reference, so it leaks too.
    assert units[1].unit_id not in evaluated
    assert len(packet.slots) == 4
    assert packet.shot_unit_ids == (shot.unit.unit_id,)
    assert json.loads(json.dumps(packet.key()))["quarantined_item_ids"] == ["item0"]


def test_shot_selection_shows_both_sides_of_the_boundary_once_per_item():
    units = make_units(n_items=6, variants=2)
    # Presence varies by item, not by variant, so half the items are genuine negatives.
    pool = [
        Exemplar(
            unit,
            Label(
                applicable=True,
                presence=int(unit.item_id[-1]) % 2 == 0,
                fidelity=3 if int(unit.item_id[-1]) % 2 == 0 else None,
                quality=2,
                span=unit.text if int(unit.item_id[-1]) % 2 == 0 else None,
            ),
        )
        for unit in units
    ]
    shots = select_shots(pool, CONSTRUCT, max_shots=4, seed=0)

    assert len(shots) == 4
    assert len({shot.unit.item_id for shot in shots}) == 4, "one exemplar per item; siblings add nothing"
    assert sum(bool(shot.label.presence) for shot in shots) == 2, "balanced across the presence label"
    assert select_shots(pool, CONSTRUCT, max_shots=4, seed=0) == shots
    assert select_shots(pool, OTHER, max_shots=4, seed=0) == []


def test_more_options_than_letters_is_refused():
    units = make_units(n_items=1, variants=27)
    with pytest.raises(ValueError, match="only 26 letters"):
        build_packet(units, SPEC, seed=0)


def test_variants_that_disagree_about_the_context_are_refused():
    """The slot renders one context, so a variant-specific one would identify its option."""
    stem = {"question": "stem 0", "student_before": "I think it is 4"}
    conflicting = [
        Unit("gene-i0-v0", "item0", CONSTRUCT, "ask them to check it", dict(stem)),
        Unit("gene-i0-v1", "item0", CONSTRUCT, "tell them the answer", {**stem, "student_before": "no idea"}),
    ]
    with pytest.raises(ValueError, match="context field 'student_before' differs"):
        build_packet(conflicting, SPEC, seed=1)

    # A field one variant simply omits is still the item's, so it comes from the sibling.
    partial = [
        Unit("gene-i0-v0", "item0", CONSTRUCT, "ask them to check it", dict(stem)),
        Unit("gene-i0-v1", "item0", CONSTRUCT, "tell them the answer", {"question": "stem 0"}),
    ]
    assert build_packet(partial, SPEC, seed=1).slots[0].context == stem


def test_a_second_seed_cannot_overwrite_the_key_that_unblinds_the_first(tmp_path):
    """A key is the only route from a letter back to a unit; clobbering one loses a round."""
    units = make_units(n_items=4, variants=2)
    written = [write_packet(build_packet(units, SPEC, seed=seed), str(tmp_path)) for seed in (1, 2)]

    assert len({path for pair in written for path in pair}) == 4, "one packet and one key per seed"
    keys = [read_key(key_path) for _, key_path in written]
    assert [key["seed"] for key in keys] == [1, 2]
    assert keys[0]["schema"] == KEY_SCHEMA
    assert set(keys[0]["slots"]) != set(keys[1]["slots"]), "each seed has its own slots to unblind"

    with pytest.raises(ValueError, match="not usable as a filename"):
        write_packet(build_packet(units, SPEC, seed=1, packet_id="../outside"), str(tmp_path))


def test_a_key_that_cannot_unblind_anything_is_refused_at_load(tmp_path):
    """A malformed key is the wrong file, not a reply to count as a parse failure."""
    broken = tmp_path / "broken.key.json"
    broken.write_text(json.dumps({"schema": KEY_SCHEMA, "slots": {"s0": {"item_id": "item0"}}}))
    with pytest.raises(ValueError, match="needs an item_id and a non-empty options map"):
        read_key(str(broken))

    wrong_schema = tmp_path / "other.key.json"
    wrong_schema.write_text(json.dumps({"schema": "something-else", "slots": {}}))
    with pytest.raises(ValueError, match="expected schema"):
        read_key(str(wrong_schema))


# -------------------------------------------------------------- strict parsing


def test_parse_label_accepts_the_contract():
    label = parse_label({"applicable": True, "presence": True, "fidelity": 2, "quality": 3, "span": "why?"})
    assert label == Label(applicable=True, presence=True, fidelity=2, quality=3, span="why?")

    absent = parse_label({"applicable": True, "presence": False, "fidelity": None, "quality": 1})
    assert absent.presence is False and absent.fidelity is None and absent.quality == 1

    declined = parse_label({"applicable": False, "presence": None, "fidelity": None, "quality": None})
    assert declined == Label(applicable=False)
    assert parse_label({"applicable": False}) == Label(applicable=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"applicable": True, "presence": 1, "quality": 2}, "presence must be a JSON boolean"),
        ({"applicable": 1, "presence": True, "fidelity": 1, "quality": 2}, "applicable must be a JSON boolean"),
        ({"presence": True, "quality": 2}, "missing required field 'applicable'"),
        ({"applicable": True, "presence": True, "fidelity": 4, "quality": 2}, "fidelity must be an integer 0-3"),
        ({"applicable": True, "presence": True, "fidelity": -1, "quality": 2}, "fidelity must be an integer 0-3"),
        ({"applicable": True, "presence": True, "fidelity": True, "quality": 2}, "fidelity must be an integer 0-3"),
        ({"applicable": True, "presence": True, "fidelity": 2.0, "quality": 2}, "fidelity must be an integer 0-3"),
        ({"applicable": True, "presence": True, "fidelity": 2, "quality": None}, "quality must be an integer 1-3"),
        ({"applicable": True, "presence": True, "fidelity": 2, "quality": 0, "span": "x"}, "quality must be an integer 1-3"),
        ({"applicable": True, "presence": False, "fidelity": 2, "quality": 2}, "fidelity must be null unless"),
        ({"applicable": False, "presence": True}, "presence must be null when applicable is false"),
        ({"applicable": False, "quality": 2}, "quality must be null when applicable is false"),
        ({"applicable": True, "presence": None, "quality": 2}, "presence is required when applicable is true"),
        ({"applicable": True, "presence": True, "fidelity": 2, "quality": 2, "score": 9}, "unknown field"),
        ({"applicable": True, "presence": True, "fidelity": 2, "quality": 2}, "span must be a non-empty quote"),
        ({"applicable": True, "presence": True, "fidelity": 2, "quality": 2, "span": 7}, "span must be a str"),
        ([1, 2], "expected an object"),
    ],
)
def test_parse_label_refuses_everything_off_contract(payload, message):
    with pytest.raises(LabelParseError, match=message):
        parse_label(payload)


def test_parse_response_requires_exactly_the_options_offered():
    good = '{"labels": {"A": {"applicable": true, "presence": true, "fidelity": 1, "quality": 2, "span": "candidate"}, "B": {"applicable": false}}}'
    parsed = parse_response(good, ["A", "B"])
    assert parsed["A"].fidelity == 1 and parsed["B"].applicable is False

    with pytest.raises(LabelParseError, match=r"no label for option\(s\) \['B'\]"):
        parse_response('{"labels": {"A": {"applicable": false}}}', ["A", "B"])
    with pytest.raises(LabelParseError, match="that were not offered"):
        parse_response('{"labels": {"A": {"applicable": false}, "Z": {"applicable": false}}}', ["A"])
    with pytest.raises(LabelParseError, match="unexpected top-level key"):
        parse_response('{"labels": {"A": {"applicable": false}}, "reasoning": "hmm"}', ["A"])
    with pytest.raises(LabelParseError, match="not a single JSON object"):
        parse_response('Sure! Here you go: {"labels": {}}', ["A"])


def test_code_fences_fail_by_default_and_unwrap_only_when_asked():
    fenced = '```json\n{"labels": {"A": {"applicable": false}}}\n```'
    with pytest.raises(LabelParseError, match="not a single JSON object"):
        parse_response(fenced, ["A"])
    assert parse_response(fenced, ["A"], allow_code_fence=True)["A"].applicable is False


def test_ingest_unblinds_by_letter_and_keeps_bad_replies_visible():
    units = make_units(n_items=3, variants=2)
    shot = Exemplar(units[0], Label(applicable=True, presence=True, fidelity=2, quality=2, span="candidate 0.0"))
    packet = build_packet(units, SPEC, shots=[shot], seed=4)
    key = packet.key()
    first, second = packet.slots[0], packet.slots[1]

    body = {
        letter: {"applicable": True, "presence": True, "fidelity": 1, "quality": 2, "span": "candidate"}
        for letter, _, _ in first.options
    }
    result = ingest_responses(
        key,
        [
            {"slot_id": first.slot_id, "text": json.dumps({"labels": body})},
            {"slot_id": second.slot_id, "text": "not json at all"},
            {"slot_id": "sdeadbeef0000", "text": "{}"},
        ],
        rater="toy_judge",
    )

    assert {record["unit_id"] for record in result.records} == {unit_id for _, unit_id, _ in first.options}
    assert all(record["item_id"] == first.item_id for record in result.records)
    assert all(record["rater"] == "toy_judge" and record["fidelity"] == 1 for record in result.records)
    assert [failure["slot_id"] for failure in result.failures] == [second.slot_id, "sdeadbeef0000"]
    assert "not in this key" in result.failures[1]["error"]

    with pytest.raises(ValueError, match="counted twice"):
        ingest_responses(key, [{"slot_id": first.slot_id, "text": "{}"}] * 2, rater="toy_judge")


def test_the_failure_rate_is_a_share_of_replies_not_of_labels():
    """One reply covers every option in its slot, so per-label would report a third of it."""
    packet = build_packet(make_units(n_items=2, variants=3), SPEC, seed=6)
    good, bad = packet.slots
    body = {
        letter: {"applicable": True, "presence": True, "fidelity": 1, "quality": 2, "span": "candidate"}
        for letter, _, _ in good.options
    }
    result = ingest_responses(
        packet.key(),
        [
            {"slot_id": good.slot_id, "text": json.dumps({"labels": body})},
            {"slot_id": bad.slot_id, "text": "I would rather explain my reasoning first."},
        ],
        rater="toy_judge",
    )

    assert (len(result.records), len(result.failures), result.n_replies) == (3, 1, 2)
    assert result.failure_rate == pytest.approx(0.5), "one of the two replies did not parse"


# -------------------------------------------------------------- agreement maths


def test_cohens_kappa_matches_hand_computation():
    assert agreement.cohens_kappa([(1, 1), (1, 0), (0, 1), (0, 0)], (0, 1)) == pytest.approx(0.0)
    assert agreement.cohens_kappa([(1, 1), (1, 1), (1, 0), (0, 0)], (0, 1)) == pytest.approx(0.5)
    assert agreement.cohens_kappa([(1, 1), (0, 0)] * 5, (0, 1)) == pytest.approx(1.0)


def test_weighted_kappa_gives_partial_credit_for_being_one_off():
    assert agreement.weighted_kappa([(0, 0), (1, 1), (2, 2), (3, 3)], FIDELITY) == pytest.approx(1.0)
    assert agreement.weighted_kappa([(0, 0), (0, 1), (1, 1), (1, 1)], FIDELITY) == pytest.approx(0.5)
    assert agreement.weighted_kappa([(0, 0), (0, 3), (3, 0), (3, 3)], FIDELITY) == pytest.approx(0.0)

    near = [(0, 1), (1, 2), (2, 3), (3, 2), (1, 1), (2, 2)]
    linear = agreement.weighted_kappa(near, FIDELITY, "linear")
    quadratic = agreement.weighted_kappa(near, FIDELITY, "quadratic")
    assert quadratic > linear, "quadratic weights forgive one-point gaps, which is why linear is the default"
    with pytest.raises(ValueError, match="linear.*quadratic"):
        agreement.weighted_kappa(near, FIDELITY, "cubic")


def test_ac1_stays_interpretable_where_kappa_collapses_on_a_rare_construct():
    """Two raters agree on 18 of 20 units and disagree about which single unit is positive."""
    pairs = [(1, 0), (0, 1)] + [(0, 0)] * 18

    assert agreement.percent_agreement(pairs) == pytest.approx(0.90)
    assert agreement.cohens_kappa(pairs, (0, 1)) == pytest.approx(-0.05263, abs=1e-4)
    assert agreement.gwet_ac1([list(pair) for pair in pairs], (0, 1)) == pytest.approx(0.88950, abs=1e-4)


def test_kappa_is_undefined_where_ac1_is_merely_unanimous():
    unanimous = [(0, 0)] * 10
    assert agreement.cohens_kappa(unanimous, (0, 1)) != agreement.cohens_kappa(unanimous, (0, 1))  # NaN
    assert agreement.gwet_ac1([[0, 0]] * 10, (0, 1)) == pytest.approx(1.0)


def test_gwet_ac1_handles_uneven_rater_counts_and_ignores_singletons():
    assert agreement.gwet_ac1([[1, 1, 1], [0, 0, 0]], (0, 1)) == pytest.approx(1.0)
    assert agreement.gwet_ac1([[1, 1], [0, 0], [1]], (0, 1)) == agreement.gwet_ac1([[1, 1], [0, 0]], (0, 1))
    assert agreement.gwet_ac1([], (0, 1)) != agreement.gwet_ac1([], (0, 1))  # NaN, not a crash


def test_a_rating_off_the_scale_is_refused_rather_than_absorbed():
    """Every chance term sums over the categories, so a stray rating returns a plausible lie."""
    with pytest.raises(ValueError, match=r"outside the scale \[0, 1\]"):
        agreement.gwet_ac1([[0, 7]] * 10, (0, 1))
    with pytest.raises(ValueError, match=r"outside the scale \[0, 1\]"):
        agreement.cohens_kappa([(0, 7)] * 10, (0, 1))
    with pytest.raises(ValueError, match="outside the scale"):
        agreement.weighted_kappa([(0, 9)], FIDELITY)

    # And an unusable weighting is a typo whether or not there is data to apply it to.
    with pytest.raises(ValueError, match="linear.*quadratic"):
        agreement.weighted_kappa([], FIDELITY, "cubic")


def test_pooled_kappa_is_a_property_of_the_data_not_of_the_file_order():
    """Ordered pairs, so with four raters whose marginals differ the row/column split matters."""

    def rating(rater: str, index: int) -> int:
        """Four raters with deliberately unlike marginals: middling, lenient, flat, erratic."""
        return {"a": index % 4, "b": min(3, index % 4 + 1), "c": 0, "d": 3 if index % 3 else 1}[rater]

    def loaded(order: tuple[str, ...]) -> agreement.FieldAgreement:
        records = [
            label_record(f"u{index}", rater, fidelity=rating(rater, index), item_id=f"item{index}")
            for index in range(16)
            for rater in order
        ]
        return agreement.field_agreement(CONSTRUCT, agreement.index_labels(records)[CONSTRUCT], "fidelity")

    canonical = loaded(("a", "b", "c", "d"))
    assert canonical.n_pairs == 16 * 6, "four raters make six pairs per unit"
    for order in (("b", "d", "a", "c"), ("d", "c", "b", "a"), ("c", "a", "d", "b")):
        shuffled = loaded(order)
        assert shuffled.kappa_w == canonical.kappa_w
        assert shuffled.cohen == canonical.cohen
        assert shuffled.exact == canonical.exact


def test_headline_statistic_follows_prevalence_not_taste():
    common = agreement.index_labels(binary_records("a", [1, 0] * 10) + binary_records("b", [1, 0] * 10))
    rare_values = [1] + [0] * 19
    rare = agreement.index_labels(binary_records("a", rare_values) + binary_records("b", rare_values))

    balanced = agreement.field_agreement(CONSTRUCT, common[CONSTRUCT], "presence")
    assert balanced.headline == "cohen" and balanced.value == pytest.approx(balanced.cohen)
    assert balanced.prevalence == pytest.approx(0.5)

    lopsided = agreement.field_agreement(CONSTRUCT, rare[CONSTRUCT], "presence")
    assert lopsided.headline == "ac1" and lopsided.value == pytest.approx(lopsided.ac1)
    assert lopsided.prevalence == pytest.approx(0.05)


def test_fidelity_pairs_exist_only_where_both_raters_saw_the_construct():
    units = agreement.index_labels(
        [
            label_record("u1", "a", presence=True, fidelity=3),
            label_record("u1", "b", presence=True, fidelity=2),
            label_record("u2", "a", presence=True, fidelity=1),
            label_record("u2", "b", presence=False),  # fidelity is null by contract
            label_record("u3", "a", applicable=False),
            label_record("u3", "b", applicable=False),
        ]
    )[CONSTRUCT]

    assert agreement.field_agreement(CONSTRUCT, units, "fidelity").n_pairs == 1
    assert agreement.field_agreement(CONSTRUCT, units, "presence").n_pairs == 2
    assert agreement.field_agreement(CONSTRUCT, units, "applicable").n_pairs == 3
    assert agreement.not_applicable_rate(units) == pytest.approx(2 / 6)


# ----------------------------------------------------------------- gate, retest


def test_gate_separates_underpowered_from_unreliable():
    agree = [1, 0] * 15
    solid = agreement.reliability_gate(binary_records("a", agree) + binary_records("b", agree), min_pairs=10)
    assert solid[CONSTRUCT].verdict == "pass"
    assert solid[CONSTRUCT].usable

    noisy = agreement.reliability_gate(
        binary_records("a", [1, 0] * 15) + binary_records("b", [0, 1] * 15), min_pairs=10
    )
    assert noisy[CONSTRUCT].verdict == "fail"
    assert not noisy[CONSTRUCT].usable
    assert any("presence" in reason for reason in noisy[CONSTRUCT].reasons)

    thin = agreement.reliability_gate(binary_records("a", [1, 0]) + binary_records("b", [1, 0]), min_pairs=10)
    assert thin[CONSTRUCT].verdict == "underpowered"
    assert not thin[CONSTRUCT].usable
    assert any("comparable pairs" in reason for reason in thin[CONSTRUCT].reasons)


def test_a_rater_who_gives_every_unit_the_same_score_cannot_pass_the_gate():
    """Perfect agreement on a constant is not evidence that two raters mean the same thing."""
    flat = [
        label_record(f"u{index}", rater, presence=True, fidelity=2, quality=2, item_id=f"item{index}")
        for index in range(30)
        for rater in ("a", "b")
    ]
    gate = agreement.reliability_gate(flat, min_pairs=10)[CONSTRUCT]

    assert gate.fields["fidelity"].exact == pytest.approx(1.0)
    assert gate.fields["fidelity"].degenerate
    assert gate.fields["fidelity"].verdict(10) == "underpowered"
    assert gate.verdict == "underpowered" and not gate.usable
    assert any("every rating identical" in reason for reason in gate.reasons)


def test_a_binary_field_nobody_varied_cannot_pass_the_gate_either():
    """Two raters call the construct absent everywhere. AC1 says 1.00; nothing has been shown."""
    absent = [
        label_record(f"u{index}", rater, presence=False, quality=2, item_id=f"item{index}")
        for index in range(30)
        for rater in ("a", "b")
    ]
    gate = agreement.reliability_gate(absent, min_pairs=10)[CONSTRUCT]
    presence = gate.fields["presence"]

    assert presence.exact == pytest.approx(1.0)
    assert presence.ac1 == pytest.approx(1.0), "still reported - it is why the field is undefined, not a pass"
    assert presence.cohen != presence.cohen, "Cohen's kappa is 0/0 here"
    assert presence.value != presence.value, "so the headline is undefined too, rather than borrowing AC1"
    assert presence.degenerate and presence.verdict(10) == "underpowered"
    assert gate.verdict == "underpowered" and not gate.usable
    assert any("agreement on a constant proves nothing" in reason for reason in gate.reasons)


def test_a_construct_almost_nobody_found_is_underpowered_rather_than_reliable():
    """Three positives in forty. AC1 is carried by the thirty-seven agreed negatives."""
    records = [
        label_record(
            f"u{index}",
            rater,
            presence=index < 3,
            fidelity=index if index < 3 else None,
            quality=2,
            item_id=f"item{index}",
        )
        for index in range(40)
        for rater in ("a", "b")
    ]
    gate = agreement.reliability_gate(records, min_pairs=10, min_positive=8)[CONSTRUCT]
    presence = gate.fields["presence"]

    assert (presence.n_pairs, presence.n_positive, presence.n_negative) == (40, 3, 37)
    assert presence.ac1 == pytest.approx(1.0) and presence.headline == "ac1"
    assert presence.verdict(10, 8) == "underpowered"
    assert presence.verdict(10, 3) == "pass", "the floor is the judgement; the statistic never moved"
    assert gate.verdict == "underpowered" and not gate.usable
    assert any("called positive" in reason for reason in gate.reasons)


def test_unanimous_applicability_is_vacuous_rather_than_unmeasured():
    """Most constructs apply to every unit, and that must not read as a construct to rewrite."""
    records = [
        label_record(
            f"u{index}",
            rater,
            presence=index % 2 == 0,
            fidelity=(index // 2) % 4 if index % 2 == 0 else None,
            quality=2,
            item_id=f"item{index}",
        )
        for index in range(40)
        for rater in ("a", "b")
    ]
    gate = agreement.reliability_gate(records, min_pairs=10)[CONSTRUCT]
    applicable = gate.fields["applicable"]

    assert applicable.n_negative == 0 and applicable.degenerate
    assert gate.verdict == "pass", "the construct is gated on presence and fidelity, and both varied"
    assert any("no rater ever declined" in reason for reason in gate.reasons)


def test_quality_is_advisory_and_does_not_sink_a_construct():
    records = []
    for index in range(30):
        presence = index % 2 == 0
        fidelity = (index // 2) % 4 if presence else None
        for rater, quality in (("a", 1), ("b", 3)):
            records.append(
                label_record(
                    f"u{index}", rater, presence=presence, fidelity=fidelity, quality=quality, item_id=f"item{index}"
                )
            )

    gate = agreement.reliability_gate(records, min_pairs=10)[CONSTRUCT]
    assert gate.fields["presence"].verdict(10) == "pass"
    assert gate.fields["fidelity"].verdict(10) == "pass"
    assert gate.fields["quality"].verdict(10) == "fail", "the raters are a full scale apart on quality"
    assert gate.verdict == "pass", "quality is an all-things-considered judgement, not part of the definition"


def test_unreliable_fidelity_keeps_reliable_binary_presence():
    records = []
    for index in range(30):
        presence = index % 2 == 0
        value = (index // 2) % 4
        records.extend(
            [
                label_record(
                    f"u{index}",
                    "a",
                    presence=presence,
                    fidelity=value if presence else None,
                    item_id=f"item{index}",
                ),
                label_record(
                    f"u{index}",
                    "b",
                    presence=presence,
                    fidelity=3 - value if presence else None,
                    item_id=f"item{index}",
                ),
            ]
        )

    gate = agreement.reliability_gate(records, min_pairs=10)[CONSTRUCT]
    assert gate.fields["presence"].verdict(10) == "pass"
    assert gate.fields["fidelity"].value < agreement.MIN_FIDELITY_KAPPA
    assert gate.verdict == "pass"
    assert gate.usable and gate.measurement_mode == "binary_only"
    assert any("retain presence and discard the ordinal" in reason for reason in gate.reasons)


def test_quarantined_units_never_reach_the_gate():
    clean = binary_records("a", [1, 0] * 10) + binary_records("b", [1, 0] * 10)
    tainted = [{**record, "quarantined": True} for record in clean]

    assert agreement.reliability_gate(clean, min_pairs=5)[CONSTRUCT].n_units == 20
    with pytest.raises(KeyError):
        _ = agreement.reliability_gate(tainted, min_pairs=5)[CONSTRUCT]
    included = agreement.reliability_gate(tainted, min_pairs=5, exclude_quarantined=False)
    assert included[CONSTRUCT].n_units == 20


def test_two_raters_on_the_same_unit_is_a_bug_not_a_retest():
    with pytest.raises(ValueError, match="use test_retest"):
        agreement.index_labels([label_record("u1", "a"), label_record("u1", "a", fidelity=1)])


def test_test_retest_pairs_a_rater_with_itself_only():
    first = binary_records("judge", [1, 0] * 5)
    identical = agreement.test_retest(first, list(first))
    assert identical[CONSTRUCT]["presence"].exact == pytest.approx(1.0)
    assert identical[CONSTRUCT]["presence"].n_pairs == 10

    # The same rater, one point higher on every unit it scored: presence held, fidelity drifted.
    drifted = [
        {**record, "fidelity": None if record["fidelity"] is None else record["fidelity"] + 1} for record in first
    ]
    retest = agreement.test_retest(first, drifted)
    assert retest[CONSTRUCT]["fidelity"].exact == pytest.approx(0.0)
    assert retest[CONSTRUCT]["fidelity"].within1 == pytest.approx(1.0)
    assert retest[CONSTRUCT]["presence"].exact == pytest.approx(1.0)

    with pytest.raises(ValueError, match="test-retest is unmeasurable"):
        agreement.test_retest(first, binary_records("someone_else", [1, 0] * 5))


# -------------------------------------------------------------------- consensus


def test_consensus_follows_the_applicability_contract():
    records = [
        label_record("u1", "a", presence=True, fidelity=3, quality=2),
        label_record("u1", "b", presence=True, fidelity=1, quality=3),
        label_record("u1", "c", presence=False, quality=1),
        label_record("u2", "a", applicable=False),
        label_record("u2", "b", applicable=False),
        label_record("u2", "c", presence=True, fidelity=2, quality=2),
    ]
    rows = {row["unit_id"]: row for row in agreement.consensus_labels(records)}

    present = rows["u1"]
    assert present["applicable"] is True and present["presence"] is True
    assert present["fidelity"] == 2, "median of the raters who saw it, kept on the rubric's scale"
    assert present["quality"] == 2
    assert present["unresolved"] == []

    declined = rows["u2"]
    assert declined["applicable"] is False
    assert declined["presence"] is None and declined["fidelity"] is None and declined["quality"] is None


def test_a_tie_is_reported_rather_than_broken():
    tied = [
        label_record("u1", "a", presence=True, fidelity=2),
        label_record("u1", "b", presence=False),
        label_record("u1", "c", presence=True, fidelity=2),
        label_record("u1", "d", presence=False),
    ]
    row = agreement.consensus_labels(tied)[0]
    assert row["presence"] is None
    assert row["unresolved"] == ["presence"]
    assert row["applicable"] is True, "they agreed it applies; only presence is unresolved"

    assert agreement.majority([1, 1, 0, 0]) is None
    assert agreement.majority([1, 1, 0]) == 1
    assert agreement.consensus_ordinal([1, 2]) == 2, "half-up, so adjacent categories are treated alike"
    assert agreement.consensus_ordinal([0, 1, 2, 3]) == 2
    assert agreement.consensus_ordinal([]) is None


def test_consensus_needs_a_second_rater():
    assert agreement.consensus_labels([label_record("u1", "a")]) == []
    assert len(agreement.consensus_labels([label_record("u1", "a")], min_raters=1)) == 1


def test_consensus_carries_the_quarantine_flag_and_honours_the_method():
    """Collapsing raters must not launder a few-shot unit into a clean-looking record."""
    flagged = label_record("u1", "c", presence=True, fidelity=3, quality=3, quarantined=True)
    tainted = [
        label_record("u1", "a", presence=True, fidelity=0, quality=1, quarantined=True),
        label_record("u1", "b", presence=True, fidelity=0, quality=1, quarantined=True),
        {**flagged, "flags": ["rubric_misfit"]},
    ]
    row = agreement.consensus_labels(tainted, exclude_quarantined=False)[0]

    assert row["quarantined"] is True, "ppi.py excludes few-shot units by default and needs the flag to do it"
    assert row["flags"] == ["rubric_misfit"], "a flag any rater raised survives the collapse"
    assert ppi.records_to_columns([row], "presence", construct=CONSTRUCT)[0] == []

    mean = agreement.consensus_labels(tainted, exclude_quarantined=False, method="mean")[0]
    assert (row["fidelity"], row["quality"]) == (0, 1), "medians of 0,0,3 and 1,1,3"
    assert (mean["fidelity"], mean["quality"]) == (1, 2), "the method reaches quality too, not only fidelity"


# -------------------------------------------------------------------------- PPI


def ppi_toy(n_items: int = 20, variants: int = 2, offset: float = 1.0):
    """A corpus whose agent proxy is wrong by exactly ``offset`` on every unit."""
    unit_ids = [f"u{item}-{variant}" for item in range(n_items) for variant in range(variants)]
    item_ids = [f"item{item}" for item in range(n_items) for _ in range(variants)]
    truth = [float(index % 4) for index in range(len(unit_ids))]
    proxy = [value - offset for value in truth]
    return unit_ids, item_ids, truth, proxy


def test_rectifier_removes_a_constant_proxy_bias_exactly():
    unit_ids, item_ids, truth, proxy = ppi_toy()
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=6, seed=1)
    gold = {unit: truth[unit_ids.index(unit)] for unit in design.unit_ids}

    estimate = ppi.ppi_mean(unit_ids, item_ids, proxy, gold, design.sampling_prob, n_boot=500, seed=0)
    true_mean = sum(truth) / len(truth)

    assert estimate.proxy_mean == pytest.approx(true_mean - 1.0), "the naive number is biased by the offset"
    assert estimate.theta == pytest.approx(true_mean), "and the rectifier removes exactly that"
    assert estimate.rectifier == pytest.approx(1.0)
    assert estimate.ci_low <= estimate.theta <= estimate.ci_high
    assert estimate.n_gold == len(gold) and estimate.n_gold_clusters == 6


def test_a_useless_proxy_degenerates_to_the_gold_only_estimator():
    """PPI is unbiased for any proxy; a constant one simply buys nothing."""
    unit_ids, item_ids, truth, _ = ppi_toy()
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=6, seed=2)
    gold = {unit: truth[unit_ids.index(unit)] for unit in design.unit_ids}

    estimate = ppi.ppi_mean(unit_ids, item_ids, [7.0] * len(unit_ids), gold, design.sampling_prob, n_boot=400, seed=0)
    assert estimate.theta == pytest.approx(estimate.gold_mean)
    assert estimate.width == pytest.approx(estimate.gold_width)
    assert estimate.gain == pytest.approx(1.0)


def test_an_informative_proxy_narrows_the_interval():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=30)
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=9, seed=3)
    gold = {unit: truth[unit_ids.index(unit)] for unit in design.unit_ids}

    helped = ppi.ppi_mean(unit_ids, item_ids, proxy, gold, design.sampling_prob, n_boot=1000, seed=0)
    assert helped.gain > 1.5
    assert helped.width < helped.gold_width


def test_bootstrap_resamples_items_so_variants_are_not_independent_evidence():
    """Three copies of a unit are one item's worth of evidence, not three."""
    values = [0.0, 1.0, 2.0, 3.0] * 2
    clustered_units = [f"u{i}" for i in range(24)]
    clustered_items = [f"item{i // 3}" for i in range(24)]
    spread_units = [f"v{i}" for i in range(24)]
    spread_items = [f"item{i}" for i in range(24)]
    proxy = [values[index // 3] for index in range(24)]

    clustered = ppi.ppi_mean(
        clustered_units,
        clustered_items,
        proxy,
        {unit: proxy[index] + 1 for index, unit in enumerate(clustered_units)},
        1.0,
        n_boot=3000,
        seed=0,
    )
    spread = ppi.ppi_mean(
        spread_units,
        spread_items,
        proxy,
        {unit: proxy[index] + 1 for index, unit in enumerate(spread_units)},
        1.0,
        n_boot=3000,
        seed=0,
    )

    assert clustered.theta == pytest.approx(spread.theta), "the same data, so the same estimate"
    assert clustered.n_clusters == 8 and spread.n_clusters == 24
    assert clustered.width > 1.4 * spread.width, "ignoring clusters would understate the width by about sqrt(3)"
    assert clustered.few_clusters and not spread.few_clusters


def test_identical_items_give_a_zero_width_interval_rather_than_an_infinite_gain():
    """Every item internally identical, so there is nothing between clusters to resample."""
    unit_ids = [f"u{index}" for index in range(20)]
    item_ids = [f"item{index // 2}" for index in range(20)]
    proxy = [1.0] * 20
    gold = {unit_ids[index]: 0.5 for index in range(8)}

    estimate = ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 0.4, n_boot=300, seed=0)
    assert estimate.theta == pytest.approx(0.5)
    assert estimate.width == 0.0 and estimate.gold_width == 0.0
    assert estimate.gain == pytest.approx(1.0), "no width either way is not a free infinite gain"
    assert "inf" not in estimate.summary()


def test_accuracy_correction_reports_being_right_not_resembling_the_judge():
    """The agent agrees with the system everywhere; humans disagree on a quarter of the slice."""
    unit_ids = [f"u{index}" for index in range(40)]
    item_ids = [f"item{index // 2}" for index in range(40)]
    system = [index % 4 != 3 for index in range(40)]
    agent = list(system)  # a judge that rubber-stamps the system
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=8, seed=7)
    human = {unit: (unit_ids.index(unit) % 4 != 3) and (unit_ids.index(unit) % 8 != 0) for unit in design.unit_ids}

    estimate = ppi.ppi_accuracy(unit_ids, item_ids, system, agent, human, design.sampling_prob, n_boot=500, seed=0)
    assert estimate.proxy_mean == pytest.approx(1.0), "agreement with the judge alone claims perfection"
    assert estimate.theta < 1.0, "and the human slice takes it back down"
    assert estimate.theta == pytest.approx(1.0 + estimate.rectifier)


def test_intervals_are_reproducible_from_the_seed():
    unit_ids, item_ids, truth, proxy = ppi_toy()
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=8, seed=4)
    gold = {unit: truth[unit_ids.index(unit)] for unit in design.unit_ids}
    args = (unit_ids, item_ids, proxy, gold, design.sampling_prob)

    first = ppi.ppi_mean(*args, n_boot=300, seed=13)
    again = ppi.ppi_mean(*args, n_boot=300, seed=13)
    different = ppi.ppi_mean(*args, n_boot=300, seed=14)

    assert (first.ci_low, first.ci_high, first.se) == (again.ci_low, again.ci_high, again.se)
    assert first.theta == again.theta == different.theta, "the estimate does not depend on the bootstrap seed"
    assert first.se != different.se, "the interval is resampled, so a new seed moves it"
    assert ppi.draw_gold_slice(unit_ids, item_ids, n_items=8, seed=4) == design


# ----------------------------------------------------------- PPI design failures


def test_invalid_sampling_probabilities_are_refused():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=10)
    gold = {unit_ids[index]: truth[index] for index in range(8)}  # items 0-3 of 10, so pi = 0.4

    ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 0.4, n_boot=200, seed=0)  # the valid design

    with pytest.raises(ppi.PPIError, match=r"must be in \(0, 1\]"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 0.0, n_boot=200)
    with pytest.raises(ppi.PPIError, match=r"must be in \(0, 1\]"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 1.4, n_boot=200)
    with pytest.raises(ppi.PPIError, match="probability 1 but no gold label"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 1.0, n_boot=200)
    with pytest.raises(ppi.PPIError, match="do not describe the slice"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 0.95, n_boot=200)


def test_per_unit_probabilities_must_name_exactly_the_units():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=10)
    gold = {unit_ids[index]: truth[index] for index in range(8)}
    good = dict.fromkeys(unit_ids, 0.4)

    ppi.ppi_mean(unit_ids, item_ids, proxy, gold, good, n_boot=200, seed=0)

    with pytest.raises(ppi.PPIError, match="missing a probability"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, {unit_ids[0]: 0.2}, n_boot=200)
    with pytest.raises(ppi.PPIError, match="ids that are not units"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, {**good, "ghost": 0.2}, n_boot=200)


def test_gold_ids_must_be_units_and_must_exist():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=10)

    with pytest.raises(ppi.PPIError, match="gold ids that are not units"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, {"typo-u0": 1.0}, 0.2, n_boot=200)
    with pytest.raises(ppi.PPIError, match="gold slice is empty"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, {}, 0.2, n_boot=200)
    with pytest.raises(ppi.PPIError, match="must be a finite number"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, {unit_ids[0]: None, unit_ids[1]: 1.0}, 0.1, n_boot=200)


def test_a_corpus_that_cannot_be_clustered_or_aligned_is_refused():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=10)
    gold = {unit_ids[0]: truth[0], unit_ids[1]: truth[1]}

    with pytest.raises(ppi.PPIError, match="must align"):
        ppi.ppi_mean(unit_ids, item_ids[:-1], proxy, gold, 0.1, n_boot=200)
    with pytest.raises(ppi.PPIError, match="duplicate unit id"):
        ppi.ppi_mean([unit_ids[0]] * len(unit_ids), item_ids, proxy, gold, 0.1, n_boot=200)
    with pytest.raises(ppi.PPIError, match="needs an item_id"):
        ppi.ppi_mean(unit_ids, [None] * len(unit_ids), proxy, gold, 0.1, n_boot=200)
    with pytest.raises(ppi.PPIError, match="proxy label missing"):
        ppi.ppi_mean(unit_ids, item_ids, [None] + proxy[1:], gold, 0.1, n_boot=200)


def test_a_gold_slice_too_thin_to_bootstrap_fails_loudly():
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=40, variants=1)
    gold = {unit_ids[0]: truth[0]}
    with pytest.raises(ppi.PPIError, match="too few to bootstrap"):
        ppi.ppi_mean(unit_ids, item_ids, proxy, gold, 1 / 40, n_boot=200, seed=0)


def test_draw_gold_slice_takes_whole_items_at_a_known_probability():
    unit_ids, item_ids, _, _ = ppi_toy(n_items=10, variants=3)
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=4, seed=0)

    assert design.sampling_prob == pytest.approx(0.4)
    assert len(design.item_ids) == 4
    assert len(design.unit_ids) == 12, "whole items, so every variant comes with it"
    chosen = set(design.item_ids)
    assert {item for unit, item in zip(unit_ids, item_ids) if unit in set(design.unit_ids)} == chosen
    assert ppi.draw_gold_slice(unit_ids, item_ids, n_items=4, seed=0).unit_ids == design.unit_ids
    assert ppi.draw_gold_slice(unit_ids, item_ids, n_items=4, seed=1).unit_ids != design.unit_ids

    with pytest.raises(ppi.PPIError, match="must be between 1 and the 10 items"):
        ppi.draw_gold_slice(unit_ids, item_ids, n_items=11)
    with pytest.raises(ppi.PPIError, match="duplicate unit id"):
        ppi.draw_gold_slice([unit_ids[0]] * len(unit_ids), item_ids, n_items=4)


def test_the_summary_says_which_way_the_proxy_moved_the_interval():
    """A proxy uncorrelated with the human widens the interval, and 0.47x must not read as a win."""
    unit_ids, item_ids, truth, proxy = ppi_toy(n_items=24)
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=8, seed=2)
    gold = {unit: truth[unit_ids.index(unit)] for unit in design.unit_ids}

    helped = ppi.ppi_mean(unit_ids, item_ids, proxy, gold, design.sampling_prob, n_boot=800, seed=0)
    assert helped.gain > 1.0
    assert "narrowed the interval" in helped.summary()

    # A proxy that is pure noise against the human: unbiased still, but it pays for it.
    noise = [3.0 * ((index * 7) % 4 == 0) for index in range(len(unit_ids))]
    hurt = ppi.ppi_mean(unit_ids, item_ids, noise, gold, design.sampling_prob, n_boot=800, seed=0)
    assert hurt.gain < 1.0
    assert "WIDENED the interval" in hurt.summary()
    assert "narrowed" not in hurt.summary()


def test_null_fields_need_an_explicit_meaning_before_they_can_be_averaged():
    records = [
        label_record("u1", "agent", presence=True, fidelity=3, item_id="item1"),
        label_record("u2", "agent", presence=False, item_id="item2"),
    ]
    with pytest.raises(ppi.PPIError, match="Pass null_as"):
        ppi.records_to_columns(records, "fidelity", construct=CONSTRUCT)

    unit_ids, item_ids, values = ppi.records_to_columns(records, "fidelity", construct=CONSTRUCT, null_as=0.0)
    assert values == [3.0, 0.0]
    assert (unit_ids, item_ids) == (["u1", "u2"], ["item1", "item2"])

    tainted = [{**records[0], "quarantined": True}, records[1]]
    assert ppi.records_to_columns(tainted, "presence", construct=CONSTRUCT)[0] == ["u2"]

    with pytest.raises(ppi.PPIError, match="no records for construct 'generativ_elicitation'"):
        ppi.records_to_columns(records, "presence", construct="generativ_elicitation")


def test_a_gold_file_holding_two_raters_is_refused_rather_than_silently_halved():
    """``dict(zip(...))`` keeps whichever rater came last and reports it as the panel."""
    two_humans = [
        label_record("u1", "sophia", presence=True, item_id="item1"),
        label_record("u1", "alex", presence=False, item_id="item1"),
    ]
    unit_ids, _, values = ppi.records_to_columns(two_humans, "presence", construct=CONSTRUCT)
    assert (unit_ids, values) == (["u1", "u1"], [1.0, 0.0])

    with pytest.raises(ppi.PPIError, match="appear more than once"):
        ppi.by_unit(unit_ids, values, what="gold labels")
    assert ppi.by_unit(["u1", "u2"], [1.0, 0.0]) == {"u1": 1.0, "u2": 0.0}


def test_a_bootstrap_that_cannot_form_an_interval_says_so():
    with pytest.raises(ppi.PPIError, match="n_boot must be at least 2"):
        ppi.ppi_mean(["u1"], ["item1"], [1.0], {"u1": 1.0}, 1.0, n_boot=1)
    with pytest.raises(ppi.PPIError, match=r"alpha must be in \(0, 1\)"):
        ppi.ppi_mean(["u1"], ["item1"], [1.0], {"u1": 1.0}, 1.0, n_boot=100, alpha=0.0)


def test_records_flow_from_ingestion_into_the_gate_and_ppi():
    """The three modules share one record shape; this is the seam that has to hold."""
    units = make_units(n_items=12, variants=1)
    packet = build_packet(units, SPEC, seed=8)
    key = packet.key()

    def replies() -> list[dict[str, object]]:
        return [
            {
                "slot_id": slot.slot_id,
                "text": json.dumps(
                    {
                        "labels": {
                            letter: {
                                "applicable": True,
                                "presence": True,
                                "fidelity": index % 4,
                                "quality": 2,
                                "span": "candidate",
                            }
                            for letter, _, _ in slot.options
                        }
                    }
                ),
            }
            for index, slot in enumerate(packet.slots)
        ]

    first = ingest_responses(key, replies(), rater="judge_a")
    second = ingest_responses(key, replies(), rater="judge_b")
    assert not first.failures and first.failure_rate == pytest.approx(0.0)

    gate = agreement.reliability_gate(first.records + second.records, min_pairs=10)[CONSTRUCT]
    assert gate.verdict == "underpowered", "perfect agreement on an all-positive field contains no negatives"
    assert gate.not_applicable_rate == pytest.approx(0.0)

    consensus = agreement.consensus_labels(first.records + second.records)
    assert len(consensus) == 12
    assert all(row["presence"] is True and row["unresolved"] == [] for row in consensus)

    # The judge is one point below the humans on every unit, so the rectifier is exactly +1.
    unit_ids, item_ids, proxy = ppi.records_to_columns(first.records, "fidelity", construct=CONSTRUCT)
    design = ppi.draw_gold_slice(unit_ids, item_ids, n_items=4, seed=0)
    by_id = dict(zip(unit_ids, proxy))
    estimate = ppi.ppi_mean(
        unit_ids,
        item_ids,
        proxy,
        {unit: by_id[unit] + 1.0 for unit in design.unit_ids},
        design.sampling_prob,
        n_boot=300,
        seed=0,
    )
    assert estimate.rectifier == pytest.approx(1.0)
    assert estimate.theta == pytest.approx(sum(proxy) / len(proxy) + 1.0)
