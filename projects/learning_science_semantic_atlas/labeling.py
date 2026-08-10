"""Per-construct judge packets, blinded, and a strict reader for what comes back.

    # corpus -> one packet per construct, plus the key that unblinds it
    python -m projects.learning_science_semantic_atlas.labeling prepare \
        --corpus data/corpus.jsonl --rubric data/rubric.json \
        --exemplars data/human_shots.jsonl --out-dir data/packets --seed 7

    # a saved judge transcript -> label records, offline
    python -m projects.learning_science_semantic_atlas.labeling ingest \
        --key data/packets/generative_elicitation-seed7.key.json \
        --responses data/raw/judge_gpt.jsonl --rater gpt --out data/labels/gpt.jsonl

THIS MODULE OPENS NO SOCKETS. It turns a corpus into files, and files back into
labels. Whoever runs the judge does so separately and saves the raw text, which
is the arrangement that makes a labelling round auditable: the exact bytes the
judge saw are on disk, and so is the exact reply it gave.

ONE CONSTRUCT PER PACKET. A packet carries one construct's question and one
construct's anchors, because a judge asked about six things in one context
couples them: its answer to the first sits in the context while it answers the
second, and a strong opinion on one bleeds into the rest. The predecessor
project measured this - two dimensions rated in one pass correlated at 0.95, and
splitting the passes was the only thing that separated them. Focused packets
cost more calls and buy independent judgements.

BLINDING IS STRUCTURAL, NOT POLITE. ``Packet.judge_view()`` is built from an
allowlist of context fields and contains no unit ids, no item ids and no
provenance - not because the caller remembers to strip them, but because the
view is assembled from scratch and nothing else can reach it. Options are
lettered after a shuffle seeded by ``(seed, construct, item_id)``, so the order
is reproducible without being informative, and a model whose output always
landed in position one cannot be recognised by position. The mapping back to
unit ids lives only in the key file, which is named after the packet so that a
second seed cannot overwrite the key that unblinds the first. Send the packet,
keep the key.

An item's context is rendered once, above its options, so it has to be identical
across that item's variants. A context field that differed between them would
show one variant's situation to every option and mark out which option it came
from, so ``build_packet`` refuses the corpus rather than picking a version.

QUARANTINE PROPAGATES BY ITEM. Few-shot exemplars are excluded from evaluation
along with every other unit sharing their ``item_id``. Variants of one item share
a stem, a reference solution and often a student turn, so showing the judge the
answer for one variant tells it most of the answer for its siblings; excluding
only the exemplar itself would leave a leak that looks like agreement. The
excluded ids are written into the key so the analysis can prove which units were
never contaminated rather than assert it.

THE LABEL CONTRACT, which the parser enforces and the prompt is generated from,
so the two cannot drift:

    applicable  JSON boolean. Does this construct apply to this unit at all?
                A judge that must answer anyway invents a reading; letting it
                decline turns that into a measurable flag rate instead.
    presence    JSON boolean, null iff not applicable. Is the construct there?
    fidelity    integer 0-3, null unless presence is true. How faithfully the
                construct is realised as the rubric defines it - absent moves
                have no fidelity to score.
    quality     integer 1-3, null iff not applicable. Instructional quality of
                the move, which is a separate question: a turn can be a textbook
                instance of a construct and still be a bad turn.
    span        short quote required when presence is true; otherwise null.

Strict means strict: one JSON object, every field typed, unknown fields
rejected, ``0``/``1`` refused where a boolean belongs. A parser that repairs
sloppy output hides a broken prompt, and the repair rate is exactly the signal
you want to see early.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

PACKET_SCHEMA = "semantic-atlas/judge-packet-v1"
KEY_SCHEMA = "semantic-atlas/packet-key-v1"
LABEL_SCHEMA = "semantic-atlas/labels-v1"

LETTERS = string.ascii_uppercase


@dataclass(frozen=True)
class Scale:
    """An ordinal field and its inclusive bounds."""

    name: str
    lo: int
    hi: int

    def describe(self) -> str:
        return f"integer {self.lo}-{self.hi}"


FIDELITY = Scale("fidelity", 0, 3)
QUALITY = Scale("quality", 1, 3)

BINARY_FIELDS: tuple[str, ...] = ("applicable", "presence")
ORDINAL_FIELDS: dict[str, Scale] = {"fidelity": FIDELITY, "quality": QUALITY}
LABEL_FIELDS: tuple[str, ...] = (*BINARY_FIELDS, *ORDINAL_FIELDS, "span")


class LabelParseError(ValueError):
    """A judge reply did not meet the contract. Carries what was wrong, verbatim."""


@dataclass(frozen=True)
class ConstructSpec:
    """One construct as the judge sees it. The rubric of record, kept as data.

    This is deliberately not imported from a construct registry. The packet is
    the artifact an eventual reader has to trust, so the wording that produced it
    lives in a file next to the labels rather than in code that has since moved
    on. Adapting a registry is ``specs_from_objects``.
    """

    key: str
    question: str
    positive_anchor: str
    negative_anchor: str
    applicability_note: str = ""

    def as_prompt_block(self) -> dict[str, str]:
        block = {
            "construct": self.key,
            "question": self.question,
            "counts_as_present": self.positive_anchor,
            "does_not_count": self.negative_anchor,
        }
        if self.applicability_note:
            block["not_applicable_when"] = self.applicability_note
        return block


@dataclass(frozen=True)
class CorpusFields:
    """Which corpus columns to read. Defaults name the columns; nothing is guessed.

    Held as data so this module does not have to agree with the corpus builder
    about spelling, and so a schema change is a call-site edit rather than a
    rewrite.
    """

    unit_id: str = "unit_id"
    item_id: str = "item_id"
    construct: str = "concept"
    option_text: str = "candidate_action"
    context: tuple[str, ...] = ("question", "reference", "student_before")


DEFAULT_FIELDS = CorpusFields()


@dataclass(frozen=True)
class Unit:
    """One candidate action, which is one thing a judge can label."""

    unit_id: str
    item_id: str
    construct: str
    text: str
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Exemplar:
    """A unit with a known label, used as few-shot and thereby disqualified as evidence."""

    unit: Unit
    label: Label


@dataclass(frozen=True)
class Label:
    """A parsed judge label. Construct one only through ``parse_label``."""

    applicable: bool
    presence: bool | None = None
    fidelity: int | None = None
    quality: int | None = None
    span: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "applicable": self.applicable,
            "presence": self.presence,
            "fidelity": self.fidelity,
            "quality": self.quality,
            "span": self.span,
        }


@dataclass(frozen=True)
class Slot:
    """One item's options, lettered. ``options`` is (letter, unit_id, text)."""

    slot_id: str
    item_id: str
    context: dict[str, str]
    options: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class Packet:
    packet_id: str
    construct: str
    seed: int
    spec: ConstructSpec
    slots: tuple[Slot, ...]
    shots: tuple[Exemplar, ...] = ()
    quarantined_item_ids: tuple[str, ...] = ()

    @property
    def shot_unit_ids(self) -> tuple[str, ...]:
        return tuple(shot.unit.unit_id for shot in self.shots)

    def letters(self, slot_id: str) -> tuple[str, ...]:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return tuple(letter for letter, _, _ in slot.options)
        raise KeyError(f"no slot {slot_id!r} in packet {self.packet_id!r}")

    def judge_view(self) -> dict[str, object]:
        """Exactly what may be shown to a judge. Assembled, not filtered."""
        return {
            "schema": PACKET_SCHEMA,
            "packet_id": self.packet_id,
            "task": self.spec.as_prompt_block(),
            "scales": {name: [scale.lo, scale.hi] for name, scale in ORDINAL_FIELDS.items()},
            "instructions": instructions(self.spec),
            "response_schema": response_schema(),
            "examples": [
                {
                    "context": shot.unit.context,
                    "options": {"A": shot.unit.text},
                    "answer": {"labels": {"A": shot.label.as_dict()}},
                }
                for shot in self.shots
            ],
            "items": [
                {
                    "slot_id": slot.slot_id,
                    "context": slot.context,
                    "options": {letter: text for letter, _, text in slot.options},
                }
                for slot in self.slots
            ],
        }

    def key(self) -> dict[str, object]:
        """The only route from a letter back to a unit."""
        return {
            "schema": KEY_SCHEMA,
            "packet_id": self.packet_id,
            "construct": self.construct,
            "seed": self.seed,
            "shot_unit_ids": list(self.shot_unit_ids),
            "quarantined_item_ids": list(self.quarantined_item_ids),
            "slots": {
                slot.slot_id: {
                    "item_id": slot.item_id,
                    "options": {letter: unit_id for letter, unit_id, _ in slot.options},
                }
                for slot in self.slots
            },
        }


@dataclass(frozen=True)
class IngestResult:
    """Label records plus the replies that did not parse. Read the failures."""

    records: tuple[dict[str, object], ...]
    failures: tuple[dict[str, str], ...]
    n_replies: int

    @property
    def failure_rate(self) -> float:
        """Share of replies that did not parse.

        Per reply, not per label: a reply covers every option in its slot, so
        dividing failures by records would divide a count of slots by a count of
        options and report a fraction of the true rate - the compliance problem
        looks smallest exactly where packets are widest.
        """
        return len(self.failures) / self.n_replies if self.n_replies else float("nan")


# ---------------------------------------------------------------- prompt text


def instructions(spec: ConstructSpec) -> str:
    """The judge-facing instruction block, generated from the label contract."""
    return (
        f"Label every lettered option below for ONE construct: {spec.key}.\n"
        f"{spec.question}\n\n"
        f"Counts as present: {spec.positive_anchor}\n"
        f"Does not count: {spec.negative_anchor}\n\n"
        "The options are presented in a randomised order and their sources are hidden. "
        "Judge each option on its own; do not rank them against each other.\n\n"
        "Reply with a single JSON object and nothing else - no prose, no markdown fence. "
        'It must have exactly one key, "labels", mapping every option letter to an object with exactly '
        f"these fields: {', '.join(LABEL_FIELDS)}.\n"
        "  applicable: true or false. False when this construct does not apply to the option at all.\n"
        "  presence:   true or false. Must be null when applicable is false.\n"
        f"  fidelity:   {FIDELITY.describe()}, how faithfully the construct is realised. "
        "Must be null unless presence is true.\n"
        f"  quality:    {QUALITY.describe()}, instructional quality of the move regardless of the construct. "
        "Must be null when applicable is false.\n"
        "  span:       a short quote from the option when presence is true; otherwise null.\n"
        "Use JSON true/false, not 0/1, and JSON null for a field that does not apply."
    )


def response_schema() -> dict[str, object]:
    return {
        "labels": {
            "<option letter>": {
                "applicable": "true or false",
                "presence": "true or false, null if applicable is false",
                "fidelity": f"{FIDELITY.describe()}, null unless presence is true",
                "quality": f"{QUALITY.describe()}, null if applicable is false",
                "span": "short quote required when presence is true; otherwise null",
            }
        }
    }


# --------------------------------------------------------------------- corpus


def _flat(value: object) -> str:
    return " ".join(str(value).split())


def read_jsonl(path: str) -> Iterator[dict[str, object]]:
    """Rows of a JSONL file, with the line number in any error."""
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")
            yield row


def _require(row: Mapping[str, object], name: str, where: str) -> str:
    value = row.get(name)
    if value is None or _flat(value) == "":
        raise ValueError(f"{where}: missing required field {name!r}; row has {sorted(row)}")
    return _flat(value)


def unit_from_row(row: Mapping[str, object], fields: CorpusFields = DEFAULT_FIELDS, where: str = "row") -> Unit:
    return Unit(
        unit_id=_require(row, fields.unit_id, where),
        item_id=_require(row, fields.item_id, where),
        construct=_require(row, fields.construct, where),
        text=_require(row, fields.option_text, where),
        context={name: _flat(row[name]) for name in fields.context if row.get(name) not in (None, "")},
    )


def read_corpus(path: str, fields: CorpusFields = DEFAULT_FIELDS) -> list[Unit]:
    """Units from a corpus JSONL. Duplicate unit ids are a fatal error, not a warning."""
    units: list[Unit] = []
    seen: dict[str, int] = {}
    for number, row in enumerate(read_jsonl(path), start=1):
        unit = unit_from_row(row, fields, where=f"{path}:{number}")
        if unit.unit_id in seen:
            first = seen[unit.unit_id]
            raise ValueError(f"{path}:{number}: duplicate unit_id {unit.unit_id!r}, already on row {first}")
        seen[unit.unit_id] = number
        units.append(unit)
    return units


def read_exemplars(path: str, fields: CorpusFields = DEFAULT_FIELDS, label_key: str = "label") -> list[Exemplar]:
    """Human-labelled units for few-shot use, validated by the same strict parser."""
    out = []
    for number, row in enumerate(read_jsonl(path), start=1):
        where = f"{path}:{number}"
        if label_key not in row:
            raise ValueError(f"{where}: exemplar has no {label_key!r} field")
        out.append(Exemplar(unit_from_row(row, fields, where=where), parse_label(row[label_key], where=where)))
    return out


# ------------------------------------------------------------------- blinding


def _rng(seed: int, *parts: str) -> random.Random:
    """A generator keyed by content, so shuffles reproduce across processes.

    ``hash()`` on strings is salted per interpreter, which would make the option
    order irreproducible from the recorded seed alone.
    """
    payload = "\x1f".join([str(seed), *parts]).encode()
    return random.Random(int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big"))


def _slot_id(packet_id: str, item_id: str) -> str:
    payload = f"{packet_id}\x1f{item_id}".encode()
    return "s" + hashlib.blake2b(payload, digest_size=6).hexdigest()


def select_shots(pool: Sequence[Exemplar], construct: str, max_shots: int, seed: int = 0) -> list[Exemplar]:
    """Few-shot exemplars for one construct, balanced across the presence label.

    Drawing at random tends to return whatever the corpus is mostly made of,
    which teaches the judge the base rate rather than the distinction. Taking
    from present and absent alternately shows the boundary instead. One exemplar
    per item, since a second variant of the same item spends a slot on
    information the first already gave.
    """
    if max_shots <= 0:
        return []
    eligible = [e for e in pool if e.unit.construct == construct and e.label.applicable]
    by_item: dict[str, Exemplar] = {}
    for exemplar in sorted(eligible, key=lambda e: e.unit.unit_id):
        by_item.setdefault(exemplar.unit.item_id, exemplar)
    rng = _rng(seed, "shots", construct)
    present = [e for e in by_item.values() if e.label.presence]
    absent = [e for e in by_item.values() if not e.label.presence]
    rng.shuffle(present)
    rng.shuffle(absent)

    chosen: list[Exemplar] = []
    for index in range(max_shots):
        first, second = (present, absent) if index % 2 == 0 else (absent, present)
        pick = first.pop() if first else (second.pop() if second else None)
        if pick is None:
            break
        chosen.append(pick)
    return chosen


def _shared_context(item_id: str, units: Sequence[Unit]) -> dict[str, str]:
    """The one context an item's options are judged against.

    A field one variant omits is taken from a sibling - the stem is the item's,
    not the option's. A field two variants spell differently is fatal: the slot
    renders a single context, so shipping it would show the wrong situation for
    every option but one, and the option it did fit would be identifiable by it.
    """
    context: dict[str, str] = {}
    source: dict[str, str] = {}
    for unit in units:
        for name, value in unit.context.items():
            if name not in context:
                context[name] = value
                source[name] = unit.unit_id
            elif context[name] != value:
                raise ValueError(
                    f"item {item_id!r}: context field {name!r} differs between units "
                    f"{source[name]!r} and {unit.unit_id!r}. Variants of one item are judged against "
                    "one shared context, so either they belong to different items or the corpus "
                    "builder is putting per-option text in a context field."
                )
    return context


def build_packet(
    units: Sequence[Unit],
    spec: ConstructSpec,
    *,
    shots: Sequence[Exemplar] = (),
    seed: int = 0,
    packet_id: str | None = None,
    extra_quarantine: Iterable[str] = (),
) -> Packet:
    """One construct's evaluation items, blinded, with the few-shot items removed.

    Every unit sharing an ``item_id`` with an exemplar is dropped, including
    exemplars for other constructs that happen to be passed in: contamination is
    a property of the item, not of the question asked about it.

    Raises when an item's variants disagree about a context field, because the
    slot can only show one context and choosing one silently would leak which
    option it belongs to.
    """
    packet_id = packet_id or f"{spec.key}-seed{seed}"
    quarantine = {shot.unit.item_id for shot in shots} | {_flat(i) for i in extra_quarantine}

    grouped: dict[str, list[Unit]] = {}
    for unit in units:
        if unit.construct != spec.key or unit.item_id in quarantine:
            continue
        grouped.setdefault(unit.item_id, []).append(unit)

    slots = []
    seen_slots: dict[str, str] = {}
    for item_id, members in grouped.items():
        if len(members) > len(LETTERS):
            raise ValueError(f"item {item_id!r} has {len(members)} options; only {len(LETTERS)} letters exist")
        order = sorted(members, key=lambda u: u.unit_id)  # a stable base the shuffle then destroys
        _rng(seed, spec.key, item_id).shuffle(order)
        slot_id = _slot_id(packet_id, item_id)
        if slot_id in seen_slots:
            raise ValueError(f"slot id collision between items {seen_slots[slot_id]!r} and {item_id!r}")
        seen_slots[slot_id] = item_id
        context = _shared_context(item_id, order)
        slots.append(
            Slot(
                slot_id=slot_id,
                item_id=item_id,
                context=context,
                options=tuple((LETTERS[i], unit.unit_id, unit.text) for i, unit in enumerate(order)),
            )
        )

    return Packet(
        packet_id=packet_id,
        construct=spec.key,
        seed=seed,
        spec=spec,
        slots=tuple(slots),
        shots=tuple(shot for shot in shots if shot.unit.construct == spec.key),
        quarantined_item_ids=tuple(sorted(quarantine)),
    )


SAFE_PACKET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def write_packet(packet: Packet, out_dir: str) -> tuple[str, str]:
    """Write the judge view and the key as separate files. Send one, keep the other.

    Named by ``packet_id``, which carries the seed. Naming them by construct
    alone means a second round overwrites the key for the first, and a key is the
    only route from a letter back to a unit: the labels for packets already sent
    would become permanently unreadable.
    """
    if not SAFE_PACKET_ID.match(packet.packet_id):
        raise ValueError(
            f"packet_id {packet.packet_id!r} is not usable as a filename; pass build_packet a packet_id "
            "of letters, digits, dot, dash and underscore"
        )
    os.makedirs(out_dir, exist_ok=True)
    packet_path = os.path.join(out_dir, f"{packet.packet_id}.packet.json")
    key_path = os.path.join(out_dir, f"{packet.packet_id}.key.json")
    with open(packet_path, "w") as handle:
        json.dump(packet.judge_view(), handle, indent=1, sort_keys=True)
    with open(key_path, "w") as handle:
        json.dump(packet.key(), handle, indent=1, sort_keys=True)
    return packet_path, key_path


def specs_from_objects(objects: Iterable[object]) -> dict[str, ConstructSpec]:
    """Adapt anything with ``key``/``question``-shaped attributes into specs.

    Here so that a construct registry elsewhere in the project can feed this
    module without this module importing it, and without a caller hand-copying
    rubric text into a second place where it can go stale.
    """
    specs = {}
    for obj in objects:
        key = getattr(obj, "key", None) or getattr(obj, "name", None)
        if not key:
            raise ValueError(f"{obj!r} has neither .key nor .name")
        question = getattr(obj, "question", "") or getattr(obj, "name", "") or str(key)
        specs[str(key)] = ConstructSpec(
            key=str(key),
            question=str(question),
            positive_anchor=str(getattr(obj, "positive_anchor", "")),
            negative_anchor=str(getattr(obj, "negative_anchor", "")),
            applicability_note=str(getattr(obj, "evidence_limit", "") or getattr(obj, "applicability_note", "")),
        )
    return specs


def read_rubric(path: str) -> dict[str, ConstructSpec]:
    """``{construct: {question, positive_anchor, negative_anchor, applicability_note}}``."""
    with open(path) as handle:
        blob = json.load(handle)
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: expected an object keyed by construct")
    specs = {}
    for key, body in blob.items():
        if not isinstance(body, dict):
            raise ValueError(f"{path}: construct {key!r} must map to an object")
        missing = [f for f in ("question", "positive_anchor", "negative_anchor") if not body.get(f)]
        if missing:
            raise ValueError(f"{path}: construct {key!r} is missing {missing}")
        specs[key] = ConstructSpec(
            key=key,
            question=str(body["question"]),
            positive_anchor=str(body["positive_anchor"]),
            negative_anchor=str(body["negative_anchor"]),
            applicability_note=str(body.get("applicability_note", "")),
        )
    return specs


# --------------------------------------------------------------------- parsing


def _require_bool(value: object, name: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise LabelParseError(f"{where}: {name} must be a JSON boolean, got {value!r}")
    return value


def _require_ordinal(value: object, scale: Scale, where: str) -> int:
    # bool is a subclass of int, so it has to be refused before the range check.
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelParseError(f"{where}: {scale.name} must be an {scale.describe()}, got {value!r}")
    if not scale.lo <= value <= scale.hi:
        raise LabelParseError(f"{where}: {scale.name} must be an {scale.describe()}, got {value!r}")
    return value


def parse_label(payload: object, where: str = "label") -> Label:
    """One option's label, or ``LabelParseError`` saying precisely what was wrong."""
    if not isinstance(payload, Mapping):
        raise LabelParseError(f"{where}: expected an object, got {type(payload).__name__}")
    unknown = sorted(set(payload) - set(LABEL_FIELDS))
    if unknown:
        raise LabelParseError(f"{where}: unknown field(s) {unknown}; allowed are {list(LABEL_FIELDS)}")
    if "applicable" not in payload:
        raise LabelParseError(f"{where}: missing required field 'applicable'")

    applicable = _require_bool(payload["applicable"], "applicable", where)
    span = payload.get("span")
    if span is not None and not isinstance(span, str):
        raise LabelParseError(f"{where}: span must be a string or null, got {span!r}")

    if not applicable:
        for name in ("presence", "fidelity", "quality"):
            if payload.get(name) is not None:
                raise LabelParseError(f"{where}: {name} must be null when applicable is false, got {payload[name]!r}")
        if span is not None:
            raise LabelParseError(f"{where}: span must be null when applicable is false, got {span!r}")
        return Label(applicable=False)

    if payload.get("presence") is None:
        raise LabelParseError(f"{where}: presence is required when applicable is true")
    presence = _require_bool(payload["presence"], "presence", where)
    quality = _require_ordinal(payload.get("quality"), QUALITY, where)

    fidelity = payload.get("fidelity")
    if presence:
        fidelity = _require_ordinal(fidelity, FIDELITY, where)
        if not (span or "").strip():
            raise LabelParseError(f"{where}: span must be a non-empty quote when presence is true")
    elif fidelity is not None:
        raise LabelParseError(f"{where}: fidelity must be null unless presence is true, got {fidelity!r}")
    elif span is not None:
        raise LabelParseError(f"{where}: span must be null unless presence is true, got {span!r}")
    return Label(applicable=True, presence=presence, fidelity=fidelity, quality=quality, span=span)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return body.rsplit("```", 1)[0].strip() if "```" in body else body.strip()


def parse_response(
    text: str, letters: Sequence[str], *, where: str = "response", allow_code_fence: bool = False
) -> dict[str, Label]:
    """A whole reply for one slot: every expected letter, no extras, no prose.

    ``allow_code_fence`` is off by default. Unwrapping a fence the instructions
    told the judge not to use silently converts a compliance problem into a
    number, and the fence rate is worth seeing while the prompt can still change.
    """
    payload = _strip_fence(text) if allow_code_fence else (text or "").strip()
    try:
        blob = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LabelParseError(f"{where}: reply is not a single JSON object ({exc}); got {payload[:120]!r}") from exc
    if not isinstance(blob, Mapping):
        raise LabelParseError(f"{where}: expected an object, got {type(blob).__name__}")
    extra_top = sorted(set(blob) - {"labels"})
    if extra_top:
        raise LabelParseError(f"{where}: unexpected top-level key(s) {extra_top}; expected only 'labels'")
    if not isinstance(blob.get("labels"), Mapping):
        raise LabelParseError(f"{where}: 'labels' must be an object keyed by option letter")

    labels = blob["labels"]
    wanted = list(letters)
    missing = [letter for letter in wanted if letter not in labels]
    if missing:
        raise LabelParseError(f"{where}: no label for option(s) {missing}")
    surplus = sorted(set(labels) - set(wanted))
    if surplus:
        raise LabelParseError(f"{where}: label(s) for option(s) {surplus} that were not offered")
    return {letter: parse_label(labels[letter], where=f"{where}[{letter}]") for letter in wanted}


# ------------------------------------------------------------------- ingestion


def _slot_parts(slot: object, slot_id: str, where: str) -> tuple[str, Mapping[str, object]]:
    """One key entry as ``(item_id, {letter: unit_id})``, or a fatal error naming it.

    A key missing either half cannot unblind anything, so this is not a per-reply
    failure to be counted - it is the wrong file.
    """
    if not isinstance(slot, Mapping):
        raise ValueError(f"{where}: slot {slot_id!r} must be an object, got {type(slot).__name__}")
    item_id = slot.get("item_id")
    options = slot.get("options")
    if not item_id or not isinstance(options, Mapping) or not options:
        raise ValueError(f"{where}: slot {slot_id!r} needs an item_id and a non-empty options map")
    return str(item_id), options


def read_key(path: str) -> dict[str, object]:
    with open(path) as handle:
        key = json.load(handle)
    if not isinstance(key, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(key).__name__}")
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError(f"{path}: expected schema {KEY_SCHEMA!r}, got {key.get('schema')!r}")
    slots = key.get("slots")
    if not isinstance(slots, dict):
        raise ValueError(f"{path}: key has no slots")
    for slot_id, slot in slots.items():
        _slot_parts(slot, str(slot_id), path)
    return key


def ingest_responses(
    key: Mapping[str, object], responses: Iterable[Mapping[str, object]], *, rater: str, allow_code_fence: bool = False
) -> IngestResult:
    """Unblind saved judge replies into label records.

    Each response needs a ``slot_id`` and the raw ``text``. A reply that does not
    parse becomes a failure record rather than an exception, so one malformed
    answer does not discard a round; a duplicated ``slot_id`` does raise, because
    it means a file was concatenated twice and would double-count.
    """
    slots = key.get("slots")
    if not isinstance(slots, Mapping):
        raise ValueError("key has no slots mapping")
    construct = str(key.get("construct", ""))
    quarantined = set(key.get("quarantined_item_ids") or ())

    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    n_replies = 0
    for response in responses:
        n_replies += 1
        slot_id = str(response.get("slot_id", ""))
        if not slot_id:
            failures.append({"slot_id": "", "error": "response has no slot_id"})
            continue
        if slot_id in seen:
            raise ValueError(f"duplicate slot_id {slot_id!r} for rater {rater!r}: responses were counted twice")
        seen.add(slot_id)
        if slot_id not in slots:
            failures.append({"slot_id": slot_id, "error": "slot_id is not in this key"})
            continue
        item_id, options = _slot_parts(slots[slot_id], slot_id, f"key for rater {rater!r}")
        try:
            parsed = parse_response(
                str(response.get("text", "")),
                list(options),
                where=f"{rater}:{slot_id}",
                allow_code_fence=allow_code_fence,
            )
        except LabelParseError as exc:
            failures.append({"slot_id": slot_id, "error": str(exc)})
            continue
        for letter, label in parsed.items():
            records.append(
                {
                    "schema": LABEL_SCHEMA,
                    "unit_id": str(options[letter]),
                    "item_id": item_id,
                    "construct": construct,
                    "rater": rater,
                    "slot_id": slot_id,
                    "option": letter,
                    "quarantined": item_id in quarantined,
                    **label.as_dict(),
                }
            )
    return IngestResult(records=tuple(records), failures=tuple(failures), n_replies=n_replies)


def write_records_jsonl(records: Iterable[Mapping[str, object]], path: str) -> int:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


# ------------------------------------------------------------------------ CLI


def _prepare(args: argparse.Namespace) -> None:
    fields = CorpusFields(
        unit_id=args.unit_id_field,
        item_id=args.item_id_field,
        construct=args.construct_field,
        option_text=args.text_field,
        context=tuple(f for f in args.context_fields.split(",") if f),
    )
    units = read_corpus(args.corpus, fields)
    specs = read_rubric(args.rubric)
    pool = read_exemplars(args.exemplars, fields) if args.exemplars else []
    wanted = [k for k in args.constructs.split(",") if k] or sorted(specs)

    print(f"{len(units)} units, {len(specs)} constructs in rubric, {len(pool)} exemplars available")
    for construct in wanted:
        if construct not in specs:
            raise SystemExit(f"construct {construct!r} is not in {args.rubric}; have {sorted(specs)}")
        shots = select_shots(pool, construct, args.max_shots, seed=args.seed)
        packet = build_packet(units, specs[construct], shots=shots, seed=args.seed)
        packet_path, key_path = write_packet(packet, args.out_dir)
        options = sum(len(slot.options) for slot in packet.slots)
        print(
            f"  {construct:<28} {len(packet.slots):>4} items {options:>5} options "
            f"{len(shots):>2} shots, {len(packet.quarantined_item_ids):>3} items quarantined"
        )
        print(f"    {packet_path}\n    {key_path}  <- do not send this one")


def _ingest(args: argparse.Namespace) -> None:
    key = read_key(args.key)
    responses = list(read_jsonl(args.responses))
    result = ingest_responses(key, responses, rater=args.rater, allow_code_fence=args.allow_code_fence)
    written = write_records_jsonl(result.records, args.out)
    print(f"{len(responses)} replies -> {written} label records for rater {args.rater!r} in {args.out}")
    if result.failures:
        print(f"\n{len(result.failures)} replies did not parse ({result.failure_rate:.0%}). These are prompt bugs:")
        for failure in result.failures[:8]:
            print(f"  {failure['slot_id']:<16} {failure['error']}")
        if len(result.failures) > 8:
            print(f"  ... and {len(result.failures) - 8} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="corpus JSONL -> blinded packets and keys")
    prepare.add_argument("--corpus", required=True)
    prepare.add_argument("--rubric", required=True, help="json: construct -> question and anchors")
    prepare.add_argument("--out-dir", default="data/packets")
    prepare.add_argument("--exemplars", default="", help="human-labelled jsonl for few-shot; ids get quarantined")
    prepare.add_argument("--constructs", default="", help="comma-separated; default is every construct in the rubric")
    prepare.add_argument("--max-shots", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--unit-id-field", default=CorpusFields.unit_id)
    prepare.add_argument("--item-id-field", default=CorpusFields.item_id)
    prepare.add_argument("--construct-field", default=CorpusFields.construct)
    prepare.add_argument("--text-field", default=CorpusFields.option_text)
    prepare.add_argument("--context-fields", default=",".join(CorpusFields.context))

    ingest = sub.add_parser("ingest", help="saved judge replies -> label records")
    ingest.add_argument("--key", required=True)
    ingest.add_argument("--responses", required=True, help="jsonl with slot_id and text")
    ingest.add_argument("--rater", required=True)
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--allow-code-fence", action="store_true", help="unwrap ```json fences instead of failing")

    args = parser.parse_args()
    if args.cmd == "prepare":
        _prepare(args)
    else:
        _ingest(args)


if __name__ == "__main__":
    main()
