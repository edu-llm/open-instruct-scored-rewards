"""Declarative-knowledge and enacted-strategy batteries over the semantic atlas.

WHAT THIS STAGE IS FOR. `analyze_representations` asks whether a construct is
readable inside the model. This asks the prior, cruder question: does the model
behave as though it has the construct at all - can it say what the principle is,
and does it prefer the artifact that performs the principle over the one that
merely talks about it? Those are rungs 1 and 2 of the README's claim ladder.
Nothing here can reach rung 3 or 4; no learner is present and no delayed
assessment exists, and the report says so rather than leaving it to be inferred.

HOW A PROBE IS SCORED. Every probe is a forced choice between two texts, scored
by the teacher-forced log-probability the model assigns to each one after the
same prompt. Nothing is sampled and nothing is judged, which matters because
this project has no human labels for this stage and an agent judge would put a
second unvalidated instrument between the model and the number. The options are
never listed inside the prompt, so there is no option order and no letter-position
bias to correct for.

WHY EVERY CONTRAST IS CROSSED. A model that always prefers the longer, hedgier
or more specific option would score well on any one-directional forced choice
while knowing nothing. So each pair of options is scored twice under two
instructions that flip which option is correct, and the primary statistic is the
difference of the two margins:

    margin(direction)  = logp(option A | direction) - logp(option B | direction)
    paired shift       = margin(first direction) - margin(second direction)

A constant preference for either text cancels exactly. The raw per-direction
accuracy is reported beside it, clearly marked as the confounded number. The one
uncrossed kind, `boundary_versus_unconditional`, exists only as a control on how
much of the crossed boundary result is a taste for hedged prose.

WHAT THE BATTERIES ARE.

- `declarative_knowledge` needs only the ontology. Within each quality-matched
  sibling pair it asks for a definition given a name, a name given a definition,
  and a name given the ontology's own `positive_anchor`. Both options are always
  real ontology prose about two constructs a labeler is expected to confuse, so
  register and specificity are matched by construction.
- `boundary_conditions` asks the ontology's boundary questions: a construct's own
  moderator statement against its sibling's, against a flat unconditional claim,
  whether the required context is missing or supplied, and whether the requested
  artifact type can carry the construct at all. The last two are the two refusals
  `schema.Applicability` keeps apart, and both come from structure - `required_context`
  and `output_types` - rather than from parsing prose. `escalation:` boundary
  entries are excluded: they say what data a higher claim level would need, which
  is a fact about this project and not about the principle.
- `enacted_strategy` and `named_versus_enacted` need the generated corpus. Within
  one scenario the corpus already holds an artifact that enacts the construct, one
  that competently enacts its sibling instead, and one that names the construct
  while enacting none of it. Those are the two contrasts: strategy against
  neighbouring strategy, and doing against saying.

WHAT IT CONSUMES, AND WHY IT NEEDS THE WITHHELD FILES. Option texts and scenario
stems come from the blinded `corpus.jsonl`. Which construct each artifact enacts
and names comes from `withheld/key.jsonl`, and the system prompt the artifact was
written under comes from `withheld/prompts.jsonl`, so an artifact is scored in the
context that produced it. This stage therefore opens the key and must run after
labelling, like every other analysis here. The withheld *user* brief is
deliberately not used as a probe prompt: it names its item's target construct and
role, so it would answer one direction of the crossing and not the other.

WHAT THE KEYING IS WORTH. The corpus key records the design's intent, not a
verified label: no rater has confirmed that a canonical positive really enacts
its construct. So every enacted result is keyed to the brief, the report records
`keying: designed_unverified`, and `--labels` narrows the battery to units whose
label agrees with the design and flips that field. Calibration is agent-grounded
and exploratory throughout; nothing here clears a confirmatory gate.

Because the keying is only as good as the generation, `enacted_design_audit`
checks the design from the text: which item role supplied each pair's enacting
artifact, and whether each says-only artifact really uses its construct's
vocabulary. Both are counted without a model, so a dry run says whether a corpus
can support these contrasts before a GPU is asked for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from projects.learning_science_semantic_atlas import build_corpus, schema

REPORT_SCHEMA = "semantic-atlas/behavior-report-v1"
PROBE_SCHEMA = "semantic-atlas/behavior-probe-v1"
RESPONSE_SCHEMA = "semantic-atlas/behavior-response-v1"
NAMING_SCHEMA = "semantic-atlas/behavior-naming-v1"

PROBES_FILE = "probes.jsonl"
RESPONSES_FILE = "responses.jsonl"
NAMING_FILE = "naming_generations.jsonl"
REPORT_FILE = "behavior_report.json"

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
DENSE_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

# Annotation only. Both arms run through the same code path; this exists so a
# report says which arm produced it without the reader matching model strings by
# eye. OLMoE and OLMo-2 are different model families, so a cross-arm difference
# is replication or non-replication and never an architecture claim.
KNOWN_ARMS: Mapping[str, str] = {DEFAULT_MODEL: "moe", DENSE_MODEL: "dense"}

DEFAULT_SALT = "behavior-v1"
DEFAULT_SEED = 1701
DEFAULT_BOOTSTRAP_SAMPLES = 5000

# There are no human labels for this stage and no reliability estimate for its
# keying, so every report carries this and no result may be described as
# confirmatory. LABELING_GUIDE.md 8 defines the gates this does not attempt.
LABEL_SOURCE = "agent_grounded_forced_choice"
CALIBRATION_STATUS = "exploratory"

# The analyst voice, used by every ontology probe. Corpus probes replace it with
# the withheld system prompt their artifacts were generated under.
ANALYST_SYSTEM_PROMPT = "You are a careful instructional designer. Answer directly, with no preamble."


class Level(str, Enum):
    """The claim-ladder rung a battery speaks to. `schema.EvidenceLevel`'s first two.

    The other two rungs are absent on purpose rather than unimplemented: a
    battery that never shows an artifact to a learner cannot be given a value
    here that implies it did.
    """

    DECLARATIVE_KNOWLEDGE = schema.EvidenceLevel.DECLARATIVE_KNOWLEDGE.value
    ENACTED_OUTPUT = schema.EvidenceLevel.ENACTED_OUTPUT.value


class ProbeKind(str, Enum):
    NAME_TO_DEFINITION = "name_to_definition"
    DEFINITION_TO_NAME = "definition_to_name"
    ANCHOR_ATTRIBUTION = "anchor_attribution"
    BOUNDARY_VERSUS_SIBLING = "boundary_versus_sibling"
    BOUNDARY_VERSUS_UNCONDITIONAL = "boundary_versus_unconditional"
    REQUIRED_CONTEXT = "required_context"
    FORMAT_APPLICABILITY = "format_applicability"
    ENACTED_STRATEGY = "enacted_strategy"
    NAMED_VERSUS_ENACTED = "named_versus_enacted"


class Direction(str, Enum):
    """Which way a crossed contrast is asked. Two directions flip the keyed option.

    Four separate word pairs rather than a generic first/second, because the
    thing being flipped differs: which member of a sibling pair the question is
    about, whether the required context was supplied, whether the requested
    artifact type can carry the construct, and whether the request is to perform
    the principle or to explain it.
    """

    TARGET = "target"
    SIBLING = "sibling"
    WITHHELD = "withheld"
    SUPPLIED = "supplied"
    ACCEPTED = "accepted"
    REFUSED = "refused"
    ENACT = "enact"
    DESCRIBE = "describe"


class OptionTag(str, Enum):
    """What an option is, so the report can name what won rather than which index."""

    TARGET_DEFINITION = "target_definition"
    SIBLING_DEFINITION = "sibling_definition"
    TARGET_NAME = "target_name"
    SIBLING_NAME = "sibling_name"
    TARGET_BOUNDARY = "target_boundary"
    SIBLING_BOUNDARY = "sibling_boundary"
    UNCONDITIONAL_CLAIM = "unconditional_claim"
    NEEDS_MORE_CONTEXT = "needs_more_context"
    CONTEXT_IS_SUFFICIENT = "context_is_sufficient"
    FORMAT_ACCEPTS = "format_accepts"
    FORMAT_REFUSES = "format_refuses"
    ENACTS_TARGET = "enacts_target"
    ENACTS_SIBLING = "enacts_sibling"
    NAMES_WITHOUT_ENACTING = "names_without_enacting"


# The crossing table, and the whole definition of "correct" in this module. For
# each kind: the directions in order, and the option tag keyed in each. Option
# order inside a probe follows this order too, so the paired shift is always
# margin(directions[0]) - margin(directions[1]) with no per-kind special cases.
CROSSING: Mapping[ProbeKind, tuple[tuple[Direction, OptionTag], ...]] = {
    ProbeKind.NAME_TO_DEFINITION: (
        (Direction.TARGET, OptionTag.TARGET_DEFINITION),
        (Direction.SIBLING, OptionTag.SIBLING_DEFINITION),
    ),
    ProbeKind.DEFINITION_TO_NAME: (
        (Direction.TARGET, OptionTag.TARGET_NAME),
        (Direction.SIBLING, OptionTag.SIBLING_NAME),
    ),
    ProbeKind.ANCHOR_ATTRIBUTION: (
        (Direction.TARGET, OptionTag.TARGET_NAME),
        (Direction.SIBLING, OptionTag.SIBLING_NAME),
    ),
    ProbeKind.BOUNDARY_VERSUS_SIBLING: (
        (Direction.TARGET, OptionTag.TARGET_BOUNDARY),
        (Direction.SIBLING, OptionTag.SIBLING_BOUNDARY),
    ),
    # Uncrossed by design: the unconditional option is written about one
    # construct, so swapping the question does not swap which option is right.
    ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL: ((Direction.TARGET, OptionTag.TARGET_BOUNDARY),),
    ProbeKind.REQUIRED_CONTEXT: (
        (Direction.WITHHELD, OptionTag.NEEDS_MORE_CONTEXT),
        (Direction.SUPPLIED, OptionTag.CONTEXT_IS_SUFFICIENT),
    ),
    ProbeKind.FORMAT_APPLICABILITY: (
        (Direction.ACCEPTED, OptionTag.FORMAT_ACCEPTS),
        (Direction.REFUSED, OptionTag.FORMAT_REFUSES),
    ),
    ProbeKind.ENACTED_STRATEGY: (
        (Direction.TARGET, OptionTag.ENACTS_TARGET),
        (Direction.SIBLING, OptionTag.ENACTS_SIBLING),
    ),
    ProbeKind.NAMED_VERSUS_ENACTED: (
        (Direction.ENACT, OptionTag.ENACTS_TARGET),
        (Direction.DESCRIBE, OptionTag.NAMES_WITHOUT_ENACTING),
    ),
}

DECLARATIVE_KINDS: tuple[ProbeKind, ...] = (
    ProbeKind.NAME_TO_DEFINITION,
    ProbeKind.DEFINITION_TO_NAME,
    ProbeKind.ANCHOR_ATTRIBUTION,
)
BOUNDARY_KINDS: tuple[ProbeKind, ...] = (
    ProbeKind.BOUNDARY_VERSUS_SIBLING,
    ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL,
    ProbeKind.REQUIRED_CONTEXT,
    ProbeKind.FORMAT_APPLICABILITY,
)
ENACTED_KINDS: tuple[ProbeKind, ...] = (ProbeKind.ENACTED_STRATEGY, ProbeKind.NAMED_VERSUS_ENACTED)

BATTERY_KINDS: Mapping[str, tuple[ProbeKind, ...]] = {
    "declarative_knowledge": DECLARATIVE_KINDS,
    "boundary_conditions": BOUNDARY_KINDS,
    "enacted_strategy": ENACTED_KINDS,
}
BATTERY_LEVELS: Mapping[str, Level] = {
    "declarative_knowledge": Level.DECLARATIVE_KNOWLEDGE,
    "boundary_conditions": Level.DECLARATIVE_KNOWLEDGE,
    "enacted_strategy": Level.ENACTED_OUTPUT,
}


class Normalization(str, Enum):
    """How a decision is read off two log-probabilities of unequal length.

    `MEAN` is per-token and is the primary reading, because option texts here
    differ in length by design - a definition against a definition is matched, an
    artifact against an artifact is not. `TOTAL` is reported beside it, and a
    contrast the two disagree about is reported as normalization-dependent rather
    than as a result.
    """

    MEAN = "mean"
    TOTAL = "total"


class BehaviorError(ValueError):
    """A battery that cannot be built, or a stored response that cannot be trusted."""


# --------------------------------------------------------------------- probes


@dataclass(frozen=True, slots=True)
class Option:
    """One of the two texts a probe scores."""

    option_id: str
    tag: OptionTag
    text: str
    keyed: bool

    def as_dict(self) -> dict[str, object]:
        return {"option_id": self.option_id, "tag": self.tag.value, "text": self.text, "keyed": self.keyed}


@dataclass(frozen=True, slots=True)
class Probe:
    """One scored decision: a prompt, two options, and which one the design keys.

    `pair_id` is the contrast, `probe_id` is one direction of it, and both are
    content digests so a rebuilt battery reproduces its own ids and a stale
    response file is detected rather than reused. `instruction` is the clause that
    differs between the two directions, kept apart from the rest of the prompt so
    the lexical-overlap control can measure that clause alone.
    """

    probe_id: str
    pair_id: str
    kind: ProbeKind
    level: Level
    direction: Direction
    construct: str
    sibling: str
    system_prompt: str
    prompt: str
    instruction: str
    options: tuple[Option, ...]
    source_units: tuple[str, ...] = ()
    max_new_tokens: int = 256
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = f"Probe[{self.kind.value}:{self.probe_id}]"
        if len(self.options) != 2:
            raise BehaviorError(f"{where}: a forced choice needs exactly two options, got {len(self.options)}")
        if self.options[0].option_id == self.options[1].option_id:
            raise BehaviorError(f"{where}: both options share the id {self.options[0].option_id!r}")
        if self.options[0].text.strip() == self.options[1].text.strip():
            raise BehaviorError(f"{where}: both options carry the same text, so the choice is undefined")
        keyed = [option for option in self.options if option.keyed]
        if len(keyed) != 1:
            raise BehaviorError(f"{where}: exactly one option must be keyed, got {len(keyed)}")
        expected = dict(CROSSING[self.kind]).get(self.direction)
        if expected is None:
            raise BehaviorError(f"{where}: {self.direction.value!r} is not a direction of this kind")
        if keyed[0].tag is not expected:
            raise BehaviorError(
                f"{where}: direction {self.direction.value!r} keys {expected.value!r}, "
                f"but the keyed option is tagged {keyed[0].tag.value!r}"
            )
        if not self.prompt.strip() or not self.instruction.strip():
            raise BehaviorError(f"{where}: prompt and instruction must not be empty")

    @property
    def keyed_option(self) -> Option:
        return next(option for option in self.options if option.keyed)

    @property
    def probe_sha(self) -> str:
        """The identity of everything scored. A changed prompt or option invalidates a response."""
        return schema.stable_digest(self.system_prompt, self.prompt, *(option.text for option in self.options))

    def as_row(self) -> dict[str, object]:
        return {
            "schema": PROBE_SCHEMA,
            "probe_id": self.probe_id,
            "pair_id": self.pair_id,
            "kind": self.kind.value,
            "level": self.level.value,
            "direction": self.direction.value,
            "construct": self.construct,
            "sibling": self.sibling,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt,
            "instruction": self.instruction,
            "options": [option.as_dict() for option in self.options],
            "source_units": list(self.source_units),
            "max_new_tokens": self.max_new_tokens,
            "probe_sha": self.probe_sha,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Battery:
    """Probes that answer one question at one claim level."""

    name: str
    level: Level
    probes: tuple[Probe, ...]

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(probe.pair_id for probe in self.probes))

    def of_kind(self, kind: ProbeKind) -> tuple[Probe, ...]:
        return tuple(probe for probe in self.probes if probe.kind is kind)


def _probe_ids(salt: str, kind: ProbeKind, parts: Sequence[str], direction: Direction) -> tuple[str, str]:
    pair_id = "bpr" + schema.stable_digest(salt, kind.value, *parts, size=7)
    return "bp" + schema.stable_digest(pair_id, direction.value, size=7), pair_id


def _option_id(pair_id: str, tag: OptionTag) -> str:
    return "bo" + schema.stable_digest(pair_id, tag.value, size=6)


def _crossed_probes(
    *,
    salt: str,
    kind: ProbeKind,
    level: Level,
    construct: str,
    sibling: str,
    id_parts: Sequence[str],
    texts: Mapping[OptionTag, str],
    prompts: Mapping[Direction, tuple[str, str]],
    system_prompt: str = ANALYST_SYSTEM_PROMPT,
    source_units: Sequence[str] = (),
    max_new_tokens: int = 256,
    metadata: Mapping[str, object] | None = None,
) -> tuple[Probe, ...]:
    """One probe per direction of a contrast, over one shared pair of option texts.

    Option order follows `CROSSING`, so the two probes of a pair present the same
    two texts in the same order and differ only in the prompt. That is the whole
    requirement for the paired shift to cancel a constant option preference, so
    it is built once here instead of being restated by every battery.
    """
    crossing = CROSSING[kind]
    _, pair_id = _probe_ids(salt, kind, id_parts, crossing[0][0])
    tags = [tag for _, tag in crossing]
    if len(tags) == 1:
        # The uncrossed kind still needs a second, unkeyed option to choose against.
        tags.append(OptionTag.UNCONDITIONAL_CLAIM)
    probes = []
    for direction, keyed_tag in crossing:
        probe_id, _ = _probe_ids(salt, kind, id_parts, direction)
        prompt, instruction = prompts[direction]
        probes.append(
            Probe(
                probe_id=probe_id,
                pair_id=pair_id,
                kind=kind,
                level=level,
                direction=direction,
                construct=construct,
                sibling=sibling,
                system_prompt=system_prompt,
                prompt=prompt,
                instruction=instruction,
                options=tuple(
                    Option(_option_id(pair_id, tag), tag, texts[tag], keyed=tag is keyed_tag) for tag in tags
                ),
                source_units=tuple(source_units),
                max_new_tokens=max_new_tokens,
                metadata=dict(metadata or {}),
            )
        )
    return tuple(probes)


# ------------------------------------------------------------- sibling pairing


@dataclass(frozen=True, slots=True)
class SiblingPair:
    """A construct and the one the ontology says it is most often mistaken for."""

    target: str
    sibling: str

    @property
    def key(self) -> str:
        return f"{self.target}|{self.sibling}"


def sibling_pairs(ontology: schema.Ontology) -> tuple[SiblingPair, ...]:
    """One pair per confusable relation, in ontology order, without duplicates.

    Five of the twenty-four links are mutual, so iterating constructs blindly
    would score those pairs twice and weight them double in a mean over pairs.
    Deduplicating on the unordered pair keeps every construct covered - each one
    appears as either the target or the sibling of something - while counting
    each contrast once.
    """
    seen: set[frozenset[str]] = set()
    pairs: list[SiblingPair] = []
    for construct in ontology:
        unordered = frozenset({construct.key, construct.sibling_key})
        if len(unordered) < 2 or unordered in seen:
            continue
        seen.add(unordered)
        pairs.append(SiblingPair(construct.key, construct.sibling_key))
    return tuple(pairs)


def moderator_conditions(construct: schema.Construct) -> tuple[str, ...]:
    """Boundary statements about when the principle holds, and how strongly.

    `escalation:` entries are dropped because they describe what evidence a
    higher claim level would need rather than a property of the principle, so a
    model cannot be right or wrong about them. This is the same filter
    `build_corpus._boundary` applies and then the opposite selection: that
    function wants a condition an artifact can visibly fall short of and prefers
    the prescriptive entries, while a knowledge probe wants exactly the moderator
    statements it discards.
    """
    return tuple(entry for entry in construct.boundary_conditions if not entry.lower().startswith("escalation"))


def _readable(text: str, ontology: schema.Ontology) -> str:
    """Ontology prose with construct keys spelled out as display names.

    Several boundary conditions cross-reference a construct by key. An
    identifier is not something a model reads in prose, and here it would also
    hand one option a marker the other lacks.
    """
    for construct in ontology:
        if construct.key in text:
            text = text.replace(construct.key, construct.display_name.lower())
    return text


def _format_phrase(output_format: schema.OutputFormat) -> str:
    return output_format.value.replace("_", " ")


def unconditional_claim(construct: schema.Construct) -> str:
    """The over-generalisation a boundary condition denies, in one fixed shape.

    Written from a template rather than by negating the ontology's prose: a
    generated negation of an arbitrary sentence is usually either trivially
    ungrammatical or accidentally true, and either way the probe would measure
    the generator.
    """
    return (
        f"{construct.display_name} improves learning for every learner, on every timescale and in every "
        "subject. There is no condition under which it helps less."
    )


# -------------------------------------------------------- declarative battery


def declarative_battery(ontology: schema.Ontology, *, salt: str = DEFAULT_SALT) -> Battery:
    """Can the model say what a construct is, and tell it from its confusable neighbour?

    Three questions per sibling pair, each crossed. Both options are always the
    ontology's own text for the two constructs, so the contrast cannot be won by
    preferring longer, more technical or more hedged prose.

    The third question shows a `positive_anchor` as the ontology writes it, which
    for half the constructs is a quoted artifact and for the other half a
    description of one, in both cases followed by the reason it qualifies. So it
    asks a model to recognise a canonical case, not to judge a raw artifact; the
    corpus batteries are where raw artifacts are scored.
    """
    probes: list[Probe] = []
    for pair in sibling_pairs(ontology):
        target = ontology.get(pair.target)
        sibling = ontology.get(pair.sibling)
        names = {OptionTag.TARGET_NAME: target.display_name, OptionTag.SIBLING_NAME: sibling.display_name}
        definitions = {
            OptionTag.TARGET_DEFINITION: _readable(target.definition, ontology),
            OptionTag.SIBLING_DEFINITION: _readable(sibling.definition, ontology),
        }
        shared = {
            "salt": salt,
            "level": Level.DECLARATIVE_KNOWLEDGE,
            "construct": pair.target,
            "sibling": pair.sibling,
        }

        asked = {}
        for direction, construct in ((Direction.TARGET, target), (Direction.SIBLING, sibling)):
            instruction = f'What does "{construct.display_name}" require of an instructional artifact?'
            asked[direction] = (f"Question: {instruction}\nAnswer:", instruction)
        probes.extend(
            _crossed_probes(
                kind=ProbeKind.NAME_TO_DEFINITION, id_parts=(pair.key,), texts=definitions, prompts=asked, **shared
            )
        )

        named = {}
        for direction, construct in ((Direction.TARGET, target), (Direction.SIBLING, sibling)):
            definition = _readable(construct.definition, ontology)
            named[direction] = (
                f"A learning-science construct is defined as follows.\n{definition}\n"
                "Question: What is the name of the construct defined above?\nAnswer:",
                definition,
            )
        probes.extend(
            _crossed_probes(
                kind=ProbeKind.DEFINITION_TO_NAME, id_parts=(pair.key,), texts=names, prompts=named, **shared
            )
        )

        attributed = {}
        for direction, construct in ((Direction.TARGET, target), (Direction.SIBLING, sibling)):
            anchor = _readable(construct.positive_anchor, ontology)
            attributed[direction] = (
                f"Consider this canonical case.\n{anchor}\n"
                "Question: Which learning-science construct does it instantiate?\nAnswer:",
                anchor,
            )
        probes.extend(
            _crossed_probes(
                kind=ProbeKind.ANCHOR_ATTRIBUTION, id_parts=(pair.key,), texts=names, prompts=attributed, **shared
            )
        )
    return Battery("declarative_knowledge", Level.DECLARATIVE_KNOWLEDGE, tuple(probes))


# ---------------------------------------------------------- boundary battery


def boundary_battery(ontology: schema.Ontology, *, salt: str = DEFAULT_SALT, conditions_per_pair: int = 2) -> Battery:
    """Does the model hold the construct's boundaries, or an unconditional version of it?

    Four questions. The first is the primary one and is crossed inside a sibling
    pair, so both options are equally hedged ontology prose and only their
    attachment differs. The second replaces the sibling's condition with a flat
    universal claim and is a control on hedging preference, not a result. The last
    two are the two ways `schema.Applicability` allows a construct to be
    inapplicable, and they are crossed on the fact that makes them so: whether the
    required context was supplied, and whether the artifact type can carry the
    construct at all.
    """
    if conditions_per_pair < 1:
        raise BehaviorError(f"conditions_per_pair must be at least 1, got {conditions_per_pair}")
    probes: list[Probe] = []
    for pair in sibling_pairs(ontology):
        target, sibling = ontology.get(pair.target), ontology.get(pair.sibling)
        target_conditions = moderator_conditions(target)
        sibling_conditions = moderator_conditions(sibling)
        if not target_conditions or not sibling_conditions:
            continue
        shared = {
            "salt": salt,
            "level": Level.DECLARATIVE_KNOWLEDGE,
            "construct": pair.target,
            "sibling": pair.sibling,
        }
        rounds = min(conditions_per_pair, len(target_conditions), len(sibling_conditions))
        for index in range(rounds):
            texts = {
                OptionTag.TARGET_BOUNDARY: _readable(target_conditions[index], ontology),
                OptionTag.SIBLING_BOUNDARY: _readable(sibling_conditions[index], ontology),
            }
            if texts[OptionTag.TARGET_BOUNDARY].strip() == texts[OptionTag.SIBLING_BOUNDARY].strip():
                continue
            probes.extend(
                _crossed_probes(
                    kind=ProbeKind.BOUNDARY_VERSUS_SIBLING,
                    id_parts=(pair.key, str(index)),
                    texts=texts,
                    prompts={Direction.TARGET: _boundary_prompt(target), Direction.SIBLING: _boundary_prompt(sibling)},
                    **shared,
                )
            )
            probes.extend(
                _crossed_probes(
                    kind=ProbeKind.BOUNDARY_VERSUS_UNCONDITIONAL,
                    id_parts=(pair.key, str(index)),
                    texts={
                        OptionTag.TARGET_BOUNDARY: texts[OptionTag.TARGET_BOUNDARY],
                        OptionTag.UNCONDITIONAL_CLAIM: unconditional_claim(target),
                    },
                    prompts={Direction.TARGET: _boundary_prompt(target)},
                    **shared,
                )
            )
    probes.extend(_applicability_probes(ontology, salt=salt))
    return Battery("boundary_conditions", Level.DECLARATIVE_KNOWLEDGE, tuple(probes))


def _boundary_prompt(construct: schema.Construct) -> tuple[str, str]:
    instruction = f'Under what conditions does "{construct.display_name}" help less, or not at all?'
    return f"Question: {instruction}\nAnswer:", instruction


def _applicability_prompt(output_format: schema.OutputFormat, display_name: str) -> tuple[str, str]:
    instruction = f'Someone wants a {_format_phrase(output_format)} that provides "{display_name}".'
    return f"{instruction}\nQuestion: Can that kind of artifact carry it?\nAnswer:", instruction


# Option texts for the two applicability probes. They name neither the construct
# nor the artifact type, because both differ between the two directions and the
# crossing only cancels a constant preference if the two texts are constant.
_NEEDS_CONTEXT = "I cannot tell from what you have sent. Deciding this needs the material you have not shown me."
_CONTEXT_SUFFICIENT = "What you have sent is enough to decide, and no further material is needed."
_FORMAT_ACCEPTS = "Yes. That kind of artifact can carry it."
_FORMAT_REFUSES = "No. That kind of artifact cannot carry it, and the request needs a different one."


def _applicability_probes(ontology: schema.Ontology, *, salt: str) -> tuple[Probe, ...]:
    probes: list[Probe] = []
    for construct in ontology:
        sibling = construct.sibling_key
        shared = {"salt": salt, "level": Level.DECLARATIVE_KNOWLEDGE, "construct": construct.key, "sibling": sibling}
        if construct.required_context:
            required = _readable(construct.required_context[0], ontology)
            withheld_instruction = (
                f'A colleague asks whether one artifact provides "{construct.display_name}". They have sent a '
                "short excerpt and nothing else."
            )
            supplied_instruction = (
                f'A colleague asks whether one artifact provides "{construct.display_name}". They have sent the '
                f"whole artifact and have also stated {required}."
            )
            probes.extend(
                _crossed_probes(
                    kind=ProbeKind.REQUIRED_CONTEXT,
                    id_parts=(construct.key,),
                    texts={
                        OptionTag.NEEDS_MORE_CONTEXT: _NEEDS_CONTEXT,
                        OptionTag.CONTEXT_IS_SUFFICIENT: _CONTEXT_SUFFICIENT,
                    },
                    prompts={
                        Direction.WITHHELD: (
                            f"{withheld_instruction}\nQuestion: What should you tell them?\nAnswer:",
                            withheld_instruction,
                        ),
                        Direction.SUPPLIED: (
                            f"{supplied_instruction}\nQuestion: What should you tell them?\nAnswer:",
                            supplied_instruction,
                        ),
                    },
                    metadata={"required_context": required},
                    **shared,
                )
            )

        accepted = construct.output_types[0]
        refused = next((fmt for fmt in schema.OutputFormat if not construct.accepts(fmt)), None)
        if refused is None:
            continue
        probes.extend(
            _crossed_probes(
                kind=ProbeKind.FORMAT_APPLICABILITY,
                id_parts=(construct.key,),
                texts={OptionTag.FORMAT_ACCEPTS: _FORMAT_ACCEPTS, OptionTag.FORMAT_REFUSES: _FORMAT_REFUSES},
                prompts={
                    Direction.ACCEPTED: _applicability_prompt(accepted, construct.display_name),
                    Direction.REFUSED: _applicability_prompt(refused, construct.display_name),
                },
                metadata={"accepted_format": accepted.value, "refused_format": refused.value},
                **shared,
            )
        )
    return tuple(probes)


# ------------------------------------------------------------- corpus intake


@dataclass(frozen=True, slots=True)
class CorpusUnit:
    """One generated artifact with the design facts this stage needs.

    Joined by unit id from the blinded corpus row, the withheld key and the
    withheld brief. Nothing is joined by line number: a resumed generation run
    reorders the generations file, and a positional join would attach the wrong
    artifact to the wrong key.
    """

    unit_id: str
    item_id: str
    concept: str
    text: str
    question: str
    student_before: str
    reference: str
    item_role: str
    output_format: str
    enacted_construct: str | None
    named_construct: str | None
    enacts: bool
    names: bool
    system_prompt: str
    prompt_sha: str
    max_new_tokens: int
    source_model: str | None
    withheld_user_prompt: str

    @property
    def enacted_key(self) -> str | None:
        """The construct this artifact performs, whatever it was written to be scored against.

        `enacts` is relative to the item's designed target, so a sibling hard
        negative reads `enacts: false` while its `enacted_construct` names the
        neighbour it competently performs. Reading only `enacts` would find no
        rival for any pair; reading only `enacted_construct` would miss the
        canonical positives, which leave it null because it equals the concept.
        """
        if self.enacted_construct:
            return self.enacted_construct
        return self.concept if self.enacts else None


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    """A corpus directory, read and cross-checked."""

    path: Path
    units: Mapping[str, CorpusUnit]
    manifest: Mapping[str, object]

    @property
    def source_models(self) -> tuple[str, ...]:
        return tuple(sorted({unit.source_model for unit in self.units.values() if unit.source_model}))

    @property
    def scaffolded(self) -> tuple[str, ...]:
        return tuple(
            sorted(unit_id for unit_id, unit in self.units.items() if unit.source_model == build_corpus.SCAFFOLD_MODEL)
        )

    def by_item(self) -> dict[str, list[CorpusUnit]]:
        grouped: dict[str, list[CorpusUnit]] = {}
        for unit in sorted(self.units.values(), key=lambda unit: unit.unit_id):
            grouped.setdefault(unit.item_id, []).append(unit)
        return grouped


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """JSON objects with errors that name the line they came from."""
    target = Path(path)
    if not target.exists():
        raise BehaviorError(f"{target} does not exist")
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BehaviorError(f"{target}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise BehaviorError(f"{target}:{number}: expected a JSON object, got {type(row).__name__}")
            rows.append(row)
    return rows


def _index(rows: Iterable[Mapping[str, Any]], *, source: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id:
            raise BehaviorError(f"{source}:{number}: row needs a unit_id")
        if unit_id in indexed:
            raise BehaviorError(f"{source}: duplicate unit_id {unit_id!r}")
        indexed[unit_id] = row
    return indexed


def _bool_field(row: Mapping[str, Any], name: str, *, where: str) -> bool:
    value = row.get(name)
    if not isinstance(value, bool):
        raise BehaviorError(f"{where}: key field {name} must be a JSON boolean, got {value!r}")
    return value


def load_bundle(corpus_dir: str | Path) -> CorpusBundle:
    """Read a corpus directory and refuse any disagreement between its three files.

    The prompt digest has to match between the key and the brief for every unit.
    A mismatch means the corpus and the briefs came from different plans, and
    pairing them would attach one item's design to another item's text - the same
    failure `build_corpus.assemble` refuses rather than works around.
    """
    root = Path(corpus_dir)
    corpus = _index(read_jsonl(root / build_corpus.CORPUS_FILE), source=str(root / build_corpus.CORPUS_FILE))
    withheld = root / build_corpus.WITHHELD_DIR
    key = _index(read_jsonl(withheld / build_corpus.KEY_FILE), source=str(withheld / build_corpus.KEY_FILE))
    prompts = _index(
        read_jsonl(withheld / build_corpus.PROMPTS_FILE), source=str(withheld / build_corpus.PROMPTS_FILE)
    )

    missing_key = sorted(set(corpus) - set(key))
    if missing_key:
        raise BehaviorError(f"{len(missing_key)} corpus row(s) have no key entry, first: {missing_key[:5]}")
    missing_prompt = sorted(set(corpus) - set(prompts))
    if missing_prompt:
        raise BehaviorError(f"{len(missing_prompt)} corpus row(s) have no withheld brief, first: {missing_prompt[:5]}")

    units: dict[str, CorpusUnit] = {}
    for unit_id, row in corpus.items():
        key_row, prompt_row = key[unit_id], prompts[unit_id]
        where = f"{root}:{unit_id}"
        if key_row.get("prompt_sha") != prompt_row.get("prompt_sha"):
            raise BehaviorError(
                f"{where}: the key records prompt {key_row.get('prompt_sha')!r} and the brief digests to "
                f"{prompt_row.get('prompt_sha')!r}. These files were written from different plans; rebuild the "
                "corpus before measuring behaviour against it."
            )
        text = str(row.get("candidate_action") or "")
        if not text.strip():
            raise BehaviorError(f"{where}: the corpus row carries no candidate_action")
        units[unit_id] = CorpusUnit(
            unit_id=unit_id,
            item_id=str(row.get("item_id") or ""),
            concept=str(row.get("concept") or key_row.get("designed_target_construct") or ""),
            text=text,
            question=str(row.get("question") or ""),
            student_before=str(row.get("student_before") or ""),
            reference=str(row.get("reference") or ""),
            item_role=str(key_row.get("item_role") or ""),
            output_format=str(key_row.get("output_format") or ""),
            enacted_construct=key_row.get("enacted_construct") or None,
            named_construct=key_row.get("named_construct") or None,
            enacts=_bool_field(key_row, "enacts", where=where),
            names=_bool_field(key_row, "names", where=where),
            system_prompt=str(prompt_row.get("system") or ANALYST_SYSTEM_PROMPT),
            prompt_sha=str(key_row.get("prompt_sha") or ""),
            max_new_tokens=int(prompt_row.get("max_new_tokens") or 256),
            source_model=(str(key_row["source_model"]) if key_row.get("source_model") else None),
            withheld_user_prompt=str(prompt_row.get("user") or ""),
        )

    manifest_path = root / build_corpus.MANIFEST_FILE
    manifest: Mapping[str, object] = {}
    if manifest_path.exists():
        blob = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(blob, Mapping):
            manifest = blob
    return CorpusBundle(root, units, manifest)


def verified_units(label_paths: Sequence[str | Path], bundle: CorpusBundle) -> dict[str, bool]:
    """Units whose label agrees with the design, keyed by unit id.

    Optional, and off by default, because no labelling round for this corpus has
    been completed. A label is read through `schema.LabelRecord` so a row that
    fills presence on an inapplicable unit is refused here rather than counted as
    agreement, and agreement is checked against `enacts` - presence tracks
    enactment and never naming.
    """
    agreement: dict[str, bool] = {}
    for path in label_paths:
        for row in read_jsonl(path):
            payload = dict(row)
            if "construct" in payload:
                payload["concept"] = payload.pop("construct")
            record = schema.LabelRecord.from_mapping(payload)
            unit = bundle.units.get(record.artifact_id)
            if unit is None:
                raise BehaviorError(f"{path}: label names unit {record.artifact_id!r}, which this corpus has not")
            if record.concept != unit.concept:
                continue
            if record.quarantined or not record.applicable:
                agreement[unit.unit_id] = False
                continue
            agrees = bool(record.presence) == unit.enacts
            agreement[unit.unit_id] = agreement.get(unit.unit_id, True) and agrees
    return agreement


# ------------------------------------------------------------ enacted battery


def _stem(unit: CorpusUnit) -> str:
    lines = [unit.question]
    if unit.student_before.strip():
        lines.append(f"The learner has said: {unit.student_before}")
    return "\n".join(line for line in lines if line.strip())


def enacted_battery(
    ontology: schema.Ontology,
    bundle: CorpusBundle,
    *,
    salt: str = DEFAULT_SALT,
    allowed_units: Mapping[str, bool] | None = None,
) -> Battery:
    """Does the model prefer the artifact that performs the strategy it was asked for?

    Both contrasts live inside one scenario, so the two options answer the same
    question for the same learner and differ in what they do about it. The corpus
    supplies the pairing: one variant enacts the construct, one competently
    enacts the sibling instead, and one names the construct while enacting none of
    it.

    Each artifact is scored under the system prompt it was generated with, so the
    text is read in the context that produced it. The withheld user brief is not
    reused: it states its own item's target and role, which would answer one
    direction of the crossing and not the other.
    """
    probes: list[Probe] = []
    for item_id, units in sorted(bundle.by_item().items()):
        usable = [unit for unit in units if allowed_units is None or allowed_units.get(unit.unit_id, False)]
        performers = [unit for unit in usable if unit.enacted_key in ontology]
        naming_only = [unit for unit in usable if unit.names and not unit.enacts and unit.named_construct in ontology]
        if not performers:
            continue
        # The canonical positive where the scenario has one: it is the variant
        # written to instantiate the construct at full fidelity, and the partial
        # positive is deliberately missing a required element.
        target_unit = min(performers, key=lambda unit: (unit.item_role != "positive_canonical", unit.unit_id))
        target = ontology.get(str(target_unit.enacted_key))
        system_prompt = target_unit.system_prompt
        metadata: dict[str, object] = {
            "item_id": item_id,
            "output_format": target_unit.output_format,
            "system_prompts_agree": len({unit.system_prompt for unit in usable}) == 1,
        }

        rival = next((unit for unit in performers if unit.enacted_key not in (None, target.key)), None)
        if rival is not None:
            sibling = ontology.get(str(rival.enacted_key))
            probes.extend(
                _crossed_probes(
                    salt=salt,
                    kind=ProbeKind.ENACTED_STRATEGY,
                    level=Level.ENACTED_OUTPUT,
                    construct=target.key,
                    sibling=sibling.key,
                    id_parts=(item_id, target_unit.unit_id, rival.unit_id),
                    texts={OptionTag.ENACTS_TARGET: target_unit.text, OptionTag.ENACTS_SIBLING: rival.text},
                    prompts={
                        Direction.TARGET: _enact_prompt(target_unit, target),
                        Direction.SIBLING: _enact_prompt(target_unit, sibling),
                    },
                    system_prompt=system_prompt,
                    source_units=(target_unit.unit_id, rival.unit_id),
                    max_new_tokens=target_unit.max_new_tokens,
                    metadata={**metadata, "roles": [target_unit.item_role, rival.item_role]},
                )
            )

        says_only = next((unit for unit in naming_only if unit.named_construct == target.key), None)
        if says_only is not None:
            probes.extend(
                _crossed_probes(
                    salt=salt,
                    kind=ProbeKind.NAMED_VERSUS_ENACTED,
                    level=Level.ENACTED_OUTPUT,
                    construct=target.key,
                    sibling=target.sibling_key,
                    id_parts=(item_id, target_unit.unit_id, says_only.unit_id),
                    texts={
                        OptionTag.ENACTS_TARGET: target_unit.text,
                        OptionTag.NAMES_WITHOUT_ENACTING: says_only.text,
                    },
                    prompts={
                        Direction.ENACT: _enact_prompt(target_unit, target),
                        Direction.DESCRIBE: _describe_prompt(target_unit, target),
                    },
                    system_prompt=system_prompt,
                    source_units=(target_unit.unit_id, says_only.unit_id),
                    max_new_tokens=target_unit.max_new_tokens,
                    metadata={**metadata, "roles": [target_unit.item_role, says_only.item_role]},
                )
            )
    return Battery("enacted_strategy", Level.ENACTED_OUTPUT, tuple(probes))


def _enact_prompt(unit: CorpusUnit, construct: schema.Construct) -> tuple[str, str]:
    instruction = f'Write the response that gives the learner "{construct.display_name}".'
    return f"{_stem(unit)}\n{instruction}\nResponse:", instruction


def _describe_prompt(unit: CorpusUnit, construct: schema.Construct) -> tuple[str, str]:
    instruction = (
        f'Write the response that explains to the learner what "{construct.display_name}" is and why it '
        "works, without giving them one."
    )
    return f"{_stem(unit)}\n{instruction}\nResponse:", instruction


def build_batteries(
    ontology: schema.Ontology,
    bundle: CorpusBundle | None = None,
    *,
    salt: str = DEFAULT_SALT,
    conditions_per_pair: int = 2,
    allowed_units: Mapping[str, bool] | None = None,
    names: Sequence[str] = (),
) -> tuple[Battery, ...]:
    """Every battery that can be built from what was supplied, in a fixed order."""
    wanted = set(names or BATTERY_KINDS)
    unknown = sorted(wanted - set(BATTERY_KINDS))
    if unknown:
        raise BehaviorError(f"unknown batter(ies) {unknown}; known are {sorted(BATTERY_KINDS)}")
    batteries: list[Battery] = []
    if "declarative_knowledge" in wanted:
        batteries.append(declarative_battery(ontology, salt=salt))
    if "boundary_conditions" in wanted:
        batteries.append(boundary_battery(ontology, salt=salt, conditions_per_pair=conditions_per_pair))
    if "enacted_strategy" in wanted:
        if bundle is None:
            raise BehaviorError("the enacted battery needs a generated corpus; pass --corpus or drop the battery")
        batteries.append(enacted_battery(ontology, bundle, salt=salt, allowed_units=allowed_units))
    return tuple(batteries)


def all_probes(batteries: Sequence[Battery]) -> tuple[Probe, ...]:
    seen: dict[str, Probe] = {}
    for battery in batteries:
        for probe in battery.probes:
            if probe.probe_id in seen:
                raise BehaviorError(f"probe id {probe.probe_id!r} is used twice; the id salt is not separating probes")
            seen[probe.probe_id] = probe
    return tuple(seen.values())


# ------------------------------------------------------------------- runners


@dataclass(frozen=True, slots=True)
class TokenScore:
    """What a runner reports for one option: its log-probability and its length.

    Both are kept because the decision depends on which is used and the report
    states the answer under each. `clean_boundary` is false when the option's
    first token merged with the prompt's last one, which makes the two options
    slightly incomparable; those are counted rather than silently accepted.
    """

    logprob: float
    n_tokens: int
    clean_boundary: bool = True

    def __post_init__(self) -> None:
        if self.n_tokens < 1:
            raise BehaviorError("an option must tokenize to at least one token")

    @property
    def mean_logprob(self) -> float:
        return self.logprob / self.n_tokens

    def value(self, normalization: Normalization) -> float:
        return self.mean_logprob if normalization is Normalization.MEAN else self.logprob


class Runner(Protocol):
    """What this stage needs from a model. Implemented for real by `HuggingFaceRunner`.

    Narrow on purpose: the CPU tests supply a toy runner, and a scoring interface
    this small is one a stub can honestly satisfy.
    """

    def score(self, system_prompt: str, prompt: str, options: Sequence[str]) -> list[TokenScore]: ...

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str: ...


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


@dataclass
class HuggingFaceRunner:
    """Teacher-forced option scoring, and sampling for the optional naming arm.

    Dense OLMo-2 and OLMoE both work unchanged: this reads next-token
    log-probabilities and never touches a router. Loading goes through
    `checkpoint_delta.load_traced_model`, so a DeepSpeed training state is refused
    with its reason before a 7B `from_pretrained`, and a LoRA is applied the same
    way the tracer applies it.
    """

    model: str
    weights: Any
    tokenizer: Any
    device: Any
    torch: Any
    adapter: str | None = None

    @classmethod
    def load(
        cls, model: str = DEFAULT_MODEL, *, adapter: str = "", revision: str = "", device_map: str = ""
    ) -> HuggingFaceRunner:
        import torch  # noqa: PLC0415

        from projects.learning_science_semantic_atlas.checkpoint_delta import load_traced_model  # noqa: PLC0415

        weights, tokenizer = load_traced_model(model, revision=revision, adapter=adapter, device_map=device_map)
        device = next(weights.parameters()).device
        return cls(model, weights, tokenizer, device, torch, adapter or None)

    @property
    def description(self) -> dict[str, object]:
        config = getattr(self.weights, "config", None)
        experts = 0
        for name in ("num_experts", "num_local_experts", "n_routed_experts"):
            experts = experts or int(getattr(config, name, 0) or 0)
        return {
            "model": self.model,
            "adapter": self.adapter,
            "model_type": getattr(config, "model_type", None),
            "n_layers": getattr(config, "num_hidden_layers", None),
            "n_experts": experts or None,
            "architecture": "moe" if experts else "dense",
            "declared_arm": KNOWN_ARMS.get(self.model),
            "device": str(self.device),
        }

    def _prefix(self, system_prompt: str, prompt: str) -> str:
        """The text an option continues, rendered the way the corpus was generated."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system_prompt}\n\n{prompt}\n\n"

    def score(self, system_prompt: str, prompt: str, options: Sequence[str]) -> list[TokenScore]:
        prefix = self._prefix(system_prompt, prompt)
        # A rendered chat template already contains its special tokens as text,
        # so asking the tokenizer for more would prepend a second BOS to the
        # prefix and to nothing else.
        specials = not getattr(self.tokenizer, "chat_template", None)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=specials)["input_ids"]
        if len(prefix_ids) < 1:
            raise BehaviorError("the probe prompt tokenized to nothing")
        scores: list[TokenScore] = []
        for option in options:
            full_ids = self.tokenizer(prefix + option, add_special_tokens=specials)["input_ids"]
            clean = list(full_ids[: len(prefix_ids)]) == list(prefix_ids)
            start = len(prefix_ids) if clean else _common_prefix_length(prefix_ids, full_ids)
            if len(full_ids) <= start:
                raise BehaviorError(f"option {option[:60]!r} adds no tokens to the prompt")
            ids = self.torch.tensor([full_ids], device=self.device)
            with self.torch.no_grad():
                logits = self.weights(ids).logits.float()
            logprobs = self.torch.log_softmax(logits[0, :-1], dim=-1)
            chosen = logprobs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            scores.append(TokenScore(float(chosen[start - 1 :].sum()), len(full_ids) - start, clean_boundary=clean))
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
        """One continuation. Only the new tokens are decoded, so the prompt cannot return with them.

        Greedy by default, and seeded either way: at a temperature above zero the
        per-probe seed is what makes a requeued job regenerate the same text
        rather than quietly change the naming rate it already reported.
        """
        self.torch.manual_seed(seed)
        encoded = self.tokenizer(self._prefix(system_prompt, prompt), return_tensors="pt").to(self.device)
        sampling: dict[str, object] = {"do_sample": False}
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p}
        with self.torch.no_grad():
            out = self.weights.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **sampling,
            )
        return self.tokenizer.decode(out[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)


# ------------------------------------------------------- running, and resuming


def probe_seed(seed: int, probe_id: str) -> int:
    """A per-probe seed derived from the run seed, so a requeued job repeats itself.

    Same construction as `build_corpus.generate`: an item regenerated after a
    preemption comes back identically, which is the property that makes resume
    safe rather than merely fast.
    """
    return int(schema.stable_digest(str(seed), probe_id, size=4), 16)


def read_responses(path: str | Path, *, repair: bool = True) -> tuple[dict[str, dict[str, Any]], int]:
    """Stored responses by probe id, keeping the newest, tolerating a truncated tail.

    A preempted job dies mid-write, so the final line of an append-only file can
    be half an object. That one line is repaired; a broken line anywhere else is a
    corrupt file and raises, on the same reasoning as
    `build_corpus.read_generations` - skipping it would drop a measurement and
    quietly change a mean.
    """
    target = Path(path)
    if not target.exists():
        return {}, 0
    lines = target.read_text(encoding="utf-8").splitlines()
    rows: dict[str, dict[str, Any]] = {}
    dropped = 0
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            if number != len(lines):
                raise BehaviorError(f"{target}:{number}: not JSON, and not the final line: {exc}") from exc
            dropped += 1
            continue
        if not isinstance(row, dict) or not row.get("probe_id"):
            raise BehaviorError(f"{target}:{number}: a response row needs a probe_id")
        rows[str(row["probe_id"])] = row
    if dropped and repair:
        kept = [line for line in lines[:-1] if line.strip()]
        target.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return rows, dropped


def response_is_usable(row: Mapping[str, Any], probe: Probe) -> bool:
    """Whether a stored row still answers this probe.

    The digest covers the system prompt, the user prompt and both option texts,
    so an edited battery invalidates its own old responses instead of mixing two
    designs into one report.
    """
    if row.get("probe_sha") != probe.probe_sha:
        return False
    scores = row.get("scores")
    if not isinstance(scores, list) or len(scores) != len(probe.options):
        return False
    stored = {str(score.get("option_id")) for score in scores if isinstance(score, Mapping)}
    return stored == {option.option_id for option in probe.options}


def pending_probes(probes: Sequence[Probe], done: Mapping[str, Mapping[str, Any]]) -> tuple[Probe, ...]:
    """Probes still to score: absent, stale, or incompletely recorded."""
    return tuple(probe for probe in probes if not response_is_usable(done.get(probe.probe_id, {}), probe))


def run_probes(
    probes: Sequence[Probe],
    runner: Runner,
    *,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
    resume: bool = True,
    limit: int = 0,
) -> dict[str, object]:
    """Score every pending probe, appending one flushed line each.

    Append-and-flush rather than collect-and-write, because this runs on a
    preemptable partition: a job killed after nine tenths of a battery should
    lose the tenth and not the nine.
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    done, dropped = read_responses(target) if resume else ({}, 0)
    if not resume and target.exists():
        target.unlink()
    todo = pending_probes(probes, done)
    if limit > 0:
        todo = todo[:limit]

    written = 0
    merged_boundaries = 0
    with target.open("a", encoding="utf-8") as handle:
        for probe in todo:
            scores = runner.score(probe.system_prompt, probe.prompt, [option.text for option in probe.options])
            if len(scores) != len(probe.options):
                raise BehaviorError(
                    f"{probe.probe_id}: the runner returned {len(scores)} score(s) for {len(probe.options)} options"
                )
            merged_boundaries += sum(1 for score in scores if not score.clean_boundary)
            row = {
                "schema": RESPONSE_SCHEMA,
                "probe_id": probe.probe_id,
                "pair_id": probe.pair_id,
                "kind": probe.kind.value,
                "direction": probe.direction.value,
                "probe_sha": probe.probe_sha,
                "seed": seed,
                "probe_seed": probe_seed(seed, probe.probe_id),
                "scores": [
                    {
                        "option_id": option.option_id,
                        "tag": option.tag.value,
                        "keyed": option.keyed,
                        "logprob": score.logprob,
                        "n_tokens": score.n_tokens,
                        "clean_boundary": score.clean_boundary,
                    }
                    for option, score in zip(probe.options, scores)
                ],
            }
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    return {
        "scored": written,
        "already_present": len(done),
        "still_pending": len(pending_probes(probes, read_responses(target)[0])),
        "repaired_tail_lines": dropped,
        "merged_option_boundaries": merged_boundaries,
    }


def run_naming_arm(
    probes: Sequence[Probe],
    runner: Runner,
    ontology: schema.Ontology,
    *,
    out_path: str | Path,
    seed: int = DEFAULT_SEED,
    resume: bool = True,
    limit: int = 0,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> dict[str, object]:
    """Generate under each enact instruction and measure only whether the output names the construct.

    Naming is lexical and can be checked; enactment cannot be checked without a
    judge, and this stage does not have one. So this arm reports a naming rate and
    nothing else, and the report says in as many words that naming is not
    evidence of enactment - that being the confusion the whole atlas exists to
    separate.
    """
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    done, _ = read_responses(target) if resume else ({}, 0)
    if not resume and target.exists():
        target.unlink()
    todo = [
        probe
        for probe in probes
        if probe.kind is ProbeKind.NAMED_VERSUS_ENACTED
        and probe.direction is Direction.ENACT
        and done.get(probe.probe_id, {}).get("probe_sha") != probe.probe_sha
    ]
    if limit > 0:
        todo = todo[:limit]

    policy = "greedy" if temperature <= 0 else f"sample-t{temperature}-p{top_p}"
    named = 0
    written = 0
    with target.open("a", encoding="utf-8") as handle:
        for probe in todo:
            item_seed = probe_seed(seed, probe.probe_id)
            text = runner.generate(
                probe.system_prompt,
                probe.prompt,
                max_new_tokens=probe.max_new_tokens,
                seed=item_seed,
                temperature=temperature,
                top_p=top_p,
            )
            construct = ontology.get(probe.construct)
            hits = names_construct(text, construct)
            named += bool(hits)
            written += 1
            handle.write(
                json.dumps(
                    {
                        "schema": NAMING_SCHEMA,
                        "probe_id": probe.probe_id,
                        "probe_sha": probe.probe_sha,
                        "construct": probe.construct,
                        "seed": seed,
                        "probe_seed": item_seed,
                        "policy": policy,
                        "text": text,
                        "named_terms": list(hits),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "generated": written,
        "already_present": len(done),
        "policy": policy,
        "named_the_construct": named,
        "naming_rate": (named / written) if written else None,
    }


def names_construct(text: str, construct: schema.Construct) -> tuple[str, ...]:
    """Which of the construct's giveaway terms the text uses.

    The vocabulary is `build_corpus.trigger_words`, which reads it off the
    ontology's `surface_confounds` - the same list an atypical positive has to
    avoid. Using one definition of "names it" across the corpus builder and this
    stage means the says-only trap and the naming rate are counting the same
    thing.
    """
    lowered = text.lower()
    terms = set(build_corpus.trigger_words(construct))
    terms.add(construct.display_name.lower())
    return tuple(sorted(term for term in terms if term and term in lowered))


# ------------------------------------------------------------------ analysis


# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "more", "most", "no", "not", "of", "on", "or", "that", "the", "their",
    "them", "then", "there", "these", "this", "to", "under", "up", "was", "what", "when", "where", "which",
    "who", "why", "will", "with", "without", "you", "your",
})
# fmt: on


def content_words(text: str) -> frozenset[str]:
    return frozenset(word for word in schema.normalize_text(text).split() if word not in _STOPWORDS)


def lexical_overlap(instruction: str, option: str) -> float:
    """Jaccard overlap of content words. The behavioural analogue of a lexical control.

    An instruction naming a construct shares vocabulary with the option that
    instantiates it, so a model matching words alone can produce a positive
    paired shift. Measuring the overlap under the same crossing gives a number
    the score can be compared against instead of a caveat that cannot.
    """
    left, right = content_words(instruction), content_words(option)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def cluster_bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float] | None:
    """Percentile interval resampling whole pairs, because the two directions are not independent.

    The resampling unit is the pair, which for the enacted batteries is also the
    scenario: an artifact and its sibling share a stem, a reference and usually a
    learner turn, so treating them as two observations would overstate the
    evidence. Seeded, so a report is reproducible from the seed it records.
    """
    data = [float(value) for value in values]
    if not data:
        return None
    if len(data) == 1:
        return data[0], data[0], data[0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(data, k=len(data))) for _ in range(max(1, samples)))
    tail = (1.0 - confidence) / 2.0
    return (
        statistics.fmean(data),
        means[_quantile_index(len(means), tail)],
        means[_quantile_index(len(means), 1 - tail)],
    )


def _quantile_index(count: int, quantile: float) -> int:
    return max(0, min(count - 1, round(quantile * (count - 1))))


def sign_test_p(successes: int, trials: int) -> float | None:
    """Two-sided exact binomial p against a coin, over pairs rather than probes.

    Exact rather than normal-approximate because the pair counts here are in the
    tens, and `math.comb` makes the exact version free.
    """
    if trials <= 0:
        return None
    tail = min(successes, trials - successes)
    mass = sum(math.comb(trials, k) for k in range(tail + 1))
    return min(1.0, 2.0 * mass / (2**trials))


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One probe's scores, joined back to the probe that asked for them."""

    probe: Probe
    scores: Mapping[str, TokenScore]

    def value(self, option: Option, normalization: Normalization) -> float:
        return self.scores[option.option_id].value(normalization)

    def margin(self, normalization: Normalization) -> float:
        """First option minus second, in the order `CROSSING` fixes."""
        return self.value(self.probe.options[0], normalization) - self.value(self.probe.options[1], normalization)

    def keyed_wins(self, normalization: Normalization) -> bool | None:
        keyed = self.probe.keyed_option
        other = next(option for option in self.probe.options if option is not keyed)
        difference = self.value(keyed, normalization) - self.value(other, normalization)
        return None if difference == 0 else difference > 0

    @property
    def overlap_margin(self) -> float:
        return lexical_overlap(self.probe.instruction, self.probe.options[0].text) - lexical_overlap(
            self.probe.instruction, self.probe.options[1].text
        )


def join_responses(probes: Sequence[Probe], responses: Mapping[str, Mapping[str, Any]]) -> list[ProbeResult]:
    """Pair probes with their stored scores, skipping probes that were never answered."""
    results: list[ProbeResult] = []
    for probe in probes:
        row = responses.get(probe.probe_id)
        if row is None or not response_is_usable(row, probe):
            continue
        scores = {
            str(score["option_id"]): TokenScore(
                float(score["logprob"]), int(score["n_tokens"]), bool(score.get("clean_boundary", True))
            )
            for score in row["scores"]
        }
        results.append(ProbeResult(probe, scores))
    return results


def _keyed_rate(
    results: Sequence[ProbeResult], normalization: Normalization, direction: Direction | None = None
) -> float | None:
    """Share of probes whose keyed option is the likelier continuation. Ties are excluded."""
    decisions = [
        result.keyed_wins(normalization)
        for result in results
        if direction is None or result.probe.direction is direction
    ]
    return _mean([float(value) for value in decisions if value is not None])


def summarize_kind(
    results: Sequence[ProbeResult],
    kind: ProbeKind,
    *,
    normalization: Normalization = Normalization.MEAN,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """One probe kind: the confounded rate, the crossed shift, and the lexical control."""
    selected = [result for result in results if result.probe.kind is kind]
    directions = [direction for direction, _ in CROSSING[kind]]
    crossed = len(directions) == 2

    decided = [result.keyed_wins(normalization) for result in selected]
    other = Normalization.TOTAL if normalization is Normalization.MEAN else Normalization.MEAN
    agreements = [
        result.keyed_wins(normalization) == result.keyed_wins(other)
        for result in selected
        if result.keyed_wins(normalization) is not None and result.keyed_wins(other) is not None
    ]

    report: dict[str, object] = {
        "kind": kind.value,
        "crossed": crossed,
        "n_probes": len(selected),
        "n_pairs": len({result.probe.pair_id for result in selected}),
        "keyed_option_rate": _keyed_rate(selected, normalization),
        "n_undecided": sum(1 for value in decided if value is None),
        "by_direction": {direction.value: _keyed_rate(selected, normalization, direction) for direction in directions},
        "normalization_agreement": _mean([float(value) for value in agreements]),
        "merged_option_boundaries": sum(
            1 for result in selected for score in result.scores.values() if not score.clean_boundary
        ),
    }
    if not crossed:
        report["claim_limit"] = (
            "one direction only, so a constant preference for the keyed option is not removed; read this as a "
            "control on the crossed contrast rather than as evidence about the construct"
        )
        return report

    shifts: dict[str, float] = {}
    overlaps: dict[str, float] = {}
    for pair_id in sorted({result.probe.pair_id for result in selected}):
        by_direction = {result.probe.direction: result for result in selected if result.probe.pair_id == pair_id}
        if set(by_direction) != set(directions):
            continue
        first, second = (by_direction[direction] for direction in directions)
        shifts[pair_id] = first.margin(normalization) - second.margin(normalization)
        overlaps[pair_id] = first.overlap_margin - second.overlap_margin

    values = [shifts[pair_id] for pair_id in sorted(shifts)]
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    interval = cluster_bootstrap_ci(values, samples=bootstrap_samples, seed=seed)
    report["paired_shift"] = {
        "n_complete_pairs": len(values),
        "mean": _mean(values),
        "ci95": None if interval is None else [interval[1], interval[2]],
        "pairs_positive": positive,
        "pairs_negative": negative,
        # A pair whose two directions score identically is a model that did not
        # move, which is evidence of nothing and is dropped from the sign test
        # rather than counted against the shift. Counting ties as failures is how
        # a model with one flat preference gets a significant negative result.
        "pairs_tied": len(values) - positive - negative,
        "pairs_positive_rate": (positive / len(values)) if values else None,
        "sign_test_p": sign_test_p(positive, positive + negative),
        "reads_the_instruction": bool(interval is not None and interval[1] > 0),
    }
    report["lexical_overlap_shift"] = {
        "mean": _mean([overlaps[pair_id] for pair_id in sorted(overlaps)]),
        "favours_the_keyed_option": bool((_mean(list(overlaps.values())) or 0.0) > 0),
    }
    report["claim_limit"] = (
        "the paired shift cancels a constant preference between the two option texts, not a preference for "
        "whichever option shares more words with the instruction; compare it against lexical_overlap_shift"
    )
    return report


def enacted_design_audit(probes: Sequence[Probe], ontology: schema.Ontology) -> dict[str, object]:
    """Whether the corpus actually built the contrasts the enacted batteries claim.

    Two facts a reader needs before reading a rate. Which role supplied each
    pair's enacting option, because a scenario whose canonical positive was
    dropped falls back to its partial positive, and a partial positive is missing
    a required element by design. And whether a says-only artifact really uses its
    construct's vocabulary: the key says it does, but that is a property of a
    brief a generator can miss, and a trap rate over pairs whose says-only text
    contains no naming is not measuring a naming trap.

    Both are counted from text and ids alone, so this runs in a dry run: a corpus
    that cannot support these contrasts should say so before a GPU is asked for.
    """
    # Counted per pair rather than per probe: the two directions of one contrast
    # score the same artifact, and a count that said four where there are two
    # pairs would not line up with any other number in the report.
    roles: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for probe in probes:
        if probe.kind not in ENACTED_KINDS or probe.pair_id in seen:
            continue
        recorded = probe.metadata.get("roles")
        if not isinstance(recorded, Sequence) or isinstance(recorded, str) or not recorded:
            continue
        seen.add(probe.pair_id)
        counts = roles.setdefault(probe.kind.value, {})
        counts[str(recorded[0])] = counts.get(str(recorded[0]), 0) + 1

    enact = [
        probe
        for probe in probes
        if probe.kind is ProbeKind.NAMED_VERSUS_ENACTED and probe.direction is Direction.ENACT
    ]
    named = 0
    vocabulary_free = 0
    for probe in enact:
        construct = ontology.get(probe.construct)
        named += bool(names_construct(probe.options[1].text, construct))
        vocabulary_free += not names_construct(probe.options[0].text, construct)
    total = len(enact)
    return {
        "enacting_option_roles": {kind: dict(sorted(counts.items())) for kind, counts in sorted(roles.items())},
        "n_pairs": total,
        "names_only_uses_the_vocabulary": named,
        "names_only_vocabulary_rate": (named / total) if total else None,
        "enacting_option_is_vocabulary_free": vocabulary_free,
        "enacting_option_vocabulary_free_rate": (vocabulary_free / total) if total else None,
        "claim_limit": (
            "naming is counted with build_corpus.trigger_words, so the rate is an upper bound. Where the "
            "names-only artifact carries none of the construct's vocabulary, its generator did not build the "
            "says-only cell the brief asked for and that pair has no naming for the model to be misled by; a "
            "trap rate over those pairs is measuring artifact difference rather than the says-only trap. Pairs "
            "listed under positive_partial rest on an artifact designed to be missing a required element."
        ),
    }


def says_only_trap(
    results: Sequence[ProbeResult], *, normalization: Normalization = Normalization.MEAN
) -> dict[str, object]:
    """How often naming the principle beats performing it when performance was asked for.

    The interpretable number of the whole stage, and the reason the corpus carries
    a says-only variant per construct: a model that has learned the vocabulary
    rather than the move prefers the artifact that discusses the principle when
    the request was to enact it.
    """
    enact = [
        result
        for result in results
        if result.probe.kind is ProbeKind.NAMED_VERSUS_ENACTED and result.probe.direction is Direction.ENACT
    ]
    trapped = 0
    decided = 0
    for result in enact:
        wins = result.keyed_wins(normalization)
        if wins is None:
            continue
        decided += 1
        trapped += not wins
    return {
        "n_decided": decided,
        "n_undecided": len(enact) - decided,
        "says_only_preferred": trapped,
        "says_only_trap_rate": (trapped / decided) if decided else None,
        "claim_limit": (
            "measured on artifacts whose naming and enactment come from the corpus design and have not been "
            "confirmed by a rater, so a trap rate is evidence about the model only as far as that design holds"
        ),
    }


# -------------------------------------------------------------------- report


def coverage(
    batteries: Sequence[Battery], results: Sequence[ProbeResult], ontology: schema.Ontology
) -> dict[str, object]:
    answered = {result.probe.probe_id for result in results}
    probes = all_probes(batteries)
    per_construct: dict[str, int] = {construct.key: 0 for construct in ontology}
    for probe in probes:
        for key in (probe.construct, probe.sibling):
            if key in per_construct:
                per_construct[key] += 1
    return {
        "n_probes": len(probes),
        "n_answered": len(answered),
        "n_unanswered": len(probes) - len(answered),
        "by_kind": {
            kind.value: sum(1 for probe in probes if probe.kind is kind)
            for kind in ProbeKind
            if any(probe.kind is kind for probe in probes)
        },
        "constructs_with_no_probe": sorted(key for key, count in per_construct.items() if count == 0),
        "probes_per_construct": dict(sorted(per_construct.items())),
    }


def calibration_block(*, keying: str, n_verified: int | None) -> dict[str, object]:
    """What the correctness of this report rests on. Never a human label.

    Recorded as a block rather than a sentence so a downstream reader can filter
    on it: nothing produced here may be pooled with a confirmatory estimate, and
    the reliability fields are null because they were not measured, not because
    they were good.
    """
    return {
        "label_source": LABEL_SOURCE,
        "status": CALIBRATION_STATUS,
        "human_labels_used": False,
        "keying": keying,
        "n_units_label_verified": n_verified,
        "inter_rater_reliability": None,
        "intra_rater_retest_reliability": None,
        "confirmatory": False,
        "note": (
            "Correct answers come from the ontology's own text and from the corpus design's withheld key. No "
            "rater has confirmed either, no reliability was estimated, and LABELING_GUIDE.md's gates are not "
            "attempted here. Every number is exploratory and agent-grounded."
        ),
    }


def required_reporting(ontology: schema.Ontology, filled: Mapping[str, object]) -> dict[str, object]:
    """The ontology's `required_reporting_fields`, answered or explicitly null.

    The ontology states what any result about a construct has to report. A stage
    that cannot fill a field says so with a null rather than omitting the key,
    because an absent field reads as an oversight and a null reads as a limit.
    """
    return {name: filled.get(name) for name in ontology.required_reporting_fields}


CLAIM_LIMITS: tuple[str, ...] = (
    "Forced choice between texts the design supplied. The model never wrote an artifact that was scored, so "
    "nothing here is evidence at the learner_enactment or learning_outcome rungs of the claim ladder.",
    "No human labels exist for this stage. Correctness is keyed to the ontology's prose and to the corpus's "
    "withheld design key, both agent-grounded and unvalidated, so every result is exploratory.",
    "Option likelihoods depend on tokenization and length. The primary decision is the per-token mean and the "
    "total-log-probability decision is reported beside it; a contrast the two disagree about is "
    "normalization-dependent rather than a result.",
    "An instruction naming a construct shares vocabulary with the option that instantiates it, so a positive "
    "paired shift can be lexical matching. Compare every shift against its lexical_overlap_shift.",
    "The enacted batteries score artifacts a model generated. Where the scored model also generated them, the "
    "model is being asked to prefer its own text; the report records both models for that comparison.",
    "OLMoE and dense OLMo-2 are different model families, so agreement between the two arms is replication and "
    "disagreement is non-replication. Neither is a claim about mixture-of-experts architectures.",
    "Boundary probes use the ontology's moderator statements. `escalation:` entries are excluded because they "
    "describe what a higher claim level would require rather than a property of the principle.",
    "The named-versus-enacted contrast uses only variants that share one scenario. The pool's vocabulary-free "
    "atypical positives sit in scenarios of their own and are not part of it.",
    "This stage reads the withheld key and briefs, so it belongs after labelling. Running it earlier would put "
    "designed answers in front of anything that is still being labelled blind.",
)


def build_report(
    batteries: Sequence[Battery],
    results: Sequence[ProbeResult],
    ontology: schema.Ontology,
    *,
    ontology_path: str,
    bundle: CorpusBundle | None,
    runner_description: Mapping[str, object],
    normalization: Normalization = Normalization.MEAN,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    salt: str = DEFAULT_SALT,
    keying: str = "designed_unverified",
    n_verified: int | None = None,
    naming: Mapping[str, object] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """The whole report: what ran, what it measured, and what it may not be read as."""
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "dry_run": dry_run,
        "model": dict(runner_description),
        "ontology": {
            "path": ontology_path,
            "construct_version": ontology.construct_version,
            "status": ontology.status,
            "n_constructs": len(ontology),
        },
        "settings": {
            "salt": salt,
            "seed": seed,
            "normalization": normalization.value,
            "bootstrap_samples": bootstrap_samples,
        },
        "batteries": {
            battery.name: {
                "level": battery.level.value,
                "n_probes": len(battery.probes),
                "n_pairs": len(battery.pair_ids),
            }
            for battery in batteries
        },
        "coverage": coverage(batteries, results, ontology),
        "calibration": calibration_block(keying=keying, n_verified=n_verified),
        "claim_limits": list(CLAIM_LIMITS),
    }

    if bundle is not None:
        scored_model = runner_description.get("model")
        report["corpus"] = {
            "path": str(bundle.path),
            "n_units": len(bundle.units),
            "construct_version": bundle.manifest.get("construct_version"),
            "salt": bundle.manifest.get("salt"),
            "source_models": list(bundle.source_models),
            "n_scaffolded_units": len(bundle.scaffolded),
            "scored_model_generated_the_corpus": bool(scored_model and scored_model in bundle.source_models),
        }

    for battery in batteries:
        measured = {
            kind.value: summarize_kind(
                results, kind, normalization=normalization, bootstrap_samples=bootstrap_samples, seed=seed
            )
            for kind in BATTERY_KINDS[battery.name]
            if battery.of_kind(kind)
        }
        report[battery.name] = {"level": battery.level.value, "kinds": measured}
    probes = all_probes(batteries)
    if any(probe.kind in ENACTED_KINDS for probe in probes):
        report["named_versus_enacted_summary"] = {
            **says_only_trap(results, normalization=normalization),
            "design_audit": enacted_design_audit(probes, ontology),
        }
    if naming is not None:
        report["naming_arm"] = {
            **dict(naming),
            "claim_limit": (
                "a naming rate counts the construct's vocabulary in generated text, and some of that "
                "vocabulary is ordinary tutor language, so the rate is an upper bound on naming. It is not "
                "evidence of enactment at all, and the two coming apart is what this project measures."
            ),
        }
    report["required_reporting_fields"] = required_reporting(
        ontology,
        {
            "construct_version": ontology.construct_version,
            "evidence_level": sorted({battery.level.value for battery in batteries}),
            "unit_of_analysis": "forced_choice_probe_pair",
            "required_context_present": "supplied by the probe; crossed in the required_context kind",
            "applicability_judgment": "probed as the two refusals of schema.Applicability, not assumed",
            "label_source": LABEL_SOURCE,
            "inter_rater_reliability": None,
            "intra_rater_retest_reliability": None,
            "says_only_false_positive_rate": (
                report.get("named_versus_enacted_summary", {}).get("says_only_trap_rate")
                if isinstance(report.get("named_versus_enacted_summary"), Mapping)
                else None
            ),
            "hard_negative_false_positive_rate": _hard_negative_rate(report),
            "natural_or_synthetic": "model_generated_from_designed_briefs",
            "known_surface_confounds_measured": ["lexical_overlap", "length_normalization", "option_preference"],
            "reward_eligibility_status": "measurement_only",
            "evidence_limit": CLAIM_LIMITS[0],
        },
    )
    return report


def _hard_negative_rate(report: Mapping[str, object]) -> float | None:
    """One minus the enacted-strategy keyed rate in the target direction, when it exists."""
    battery = report.get("enacted_strategy")
    if not isinstance(battery, Mapping):
        return None
    kinds = battery.get("kinds")
    if not isinstance(kinds, Mapping):
        return None
    strategy = kinds.get(ProbeKind.ENACTED_STRATEGY.value)
    if not isinstance(strategy, Mapping):
        return None
    by_direction = strategy.get("by_direction")
    if not isinstance(by_direction, Mapping):
        return None
    rate = by_direction.get(Direction.TARGET.value)
    return None if rate is None else 1.0 - float(rate)


# ------------------------------------------------------------------------ cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score declarative-knowledge and enacted-strategy batteries over the semantic atlas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ontology", default=None, help="ontology.yaml; the packaged one by default")
    parser.add_argument(
        "--corpus", default=None, help="a build_corpus output directory; needed by the enacted battery"
    )
    parser.add_argument("--out", required=True, help="output directory for probes, responses and the report")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"the dense replication is {DENSE_MODEL}")
    parser.add_argument("--adapter", default="", help="a PEFT adapter directory or zip applied on top of --model")
    parser.add_argument("--revision", default="")
    parser.add_argument("--device-map", default="", help="for example 'auto'; empty puts the model on one device")
    parser.add_argument(
        "--batteries",
        nargs="+",
        default=[],
        choices=sorted(BATTERY_KINDS),
        help="default is every battery the inputs allow",
    )
    parser.add_argument("--labels", nargs="+", default=[], help="restrict enacted probes to label-verified units")
    parser.add_argument("--salt", default=DEFAULT_SALT, help="probe id salt; a new salt is a new, non-colliding set")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conditions-per-pair", type=int, default=2, help="boundary statements per sibling pair")
    parser.add_argument(
        "--normalization",
        default=Normalization.MEAN.value,
        choices=[member.value for member in Normalization],
        help="how unequal option lengths are compared; the other reading is reported either way",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--limit", type=int, default=0, help="score at most this many pending probes")
    parser.add_argument("--generate", action="store_true", help="also run the naming arm, which writes text")
    parser.add_argument("--temperature", type=float, default=0.0, help="naming arm only; 0 is greedy")
    parser.add_argument("--top-p", type=float, default=0.9, help="naming arm only, above temperature 0")
    parser.add_argument("--no-resume", action="store_true", help="discard an existing responses file and start again")
    parser.add_argument("--dry-run", action="store_true", help="build and write the probes, load no weights")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ontology_path = Path(args.ontology) if args.ontology else schema.default_ontology_path()
    ontology = schema.load_ontology(ontology_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keying, verified, allowed = "designed_unverified", None, None
    # A refused corpus, an unreadable label file and an impossible battery are all
    # things the caller passed in, so they exit with their reason rather than a
    # traceback. Every one of those messages already names the file and the fix.
    try:
        bundle = load_bundle(args.corpus) if args.corpus else None
        if args.labels:
            if bundle is None:
                raise SystemExit("--labels describes corpus units, so it needs --corpus")
            allowed = verified_units(args.labels, bundle)
            verified = sum(1 for agrees in allowed.values() if agrees)
            keying = "label_verified"
        batteries = build_batteries(
            ontology,
            bundle,
            salt=args.salt,
            conditions_per_pair=args.conditions_per_pair,
            allowed_units=allowed,
            names=args.batteries or ([] if args.corpus else ["declarative_knowledge", "boundary_conditions"]),
        )
    except BehaviorError as exc:
        raise SystemExit(str(exc)) from exc
    probes = all_probes(batteries)
    if not probes:
        raise SystemExit("no probes were built; check --batteries and the corpus directory")
    schema.write_jsonl((probe.as_row() for probe in probes), out / PROBES_FILE)
    print(
        f"built {len(probes)} probes in {len(batteries)} batter(ies) from {ontology_path} "
        f"at construct_version {ontology.construct_version}"
    )
    for battery in batteries:
        print(f"  {battery.name}: {len(battery.probes)} probes over {len(battery.pair_ids)} pairs")

    normalization = Normalization(args.normalization)
    shared = {
        "ontology_path": str(ontology_path),
        "bundle": bundle,
        "normalization": normalization,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "salt": args.salt,
        "keying": keying,
        "n_verified": verified,
    }

    if args.dry_run:
        report = build_report(
            batteries, [], ontology, runner_description={"model": args.model, "loaded": False}, dry_run=True, **shared
        )
        _write_report(report, out / REPORT_FILE)
        print(f"dry run: wrote {out / PROBES_FILE} and {out / REPORT_FILE}; no weights were loaded")
        return 0

    runner = HuggingFaceRunner.load(
        args.model, adapter=args.adapter, revision=args.revision, device_map=args.device_map
    )
    print(f"loaded {json.dumps(runner.description, sort_keys=True)}")
    progress = run_probes(
        probes, runner, out_path=out / RESPONSES_FILE, seed=args.seed, resume=not args.no_resume, limit=args.limit
    )
    print(f"scored {progress['scored']} probes; {progress}")

    naming = None
    if args.generate:
        naming = run_naming_arm(
            probes,
            runner,
            ontology,
            out_path=out / NAMING_FILE,
            seed=args.seed,
            resume=not args.no_resume,
            limit=args.limit,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"naming arm: {naming}")

    responses, dropped = read_responses(out / RESPONSES_FILE)
    if dropped:
        print(f"repaired {dropped} truncated line(s) at the end of {out / RESPONSES_FILE}")
    results = join_responses(probes, responses)
    report = build_report(batteries, results, ontology, runner_description=runner.description, naming=naming, **shared)
    _write_report(report, out / REPORT_FILE)
    print(f"wrote {out / REPORT_FILE} over {len(results)} of {len(probes)} probes")
    if progress["still_pending"]:
        print(
            f"WARNING: {progress['still_pending']} probe(s) are still unscored, so every mean in the report is "
            "over a partial battery. Rerun to resume.",
            file=sys.stderr,
        )
    return 0


def _write_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
