"""The atlas's typed vocabulary: the ontology as objects, and the two record types.

Everything downstream of `ontology.yaml` reads it through here, so the file is
parsed and checked once and a construct key that does not exist fails at load
rather than as a missing dict key three modules later.

WHAT THIS MODULE IS FOR. `ontology.yaml` is prose meant for a person, and prose
is where a labelling protocol quietly drifts from the code that implements it.
This module turns the parts a program depends on into frozen dataclasses and
closed enumerations, and then asserts that the two agree: if someone adds a
ninth `output_types` entry to the YAML without adding it to `OutputFormat`,
`load_ontology` refuses the file and names the value. That check exists because
the ontology's own §10 says to freeze the file before labelling begins, and a
freeze nobody can verify is a request rather than a guarantee.

THE 2x2 IS THE POINT OF THE PROJECT, so it is a type rather than a convention.
`NamesEnacts` carries two independent booleans - does the artifact NAME the
principle, does the artifact ENACT it - and refuses to let either stand in for
the other. The four cells all exist and all get built:

                     enacts = False          enacts = True
    names = False    neither                 enacts_only
    names = True     names_only              names_and_enacts
                     ^ the says-only trap

`names_only` is the case the atlas exists to detect: "research shows quizzing
beats rereading, so quiz yourself" is correct declarative content about
retrieval practice containing no retrieval practice. Every construct in the
ontology is scored at `enacted_output`, so `expected_presence` is `enacts` and
never `names`. A pipeline that collapses the two cannot express the distinction
it is being asked to measure, which is why this is a dataclass with an
invariant and not a pair of loose kwargs.

`names_and_enacts` is easy to forget and load-bearing. Without it, naming is
perfectly anti-correlated with enacting across the corpus, and a probe reaches
the right answer by detecting topic and inverting - the same failure as reading
topic for structure, wearing the opposite sign. See `build_corpus`, which builds
that cell on purpose and tests that both marginals stay mixed.

NULLS MEAN THINGS HERE. `LabelRecord` enforces LABELING_GUIDE.md §9 exactly:
`presence` is null iff the construct did not apply, `fidelity` is null unless
presence is true, and a record breaking either is rejected rather than repaired.
Imputing a zero would turn "the question did not apply" into "the artifact did
it badly", which are different observations that a mean cannot separate again.

BOOLEANS ARE NOT SMALL INTEGERS. `bool` subclasses `int` in Python, so a bare
`isinstance(value, int)` accepts `True` for a 0-3 ordinal and stores a fidelity
of 1 that nobody typed. Both ordinal fields refuse `bool` before they check
range, and both boolean fields refuse `0`/`1`, matching `labeling.parse_label`.

YAML WITHOUT PYYAML. PyYAML is used when it is importable and a bundled parser
for the subset this ontology actually uses runs when it is not, so the schema
can be loaded in a stripped environment. The fallback is deliberately narrow: it
implements block mappings, block sequences, flow sequences, folded and literal
block scalars, and refuses anything else by name rather than guessing. Test
`test_schema.py::test_fallback_parser_matches_pyyaml` parses the real
`ontology.yaml` both ways and compares, so the fallback is checked against the
file it exists to read rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from types import MappingProxyType

try:
    import yaml

    PYYAML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where PyYAML is absent
    yaml = None
    PYYAML_AVAILABLE = False

SCHEMA_VERSION = 1
ONTOLOGY_FILENAME = "ontology.yaml"

FIDELITY_RANGE = (0, 3)
QUALITY_RANGE = (1, 3)


class OntologyError(ValueError):
    """`ontology.yaml` is missing something, or says something the code cannot honour."""


class RecordError(ValueError):
    """An artifact or label record broke an invariant. Carries the field and the value."""


class YamlSubsetError(ValueError):
    """The bundled parser met YAML it does not implement. Install PyYAML."""


# --------------------------------------------------------------- yaml fallback
#
# A parser for the subset `ontology.yaml` uses, so that `schema` imports in an
# environment with no third-party packages at all. It is not a YAML
# implementation and does not try to be; every construct it has not implemented
# raises `YamlSubsetError` naming the line, because a loader that guesses at
# syntax it does not understand produces a plausible ontology with a silently
# missing field, and that is worse than not loading.

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*:(\s|$)")
_BLOCK_SCALAR_RE = re.compile(r"^[|>]([-+]?)$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "0": "\0"}


def _unescape(text: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            out.append(_ESCAPES.get(nxt, "\\" + nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _split_quoted(text: str, line_no: int) -> tuple[str, str]:
    """A leading quoted scalar and whatever follows it."""
    quote = text[0]
    index = 1
    while index < len(text):
        char = text[index]
        if quote == '"' and char == "\\":
            index += 2
            continue
        if char == quote:
            if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            body = text[1:index]
            return (_unescape(body) if quote == '"' else body.replace("''", "'")), text[index + 1 :]
        index += 1
    raise YamlSubsetError(f"line {line_no}: unterminated {quote} quoted scalar: {text!r}")


def _strip_comment(text: str) -> str:
    """Drop a trailing ``# comment``. YAML needs whitespace before the ``#``."""
    if text.startswith("#"):
        return ""
    cut = text.find(" #")
    return text[:cut] if cut >= 0 else text


def _scalar(token: str, line_no: int) -> object:
    text = token.strip()
    if text[:1] in ('"', "'"):
        value, rest = _split_quoted(text, line_no)
        trailing = _strip_comment(rest.strip()).strip()
        if trailing:
            raise YamlSubsetError(f"line {line_no}: trailing {trailing!r} after a quoted scalar")
        return value
    text = _strip_comment(text).strip()
    if text in ("", "null", "~"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    if text[0] in "&*!%@`":
        raise YamlSubsetError(
            f"line {line_no}: {text[0]!r} (anchors, aliases, tags) is not implemented; install PyYAML"
        )
    return text


def _split_flow(inner: str, line_no: int) -> list[str]:
    """Split a flow sequence's body on commas that are not inside quotes."""
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in inner:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == ",":
            parts.append("".join(current))
            current = []
            continue
        if char in "[]{}":
            raise YamlSubsetError(f"line {line_no}: nested flow collections are not implemented; install PyYAML")
        current.append(char)
    if quote:
        raise YamlSubsetError(f"line {line_no}: unterminated quote in flow sequence")
    parts.append("".join(current))
    return parts


@dataclass
class _Cursor:
    """A line cursor that knows which lines YAML would ignore."""

    lines: list[str]
    index: int = 0

    def _significant(self, line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and not stripped.startswith("#") and stripped not in ("---", "...")

    def peek(self) -> tuple[int, str, int] | None:
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if self._significant(line):
                indent = len(line) - len(line.lstrip(" "))
                return indent, _strip_comment(line.strip()).rstrip() or line.strip(), self.index + 1
            self.index += 1
        return None

    def take(self) -> tuple[int, str, int]:
        item = self.peek()
        if item is None:  # pragma: no cover - callers peek first
            raise YamlSubsetError("unexpected end of file")
        self.index += 1
        return item


def _parse_block_scalar(cursor: _Cursor, parent_indent: int, indicator: str) -> str:
    match = _BLOCK_SCALAR_RE.match(indicator)
    if not match:
        raise YamlSubsetError(f"block scalar {indicator!r} is not implemented; install PyYAML")
    style, chomp = indicator[0], match.group(1)

    raw: list[str] = []
    while cursor.index < len(cursor.lines):
        line = cursor.lines[cursor.index]
        if not line.strip():
            raw.append("")
            cursor.index += 1
            continue
        if len(line) - len(line.lstrip(" ")) <= parent_indent:
            break
        raw.append(line)
        cursor.index += 1
    while raw and not raw[-1]:
        raw.pop()
    if not raw:
        return "" if chomp == "-" else "\n"

    block_indent = min(len(line) - len(line.lstrip(" ")) for line in raw if line.strip())
    body = [line[block_indent:].rstrip() if line.strip() else "" for line in raw]

    if style == "|":
        text = "\n".join(body)
    else:
        groups: list[list[str]] = [[]]
        for line in body:
            if line:
                groups[-1].append(line)
            else:
                groups.append([])
        rendered = []
        for group in groups:
            if not group:
                rendered.append("")
            elif any(line.startswith(" ") for line in group):
                rendered.append("\n".join(group))  # more-indented lines stay literal
            else:
                rendered.append(" ".join(group))
        text = "\n".join(rendered)

    if chomp == "-":
        return text.rstrip("\n")
    if chomp == "+":
        return text + "\n"
    return text.rstrip("\n") + "\n"


def _parse_flow_sequence(text: str, line_no: int) -> list[object]:
    if not text.endswith("]"):
        raise YamlSubsetError(f"line {line_no}: a flow sequence must close on its own line")
    inner = text[1:-1].strip()
    if not inner:
        return []
    return [_scalar(part, line_no) for part in _split_flow(inner, line_no)]


def _split_key(content: str, line_no: int) -> tuple[str, str]:
    key, separator, rest = content.partition(":")
    if not separator:
        raise YamlSubsetError(f"line {line_no}: expected 'key: value', got {content!r}")
    return key.strip(), rest.strip()


def _parse_value(cursor: _Cursor, parent_indent: int, rest: str, line_no: int) -> object:
    if _BLOCK_SCALAR_RE.match(rest):
        return _parse_block_scalar(cursor, parent_indent, rest)
    if rest.startswith("["):
        return _parse_flow_sequence(rest, line_no)
    if rest.startswith("{"):
        raise YamlSubsetError(f"line {line_no}: flow mappings are not implemented; install PyYAML")
    if rest:
        return _scalar(rest, line_no)

    nxt = cursor.peek()
    if nxt is None:
        return None
    indent, content, _ = nxt
    if indent > parent_indent:
        return _parse_block(cursor, indent)
    if indent == parent_indent and (content == "-" or content.startswith("- ")):
        return _parse_sequence(cursor, indent)
    return None


def _parse_mapping(cursor: _Cursor, indent: int) -> dict[str, object]:
    out: dict[str, object] = {}
    while True:
        nxt = cursor.peek()
        if nxt is None:
            return out
        line_indent, content, line_no = nxt
        if line_indent < indent:
            return out
        if line_indent > indent:
            raise YamlSubsetError(f"line {line_no}: unexpected indent {line_indent}, expected {indent}: {content!r}")
        if content == "-" or content.startswith("- "):
            return out
        key, rest = _split_key(content, line_no)
        cursor.take()
        if key in out:
            raise YamlSubsetError(f"line {line_no}: duplicate key {key!r}")
        out[key] = _parse_value(cursor, indent, rest, line_no)


def _parse_sequence(cursor: _Cursor, indent: int) -> list[object]:
    out: list[object] = []
    while True:
        nxt = cursor.peek()
        if nxt is None:
            return out
        line_indent, content, line_no = nxt
        if line_indent < indent or not (content == "-" or content.startswith("- ")):
            return out
        if line_indent > indent:
            raise YamlSubsetError(f"line {line_no}: unexpected indent {line_indent}, expected {indent}")
        cursor.take()
        after = content[1:]
        body = after.lstrip(" ")
        body_indent = indent + 1 + (len(after) - len(body))

        if not body:
            nested = cursor.peek()
            out.append(_parse_block(cursor, nested[0]) if nested and nested[0] > indent else None)
            continue
        if _KEY_RE.match(body):
            key, rest = _split_key(body, line_no)
            item: dict[str, object] = {key: _parse_value(cursor, body_indent, rest, line_no)}
            for extra_key, extra_value in _parse_mapping(cursor, body_indent).items():
                if extra_key in item:
                    raise YamlSubsetError(f"line {line_no}: duplicate key {extra_key!r}")
                item[extra_key] = extra_value
            out.append(item)
            continue
        out.append(_parse_value(cursor, body_indent, body, line_no))


def _parse_block(cursor: _Cursor, indent: int) -> object:
    nxt = cursor.peek()
    if nxt is None:
        return None
    _, content, _ = nxt
    if content == "-" or content.startswith("- "):
        return _parse_sequence(cursor, indent)
    return _parse_mapping(cursor, indent)


def parse_yaml_subset(text: str) -> object:
    """Parse the YAML subset `ontology.yaml` uses. Raises on anything else."""
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if "\t" in line:
            raise YamlSubsetError(f"line {number}: tab in indentation is illegal in YAML")
    cursor = _Cursor(lines)
    first = cursor.peek()
    if first is None:
        return None
    value = _parse_block(cursor, first[0])
    trailing = cursor.peek()
    if trailing is not None:
        raise YamlSubsetError(f"line {trailing[2]}: unparsed trailing content {trailing[1]!r}")
    return value


def parse_yaml(text: str, *, prefer_pyyaml: bool = True) -> object:
    """PyYAML when it is importable, the bundled subset parser otherwise.

    `prefer_pyyaml=False` forces the fallback, which is how the test suite
    checks the two against each other on the real ontology.
    """
    if prefer_pyyaml and PYYAML_AVAILABLE:
        return yaml.safe_load(text)
    return parse_yaml_subset(text)


def yaml_backend(*, prefer_pyyaml: bool = True) -> str:
    return "pyyaml" if prefer_pyyaml and PYYAML_AVAILABLE else "bundled-subset"


# -------------------------------------------------------------- vocabularies
#
# Closed sets. Each of the first six mirrors a block of `ontology.yaml` and is
# checked against it at load time, so the file and the code cannot drift apart
# without one of them failing loudly.


class EvidenceLevel(str, Enum):
    """The claim ladder. Which of four different things a label is evidence for."""

    DECLARATIVE_KNOWLEDGE = "declarative_knowledge"
    ENACTED_OUTPUT = "enacted_output"
    LEARNER_ENACTMENT = "learner_enactment"
    LEARNING_OUTCOME = "learning_outcome"


class EvidenceGrade(str, Enum):
    """How good the evidence for the PRINCIPLE is. Says nothing about labelling it."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Family(str, Enum):
    PRACTICE_SCHEDULING_AND_GENERATION = "practice_scheduling_and_generation"
    EXAMPLE_BASED_AND_PROBLEM_FIRST = "example_based_and_problem_first"
    GENERATIVE_EXPLANATION = "generative_explanation"
    FEEDBACK_AND_ASSISTANCE = "feedback_and_assistance"
    METACOGNITIVE_AND_MOTIVATIONAL = "metacognitive_and_motivational"


class OutputFormat(str, Enum):
    """The kind of artifact. A hard constraint, not a style: a single turn cannot hold a schedule."""

    TUTOR_TURN = "tutor_turn"
    DIALOGUE_EPISODE = "dialogue_episode"
    EXPLANATION_TEXT = "explanation_text"
    WORKED_SOLUTION = "worked_solution"
    PRACTICE_ITEM_SET = "practice_item_set"
    STUDY_SCHEDULE = "study_schedule"
    ASSESSMENT_ITEM = "assessment_item"
    CURRICULUM_PLAN = "curriculum_plan"


class LabelabilityBand(str, Enum):
    """How reliably two careful raters can answer the question, predicted in advance."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    CONTEXT_DEPENDENT = "context_dependent"


class RewardStatus(str, Enum):
    ELIGIBLE_WITH_GUARD = "eligible_with_guard"
    NOT_ELIGIBLE_UNRELIABLE = "not_eligible_unreliable"
    NOT_ELIGIBLE_WRONG_LEVEL = "not_eligible_wrong_level"
    NOT_ELIGIBLE_DEGENERATE = "not_eligible_degenerate"
    MEASUREMENT_ONLY = "measurement_only"


# The remaining vocabularies are corpus-design choices and have no counterpart
# in the ontology, which describes constructs rather than the pool built to
# measure them.


class Domain(str, Enum):
    """Subject matter. A balance axis, so that a construct is not confounded with a topic."""

    ALGEBRA = "algebra"
    CHEMISTRY = "chemistry"
    CELL_BIOLOGY = "cell_biology"
    PROGRAMMING = "programming"
    STATISTICS = "statistics"
    HISTORY = "history"


class LearnerState(str, Enum):
    """What the scenario DECLARES about the learner, never what a labeler infers.

    `expertise_reversal_adaptation` is unlabelable unless expertise is stated as
    a field, so this axis is a required input to that construct rather than
    colour: LABELING_GUIDE.md §3.1 makes inferring it from vocabulary or tone a
    protocol violation.
    """

    NO_ATTEMPT_YET = "no_attempt_yet"
    NOVICE_DECLARED = "novice_declared"
    PARTIAL_ATTEMPT = "partial_attempt"
    MISCONCEPTION_STATED = "misconception_stated"
    ADVANCED_DECLARED = "advanced_declared"


class PromptMode(str, Enum):
    """How the artifact was asked for. Carried so elicitation cannot be confounded with construct."""

    DIRECT_INSTRUCTION_REQUEST = "direct_instruction_request"
    TUTOR_ROLEPLAY = "tutor_roleplay"
    LESSON_AUTHORING = "lesson_authoring"
    FEEDBACK_REQUEST = "feedback_request"


class ItemRole(str, Enum):
    """Why an item is in the pool. Withheld from raters; see LABELING_GUIDE.md §5.

    The first six roles are written against a target construct. The last two are
    the pool's other two blocks, which have no designed target: a debunked
    control is in the pool to be scored 0 by every construct it is shown to, and
    an open-pool item is a natural artifact whose only job is to supply the base
    rates a targeted candidate set cannot (LABELING_GUIDE.md §2.3).
    """

    POSITIVE_CANONICAL = "positive_canonical"
    POSITIVE_ATYPICAL = "positive_atypical"
    POSITIVE_PARTIAL = "positive_partial"
    NAMED_AND_ENACTED = "named_and_enacted"
    SIBLING_HARD_NEGATIVE = "sibling_hard_negative"
    SAYS_ONLY = "says_only"
    DEBUNKED_CONTROL = "debunked_control"
    OPEN_POOL = "open_pool"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Applicability(str, Enum):
    """LABELING_GUIDE.md §3.1. The two refusals are different observations and stay apart."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE_MISSING_CONTEXT = "not_applicable_missing_context"
    NOT_APPLICABLE_WRONG_ARTIFACT = "not_applicable_wrong_artifact"


class Flag(str, Enum):
    RUBRIC_MISFIT = "rubric_misfit"
    FACTUAL_ERROR = "factual_error"
    SAYS_ONLY = "says_only"
    DEBUNKED_CLAIM = "debunked_claim"


class EnactmentCell(str, Enum):
    """The four cells of the names/enacts 2x2. `NAMES_ONLY` is the says-only trap."""

    NEITHER = "neither"
    ENACTS_ONLY = "enacts_only"
    NAMES_ONLY = "names_only"
    NAMES_AND_ENACTS = "names_and_enacts"


# The fields an item's key holds back from raters and agents alike. Assembled
# into `Artifact.as_key_row`; `as_corpus_row` is built from an allowlist that
# cannot reach them.
WITHHELD_FIELDS: tuple[str, ...] = (
    "designed_target_construct",
    "item_role",
    "names",
    "enacts",
    "enactment_cell",
    "enacted_construct",
    "named_construct",
    "expected_presence",
    "expected_fidelity",
    "split",
)


def _require_bool(value: object, name: str, where: str) -> bool:
    """A JSON boolean. `0` and `1` are refused; they are how a scale leaks into a flag."""
    if not isinstance(value, bool):
        raise RecordError(f"{where}: {name} must be a boolean, got {value!r} ({type(value).__name__})")
    return value


def _require_ordinal(value: object, name: str, bounds: tuple[int, int], where: str) -> int:
    """An integer in range. `bool` is refused FIRST because it subclasses `int`."""
    low, high = bounds
    if isinstance(value, bool):
        raise RecordError(f"{where}: {name} must be an integer {low}-{high}, got the boolean {value!r}")
    if not isinstance(value, int):
        raise RecordError(f"{where}: {name} must be an integer {low}-{high}, got {value!r} ({type(value).__name__})")
    if not low <= value <= high:
        raise RecordError(f"{where}: {name} must be an integer {low}-{high}, got {value!r}")
    return value


# ------------------------------------------------------------- the 2x2 itself


@dataclass(frozen=True, slots=True)
class NamesEnacts:
    """Whether an artifact NAMES a principle and whether it ENACTS it, kept apart.

    Two booleans rather than one label, because they are independent and the
    whole project is the observation that they come apart. `expected_presence`
    is `enacts` alone: every construct in the ontology is scored at
    `enacted_output`, where naming is not evidence.
    """

    names: bool
    enacts: bool

    def __post_init__(self) -> None:
        _require_bool(self.names, "names", "NamesEnacts")
        _require_bool(self.enacts, "enacts", "NamesEnacts")

    @property
    def cell(self) -> EnactmentCell:
        if self.names and self.enacts:
            return EnactmentCell.NAMES_AND_ENACTS
        if self.names:
            return EnactmentCell.NAMES_ONLY
        if self.enacts:
            return EnactmentCell.ENACTS_ONLY
        return EnactmentCell.NEITHER

    @property
    def is_says_only(self) -> bool:
        """Names the principle, enacts none of it. The false positive the atlas is built to count."""
        return self.names and not self.enacts

    @property
    def expected_presence(self) -> bool:
        """Presence tracks enactment and never naming. See `EvidenceLevel.ENACTED_OUTPUT`."""
        return self.enacts

    def as_dict(self) -> dict[str, object]:
        return {"names": self.names, "enacts": self.enacts, "enactment_cell": self.cell.value}


# ------------------------------------------------------------------- ontology


@dataclass(frozen=True, slots=True)
class SiblingConfusable:
    """The construct this one is most often mistaken for, and the rule that separates them."""

    key: str
    discriminator: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Labelability:
    band: LabelabilityBand
    expected_kappa_w: str
    basis: str
    failure_mode: str


@dataclass(frozen=True, slots=True)
class RewardEligibility:
    status: RewardStatus
    rationale: str
    guard: str = ""


@dataclass(frozen=True, slots=True)
class Citation:
    ref: str
    claim: str
    doi: str = ""


@dataclass(frozen=True, slots=True)
class Construct:
    """One construct, with the text a labeler and a corpus builder both need."""

    key: str
    display_name: str
    family: Family
    definition: str
    evidence_grade: EvidenceGrade
    evidence_grade_rationale: str
    evidence_level: EvidenceLevel
    output_types: tuple[OutputFormat, ...]
    required_context: tuple[str, ...]
    sibling_confusable: SiblingConfusable
    boundary_conditions: tuple[str, ...]
    positive_anchor: str
    hard_negative: str
    says_only_negative: str
    surface_confounds: tuple[str, ...]
    labelability: Labelability
    reward_eligibility: RewardEligibility
    citations: tuple[Citation, ...]

    @property
    def sibling_key(self) -> str:
        return self.sibling_confusable.key

    def accepts(self, output_format: OutputFormat) -> bool:
        """Whether this construct can be instantiated in that artifact type at all."""
        return output_format in self.output_types


@dataclass(frozen=True, slots=True)
class DebunkedControl:
    """A negative control that a naive quality model should not reward."""

    key: str
    display_name: str
    status: str
    expected_label: str
    why: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True, slots=True)
class ExcludedConstruct:
    """Something important that is deliberately not here, and what would change that."""

    key: str
    display_name: str
    why_excluded: str
    evidence: str
    what_would_change_this: str


@dataclass(frozen=True, slots=True)
class Ontology:
    """`ontology.yaml`, parsed and checked. Build it with `load_ontology`."""

    schema_version: int
    construct_version: str
    status: str
    constructs: tuple[Construct, ...]
    debunked_controls: tuple[DebunkedControl, ...]
    excluded_latent_constructs: tuple[ExcludedConstruct, ...]
    required_reporting_fields: tuple[str, ...]
    source_path: str = ""
    _by_key: Mapping[str, Construct] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_key", MappingProxyType({c.key: c for c in self.constructs}))

    def __len__(self) -> int:
        return len(self.constructs)

    def __iter__(self) -> Iterator[Construct]:
        return iter(self.constructs)

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._by_key)

    def get(self, key: str) -> Construct:
        try:
            return self._by_key[key]
        except KeyError:
            raise OntologyError(f"no construct {key!r}; the ontology has {list(self._by_key)}") from None

    def sibling_of(self, key: str) -> Construct:
        return self.get(self.get(key).sibling_key)

    def accepting(self, output_format: OutputFormat) -> tuple[Construct, ...]:
        return tuple(c for c in self.constructs if c.accepts(output_format))


def _as_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OntologyError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _text(body: Mapping[str, object], name: str, where: str, *, required: bool = True) -> str:
    value = body.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise OntologyError(f"{where}: missing required field {name!r}; present are {sorted(body)}")
        return ""
    if not isinstance(value, str):
        raise OntologyError(f"{where}: {name} must be text, got {type(value).__name__}")
    return " ".join(value.split())


def _text_list(body: Mapping[str, object], name: str, where: str) -> tuple[str, ...]:
    value = body.get(name)
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise OntologyError(f"{where}: {name} must be a non-empty list, got {value!r}")
    out = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise OntologyError(f"{where}: {name}[{index}] must be non-empty text, got {entry!r}")
        out.append(" ".join(entry.split()))
    return tuple(out)


def _enum(kind: type[Enum], value: object, name: str, where: str) -> Enum:
    try:
        return kind(value)
    except ValueError:
        allowed = [member.value for member in kind]
        raise OntologyError(f"{where}: {name} is {value!r}, which is not one of {allowed}") from None


def _check_vocabulary(declared: object, kind: type[Enum], name: str, where: str) -> None:
    """The YAML's vocabulary block and the enum must name the same set.

    Drift here is the failure this catches: a value added to the file and not to
    the code is silently unusable, and a value in the code and not the file is a
    construct nobody defined.
    """
    if not isinstance(declared, Mapping):
        raise OntologyError(f"{where}: {name} must be a mapping of value -> description")
    in_file = set(declared)
    in_code = {member.value for member in kind}
    missing_here = sorted(in_file - in_code)
    missing_there = sorted(in_code - in_file)
    if missing_here or missing_there:
        raise OntologyError(
            f"{where}: {name} disagrees with {kind.__name__}: "
            f"in the file but not the enum {missing_here}, in the enum but not the file {missing_there}"
        )


def _citations(body: Mapping[str, object], where: str) -> tuple[Citation, ...]:
    raw = body.get("citations")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise OntologyError(f"{where}: citations must be a non-empty list")
    out = []
    for index, entry in enumerate(raw):
        cite = _as_mapping(entry, f"{where}.citations[{index}]")
        out.append(
            Citation(
                ref=_text(cite, "ref", f"{where}.citations[{index}]"),
                claim=_text(cite, "claim", f"{where}.citations[{index}]"),
                doi=_text(cite, "doi", f"{where}.citations[{index}]", required=False),
            )
        )
    return tuple(out)


def _construct(body: Mapping[str, object], where: str) -> Construct:
    key = _text(body, "key", where)
    at = f"{where}[{key}]"

    raw_formats = body.get("output_types")
    if not isinstance(raw_formats, Sequence) or isinstance(raw_formats, str) or not raw_formats:
        raise OntologyError(f"{at}: output_types must be a non-empty list, got {raw_formats!r}")
    formats = tuple(_enum(OutputFormat, value, "output_types", at) for value in raw_formats)
    if len(set(formats)) != len(formats):
        raise OntologyError(f"{at}: output_types repeats a value: {[f.value for f in formats]}")

    sibling = _as_mapping(body.get("sibling_confusable"), f"{at}.sibling_confusable")
    labelability = _as_mapping(body.get("labelability"), f"{at}.labelability")
    reward = _as_mapping(body.get("reward_eligibility"), f"{at}.reward_eligibility")

    status = _enum(RewardStatus, reward.get("status"), "status", f"{at}.reward_eligibility")
    guard = _text(reward, "guard", f"{at}.reward_eligibility", required=False)
    if status is RewardStatus.ELIGIBLE_WITH_GUARD and not guard:
        raise OntologyError(f"{at}: reward status is {status.value!r} and names no guard; the guard is the point")

    return Construct(
        key=key,
        display_name=_text(body, "display_name", at),
        family=_enum(Family, body.get("family"), "family", at),
        definition=_text(body, "definition", at),
        evidence_grade=_enum(EvidenceGrade, body.get("evidence_grade"), "evidence_grade", at),
        evidence_grade_rationale=_text(body, "evidence_grade_rationale", at),
        evidence_level=_enum(EvidenceLevel, body.get("evidence_level"), "evidence_level", at),
        output_types=formats,
        required_context=_text_list(body, "required_context", at),
        sibling_confusable=SiblingConfusable(
            key=_text(sibling, "key", f"{at}.sibling_confusable"),
            discriminator=_text(sibling, "discriminator", f"{at}.sibling_confusable"),
            note=_text(sibling, "note", f"{at}.sibling_confusable", required=False),
        ),
        boundary_conditions=_text_list(body, "boundary_conditions", at),
        positive_anchor=_text(body, "positive_anchor", at),
        hard_negative=_text(body, "hard_negative", at),
        says_only_negative=_text(body, "says_only_negative", at),
        surface_confounds=_text_list(body, "surface_confounds", at),
        labelability=Labelability(
            band=_enum(LabelabilityBand, labelability.get("band"), "band", f"{at}.labelability"),
            expected_kappa_w=_text(labelability, "expected_kappa_w", f"{at}.labelability"),
            basis=_text(labelability, "basis", f"{at}.labelability"),
            failure_mode=_text(labelability, "failure_mode", f"{at}.labelability"),
        ),
        reward_eligibility=RewardEligibility(
            status=status, rationale=_text(reward, "rationale", f"{at}.reward_eligibility"), guard=guard
        ),
        citations=_citations(body, at),
    )


def ontology_from_mapping(blob: Mapping[str, object], *, source_path: str = "") -> Ontology:
    """Check a parsed ontology and freeze it. Every failure names the field."""
    where = source_path or "ontology"

    version = blob.get("schema_version")
    if version != SCHEMA_VERSION:
        raise OntologyError(f"{where}: schema_version is {version!r}, this code implements {SCHEMA_VERSION}")

    for name, kind in (
        ("evidence_levels", EvidenceLevel),
        ("evidence_grades", EvidenceGrade),
        ("families", Family),
        ("output_types", OutputFormat),
        ("labelability_bands", LabelabilityBand),
        ("reward_status", RewardStatus),
    ):
        _check_vocabulary(blob.get(name), kind, name, where)

    raw_constructs = blob.get("constructs")
    if not isinstance(raw_constructs, Sequence) or isinstance(raw_constructs, str) or not raw_constructs:
        raise OntologyError(f"{where}: constructs must be a non-empty list")

    constructs = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_constructs):
        construct = _construct(_as_mapping(entry, f"{where}.constructs[{index}]"), f"{where}.constructs")
        if construct.key in seen:
            raise OntologyError(f"{where}: duplicate construct key {construct.key!r}")
        seen.add(construct.key)
        constructs.append(construct)

    declared = blob.get("n_constructs")
    if declared is not None and declared != len(constructs):
        raise OntologyError(f"{where}: n_constructs says {declared!r}, the file carries {len(constructs)}")

    for construct in constructs:
        sibling = construct.sibling_key
        if sibling == construct.key:
            raise OntologyError(
                f"{where}: {construct.key!r} is its own sibling_confusable, which discriminates nothing"
            )
        if sibling not in seen:
            raise OntologyError(f"{where}: {construct.key!r} names sibling {sibling!r}, which is not a construct")

    controls = []
    for index, entry in enumerate(blob.get("debunked_controls") or ()):
        at = f"{where}.debunked_controls[{index}]"
        body = _as_mapping(entry, at)
        controls.append(
            DebunkedControl(
                key=_text(body, "key", at),
                display_name=_text(body, "display_name", at),
                status=_text(body, "status", at),
                expected_label=_text(body, "expected_label", at),
                why=_text(body, "why", at),
                citations=_citations(body, at),
            )
        )

    excluded = []
    for index, entry in enumerate(blob.get("excluded_latent_constructs") or ()):
        at = f"{where}.excluded_latent_constructs[{index}]"
        body = _as_mapping(entry, at)
        excluded.append(
            ExcludedConstruct(
                key=_text(body, "key", at),
                display_name=_text(body, "display_name", at),
                why_excluded=_text(body, "why_excluded", at),
                evidence=_text(body, "evidence", at),
                what_would_change_this=_text(body, "what_would_change_this", at),
            )
        )

    overlap = sorted({c.key for c in excluded} & seen)
    if overlap:
        raise OntologyError(f"{where}: {overlap} are both scored constructs and excluded latent constructs")

    return Ontology(
        schema_version=SCHEMA_VERSION,
        construct_version=_text(blob, "construct_version", where),
        status=_text(blob, "status", where, required=False),
        constructs=tuple(constructs),
        debunked_controls=tuple(controls),
        excluded_latent_constructs=tuple(excluded),
        required_reporting_fields=_text_list(blob, "required_reporting_fields", where),
        source_path=source_path,
    )


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parent / ONTOLOGY_FILENAME


def load_ontology(path: str | Path | None = None, *, prefer_pyyaml: bool = True) -> Ontology:
    """Read, parse and check `ontology.yaml`.

    `prefer_pyyaml=False` forces the bundled subset parser even where PyYAML is
    installed, which is what the parity test uses.
    """
    resolved = Path(path) if path is not None else default_ontology_path()
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise OntologyError(f"cannot read the ontology at {resolved}: {exc}") from exc
    blob = parse_yaml(text, prefer_pyyaml=prefer_pyyaml)
    return ontology_from_mapping(_as_mapping(blob, str(resolved)), source_path=str(resolved))


# ----------------------------------------------------------- ids and matching


def stable_digest(*parts: str, size: int = 8) -> str:
    """A content-keyed digest that reproduces across processes.

    Python salts `hash()` per interpreter, so an id built from it changes
    between runs and a corpus cannot be regenerated from its recorded seed.
    """
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).hexdigest()


def lineage_id(
    concept: str,
    domain: Domain,
    output_format: OutputFormat,
    learner_state: LearnerState,
    prompt_mode: PromptMode,
    scenario_index: int,
    *,
    salt: str = "",
) -> str:
    """The id of the SCENARIO, which every variant written from it shares.

    Variants of one lineage share a stem, a reference solution and usually a
    learner turn, so they are one unit for every purpose that cares about
    contamination: they split together, they quarantine together, and they reach
    a judge as options under a single item.
    """
    return "ln" + stable_digest(
        salt,
        "lineage",
        concept,
        domain.value,
        output_format.value,
        learner_state.value,
        prompt_mode.value,
        str(scenario_index),
        size=7,
    )


def artifact_id(lineage: str, role: ItemRole, variant_index: int = 0, *, salt: str = "") -> str:
    """The id of one variant, derived from its lineage so the two can never disagree.

    A digest rather than a readable string because ids reach raters: an id
    carrying `says_only` would answer the question it was shown to ask.
    """
    return "it" + stable_digest(salt, "artifact", lineage, role.value, str(variant_index), size=7)


_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Casefolded, unpunctuated, whitespace-collapsed text, for near-duplicate matching.

    Exact matching alone misses the duplicate that actually happens: the same
    artifact re-wrapped, re-quoted or re-cased on its way through a pipeline.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_PUNCTUATION.sub(" ", folded).split())


# --------------------------------------------------------------- the records


@dataclass(frozen=True, slots=True)
class Scenario:
    """The shared stem a lineage's variants are written from.

    Its five categorical fields are the corpus's balance axes and, with the
    concept and an index, they are exactly what `lineage_id` hashes.
    """

    domain: Domain
    output_format: OutputFormat
    learner_state: LearnerState
    prompt_mode: PromptMode
    index: int
    topic: str
    target: str
    task: str
    reference: str
    student_before: str
    misstep: str = ""
    worked_step: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise RecordError(f"Scenario: index must be non-negative, got {self.index!r}")
        for name in ("topic", "target", "task", "reference"):
            if not str(getattr(self, name)).strip():
                raise RecordError(f"Scenario: {name} must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain.value,
            "output_format": self.output_format.value,
            "learner_state": self.learner_state.value,
            "prompt_mode": self.prompt_mode.value,
            "scenario_index": self.index,
            "topic": self.topic,
            "target": self.target,
        }


@dataclass(frozen=True, slots=True)
class Artifact:
    """One candidate action: the thing a rater judges and the row a corpus carries.

    `enactment` is the designed truth about this item and `role` says why it was
    written; both are withheld from raters. `as_corpus_row` is assembled from an
    allowlist rather than filtered, on the same reasoning as
    `labeling.Packet.judge_view`: a view built from scratch cannot leak a field
    somebody forgot to strip.
    """

    artifact_id: str
    lineage_id: str
    concept: str
    role: ItemRole
    enactment: NamesEnacts
    text: str
    scenario: Scenario
    enacted_construct: str | None = None
    named_construct: str | None = None
    expected_fidelity: int | None = None
    variant_index: int = 0

    def __post_init__(self) -> None:
        where = f"Artifact[{self.artifact_id}]"
        if not self.text.strip():
            raise RecordError(f"{where}: text must not be empty")
        if not self.concept.strip():
            raise RecordError(f"{where}: concept must not be empty")
        if self.expected_fidelity is not None:
            _require_ordinal(self.expected_fidelity, "expected_fidelity", FIDELITY_RANGE, where)
            if not self.enactment.enacts:
                raise RecordError(
                    f"{where}: expected_fidelity is {self.expected_fidelity!r} on an item that enacts nothing; "
                    "an absent construct has no fidelity to score"
                )
        elif self.enactment.enacts:
            raise RecordError(f"{where}: an item that enacts its concept must declare an expected_fidelity")
        if self.enactment.enacts and self.enacted_construct not in (None, self.concept):
            raise RecordError(
                f"{where}: enacts is true for {self.concept!r} but enacted_construct is {self.enacted_construct!r}"
            )
        if self.enactment.names and not self.named_construct:
            raise RecordError(f"{where}: names is true but no named_construct is recorded")
        if not self.enactment.names and self.named_construct:
            raise RecordError(f"{where}: named_construct is {self.named_construct!r} but names is false")

    @property
    def cell(self) -> EnactmentCell:
        return self.enactment.cell

    @property
    def expected_presence(self) -> bool:
        return self.enactment.expected_presence

    def as_corpus_row(self) -> dict[str, object]:
        """The row a rater or a judge may see. Column names are `labeling.CorpusFields`.

        `item_id` is the lineage, so `labeling.build_packet` groups a scenario's
        variants into one blinded item and quarantines them together.
        """
        return {
            "unit_id": self.artifact_id,
            "item_id": self.lineage_id,
            "concept": self.concept,
            "candidate_action": self.text,
            "question": self.scenario.task,
            "reference": self.scenario.reference,
            "student_before": self.scenario.student_before,
        }

    def as_key_row(self, split: Split | None = None) -> dict[str, object]:
        """Everything withheld from raters. Kept beside the corpus, opened after labelling."""
        return {
            "unit_id": self.artifact_id,
            "item_id": self.lineage_id,
            "designed_target_construct": self.concept,
            "item_role": self.role.value,
            "enacted_construct": self.enacted_construct,
            "named_construct": self.named_construct,
            "expected_presence": self.expected_presence,
            "expected_fidelity": self.expected_fidelity,
            "split": split.value if split else None,
            **self.enactment.as_dict(),
            **self.scenario.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One rater's judgment of one construct on one artifact, per LABELING_GUIDE.md §9.

    The null rules are invariants and not conventions. A construct marked
    inapplicable has no presence, and one marked absent has no fidelity; a
    record that fills either with a zero is rejected here rather than averaged
    into a prevalence estimate later.
    """

    artifact_id: str
    lineage_id: str
    concept: str
    rater: str
    applicable: bool
    applicability: Applicability
    presence: bool | None = None
    fidelity: int | None = None
    quality: int | None = None
    span: str | None = None
    flags: tuple[Flag, ...] = ()
    quarantined: bool = False

    def __post_init__(self) -> None:
        where = f"LabelRecord[{self.rater}:{self.artifact_id}:{self.concept}]"
        _require_bool(self.applicable, "applicable", where)
        _require_bool(self.quarantined, "quarantined", where)
        if self.applicable != (self.applicability is Applicability.APPLICABLE):
            raise RecordError(
                f"{where}: applicable is {self.applicable!r} but applicability is {self.applicability.value!r}; "
                "the two may not disagree"
            )
        if self.quality is not None:
            _require_ordinal(self.quality, "quality", QUALITY_RANGE, where)

        if not self.applicable:
            for name in ("presence", "fidelity"):
                if getattr(self, name) is not None:
                    raise RecordError(
                        f"{where}: {name} must be null when applicable is false, got {getattr(self, name)!r}"
                    )
            if self.span:
                raise RecordError(f"{where}: span must be empty when applicable is false, got {self.span!r}")
            return

        if self.presence is None:
            raise RecordError(f"{where}: presence is required when applicable is true")
        _require_bool(self.presence, "presence", where)

        if self.presence:
            _require_ordinal(self.fidelity, "fidelity", FIDELITY_RANGE, where)
            if not (self.span or "").strip():
                raise RecordError(f"{where}: presence is true and no span was quoted; no span, no presence")
        elif self.fidelity is not None:
            raise RecordError(f"{where}: fidelity must be null unless presence is true, got {self.fidelity!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.artifact_id,
            "item_id": self.lineage_id,
            "concept": self.concept,
            "rater": self.rater,
            "applicable": self.applicable,
            "applicability": self.applicability.value,
            "presence": self.presence,
            "fidelity": self.fidelity,
            "quality": self.quality,
            "span": self.span,
            "flags": [flag.value for flag in self.flags],
            "quarantined": self.quarantined,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, rater: str | None = None) -> LabelRecord:
        """Parse a stored label. Unknown fields are rejected rather than ignored."""
        allowed = {f.name for f in fields(cls)} | {"unit_id", "item_id"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise RecordError(f"label record has unknown field(s) {unknown}; allowed are {sorted(allowed)}")

        raw_applicability = payload.get("applicability")
        try:
            applicability = Applicability(raw_applicability)
        except ValueError:
            allowed_values = [member.value for member in Applicability]
            raise RecordError(f"applicability is {raw_applicability!r}, not one of {allowed_values}") from None

        raw_flags = payload.get("flags") or ()
        if isinstance(raw_flags, str):
            raise RecordError(f"flags must be a list, got the string {raw_flags!r}")
        try:
            flags = tuple(Flag(flag) for flag in raw_flags)
        except ValueError as exc:
            raise RecordError(f"unknown flag: {exc}") from None

        return cls(
            artifact_id=str(payload.get("artifact_id") or payload.get("unit_id") or ""),
            lineage_id=str(payload.get("lineage_id") or payload.get("item_id") or ""),
            concept=str(payload.get("concept") or ""),
            rater=str(rater or payload.get("rater") or ""),
            applicable=payload.get("applicable"),
            applicability=applicability,
            presence=payload.get("presence"),
            fidelity=payload.get("fidelity"),
            quality=payload.get("quality"),
            span=payload.get("span"),
            flags=flags,
            quarantined=payload.get("quarantined", False),
        )


def gold_label(artifact: Artifact, *, rater: str = "designed") -> LabelRecord:
    """The label the corpus's design entails, for checking a labelling round against.

    Presence is `enacts`, so a says-only item is gold-absent however loudly it
    names the principle. This is the reference `says_only_fp` (gate G4) is
    computed against.
    """
    presence = artifact.expected_presence
    return LabelRecord(
        artifact_id=artifact.artifact_id,
        lineage_id=artifact.lineage_id,
        concept=artifact.concept,
        rater=rater,
        applicable=True,
        applicability=Applicability.APPLICABLE,
        presence=presence,
        fidelity=artifact.expected_fidelity if presence else None,
        quality=None,
        span=artifact.text[:160] if presence else None,
    )


def write_jsonl(rows: Iterable[Mapping[str, object]], path: str | Path) -> int:
    """Write records as JSONL with sorted keys, so a diff of two runs is a diff of content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count
