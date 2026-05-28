"""
spec.py — Parse and validate eval YAML specs.

Produces a typed Spec object that query.py and grader.py both consume.
Validation happens at parse time so misconfigured specs fail fast, before
any tokens are spent or any results are written.

Public API:
    load_spec(path: str | Path) -> Spec

Required PyYAML (`pip install pyyaml`); a clear ImportError fires below
if it's missing.
"""

# --- Imports --------------------------

# dataclass is the cleanest way to declare a typed record-like object.
from dataclasses import dataclass
from pathlib import Path

# PyYAML's safe_load parses YAML into Python primitives (dict, list, str,
# int, float, bool). Always prefer safe_load over load — plain load can
# execute arbitrary Python via YAML tags, which is a security footgun.
import yaml


# --- Constants --------------------------

# Grader types the parser will accept. Anything else fails validation.
# Keeping the set here makes it the single source of truth for what's valid.
ALLOWED_GRADER_TYPES = {"exact", "numeric", "judge", "none", "structured"}

# Top-level YAML keys that must be present. The parser rejects specs
# missing any of these before trying to construct the typed object.
REQUIRED_TOP_LEVEL = (
    "name", "version", "models", "runs", "temperature",
    "grader", "topics", "queries", "output",
)

# Fields each topic dict must contain. Validated per-topic in Spec.__post_init__.
REQUIRED_TOPIC_FIELDS = ("league", "year", "truth")


# --- Typed config objects --------------------------

# Each section of the YAML spec becomes a small dataclass. This gives us:
#   1. Attribute access (`spec.queries.pre_query.text`) — what query.py expects
#   2. Field declarations that read like a schema
#   3. A natural place to hang per-section validation via __post_init__
#
# The dataclasses nest: Spec contains QueriesConfig, which contains
# QueryConfig, etc. load_spec() constructs them from inside-out.

@dataclass
class QueryConfig:
    """One query definition — either the pre_query or the primary query."""
    text: str
    web_search: bool
    system_prompt: str = ""

    def __post_init__(self):
        # Defensive: catch obvious bad inputs early. {league}/{year}
        # placeholder validation happens in query.py at .format() time.
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be a non-empty string")
        if not isinstance(self.web_search, bool):
            raise ValueError(f"web_search must be true or false; got {self.web_search!r}")


@dataclass
class QueriesConfig:
    """The two queries the harness runs per topic: coverage check + primary."""
    pre_query: QueryConfig
    query: QueryConfig


@dataclass
class GraderConfig:
    """Grader settings — type selector plus type-specific options."""
    type: str
    expected_unit: str
    refusal_patterns: list

    def __post_init__(self):
        if self.type not in ALLOWED_GRADER_TYPES:
            raise ValueError(
                f"grader.type must be one of {sorted(ALLOWED_GRADER_TYPES)}; "
                f"got {self.type!r}"
            )
        if not self.expected_unit:
            raise ValueError("grader.expected_unit must be set (e.g. 'hours')")
        if not self.refusal_patterns:
            raise ValueError("grader.refusal_patterns must not be empty")


@dataclass
class OutputConfig:
    """Where to write the harness output. Defaults to JSON format."""
    path: str
    format: str = "json"

    def __post_init__(self):
        if self.format != "json":
            # v1.0 only supports JSON. The field is here for forward
            # compatibility (CSV, JSONL, etc.) but reject anything else now.
            raise ValueError(f"output.format must be 'json' for v1.0; got {self.format!r}")


@dataclass
class Spec:
    """
    A complete, validated eval spec.

    All required fields are positional; defaults are reserved for things
    that genuinely have a sensible default (`description`, for example).
    The __post_init__ catches constraint violations that don't show up
    in individual sub-object validation — temperature range, list emptiness,
    per-topic field presence.
    """
    name: str
    version: str
    models: list
    runs: int
    temperature: float
    grader: GraderConfig
    topics: list
    queries: QueriesConfig
    output: OutputConfig
    description: str = ""
    soft_token_budget: int = None

    def __post_init__(self):
        # temperature > 0: temperature=0 trivialises stability (same response
        # every time defeats the whole point of measuring answer variation).
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0 (got {self.temperature}); "
                "temperature=0 trivialises stability"
            )

        if self.runs < 1:
            raise ValueError(f"runs must be >= 1 (got {self.runs})")

        if not self.models:
            raise ValueError("models list must not be empty")

        if not self.topics:
            raise ValueError("topics list must not be empty")

        # Each topic dict must carry the fields query.py and grader.py read.
        # Catching this here is much friendlier than a KeyError mid-run.
        for i, topic in enumerate(self.topics):
            if not isinstance(topic, dict):
                raise ValueError(f"topics[{i}] must be a dict; got {type(topic).__name__}")
            missing = [f for f in REQUIRED_TOPIC_FIELDS if f not in topic]
            if missing:
                raise ValueError(
                    f"topics[{i}] missing required field(s): {missing} "
                    f"(needs {list(REQUIRED_TOPIC_FIELDS)})"
                )


# --- Public API --------------------------

def load_spec(path) -> Spec:
    """
    Load and validate an eval spec from a YAML file.

    Strategy:
      1. Read the file. Fail fast on I/O or YAML errors.
      2. Verify required top-level keys exist in the raw dict before
         constructing typed objects. This gives clearer error messages
         than letting dataclass(**raw) blow up on a missing kwarg.
      3. Construct nested dataclasses from inside-out. Each __post_init__
         runs its own validation as it's built.
      4. Return the fully validated Spec.

    Raises:
        FileNotFoundError: spec file path doesn't exist
        ValueError:        required field missing or constraint violated
        yaml.YAMLError:    file isn't valid YAML
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    # `with` ensures the file handle closes even if YAML parsing throws.
    with open(path) as f:
        raw = yaml.safe_load(f)

    # yaml.safe_load returns None for an empty file. Catch that explicitly.
    if raw is None:
        raise ValueError(f"Spec file is empty: {path}")

    if not isinstance(raw, dict):
        raise ValueError(
            f"Spec file root must be a mapping (key/value pairs); got {type(raw).__name__}"
        )

    # Check all required top-level keys before going further. Reporting
    # them all in one error is friendlier than failing on the first one.
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ValueError(
            f"Spec missing required top-level field(s): {missing}. "
            f"Required: {list(REQUIRED_TOP_LEVEL)}"
        )

    # queries must have both pre_query (the coverage check) and query.
    # The pre_query requirement is the punch-list constraint — reject
    # specs that don't include a coverage check.
    if "pre_query" not in raw["queries"]:
        raise ValueError(
            "Spec missing required field: queries.pre_query "
            "(the coverage_check query is required)"
        )
    if "query" not in raw["queries"]:
        raise ValueError("Spec missing required field: queries.query")

    # Construct nested dataclasses. The ** unpacking converts a dict's keys
    # into keyword arguments — works as long as the dict keys exactly match
    # the dataclass field names, which they do by spec design.
    queries = QueriesConfig(
        pre_query=QueryConfig(**raw["queries"]["pre_query"]),
        query=QueryConfig(**raw["queries"]["query"]),
    )
    grader = GraderConfig(**raw["grader"])
    output = OutputConfig(**raw["output"])

    # version comes in as a string from YAML (we quoted "1.0") so we cast
    # nothing; if you ever decide version should be a tuple or semver type,
    # do the conversion here.
    return Spec(
        name=raw["name"],
        version=str(raw["version"]),
        description=raw.get("description", ""),  # description is optional
        models=raw["models"],
        runs=raw["runs"],
        temperature=raw["temperature"],
        grader=grader,
        topics=raw["topics"],
        queries=queries,
        output=output,
        soft_token_budget=raw.get("soft_token_budget"),
    )