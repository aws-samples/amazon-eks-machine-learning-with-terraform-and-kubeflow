"""The reasoner: a text-in, text-out model, and the record of what it was asked.

Three implementations, and the boring two matter as much as the real one:

  - `BedrockReasoner` calls Bedrock's Converse API. The only part of the agent that needs the
    network for a reason other than talking to the cluster.
  - `ReplayReasoner` returns canned replies in order. It is what lets `selftest.py` exercise the
    whole loop -- validation, retry, fallback -- offline, deterministically, and for nothing.
  - `NullReasoner` raises. Used to prove that a code path claimed not to consult the model
    really does not.

`Transcript` is not debug output. A record whose placement was chosen by a model, without the
prompt and reply that chose it, is a measurement nobody can audit: the search is only
reproducible if what the model saw is recoverable. So the transcript travels into the
observation's provenance, and the store is where it lives.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The cross-region inference profile, not the bare model id. `anthropic.claude-opus-4-5-...`
# exists in `list-foundation-models` but on-demand invocation of these models goes through the
# `us.` profile; calling the bare id fails with a ValidationException that names throughput
# rather than the prefix, which is a slow thing to work out from the error alone.
DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"
DEFAULT_REGION = "us-west-2"

# Generous, because this model thinks before it answers and the thinking is charged to the same
# budget. Measured: a 64-token cap returned a reply that was 100% reasoning and 0% text, and the
# only symptom downstream was "no JSON object in the reply" -- a truncation that reads as a model
# unable to follow instructions. `stopReason == "max_tokens"` is logged for the same reason.
DEFAULT_MAX_TOKENS = 16384

# What stands in for a turn with no text in it. Deliberately not "": Bedrock refuses an empty text
# block outright ("text content blocks must be non-empty"), and because every call re-sends the whole
# conversation, one empty assistant turn does not fail the turn that produced it -- it fails every
# call after it. A run lost three quarters of its budget that way with the work already done, and the
# log said "the reasoner could not be reached" about a model that had answered.
EMPTY_REPLY = "(no text in this reply)"

THROTTLE_CODES = ("ThrottlingException", "TooManyRequestsException",
                  "ServiceUnavailableException", "ModelTimeoutException",
                  "InternalServerException")


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    text: str


@dataclass
class Transcript:
    """One exchange, kept so a decision can be re-read later."""

    reasoner: str
    system: str
    turns: list = field(default_factory=list)      # list of Turn, in order
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    retries: int = 0

    def to_dict(self, *, include_prompt: bool = True) -> dict:
        """Provenance form. `include_prompt=False` keeps only what the model *said*.

        The prompt is large and is a deterministic function of the store, so a caller with the
        store can regenerate it. The replies are not recoverable from anything and are always
        kept.
        """
        out = {"reasoner": self.reasoner, "usage": dict(self.usage),
               "latency_s": round(self.latency_s, 2), "retries": self.retries,
               "replies": [t.text for t in self.turns if t.role == "assistant"]}
        if include_prompt:
            out["system"] = self.system
            out["turns"] = [{"role": t.role, "text": t.text} for t in self.turns]
        return out


class Reasoner:
    """Anything that can answer a question in text."""

    name = "reasoner"

    def converse(self, system: str, turns: list) -> tuple[str, dict]:
        """Return `(reply_text, usage)` for a conversation ending in a user turn."""
        raise NotImplementedError


class NullReasoner(Reasoner):
    name = "null"

    def converse(self, system: str, turns: list) -> tuple[str, dict]:
        raise AssertionError("a NullReasoner was asked to think; this path was supposed to "
                             "reach no model at all")


class ReplayReasoner(Reasoner):
    """Canned replies, in order. Records what it was asked so a test can assert on the prompt."""

    def __init__(self, replies, name: str = "replay"):
        self.replies = list(replies)
        self.name = name
        self.asked: list = []          # the `turns` of every call, in order

    def converse(self, system: str, turns: list) -> tuple[str, dict]:
        self.asked.append(list(turns))
        if not self.replies:
            raise AssertionError(
                f"a ReplayReasoner ran out of replies after {len(self.asked)} call(s). The policy "
                "asked more times than the test expected, which is itself the finding -- add a "
                "reply only once you know why it asked again.")
        return self.replies.pop(0), {"replay": True}


class BedrockReasoner(Reasoner):
    """Bedrock Converse. Constructed lazily so importing this module needs no credentials."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, region: str = DEFAULT_REGION,
                 max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float | None = 0.0,
                 max_attempts: int = 5, client=None):
        self.model_id = model_id
        self.region = region
        self.max_tokens = int(max_tokens)
        # Zero when the model accepts it. A search whose choices cannot be reproduced from its
        # store is a search whose conclusions cannot be checked, and sampling is not where the
        # exploration in this agent is supposed to come from -- the reasoner's own tool calls are.
        # Newer models reject the parameter outright ("`temperature` is deprecated for this
        # model"), so it is dropped on that error rather than made a per-model constant here:
        # a table of which models take which parameters would be wrong by the next release.
        self.temperature = temperature
        self.max_attempts = int(max_attempts)
        self._client = client
        self.name = f"bedrock:{model_id}"

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def converse(self, system: str, turns: list) -> tuple[str, dict]:
        kwargs = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            # Through `_content`, not `{"text": t.text}` directly: the API cannot represent an empty
            # text block, and dropping such a turn instead would break the strict user/assistant
            # alternation these models require. So it is filled in rather than removed.
            "messages": [{"role": t.role, "content": _content(t.text)} for t in turns],
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if self.temperature is not None:
            kwargs["inferenceConfig"]["temperature"] = self.temperature

        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.converse(**kwargs)
                break
            except Exception as exc:                        # noqa: BLE001 - classified below
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
                if (code == "ValidationException" and "temperature" in str(exc)
                        and "temperature" in kwargs["inferenceConfig"]):
                    log.info("%s rejects `temperature`; dropping it and retrying", self.model_id)
                    kwargs["inferenceConfig"].pop("temperature")
                    self.temperature = None
                    continue
                if code not in THROTTLE_CODES or attempt == self.max_attempts:
                    raise
                last, wait = exc, min(2.0 ** attempt, 30.0)
                log.warning("bedrock %s (attempt %d/%d); retrying in %.0fs",
                            code, attempt, self.max_attempts, wait)
                time.sleep(wait)
        else:                                               # pragma: no cover - loop always breaks
            raise last

        text = _text_of(resp)
        usage = dict(resp.get("usage") or {})
        usage["stopReason"] = resp.get("stopReason")
        if resp.get("stopReason") == "max_tokens":
            # Named rather than swallowed: a truncated reply usually fails JSON parsing several
            # layers later, where it reads as a model that cannot follow instructions.
            log.warning("bedrock stopped at maxTokens=%d, so the reply is truncated",
                        self.max_tokens)
        if not text:
            # A response with no text block at all: all reasoning, or an empty content list. Returned
            # as a marker rather than as "" so that it fails where it happened. The caller appends
            # whatever comes back as an assistant turn and then parses it, so a marker takes the
            # existing invalid-reply path -- the loop tells the model its reply was unusable and asks
            # again, which is a recoverable turn. An empty string instead is accepted by the parser's
            # caller, stored, and then makes every later call in the conversation fail inside the API
            # for a reason that names neither the turn nor the model.
            log.warning("bedrock returned no text block (stopReason=%s), so there is nothing to "
                        "parse; handing back a marker so the loop can ask again",
                        resp.get("stopReason"))
            text = f"{EMPTY_REPLY} stopReason={resp.get('stopReason')!r}"
        return text, usage


def _content(text: str) -> list:
    """One Converse content list for a turn's text, with an empty turn filled in.

    Whitespace counts as empty here: the API's own check is on the block being non-empty, but a turn
    that is only whitespace carries nothing and reads to the model as a mistake either way.
    """
    return [{"text": text if (text or "").strip() else EMPTY_REPLY}]


def _text_of(resp) -> str:
    """Concatenate the text blocks of a Converse response, ignoring the rest.

    Reasoning blocks are skipped on purpose. They are the model's scratch work, they are not
    part of the contract in `prompt.SYSTEM`, and letting them into the parsed text would mean a
    reply's validity depended on whether a given model happened to emit them.
    """
    content = ((resp.get("output") or {}).get("message") or {}).get("content") or []
    return "".join(b["text"] for b in content if isinstance(b, dict) and "text" in b).strip()


def ask(reasoner: Reasoner, system: str, prompt: str, *, validate,
        max_retries: int = 2) -> tuple[object, Transcript]:
    """Ask, validate, and on an invalid reply hand the reason back and ask again.

    `validate(text)` returns the parsed decision or raises with a message meant for the model.
    The failed reply stays in the conversation: the model is shown its own words and told what
    was wrong with them, which is the only form of feedback that reliably fixes a format error.

    Raises the last validation error when the retries run out. That is deliberately not a
    fallback -- deciding what to do when the reasoner cannot answer belongs to the caller, which is
    the thing that knows whether the work can continue without it. `packer.py` drives the search's
    own conversation and so does not use this; it is here for the single questions Phase 7's
    hosting asks, and it is where the feedback rule was first written down.
    """
    t0 = time.time()
    transcript = Transcript(reasoner=reasoner.name, system=system,
                            turns=[Turn("user", prompt)])
    error = None
    for attempt in range(max_retries + 1):
        text, usage = reasoner.converse(system, transcript.turns)
        transcript.turns.append(Turn("assistant", text))
        transcript.usage = usage
        transcript.latency_s = time.time() - t0
        try:
            decision = validate(text)
        except Exception as exc:                            # noqa: BLE001 - handed to the model
            error = exc
            transcript.retries = attempt + 1
            if attempt == max_retries:
                break
            log.warning("the reasoner's reply was rejected (%s); asking again", exc)
            transcript.turns.append(Turn(
                "user", f"That reply cannot be used: {exc}\n\nSend the corrected JSON object, "
                        "and nothing else."))
            continue
        return decision, transcript
    raise error
