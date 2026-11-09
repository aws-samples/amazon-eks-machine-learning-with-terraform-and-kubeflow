"""Fixed workload construction (impl §6.2).

Input and output length are both *inputs* to the measurement, not outputs. Output
length is pinned by `min_tokens` + `ignore_eos`; input length has to be built here,
because a prompt of "about" the right length makes prefill cost a hidden variable.

**How input length is pinned, and why it is not computed here.** A tokenizer can
count text. It cannot count what a *multimodal* request costs: an image becomes some
number of visual tokens through the model's own preprocessor -- patch size, spatial
merge, a resize rule with its own rounding -- and the chat template adds its own
tokens around every part. Reimplementing that arithmetic here would put a copy of
each model family's preprocessor in the load driver, which is the same mistake
`agent/onboard.py` exists to refuse for the KV page size: predicting a number the
engine will state exactly.

So the length is *measured*. Before the run, each tenant is probed with the real
payload -- real images, real template -- at `max_tokens: 1`, and the server's own
`usage.prompt_tokens` says what the request costs. The text filler is then resized
and the probe repeated until the server agrees with `isl_tokens` exactly. Two
requests in the common case, none of them inside the measurement window, and the
harness learns nothing about any architecture.

That also closes a gap the text-only path always had: `isl_tokens_achieved` used to
be the *client's* tokenizer count, which differs from the server's whenever the
endpoint adds special tokens. `load.RequestResult.tokens` already made this call for
the output side -- the server's count is ground truth, the client's is a cross-check.
This is that rule applied to prefill.

**Images are generated, never shipped.** Cost follows resolution, not content: the
preprocessor turns pixels into visual tokens and the vision tower does the same work
whichever colours they are. So a synthetic gradient at the stated pixel count is a
faithful image payload, and it keeps the container free of a corpus.
"""

from __future__ import annotations

import base64
import io
import logging
import math

from . import engines

log = logging.getLogger(__name__)

# Filler with no natural stopping point. Content is irrelevant -- only token count
# is -- but it must tokenize densely and not trip a chat template.
_FILLER = (
    "the quantity of memory a tenant receives is decided at admission and does not "
    "change afterwards while the engine continues to serve requests at a fixed rate "
)

# How many distinct images to pre-render *beyond* one request's worth. The pool is
# `image_count + _POOL`, and both terms are there for a different reason.
#
# Why more than one image at all: an engine that hashes multimodal inputs can serve a
# repeated image out of a cache, and then the vision tower -- the entire reason for
# sending an image -- does no work while the record still says images were sent. The
# fleet file is expected to disable that cache (`--mm-processor-cache-gb=0`, and prefix
# caching with it), but a measurement that is only valid because two flags were
# remembered is the kind that gets voided later, and this repo has voided one already.
#
# Why `image_count` of them: a request must not repeat an image *within itself*. With a
# fixed pool of 8, a 24-image tenant sent variant `i % 8` three times per request --
# eight distinct images doing the work of twenty-four, in the one tenant where the
# encoder budget is the binding constraint. Sizing the pool above the request's own
# demand makes the indices of one request consecutive-and-distinct by construction.
#
# Why `+ _POOL` on top: it makes the pool length coprime-ish with the request length, so
# successive requests start at a different offset and send a different set rather than
# re-sending the first request's images forever.
#
# Why not one render per request: encoding a megapixel PNG costs tens of milliseconds on
# the driver, and the driver must never spend its schedule budget on itself (`load.py`).
# Rendering is start-up cost only.
_POOL = 8

# Probe attempts before giving up on an exact ISL. Each attempt is one request and the
# adjustment is exact to first order (one added text token adds one prompt token), so
# two is the normal case and more than three means something non-linear is happening --
# which is worth recording rather than looping over.
_MAX_PROBES = 6

# Text tokens for the first probe. Small on purpose: the first probe's only job is to
# price the images and the template, and a first guess of the full `isl_tokens` would
# ask a 24-image tenant for its images *plus* a full-length text prompt, which exceeds
# `max_model_len` and comes back as an HTTP 400 instead of a number.
_SEED_TEXT_TOKENS = 8


class ImagePool:
    """Pre-rendered synthetic images, all at the same pixel count.

    The pixel count is the whole specification. `pixels` is what the tenant's engine
    flags allow (`min_pixels`/`max_pixels`), and staying inside them is what stops the
    server's preprocessor from resizing -- a resize would leave the driver sending one
    resolution and the engine costing another, silently.
    """

    def __init__(self, pixels: int, count: int):
        self.pixels = int(pixels)
        if int(count) < 1:
            raise ValueError(f"image pool count must be >= 1, got {count}")
        # Square, so one number describes it and the aspect ratio cannot drift between
        # tenants. `isqrt` floors, so the rendered image is at or just under `pixels`
        # -- never over, which is the side that would trip `max_pixels`.
        self.side = max(1, math.isqrt(self.pixels))
        self.actual_pixels = self.side * self.side
        self.variants = [self._render(i) for i in range(count)]
        self.bytes_each = len(self.variants[0]) if self.variants else 0

    def _render(self, seed: int) -> str:
        from PIL import Image

        # Smooth blocks, not noise. Both cost the same on the GPU -- the tower sees a
        # fixed number of patches either way -- but noise does not compress, and a
        # 24-image request of incompressible megapixels would put tens of megabytes on
        # the wire per request and measure the upload inside TTFT.
        #
        # Drawn small and scaled up with NEAREST rather than filled pixel by pixel: a
        # megapixel `putdata` is a million-tuple Python loop per variant, which is
        # start-up time spent on nothing. The seed shifts every channel, so each variant
        # is a genuinely different image -- different bytes, different hash -- at an
        # identical pixel count and an identical decode cost.
        off = (seed * 37) % 256
        seed_img = Image.new("RGB", (16, 16))
        seed_img.putdata([((x * 16 + off) % 256, (y * 16 + off) % 256, off)
                          for y in range(16) for x in range(16)])
        img = seed_img.resize((self.side, self.side), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def data_url(self, i: int) -> str:
        return "data:image/png;base64," + self.variants[i % len(self.variants)]

    def describe(self) -> dict:
        return {
            "image_pixels_requested": self.pixels,
            "image_pixels_rendered": self.actual_pixels,
            "image_side_px": self.side,
            "image_variants": len(self.variants),
            "image_b64_bytes_each": self.bytes_each,
        }


class Workload:
    """A request whose input length the *server* agrees with, plus the output pin.

    Construction is two-phase on purpose. `__init__` is offline and cheap: it loads the
    tokenizer and renders the images. `calibrate` needs a live endpoint, because only
    the engine can say what the payload costs. Nothing that reads `payload` before
    `calibrate` gets a length anybody has checked, so `calibrate` is not optional --
    `__main__` runs it for every tenant after the readiness check and before load.
    """

    def __init__(self, isl_tokens: int, osl_tokens: int, tokenizer_path: str | None,
                 image_count: int = 0, image_pixels: int = 0, engine: str | None = None):
        # Which serving stack this request is addressed to. Held on the workload rather than passed to
        # `payload` so `load.py` is untouched by the seam -- and it is therefore part of the sharing
        # key in `build_workloads`, because two tenants on two engines do not send the same bytes.
        self.engine = engines.get(engine)
        self.isl_tokens = int(isl_tokens)
        self.osl_tokens = int(osl_tokens)
        # Kept, not just consumed: it goes into the record via `describe`, so a reader can see
        # which tokenizer produced this tenant's prompt without inferring it from the model.
        self.tokenizer_path = tokenizer_path
        self.image_count = int(image_count)
        if self.image_count and int(image_pixels) < 1:
            raise ValueError(
                f"image_count={self.image_count} with image_pixels={image_pixels}: an image "
                "request has to state its pixels, because that is what decides both the visual "
                "token count and the work the vision tower does.")
        # Asked now and thrown away, so that an engine which cannot serve this payload says so here
        # rather than at the first request. The engine decides -- this line does not know what any of
        # them refuse -- but the *timing* is not the engine's business: a refusal raised during load
        # has already spent the deploy, and one raised while assembling the workloads costs nothing.
        self.endpoint  # noqa: B018
        # Bigger than one request's demand, so a single request never repeats an image
        # and successive requests start at a different offset. See `_POOL`.
        self.images = (ImagePool(image_pixels, self.image_count + _POOL)
                       if self.image_count else None)

        self._tok = self._load_tokenizer(tokenizer_path)
        # Text length in tokens, as the client counts them. `calibrate` moves this until the
        # SERVER's total matches `isl_tokens`; the two differ by the template and the images.
        self.text_tokens = _SEED_TEXT_TOKENS if self.image_count else self.isl_tokens
        self.text = self._build_text(self.text_tokens)
        self.isl_achieved = -1
        self.isl_exact = False
        self.calibrated = False
        self.probes: list[dict] = []

    # -- construction -------------------------------------------------------------

    def _load_tokenizer(self, path: str | None):
        if not path:
            # Recorded, not silently tolerated: without a tokenizer the text length is
            # approximate, so the probe has a coarser lever and may not land exactly.
            log.warning("no tokenizer_path -- text length is approximate")
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, local_files_only=True)

    def _build_text(self, n_tokens: int) -> str:
        """Filler of exactly `n_tokens` tokens, as this tenant's tokenizer counts them."""
        n = max(1, int(n_tokens))
        if self._tok is None:
            # ~1 token per word is wrong but stable, which is the point: a wrong-and-
            # stable prompt still holds prefill constant across the tenants in one run.
            words = _FILLER.split()
            return " ".join((words * (n // len(words) + 2))[:n])

        tok = self._tok
        # Build long, slice to length, then correct. The slice-decode-reencode round
        # trip is not token-exact -- decoding merges or splits pieces at the
        # boundary -- so measure the result and adjust rather than trusting it.
        ids = tok(_FILLER * (n // 8 + 4), add_special_tokens=False)["input_ids"]
        if len(ids) < n:
            ids = (ids * (n // len(ids) + 2))[: n * 2]

        cut = n
        text = ""
        for _ in range(24):
            text = tok.decode(ids[:cut], skip_special_tokens=True)
            got = len(tok(text, add_special_tokens=False)["input_ids"])
            if got == n:
                return text
            cut += n - got
            cut = max(1, min(cut, len(ids)))
        log.warning("text build: wanted %d tokens, got %d", n, got)
        return text

    # -- what gets sent -----------------------------------------------------------

    @property
    def endpoint(self) -> str:
        """Where this request goes, which is the engine's business and depends on the payload."""
        return self.engine.endpoint(self.image_count)

    @property
    def osl_pinned(self) -> bool:
        """Whether output length is an input to this measurement or an observation of it.

        False is legitimate and is not a failure: an engine that cannot fix output length gets no
        hidden approximation, the achieved distribution is measured instead, and the comparison gets
        weaker while staying honest. What is NOT legitimate is reporting a pinned OSL that nothing
        pinned, so `describe` writes this into the record either way.
        """
        return bool(self.engine.pins_output and self.engine.pin_output(self.osl_tokens))

    def payload(self, served_name: str, index: int = 0) -> dict:
        """The request body. `index` selects which image variant, so successive requests
        differ in bytes while costing an identical amount of vision work."""
        body = {"model": served_name, "temperature": 0.0}
        # The output pin and the streaming fields are the engine's vocabulary; everything else in
        # this body is the request itself. See `engines/` for why the pin is load-bearing.
        body.update(self.engine.pin_output(self.osl_tokens))
        body.update(self.engine.stream_fields())
        body.update(self._input_fields(index))
        return body

    def _input_fields(self, index: int) -> dict:
        if not self.image_count:
            return {"prompt": self.text}
        parts = [{"type": "image_url",
                  "image_url": {"url": self.images.data_url(index * self.image_count + i)}}
                 for i in range(self.image_count)]
        parts.append({"type": "text", "text": self.text})
        return {"messages": [{"role": "user", "content": parts}]}

    def _probe_payload(self, served_name: str) -> dict:
        """The same input, priced rather than driven: one token out, no streaming.

        Everything that contributes to prompt length is identical to `payload` -- same
        images, same message structure, same text -- because a probe of a different
        request measures a different request.
        """
        body = {"model": served_name, "max_tokens": 1, "temperature": 0.0}
        body.update(self._input_fields(0))
        return body

    # -- calibration --------------------------------------------------------------

    async def calibrate(self, price, served_name: str) -> None:
        """Resize the text until the server's `prompt_tokens` equals `isl_tokens`.

        `price(payload) -> int | None` issues one non-streaming request and returns the
        server's prompt token count. Injected rather than imported so the offline tests
        can drive this loop without a server.

        The adjustment is exact to first order: one more text token is one more prompt
        token, and everything else about the request is fixed. What the loop is really
        absorbing is the constant nobody here can compute -- visual tokens plus template.
        """
        for attempt in range(_MAX_PROBES):
            got = await price(self._probe_payload(served_name))
            self.probes.append({"attempt": attempt + 1, "text_tokens": self.text_tokens,
                                "prompt_tokens": got})
            if got is None:
                log.error("probe %d returned no usage.prompt_tokens; ISL is unverified",
                          attempt + 1)
                break
            self.isl_achieved = int(got)
            if got == self.isl_tokens:
                self.isl_exact = True
                break
            nxt = self.text_tokens + (self.isl_tokens - got)
            if nxt < 1:
                # Not clamped to 1 and carried on with: the request cannot be built at
                # all, and the fleet file is where that is fixed. A tenant whose images
                # already cost more than its stated ISL is asking for a shape that does
                # not exist, and driving the nearest one that does would measure a
                # different workload under the stated one's name.
                raise ValueError(
                    f"cannot hit isl_tokens={self.isl_tokens}: this request costs "
                    f"{got - self.text_tokens} tokens before any text at all "
                    f"({self.image_count} image(s), plus whatever the endpoint's template "
                    f"adds), leaving no room for text. Raise isl_tokens above that, lower "
                    f"the image count, or lower the pixels per image.")
            self.text_tokens = nxt
            self.text = self._build_text(nxt)
        else:
            log.warning("ISL not exact after %d probes: wanted %d, last %d",
                        _MAX_PROBES, self.isl_tokens, self.isl_achieved)
        self.calibrated = True
        if not self.isl_exact:
            log.warning("isl_exact=False: prefill cost is approximate for this tenant")

    # -- provenance ---------------------------------------------------------------

    def describe(self) -> dict:
        d = {
            "isl_tokens_requested": self.isl_tokens,
            # The SERVER's count of the request it was sent, not the client's count of the
            # text it built. Those differ by the template and the images, and the difference
            # is the whole reason this is measured rather than computed.
            "isl_tokens_achieved": self.isl_achieved,
            "isl_exact": self.isl_exact,
            "isl_calibrated": self.calibrated,
            "isl_probes": self.probes,
            "text_tokens": self.text_tokens,
            "osl_tokens": self.osl_tokens,
            # Whether that OSL was an input or an observation. Recorded rather than assumed, because
            # every row in the store reads `osl_tokens` as a controlled variable and an engine that
            # cannot pin it makes that reading false. A reader seeing `osl_pinned: false` knows to
            # treat the figure as achieved rather than requested; a reader seeing no field at all is
            # looking at a record from when only vLLM existed, which did pin it.
            "osl_pinned": self.osl_pinned,
            "tokenizer_path": self.tokenizer_path,
            "engine": self.engine.name,
            "endpoint": self.endpoint,
            "image_count": self.image_count,
        }
        if self.images is not None:
            d.update(self.images.describe())
        return d
