"""Streaming load generator for an OpenAI-compatible endpoint.

Reports time-to-first-token, inter-token latency and output throughput at a fixed
concurrency. Standard library only, so it runs in any python image without a pip install.

Run it inside the cluster, against the frontend Service: a kubectl port-forward tunnel becomes
the bottleneck well before the GPUs do.

  python3 loadgen.py --host dyn-qwen3-8b-frontend --port 8000 \
      --model qwen3-8b --prompt-tokens 1000 --max-tokens 128 \
      --concurrency 1,4,16,32 --requests 64

By default every request carries a unique prefix, which is what you want for measuring topology:
shared prefill work would let one arm look faster because it recomputed less.
`--shared-prefix-tokens` inverts that, for measuring KV-aware *routing*. See build_prompt.
"""

import argparse
import hashlib
import http.client
import json
import statistics
import sys
import threading
import time

# Prose rather than random tokens: random strings tokenize badly and defeat the prefix cache in
# ways real traffic does not.
#
# "Unique prefix" means unique in the first block only -- the body is this text repeated, so two
# prompts diverge just in how the salt shifted the block alignment. Enough to keep requests from
# sharing prefill in practice, which is why the run marker matters. See build_prompt.
FILLER = (
    "The transformer architecture processes sequences by attending over all positions "
    "simultaneously, which makes the prefill phase compute bound and the decode phase "
    "memory bandwidth bound. This asymmetry is the motivation for serving the two phases "
    "on separate accelerators. "
)


# Different text from FILLER so a shared prefix cannot be confused with repeated filler.
SHARED = (
    "You are a deployment assistant for a Kubernetes cluster that serves large language "
    "models. Answer using only the cluster's own conventions: workloads are scheduled by "
    "Karpenter onto NodePools, GPU nodes carry a taint that pods must tolerate, and model "
    "weights are read from a shared filesystem mounted at /fsx. "
)


def pick_group(index, groups):
    """Which shared-prefix group request `index` belongs to.

    Hashed rather than `index % groups`. Round-robin sends request i to worker `i % n_workers`,
    so if n_workers divides n_groups each group lands on exactly one worker forever and
    round-robin gets perfect cache affinity for free -- scoring identically to a KV-aware router.

    Hashing decouples the group sequence from the router's phase while staying deterministic.
    """
    if groups <= 1:
        return 0
    digest = hashlib.blake2b(str(index).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % groups


def build_prompt(target_tokens, salt, shared_tokens=0, group=0, run=""):
    # ~4 characters per token is close enough; exact lengths come back in the usage field.
    if shared_tokens <= 0:
        # The `run` marker belongs here too, not only in shared-prefix mode: without it two
        # invocations at the same concurrency generate byte-identical prompts, and the second
        # reads as a near-total prefix cache hit off the first.
        reps = max(1, (target_tokens * 4) // len(FILLER))
        return f"[request {run} {salt}] " + (FILLER * reps)

    # Shared-prefix mode, for exercising KV-aware routing rather than topology. The shared block
    # comes FIRST and is byte-identical within a group, so it is a true prefix; the unique tail
    # keeps no two requests identical.
    #
    # Use more groups than workers. With one shared prefix every worker ends up holding it and
    # any router looks good; the same happens when groups == worker count. Letting the group
    # sequence line up with the router's phase is the other trap -- see pick_group. Neither shows
    # up as an error, only as two routing modes scoring the same.
    #
    # The `run` marker, seeded from --tag, makes each invocation's prefixes new text, so each run
    # starts from a cold prefix cache without restarting the workers between arms.
    shared_reps = max(1, (shared_tokens * 4) // len(SHARED))
    prefix = f"[context {run} group {group}] " + (SHARED * shared_reps)
    unique_tokens = max(1, target_tokens - shared_tokens)
    reps = max(1, (unique_tokens * 4) // len(FILLER))
    return prefix + f" [request {salt}] " + (FILLER * reps)


class Result:
    __slots__ = ("ttft", "e2e", "tokens", "itls", "error")

    def __init__(self):
        self.ttft = None
        self.e2e = None
        self.tokens = 0
        self.itls = []
        self.error = None


def one_request(host, port, model, prompt, max_tokens, timeout):
    r = Result()
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            # Greedy and ignoring EOS, so every request emits exactly max_tokens -- otherwise
            # output length varies and throughput stops being comparable.
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
    )
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.perf_counter()
    last = start
    try:
        conn.request(
            "POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        if resp.status != 200:
            r.error = f"HTTP {resp.status}: {resp.read()[:200]!r}"
            return r
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if not text:
                continue
            now = time.perf_counter()
            if r.ttft is None:
                r.ttft = now - start
            else:
                r.itls.append(now - last)
            last = now
            r.tokens += 1
        r.e2e = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - any failure is just a failed sample
        r.error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return r


def run_phase(args, concurrency, n_requests, label):
    results = []
    lock = threading.Lock()
    counter = {"i": 0}

    def worker():
        while True:
            with lock:
                i = counter["i"]
                if i >= n_requests:
                    return
                counter["i"] = i + 1
            res = one_request(
                args.host,
                args.port,
                args.model,
                build_prompt(
                    args.prompt_tokens,
                    f"{label}-{i}",
                    args.shared_prefix_tokens,
                    pick_group(i, max(1, args.shared_prefix_groups)),
                    args.tag,
                ),
                args.max_tokens,
                args.timeout,
            )
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start
    return results, wall


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(tag, concurrency, results, wall):
    ok = [r for r in results if r.error is None and r.ttft is not None]
    bad = [r for r in results if r.error is not None]
    if not ok:
        print(f"{tag} concurrency={concurrency}: ALL {len(results)} REQUESTS FAILED")
        for r in bad[:3]:
            print(f"    {r.error}")
        return None
    ttfts = [r.ttft * 1000 for r in ok]
    e2es = [r.e2e * 1000 for r in ok]
    itls = [v * 1000 for r in ok for v in r.itls]
    total_tokens = sum(r.tokens for r in ok)
    row = {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(bad),
        "ttft_mean_ms": round(statistics.mean(ttfts), 1),
        "ttft_p50_ms": round(pct(ttfts, 50), 1),
        "ttft_p99_ms": round(pct(ttfts, 99), 1),
        "itl_mean_ms": round(statistics.mean(itls), 2) if itls else None,
        "itl_p99_ms": round(pct(itls, 99), 2) if itls else None,
        "e2e_mean_ms": round(statistics.mean(e2es), 1),
        "output_tok_per_s": round(total_tokens / wall, 1),
        "req_per_s": round(len(ok) / wall, 3),
        "tokens_per_req": round(total_tokens / len(ok), 1),
    }
    print(f"{tag} " + json.dumps(row))
    if bad:
        print(f"    {len(bad)} failed, first: {bad[0].error}")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-tokens", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", default="1,4,16,32")
    p.add_argument("--requests", type=int, default=64)
    # 0 keeps the default unique-prefix behaviour, so existing invocations are unaffected.
    p.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=0,
        help="prepend this many tokens of identical text per group, for KV-routing tests",
    )
    p.add_argument(
        "--shared-prefix-groups",
        type=int,
        default=2,
        help="number of distinct shared prefixes; use several times the worker replica count, "
        "not equal to it -- see build_prompt",
    )
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--tag", default="RESULT")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    if args.shared_prefix_tokens >= args.prompt_tokens:
        print("# --shared-prefix-tokens must be less than --prompt-tokens")
        sys.exit(2)

    print(
        f"# host={args.host}:{args.port} model={args.model} "
        f"prompt_tokens~{args.prompt_tokens} max_tokens={args.max_tokens} "
        f"shared_prefix_tokens={args.shared_prefix_tokens} "
        f"shared_prefix_groups={args.shared_prefix_groups if args.shared_prefix_tokens else 0}",
        flush=True,
    )

    if args.warmup:
        print(f"# warmup: {args.warmup} requests", flush=True)
        wres, _ = run_phase(args, min(args.warmup, 4), args.warmup, "warmup")
        failed = [r for r in wres if r.error]
        if len(failed) == len(wres):
            print("# warmup failed entirely, aborting")
            for r in failed[:3]:
                print(f"#   {r.error}")
            sys.exit(1)
        # Let the engine settle. In shared-prefix mode warmup deliberately carries over: it seeds
        # each group's prefix onto whichever worker served it, which is the steady state a KV
        # router is meant to exploit.
        time.sleep(5)

    rows = []
    for c in levels:
        # Scale total requests with concurrency so every level reaches steady state.
        n = max(args.requests, c * 4)
        res, wall = run_phase(args, c, n, f"c{c}")
        row = summarize(args.tag, c, res, wall)
        if row:
            rows.append(row)
        time.sleep(3)

    print("# JSON_SUMMARY " + json.dumps({"tag": args.tag, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
