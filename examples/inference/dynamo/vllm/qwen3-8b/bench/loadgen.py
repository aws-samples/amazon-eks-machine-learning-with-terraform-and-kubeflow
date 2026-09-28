"""Streaming load generator for an OpenAI-compatible endpoint.

Reports time-to-first-token, inter-token latency and output throughput at a fixed
concurrency. Standard library only, so it runs in any python image without a pip install.

Run it inside the cluster, against the frontend Service. A kubectl port-forward tunnel
becomes the bottleneck well before the GPUs do and would flatten the differences this is
meant to measure.

  python3 loadgen.py --host dyn-qwen3-8b-frontend --port 8000 \
      --model qwen3-8b --prompt-tokens 1000 --max-tokens 128 \
      --concurrency 1,4,16,32 --requests 64
"""

import argparse
import http.client
import json
import statistics
import sys
import threading
import time

# Deliberately prose rather than random tokens: random strings tokenize badly and defeat
# the prefix cache in ways real traffic does not. Each request gets a unique prefix so
# requests do not share cached prefill work with each other.
FILLER = (
    "The transformer architecture processes sequences by attending over all positions "
    "simultaneously, which makes the prefill phase compute bound and the decode phase "
    "memory bandwidth bound. This asymmetry is the motivation for serving the two phases "
    "on separate accelerators. "
)


def build_prompt(target_tokens, salt):
    # ~4 characters per token is close enough for a synthetic prompt; the exact length is
    # reported back from the server's usage field where available.
    reps = max(1, (target_tokens * 4) // len(FILLER))
    return f"[request {salt}] " + (FILLER * reps)


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
            # Greedy, and ignore the EOS token, so every request emits exactly max_tokens.
            # Without this, output length varies per request and throughput numbers stop
            # being comparable between configurations.
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
                build_prompt(args.prompt_tokens, f"{label}-{i}"),
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
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--tag", default="RESULT")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    print(
        f"# host={args.host}:{args.port} model={args.model} "
        f"prompt_tokens~{args.prompt_tokens} max_tokens={args.max_tokens}",
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
        # Let the engine settle and any prefix cache from warmup age out of relevance.
        time.sleep(5)

    rows = []
    for c in levels:
        # Scale total requests with concurrency so every level runs long enough to be
        # steady state rather than dominated by ramp-up.
        n = max(args.requests, c * 4)
        res, wall = run_phase(args, c, n, f"c{c}")
        row = summarize(args.tag, c, res, wall)
        if row:
            rows.append(row)
        time.sleep(3)

    print("# JSON_SUMMARY " + json.dumps({"tag": args.tag, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
