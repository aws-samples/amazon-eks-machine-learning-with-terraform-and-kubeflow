# CLM-8B load test

`loadtest.py` checks that `clm-serve` and the encoder complete every request under concurrent load. At each concurrency level it sends 96 requests. Each asks three typed questions, one `choice`, one `score` and one `noul`, about a state from [states.jsonl](./states.jsonl). Each state gets a unique prefix, so no request is answered from the state cache.

## Run

Run it inside the cluster, next to the services. Through `kubectl port-forward`, concurrent connections break the tunnel, and the errors measure the tunnel instead of `clm-serve`. Step 7 of [serve.ipynb](../serve.ipynb) runs it as a Kubernetes Job and copies `results/loadtest.json` back. Inside the cluster, the command is:

```bash
pip install -r requirements.txt
python loadtest.py --url http://clm-serve:8700 --encoder-metrics-url http://clm-encoder:8000/metrics \
    --concurrency 1,2,4,8,16,32
```

If `clm-serve` was installed with `CLM_API_KEY`, export the same `CLM_API_KEY` before running.

## What it reports

For each concurrency level, the number of requests and the number of errors, with up to three error messages. A request counts as an error if it fails, returns an incomplete answer, or does not complete within the client's 60-second timeout. The results go to `results/loadtest.json`, which is git-ignored. The script does not record latency or throughput.

With `--encoder-metrics-url`, the script also reads the encoder's `vllm:num_requests_running` five seconds after each level. It should be 0. A non-zero value on an idle server is the symptom that made `clm-encoder.yaml` turn off chunked prefill.

## Mock encoder

The upstream repository ships `tools/playground_mock.py`, which runs the real `clm-serve` app on character n-grams instead of Qwen3-8B. Use it to work on the script without a GPU. `/health` reports `"mock": true`; the script then prints a warning and sets `"mock": true` in `loadtest.json`. A passing run against the mock says nothing about CLM-8B.
