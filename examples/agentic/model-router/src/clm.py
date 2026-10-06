"""Client for clm-serve's /v1/systemone endpoint."""
from __future__ import annotations

import time

import requests


class CLMError(RuntimeError):
    """clm-serve failed or returned something unusable."""


class CLM:
    def __init__(self, url: str, model: str = "clm-latest", api_key: str | None = None, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def healthy(self) -> bool:
        """True when clm-serve is up and can reach its encoder."""
        try:
            return bool(self.session.get(f"{self.url}/health", timeout=5).json().get("embedder"))
        except (requests.RequestException, ValueError):
            return False

    def ask(self, state: str, questions: dict) -> tuple[dict, float]:
        """Ask typed questions about a state. Returns the answers and the call's wall time in ms."""
        body = {"state": state, "model": self.model, "questions": questions}
        t0 = time.perf_counter()
        try:
            # A call has no side effects, so retry connection errors: a pooled keep-alive connection
            # the server already closed, or a kubectl port-forward that is reconnecting.
            for attempt in range(4):
                try:
                    r = self.session.post(f"{self.url}/v1/systemone", json=body, timeout=self.timeout)
                    break
                except requests.ConnectionError:
                    if attempt == 3:
                        raise
                    time.sleep(2 * attempt)
            r.raise_for_status()
            answers = r.json()["answers"]
        except (requests.RequestException, KeyError, ValueError) as e:
            raise CLMError(f"clm-serve: {e}") from e
        return answers, (time.perf_counter() - t0) * 1000
