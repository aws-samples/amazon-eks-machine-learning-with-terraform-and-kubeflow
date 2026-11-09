"""Batch means: one estimator for both the steady-state criterion and the window
length rule (impl §6.4).

Steady-state output from a queueing system is autocorrelated, so the standard error
of the mean understates the true uncertainty -- consecutive samples carry much of
the same information. Batch means recovers an honest interval by averaging within
batches first: the batch means are far closer to independent than the raw samples,
so their spread is a usable estimate of run-to-run spread.
"""

from __future__ import annotations

import numpy as np

# impl §6.4 fixes k=10. Do not tune it per signal: the number of batches sets the
# degrees of freedom of the interval, so varying k would make thresholds on
# different signals mean different confidence levels.
DEFAULT_K = 10

# Known blind spot, recorded so it is not mistaken for a bug: averaging within batches
# aliases. A signal oscillating with a period that divides the batch length gives every
# batch the same mean, so the spread between batch means is zero and the series looks
# settled. This is why the steady-state criterion does not rest on one signal -- the
# in-flight count drifts monotonically under overload rather than periodically, so it
# catches the case the estimator cannot see.


def batch_means_ci(samples, k: int = DEFAULT_K) -> float | None:
    """Confidence half-width for correlated steady-state output.

    Returns None when there is not enough data to form k batches of at least 2
    each -- that is "not yet known", which the callers must treat differently from
    a wide interval. Collapsing the two would let a run converge by having no data.
    """
    n = len(samples) // k
    if n < 2:
        return None
    arr = np.asarray(samples, dtype=float)
    batches = [arr[i * n:(i + 1) * n].mean() for i in range(k)]
    return float(1.96 * np.std(batches, ddof=1) / np.sqrt(k))


def relative_halfwidth(samples, k: int = DEFAULT_K) -> float | None:
    """Half-width as a fraction of the mean.

    Used for the steady-state signals whose absolute scale is not known in advance
    (delivered rate, in-flight count). A fixed absolute threshold on those would
    mean something different at 2 req/s than at 200.
    """
    hw = batch_means_ci(samples, k)
    if hw is None:
        return None
    mean = float(np.mean(samples))
    if mean == 0.0:
        # Zero mean with zero spread is settled; zero mean with spread is not.
        return 0.0 if hw == 0.0 else float("inf")
    return hw / abs(mean)
