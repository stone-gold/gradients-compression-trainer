"""Landing rule for the freeform track: which calibration ratio to aim at, and which candidate to ship.

Score (docs/miner_compression.md#freeform): S = bytes * exp(curv * max(0, x)^2), x = r - knee, r = test_loss / reference;
over the cap an entry ranks behind every entry under it. The penalty has zero slope at the knee, while bytes fall
steadily with r (a uniform quantiser with an entropy coder loses ~6 dB of noise per bit: ln bytes drops by about
k = 6-12 per unit of r near r - 1 = 0.04 on 1-8B models). So the deterministic optimum sits a little PAST the knee, and with
noise on our estimate of the test r the optimum aim is the knee plus delta, where delta solves

    d/dmu E[curv * max(0, mu + e)^2] = -k,  e ~ N(0, sigma^2)   <=>   mu Phi(mu/sigma) + sigma phi(mu/sigma) = -k / (2 curv)

sigma combines the sampling error of the calibration mean (from the per-row loss changes we measure), the same for the
scored set (QUANT_FF_TEST_ROWS rows, unknown: 1,000 by the eval-set rule), and a floor for engine / GPU differences
(HF here vs vLLM on the scorer's SKU). Each candidate is then ranked by its expected ln S under the same noise model.

The only place that knows how the target is expressed is `Target` below: today the knee is a ratio of the reference
loss (1.04); from 14 Dec 2026 the roadmap (incoming_changes.yaml, compression-nats-budget) replaces it with a fixed
budget in nats per token. Switching is the constant TARGET_MODE (or QUANT_FF_TARGET_MODE) plus the budget field.
"""

import math
import os
from dataclasses import dataclass

# "ratio": knee and cap are ratios of the reference loss (QUANT_TARGET knee / cap), penalty on (r - knee).
# "nats":  knee = reference + budget nats (QUANT_TARGET knee_loss, or reference_loss + QUANT_FF_NATS_BUDGET), penalty on
#          the excess in nats. All search arithmetic stays in ratio space: knee_r = knee_loss / reference_loss.
TARGET_MODE = os.environ.get("QUANT_FF_TARGET_MODE", "ratio")
CAP_COST = 20.0  # nats charged per unit probability of landing over the cap (exp(20): never worth it)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_sq_excess(mu: float, sigma: float) -> float:
    """E[max(0, X)^2], X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        return max(0.0, mu) ** 2
    z = mu / sigma
    return (mu * mu + sigma * sigma) * _Phi(z) + mu * sigma * _phi(z)


@dataclass
class Target:
    knee_r: float  # knee as a ratio of the reference loss
    cap_r: float
    curv: float  # per unit of the excess measure (ratio, or nats in "nats" mode)
    scale: float  # excess measure per unit of r: 1 (ratio) or reference_loss (nats)

    @classmethod
    def from_quant_target(cls, t: dict) -> "Target":
        ref = t.get("reference_loss")
        curv = float(t.get("penalty_curvature") if t.get("penalty_curvature") is not None else 2500.0)
        if TARGET_MODE == "nats" and ref:
            ref = float(ref)
            budget = os.environ.get("QUANT_FF_NATS_BUDGET")
            knee_loss = float(t["knee_loss"]) if t.get("knee_loss") else ref + float(budget or 0.04 * ref)
            cap_loss = float(t["cap_loss"]) if t.get("cap_loss") else knee_loss + 0.04 * ref
            return cls(knee_r=knee_loss / ref, cap_r=cap_loss / ref, curv=curv, scale=ref)
        return cls(knee_r=float(t.get("knee") or 1.04), cap_r=float(t.get("cap") or 1.08), curv=curv, scale=1.0)

    def penalty_ln(self, r: float) -> float:
        return self.curv * (max(0.0, r - self.knee_r) * self.scale) ** 2

    def expected_ln_s(self, nbytes: int, r_pred: float, sigma: float) -> float:
        """E[ln S] for a candidate whose test r is ~ N(r_pred, sigma^2), plus CAP_COST nats times P(r > cap) (over the
        cap an entry ranks behind every under-cap one). Never inf below the cap: a noisy estimate must not make every
        compressed candidate lose to the W8 fallback (t15 with sigma 0.013 did, before this)."""
        if r_pred > self.cap_r:
            return float("inf")
        return (math.log(max(1, nbytes)) + self.curv * self.scale**2 * expected_sq_excess(r_pred - self.knee_r, sigma)
                + CAP_COST * self.p_over(r_pred, sigma, self.cap_r))

    def p_over(self, r_pred: float, sigma: float, level: float) -> float:
        """P(test r > level) under the noise model."""
        if sigma <= 0:
            return float(r_pred > level)
        return 1.0 - _Phi((level - r_pred) / sigma)

    def aim(self, sigma: float, slope: float) -> float:
        """Calibration r to aim at: knee + delta with delta from the first-order condition above (slope = d ln bytes / dr < 0)."""
        need = -slope / (2.0 * self.curv * self.scale**2)
        lo, hi = -0.05, 0.05
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            g = mid * _Phi(mid / sigma) + sigma * _phi(mid / sigma) if sigma > 0 else max(0.0, mid)
            lo, hi = (mid, hi) if g < need else (lo, mid)
        return self.knee_r + 0.5 * (lo + hi)


def sigma_of(deltas: list[float], ref_mean: float, n_test: int, floor: float) -> float:
    """Std of (test r - calibration r): sampling of both means (per-row loss change / reference mean) plus a floor."""
    n = len(deltas)
    if n < 2 or ref_mean <= 0:
        return math.sqrt(floor**2 + 0.003**2)
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var) / ref_mean
    return math.sqrt(sd * sd / n + sd * sd / max(1, n_test) + floor * floor)


def bytes_slope(points: list[tuple[float, int]], default: float = -8.0) -> float:
    """d ln bytes / dr from (r, bytes) candidates near the knee (least squares), clamped to [-40, -2]."""
    pts = [(r, math.log(b)) for r, b in points if 1.005 < r < 1.08 and b > 0]
    if len(pts) < 2:
        return default
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx < 1e-10:
        return default
    k = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return max(-40.0, min(-2.0, k))
