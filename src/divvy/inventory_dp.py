from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

COUNT_BUCKETS = ("0", "1", "2", "3", "4", "5_plus")


@dataclass(frozen=True)
class InventoryRolloutResult:
    p_has_ebike: float
    p_zero: float
    expected_ebikes: float
    expected_total_bikes: float
    p_count_ebikes: dict[str, float]
    p_count_total: dict[str, float]
    p_capacity_violation: float
    p_dock_constrained_arrival: float
    expected_ebike_departures: float
    expected_classic_departures: float
    expected_ebike_arrivals: float
    expected_classic_arrivals: float
    p_joint_e_q: np.ndarray | None = None
    p_count_ebikes_full: list[float] | None = None


def _safe_int(value: int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if not math.isfinite(float(value)):
            return default
    except (TypeError, ValueError):
        return default
    return int(round(float(value)))


def _bounded_poisson_support(
    mean: float,
    capacity: int,
    *,
    max_support: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact Poisson support with the tail mapped to capacity."""
    capacity = max(0, int(capacity))
    if capacity == 0:
        return np.array([0], dtype=int), np.array([1.0], dtype=float)

    mean = float(mean) if math.isfinite(float(mean)) else 0.0
    mean = max(0.0, mean)
    dynamic_support = int(math.ceil(mean + 6.0 * math.sqrt(mean + 1.0)))
    support_limit = max(2, min(capacity, max_support, dynamic_support))

    if support_limit >= capacity:
        counts = np.arange(capacity + 1, dtype=int)
        exact_count_limit = capacity
    else:
        counts = np.concatenate(
            [np.arange(support_limit, dtype=int), np.array([capacity], dtype=int)]
        )
        exact_count_limit = support_limit - 1

    probs = np.zeros(len(counts), dtype=float)
    if mean <= 1e-12:
        probs[0] = 1.0
        return counts, probs

    p = math.exp(-min(mean, 700.0))
    running = 0.0
    for idx, k in enumerate(range(exact_count_limit + 1)):
        if k == 0:
            p = math.exp(-min(mean, 700.0))
        elif p == 0.0:
            p = 0.0
        else:
            p *= mean / k
        probs[idx] = p
        running += p

    tail = max(0.0, 1.0 - running)
    if exact_count_limit < capacity:
        probs[-1] = tail
    else:
        probs[-1] += tail

    total = probs.sum()
    if not math.isfinite(float(total)) or total <= 0:
        probs[:] = 0.0
        probs[0] = 1.0
    else:
        probs /= total
    return counts, probs


def _normalize_pmf(pmf: np.ndarray) -> np.ndarray:
    arr = np.asarray(pmf, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if total <= 0.0:
        arr[:] = 0.0
        if arr.size:
            arr.flat[0] = 1.0
        return arr
    return arr / total


def nb_pmf(mean: float, theta: float, max_k: int) -> np.ndarray:
    """Negative-binomial PMF with finite support and tail mass in the last bin."""
    max_k = max(0, int(max_k))
    mean = float(mean) if np.isfinite(float(mean)) else 0.0
    theta = float(theta) if np.isfinite(float(theta)) else 0.0
    mean = max(0.0, mean)
    theta = max(1e-6, theta)
    if max_k == 0 or mean <= 1e-12:
        out = np.zeros(max_k + 1, dtype=float)
        out[0] = 1.0
        return out

    p = theta / (theta + mean)
    log_p = math.log(max(1e-12, min(1.0 - 1e-12, p)))
    log_1mp = math.log(max(1e-12, 1.0 - p))
    probs = np.zeros(max_k + 1, dtype=float)
    running = 0.0
    for k in range(max_k):
        log_prob = (
            math.lgamma(k + theta)
            - math.lgamma(theta)
            - math.lgamma(k + 1)
            + theta * log_p
            + k * log_1mp
        )
        value = math.exp(max(-745.0, min(80.0, log_prob)))
        probs[k] = value
        running += value
    probs[max_k] = max(0.0, 1.0 - running)
    return _normalize_pmf(probs)


def zinb_pmf(mean: float, theta: float, zero_inflation: float, max_k: int) -> np.ndarray:
    """Zero-inflated negative-binomial PMF with finite support."""
    zeta = float(zero_inflation) if np.isfinite(float(zero_inflation)) else 0.0
    zeta = max(0.0, min(1.0, zeta))
    base = nb_pmf(mean, theta, max_k)
    out = (1.0 - zeta) * base
    if out.size:
        out[0] += zeta
    return _normalize_pmf(out)


def collapse_count_distribution(pmf: np.ndarray) -> dict[str, float]:
    """Collapse a full count PMF into stable API buckets 0, 1, 2, 3, 4, 5_plus."""
    arr = np.asarray(pmf, dtype=float)
    if arr.size == 0 or arr.sum() <= 0:
        return {bucket: 0.0 for bucket in COUNT_BUCKETS}
    arr = arr / arr.sum()
    out = {
        "0": float(arr[0]) if arr.size > 0 else 0.0,
        "1": float(arr[1]) if arr.size > 1 else 0.0,
        "2": float(arr[2]) if arr.size > 2 else 0.0,
        "3": float(arr[3]) if arr.size > 3 else 0.0,
        "4": float(arr[4]) if arr.size > 4 else 0.0,
        "5_plus": float(arr[5:].sum()) if arr.size > 5 else 0.0,
    }
    total = sum(out.values())
    if total > 0:
        out = {key: float(value / total) for key, value in out.items()}
    return out


def _support_from_pmf(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _normalize_pmf(np.asarray(pmf, dtype=float))
    return np.arange(probs.size, dtype=int), probs


def censored_departure_log_prob(pmf_depart: np.ndarray, observed_completed: int, available: int) -> float:
    """Log P(min(U, available)=observed_completed) for censored departures."""
    pmf = _normalize_pmf(pmf_depart)
    x = max(0, int(observed_completed))
    available = max(0, int(available))
    if x < available:
        prob = pmf[x] if x < pmf.size else 0.0
    elif x == available:
        prob = float(pmf[available:].sum()) if available < pmf.size else 0.0
    else:
        prob = 0.0
    return math.log(max(1e-12, prob))


def censored_arrival_log_prob(pmf_arrive: np.ndarray, observed_accepted: int, open_docks: int) -> float:
    """Log P(min(A, open_docks)=observed_accepted) for censored arrivals."""
    return censored_departure_log_prob(pmf_arrive, observed_accepted, open_docks)


def censored_departure_nll(pmf_depart: np.ndarray, observed_completed: int, available: int) -> float:
    return -censored_departure_log_prob(pmf_depart, observed_completed, available)


def censored_arrival_nll(pmf_arrive: np.ndarray, observed_accepted: int, open_docks: int) -> float:
    return -censored_arrival_log_prob(pmf_arrive, observed_accepted, open_docks)


def transition_distribution(
    capacity: int,
    pi: np.ndarray,
    pmf_ebike_depart: np.ndarray,
    pmf_classic_depart: np.ndarray,
    pmf_ebike_arrive: np.ndarray,
    pmf_classic_arrive: np.ndarray,
    is_renting: bool = True,
    is_returning: bool = True,
) -> tuple[np.ndarray, dict]:
    """One-minute finite-state transition over (eBike count, total bike count).

    All output mass is explicitly constrained to states satisfying
    ``0 <= E_next <= Q_next <= capacity``.
    """
    c = max(0, int(capacity))
    state = np.asarray(pi, dtype=float)
    if state.shape != (c + 1, c + 1):
        raise ValueError("pi must have shape (capacity + 1, capacity + 1)")

    valid_mask = np.zeros_like(state, dtype=bool)
    for e in range(c + 1):
        valid_mask[e, e : c + 1] = True
    state = np.where(valid_mask, state, 0.0)
    state = _normalize_pmf(state)

    if not is_renting:
        pmf_ebike_depart = np.array([1.0], dtype=float)
        pmf_classic_depart = np.array([1.0], dtype=float)
    if not is_returning:
        pmf_ebike_arrive = np.array([1.0], dtype=float)
        pmf_classic_arrive = np.array([1.0], dtype=float)

    de_counts, de_probs = _support_from_pmf(pmf_ebike_depart)
    dc_counts, dc_probs = _support_from_pmf(pmf_classic_depart)
    ae_counts, ae_probs = _support_from_pmf(pmf_ebike_arrive)
    ac_counts, ac_probs = _support_from_pmf(pmf_classic_arrive)

    out = np.zeros_like(state)
    p_dock_constrained = 0.0
    expected_de = 0.0
    expected_dc = 0.0
    expected_ae = 0.0
    expected_ac = 0.0

    for e in range(c + 1):
        for q in range(e, c + 1):
            state_mass = float(state[e, q])
            if state_mass <= 0.0:
                continue
            classic = q - e
            for ue, p_ue in zip(de_counts, de_probs):
                if p_ue <= 0.0:
                    continue
                x_e = min(e, int(ue))
                for uc, p_uc in zip(dc_counts, dc_probs):
                    depart_mass = state_mass * float(p_ue) * float(p_uc)
                    if depart_mass <= 0.0:
                        continue
                    x_c = min(classic, int(uc))
                    open_docks = c - q + x_e + x_c
                    for ae, p_ae in zip(ae_counts, ae_probs):
                        e_arrive_mass = depart_mass * float(p_ae)
                        if e_arrive_mass <= 0.0:
                            continue
                        r_e = min(int(ae), open_docks)
                        remaining_docks = open_docks - r_e
                        for ac, p_ac in zip(ac_counts, ac_probs):
                            mass = e_arrive_mass * float(p_ac)
                            if mass <= 0.0:
                                continue
                            r_c = min(int(ac), remaining_docks)
                            e_next = e - x_e + r_e
                            q_next = q - x_e - x_c + r_e + r_c
                            if 0 <= e_next <= q_next <= c:
                                out[e_next, q_next] += mass
                            expected_de += mass * x_e
                            expected_dc += mass * x_c
                            expected_ae += mass * r_e
                            expected_ac += mass * r_c
                            if int(ae) + int(ac) > open_docks:
                                p_dock_constrained += mass

    out = _normalize_pmf(out)
    invalid_mass = float(out[~valid_mask].sum())
    if invalid_mass:
        out[~valid_mask] = 0.0
        out = _normalize_pmf(out)
    metrics = {
        "p_capacity_violation": 0.0,
        "p_dock_constrained_arrival": float(min(1.0, max(0.0, p_dock_constrained))),
        "expected_ebike_departures": float(expected_de),
        "expected_classic_departures": float(expected_dc),
        "expected_ebike_arrivals": float(expected_ae),
        "expected_classic_arrivals": float(expected_ac),
    }
    return out, metrics


def _intensity_value(intensity: Any, *keys: str, default: float) -> float:
    if isinstance(intensity, dict):
        value = next((intensity[key] for key in keys if key in intensity), default)
    else:
        value = next((getattr(intensity, key) for key in keys if hasattr(intensity, key)), default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return value


def _flow_pmf(intensity: Any, prefix: str, capacity: int) -> np.ndarray:
    mean = _intensity_value(
        intensity,
        f"{prefix}_mean",
        f"mean_{prefix}",
        f"{prefix}ure_mean" if prefix.endswith("depart") else f"{prefix}_mean",
        default=0.0,
    )
    aliases = {
        "ebike_depart": ("ebike_departure_mean", "mu_ebike_depart", "mu_e_depart"),
        "classic_depart": ("classic_departure_mean", "mu_classic_depart", "mu_c_depart"),
        "ebike_arrive": ("ebike_arrival_mean", "mu_ebike_arrive", "mu_e_arrive"),
        "classic_arrive": ("classic_arrival_mean", "mu_classic_arrive", "mu_c_arrive"),
    }
    if prefix in aliases:
        mean = _intensity_value(intensity, *aliases[prefix], f"{prefix}_mean", default=mean)
    theta = _intensity_value(intensity, f"{prefix}_theta", f"theta_{prefix}", default=20.0)
    zeta = _intensity_value(
        intensity,
        f"{prefix}_zero_inflation",
        f"{prefix}_zeta",
        f"zeta_{prefix}",
        default=0.0,
    )
    support = max(1, min(int(capacity), int(math.ceil(mean + 6.0 * math.sqrt(mean + 1.0))), 5))
    return zinb_pmf(mean, theta, zeta, support)


def rollout_inventory_distribution_multistep(
    capacity,
    current_ebikes,
    current_total_bikes,
    intensity_sequence,
    max_capacity: int = 80,
    return_joint: bool = False,
) -> InventoryRolloutResult:
    """Roll one-minute constrained transitions for a sequence of flow intensities."""
    c = _safe_int(capacity, 1)
    c = max(1, min(max_capacity, c))
    e0 = max(0, min(c, _safe_int(current_ebikes, 0)))
    q0 = max(e0, min(c, _safe_int(current_total_bikes, e0)))

    if isinstance(intensity_sequence, dict):
        sequence = [intensity_sequence]
    else:
        sequence = list(intensity_sequence or [])
    if not sequence:
        sequence = [{}]

    state = np.zeros((c + 1, c + 1), dtype=float)
    state[e0, q0] = 1.0
    constrained_survival = 1.0
    expected_de = 0.0
    expected_dc = 0.0
    expected_ae = 0.0
    expected_ac = 0.0

    for intensity in sequence:
        state, metrics = transition_distribution(
            c,
            state,
            _flow_pmf(intensity, "ebike_depart", c),
            _flow_pmf(intensity, "classic_depart", c),
            _flow_pmf(intensity, "ebike_arrive", c),
            _flow_pmf(intensity, "classic_arrive", c),
            is_renting=bool(_intensity_value(intensity, "is_renting", default=1.0)),
            is_returning=bool(_intensity_value(intensity, "is_returning", default=1.0)),
        )
        constrained_survival *= 1.0 - float(metrics["p_dock_constrained_arrival"])
        expected_de += float(metrics["expected_ebike_departures"])
        expected_dc += float(metrics["expected_classic_departures"])
        expected_ae += float(metrics["expected_ebike_arrivals"])
        expected_ac += float(metrics["expected_classic_arrivals"])

    e_pmf = state.sum(axis=1)
    q_pmf = state.sum(axis=0)
    counts = np.arange(c + 1, dtype=float)
    expected_e = float(np.dot(counts, e_pmf))
    expected_q = float(np.dot(counts, q_pmf))
    p_zero = float(e_pmf[0])
    return InventoryRolloutResult(
        p_has_ebike=float(1.0 - p_zero),
        p_zero=p_zero,
        expected_ebikes=expected_e,
        expected_total_bikes=expected_q,
        p_count_ebikes=collapse_count_distribution(e_pmf),
        p_count_total=collapse_count_distribution(q_pmf),
        p_capacity_violation=0.0,
        p_dock_constrained_arrival=float(min(1.0, max(0.0, 1.0 - constrained_survival))),
        expected_ebike_departures=float(expected_de),
        expected_classic_departures=float(expected_dc),
        expected_ebike_arrivals=float(expected_ae),
        expected_classic_arrivals=float(expected_ac),
        p_joint_e_q=state.copy() if return_joint else None,
        p_count_ebikes_full=e_pmf.tolist(),
    )


def rollout_inventory_distribution(
    *,
    capacity: int | float | None,
    current_ebikes: int | float | None,
    current_total_bikes: int | float | None,
    ebike_departure_mean: float,
    classic_departure_mean: float,
    ebike_arrival_mean: float,
    classic_arrival_mean: float,
    max_capacity: int = 80,
) -> InventoryRolloutResult:
    """Roll one bounded stochastic inventory transition over the target horizon."""
    c = _safe_int(capacity, 1)
    c = max(1, min(max_capacity, c))
    e0 = max(0, min(c, _safe_int(current_ebikes, 0)))
    q0 = max(e0, min(c, _safe_int(current_total_bikes, e0)))
    classic0 = max(0, q0 - e0)

    pi = np.zeros((c + 1, c + 1), dtype=float)
    pi[e0, q0] = 1.0
    _, de_probs = _bounded_poisson_support(ebike_departure_mean, c)
    _, dc_probs = _bounded_poisson_support(classic_departure_mean, c)
    _, ae_probs = _bounded_poisson_support(ebike_arrival_mean, c)
    _, ac_probs = _bounded_poisson_support(classic_arrival_mean, c)
    state, metrics = transition_distribution(c, pi, de_probs, dc_probs, ae_probs, ac_probs)

    e_pmf = state.sum(axis=1)
    q_pmf = state.sum(axis=0)
    counts = np.arange(c + 1, dtype=float)
    expected_e = float(np.dot(counts, e_pmf))
    expected_q = float(np.dot(counts, q_pmf))
    p_zero = float(e_pmf[0])
    return InventoryRolloutResult(
        p_has_ebike=float(1.0 - p_zero),
        p_zero=p_zero,
        expected_ebikes=expected_e,
        expected_total_bikes=expected_q,
        p_count_ebikes=collapse_count_distribution(e_pmf),
        p_count_total=collapse_count_distribution(q_pmf),
        p_capacity_violation=0.0,
        p_dock_constrained_arrival=float(metrics["p_dock_constrained_arrival"]),
        expected_ebike_departures=float(metrics["expected_ebike_departures"]),
        expected_classic_departures=float(metrics["expected_classic_departures"]),
        expected_ebike_arrivals=float(metrics["expected_ebike_arrivals"]),
        expected_classic_arrivals=float(metrics["expected_classic_arrivals"]),
        p_count_ebikes_full=e_pmf.tolist(),
    )
