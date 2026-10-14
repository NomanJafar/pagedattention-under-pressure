"""Timeline analysis for a vLLM server under KV cache pressure (CONCEPTS.md section 4).

Input rows are /metrics samples as tokenbench records them:
{"t", "kv_cache_usage", "running", "waiting", "preemptions", "generation_tokens_total"}.
"""

from __future__ import annotations

import itertools

FULL = 0.98  # cache usage at or above this counts as "full"


def server_args(utilization: float, block_size: int, prefix_caching: bool, max_num_seqs: int) -> list[str]:
    return ["--gpu-memory-utilization", str(utilization), "--block-size", str(block_size),
            "--max-num-seqs", str(max_num_seqs),
            "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching"]


def preemption_events(rows: list[dict]) -> list[dict]:
    """One event per sample where the cumulative preemption counter increased."""
    events, prev = [], None
    for r in rows:
        p = r.get("preemptions")
        if p is None:
            continue
        if prev is not None and p > prev:
            events.append({"t": r["t"], "count": int(p - prev), "running": r.get("running"), "waiting": r.get("waiting")})
        prev = p
    return events


def _rates(rows: list[dict]) -> list[float | None]:
    out = [None]
    for a, b in itertools.pairwise(rows):
        ga, gb = a.get("generation_tokens_total"), b.get("generation_tokens_total")
        dt = b["t"] - a["t"]
        out.append((gb - ga) / dt if ga is not None and gb is not None and dt > 0 else None)
    return out


def annotate(rows: list[dict]) -> list[dict]:
    """Tag each sample: filling | full | preempting | recovering | idle."""
    events = {e["t"] for e in preemption_events(rows)}
    tagged, last_event_t = [], None
    for r in rows:
        u = r.get("kv_cache_usage") or 0.0
        if r["t"] in events:
            phase, last_event_t = "preempting", r["t"]
        elif last_event_t is not None and r["t"] - last_event_t <= 2 and u < FULL:
            phase = "recovering"
        elif u >= FULL:
            phase = "full"
        elif (r.get("running") or 0) == 0:
            phase = "idle"
        else:
            phase = "filling"
        tagged.append({**r, "phase": phase})
    return tagged


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0}
    rates = _rates(rows)
    events = preemption_events(rows)
    event_ts = [e["t"] for e in events]
    after = [rt for r, rt in zip(rows, rates, strict=True) if rt is not None and any(0 < r["t"] - t <= 2 for t in event_ts)]
    steady = [rt for r, rt in zip(rows, rates, strict=True) if rt is not None and not any(-1 <= r["t"] - t <= 3 for t in event_ts)]
    total_tokens = (rows[-1].get("generation_tokens_total") or 0) - (rows[0].get("generation_tokens_total") or 0)
    span = rows[-1]["t"] - rows[0]["t"]
    full = sum(1 for r in rows if (r.get("kv_cache_usage") or 0) >= FULL)
    return {
        "samples": len(rows),
        "duration_s": span,
        "preemptions_total": int((rows[-1].get("preemptions") or 0) - (rows[0].get("preemptions") or 0)),
        "preemption_events": len(events),
        "time_at_full_cache_pct": 100 * full / len(rows),
        "peak_kv_usage": max((r.get("kv_cache_usage") or 0) for r in rows),
        "peak_running": max((r.get("running") or 0) for r in rows),
        "peak_waiting": max((r.get("waiting") or 0) for r in rows),
        "mean_tokens_per_s": total_tokens / span if span > 0 else None,
        "tokens_per_s_steady": sum(steady) / len(steady) if steady else None,
        "tokens_per_s_after_preemption": sum(after) / len(after) if after else None,
    }
