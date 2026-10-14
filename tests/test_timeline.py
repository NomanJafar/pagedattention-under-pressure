import pytest

from paged_pressure.timeline import annotate, preemption_events, server_args, summarize


def _tl():
    # 12 samples at 1 s: pressure builds, two preemption bursts, throughput dips after each
    rows = []
    usage = [0.2, 0.5, 0.8, 1.0, 1.0, 0.7, 1.0, 1.0, 0.6, 0.9, 1.0, 1.0]
    running = [4, 8, 12, 16, 16, 10, 14, 16, 9, 13, 16, 16]
    waiting = [0, 0, 0, 0, 2, 6, 3, 2, 7, 4, 2, 3]
    preempt = [0, 0, 0, 0, 0, 6, 6, 6, 13, 13, 13, 13]
    gen = [0, 300, 600, 900, 1200, 1300, 1500, 1800, 1900, 2100, 2400, 2700]
    for i in range(12):
        rows.append({"t": float(i), "kv_cache_usage": usage[i], "running": running[i], "waiting": waiting[i],
                     "preemptions": preempt[i], "generation_tokens_total": gen[i]})
    return rows


def test_preemption_events_are_increments_of_the_counter():
    ev = preemption_events(_tl())
    assert [(e["t"], e["count"]) for e in ev] == [(5.0, 6), (8.0, 7)]


def test_summarize_reports_pressure_signature():
    s = summarize(_tl())
    assert s["samples"] == 12 and s["preemptions_total"] == 13
    assert s["time_at_full_cache_pct"] == pytest.approx(100 * 6 / 12)      # 6 samples at usage 1.0
    assert s["peak_waiting"] == 7 and s["peak_running"] == 16
    assert s["mean_tokens_per_s"] == pytest.approx(2700 / 11)
    assert s["tokens_per_s_after_preemption"] < s["tokens_per_s_steady"]   # throughput dips right after a burst


def test_annotate_tags_samples_with_phases():
    tagged = annotate(_tl())
    assert tagged[0]["phase"] == "filling" and tagged[3]["phase"] == "full"
    assert tagged[5]["phase"] == "preempting"


def test_server_args_builds_pressure_configuration():
    a = server_args(utilization=0.35, block_size=32, prefix_caching=False, max_num_seqs=256)
    assert a == ["--gpu-memory-utilization", "0.35", "--block-size", "32", "--max-num-seqs", "256", "--no-enable-prefix-caching"]
    assert "--enable-prefix-caching" in server_args(0.9, 16, True, 256)
