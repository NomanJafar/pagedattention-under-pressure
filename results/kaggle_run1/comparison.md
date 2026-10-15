| config | kv_tokens | preemptions | events | full_pct | peak_waiting | tok_s | tok_s_steady | tok_s_after_preempt | ttft_p50_ms | itl_p50_ms | itl_p99_ms | e2e_p95_s | requests | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| healthy (util 0.9, block 16, prefix on) | 340064 | 0 | 0 | 0 | 0 | 1246 | 1045 | 0 | 20102 | 74.2 | 150 | 73.3 | 522 | 0 |
| pressure (util 0.35, block 16, prefix on) | 52032 | 0 | 0 | 0 | 0 | 1223 | 1024 | 0 | 21695 | 75.9 | 153 | 76.4 | 522 | 0 |
| pressure, block 32 | 40096 | 0 | 0 | 0 | 0 | 1234 | 1026 | 0 | 21663 | 74.4 | 146 | 75.0 | 522 | 0 |
| pressure, prefix caching off | 52032 | 203 | 8 | 24 | 32.0 | 714 | 387 | 1302 | 46537 | 61.8 | 486 | 134.1 | 522 | 171 |
