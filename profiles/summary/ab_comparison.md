# Nsight A/B 对比

| Profile | 场景 | 并发 | TTFT ms | E2E P50 ms | 聚合 tok/s |
|---|---|---:|---:|---:|---:|
| bf16_steady_c8_offline | steady | 8 | N/A | 3265.1684320007917 | 627.2264486965686 |
| bf16_steady_offline | steady | 1 | N/A | 2838.167446985608 | 90.19904737188614 |
| fp8_steady_c8_offline | steady | 8 | N/A | 2142.166897014249 | 956.0412883116162 |
| fp8_steady_offline | steady | 1 | N/A | 1857.941889000358 | 137.7868713309099 |

> 只有在输入、缓存状态和采样设置一致时才比较 Profile；低于 5% 的差异不写成优化收益。
