# Nsight Compute 与 Roofline 实战指南

## 使用边界

Nsight Compute 回答“某个 Kernel 为什么慢”。必须先用 Systems 选 Top Kernel，再限制名称和 launch count；对整个 vLLM 使用 `--set full` 会产生巨量重放和严重扰动。

```bash
make profile-permissions
sudo -E PROFILE_NCU_SET=basic PROFILE_NCU_LAUNCH_COUNT=1 make profile-ncu
sudo -E PROFILE_NCU_SET=roofline PROFILE_NCU_LAUNCH_COUNT=1 make profile-ncu
```

当前机器普通用户实际触发 `ERR_NVGPUCTRPERM`，因此 `.ncu-rep` 必须在交互式 sudo 会话中生成。项目不会永久修改 NVIDIA 驱动权限。`profiling/ncu_probe.py` 是独立 BF16 GEMM 教学负载，不是 vLLM Kernel 替代品。

## 指标判断框架

- Speed of Light：SM 高、DRAM 低通常偏 compute-bound；DRAM 高、SM 低通常偏 memory-bound。
- Roofline：Arithmetic Intensity 低且贴近带宽斜线为带宽受限；高强度且接近水平计算峰值为算力受限。
- Occupancy：低 occupancy 可能由寄存器、共享内存或 block shape 限制，但高 occupancy 不等于高性能。
- Warp Stall：Long Scoreboard 常指等待显存/L2；Not Selected 表示可运行 warp 多但调度没选中；Barrier 指同步压力。
- Memory Workload：结合 L1/L2 hit rate、DRAM throughput 和访问合并判断，不能只看一个百分比。
- Launch Statistics：核对 grid/block、寄存器、shared memory 与理论/实际 occupancy。

## 项目中的 Kernel 选择

优先分析 Systems 中的 FlashAttention、BF16 GEMV，以及 FP8 `enable_sm120_family` CUTLASS GEMM。每个 Kernel 卡片记录：所属阶段、GPU 时间占比、SOL/带宽、occupancy、主要 stall、瓶颈结论和可操作建议。

预期假设而非预设结论：c1 Decode 的 GEMV 更可能受权重带宽限制；FP8 减少权重流量后吞吐提高；Attention 占比随之上升。最终结论必须由 `.ncu-rep` 指标确认。

