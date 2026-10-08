#!/usr/bin/env bash
set -euo pipefail

mode="${1:-status}"
case "${mode}" in
  status)
    sysctl kernel.perf_event_paranoid
    grep -E 'RestrictProfilingToAdminUsers' /proc/driver/nvidia/params 2>/dev/null || true
    echo "临时启用 CPU sampling: sudo sysctl kernel.perf_event_paranoid=2"
    echo "恢复: sudo sysctl kernel.perf_event_paranoid=4"
    echo "GPU 计数器若报 ERR_NVGPUCTRPERM，请用 sudo -E make profile-ncu；不永久修改驱动参数。"
    ;;
  enable-cpu) sudo sysctl kernel.perf_event_paranoid=2 ;;
  restore-cpu) sudo sysctl kernel.perf_event_paranoid=4 ;;
  *) echo "用法: $0 status|enable-cpu|restore-cpu" >&2; exit 2 ;;
esac
