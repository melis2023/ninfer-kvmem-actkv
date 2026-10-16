#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvmem-ninfer 图形化启动器（本地 Web 控制台）
=============================================
- 本机 Web 仪表盘（默认 http://127.0.0.1:18300）
- 管理 infer-kvmem-server.exe 进程：启动 / 停止
- 展示后端 /v1/models、/stats、/metrics、/props、/health（原样代理）
- 代理 /v1/chat/completions（SSE 流式直通）
- 预设保存与切换（config.json，与本脚本同目录）
- /api/browse：弹出系统原生文件 / 目录选择框（子进程 pick.py，Tk，标准库）
- /api/sys：本机 CPU / 内存 / GPU 显存占用（ctypes + nvidia-smi，标准库实现）

仅依赖 Python 3 标准库，绑定 127.0.0.1。
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
PICK_SCRIPT = BASE_DIR / "pick.py"      # 原生选择框子进程（见 /api/browse）

# 默认 worker 目录：本启动器位于仓库根目录 kvmem-launcher/，优先取仓库内 workers/ninfer；
# 仓库外使用时回退到历史绝对路径。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKER_IN_REPO = _REPO_ROOT / "workers" / "ninfer"
WORKER_DIR = str(_WORKER_IN_REPO) if _WORKER_IN_REPO.is_dir() else r"I:\llama\kvmem-llama.cpp-multi-backend-framework\workers\ninfer"
WORKER_EXE = WORKER_DIR + r"\infer-kvmem-server.exe"

# --------------------------------------------------------------------------
# 参数 schema：表单字段 → CLI 标志（顺序与 start-kvmem-ninfer.bat 一致）
#   kind: text | num | bool
#   bool 字段为开关：勾选时输出标志，不勾则省略
# --------------------------------------------------------------------------
FIELD_ORDER = [
    ("model",                 None,                        "text"),
    ("port",                  "--port",                    "num"),
    ("max_context",           "--max-context",             "num"),
    # GPU 选择：单卡用 --device（0 = 第一块卡，省略即默认 0）；
    # 多卡流水线是 --devices A,B,... + --stage-layers，那两个标志不放这里（走附加参数）。
    ("device",                "--device",                  "num"),
    # KVMem 总开关（无对应 CLI 标志）：关闭时 build_command 丢弃全部 --kvmem-*，
    # 退回 ninfer 原版启动方式（上游在没有 --kvmem-budget 时会拒绝任何 KVMem 辅助参数）。
    ("kvmem_enabled",         None,                        "bool"),
    ("kvmem_recent_tokens",   "--kvmem-recent-tokens",     "num"),
    ("prefill_chunk",         "--prefill-chunk",           "num"),
    ("kvmem_budget",          "--kvmem-budget",            "num"),
    ("kvmem_gen_reserve",     "--kvmem-gen-reserve",       "num"),
    ("kvmem_host_mib",        "--kvmem-host-mib",          "num"),
    # 单次最大输出（服务端口径）：请求不带 max_tokens 时的上限。留空 = 不限（用 max_context）。
    ("default_max_tokens",    "--default-max-tokens",      "num"),
    ("kvmem_action_lrfu",     "--kvmem-lrfu",       "bool"),
    ("model_id",              "--model-id",                "text"),
    ("kv_dtype",              "--kv-dtype",                "text"),
    ("temperature",           "--temperature",             "num"),
    ("top_p",                 "--top-p",                   "num"),
    ("top_k",                 "--top-k",                   "num"),
    ("min_p",                 "--min-p",                   "num"),
    ("spec",                  "--spec",                    "text"),
    ("draft_tokens",          "--draft-tokens",            "num"),
    ("ngram_draft_tokens",    "--ngram-draft-tokens",      "num"),
    ("adaptive_mtp",          "--adaptive-mtp",            "bool"),
    ("vision",                "--vision",                  "bool"),
    ("vision_residency",      "--vision-residency",        "text"),
    ("vision_max_merged",     "--vision-max-merged",       "num"),
    ("default_reasoning_effort", "--default-reasoning-effort", "text"),
    ("default_thinking_budget",  "--default-thinking-budget",  "num"),
    ("extra_args",            None,                        "text"),
]

# --------------------------------------------------------------------------
# 进阶参数（第三批，来自 serve_options.cpp）
#   第 1 梯队 KVMem/上下文调优、第 2 梯队 精度与内核、第 3 梯队 RoPE 长上下文。
#   全部与当前 KVMem 配置兼容；与 KVMem 互斥的标志（--model-suspend / --graft /
#   --lookup-ngram / router 模式）在源码里被明确禁止，这里一律不暴露。
# --------------------------------------------------------------------------
ADVANCED_ORDER = [
    # ---- 第 1 梯队：KVMem / 上下文 ----
    ("max_concurrency",       "--max-concurrency",         "num"),
    ("kvmem_sink_tokens",     "--kvmem-sink-tokens",       "num"),
    ("kvmem_sessions",        "--kvmem-sessions",          "num"),
    ("kvmem_action_region",   "--kvmem-action-region",     "bool"),
    ("kvmem_query_union",     "--kvmem-query-union",       "bool"),
    # ---- ActKV 保留策略：§8.1 语义保护 ----
    # 注：§5.1 AAV（--kvmem-aav-protect / -lambda-intermittent / -min-hits）已被上游移除，
    # 传入会直接报错（AAV 只保留为 KVMEM_AAV_TRACE 诊断，不再是一种保留策略）。
    ("kvmem_protect_semantics",       "--kvmem-protect-semantics",       "bool"),
    ("kvmem_lrfu_lambda",     "--kvmem-lrfu-lambda",       "num"),
    ("kvmem_lrfu_top_frac",   "--kvmem-lrfu-top-frac",     "num"),
    ("kvmem_conf_budget",     "--kvmem-conf-budget",       "num"),
    ("kvmem_conf_scale_up",   "--kvmem-conf-scale-up",     "num"),
    ("kvmem_conf_cooldown",   "--kvmem-conf-cooldown",     "num"),
    ("kvmem_conf_pool",       "--kvmem-conf-pool",         "num"),
    ("kvmem_conf_stride",     "--kvmem-conf-stride",       "num"),
    ("kvmem_conf_bottom",     "--kvmem-conf-bottom",       "num"),
    ("kvmem_disk_mib",        "--kvmem-disk-mib",          "num"),
    ("kv_capacity",           "--kv-capacity",             "text"),
    ("no_prefix_reuse",       "--no-prefix-reuse",         "bool"),
    # ---- 第 2 梯队：精度与内核 ----
    ("fast_prefill_kernel",   "--fast-prefill-kernel",     "bool"),
    ("prefill_cublas",        "--prefill-cublas",          "bool"),
    ("mlp_a8_decode",         "--mlp-a8-decode",           "bool"),
    ("no_prefill_a8",         "--no-prefill-a8",           "bool"),
    ("lm_head_q4",            "--lm-head-q4",              "bool"),
    ("lm_head_q6",            "--lm-head-q6",              "bool"),
    ("embedding_q4",          "--embedding-q4",            "bool"),
    ("embedding_q6",          "--embedding-q6",            "bool"),
    ("gdn_state_fp16",        "--gdn-state-fp16",          "bool"),
    ("no_cuda_graph",         "--no-cuda-graph",           "bool"),
    ("cuda_graph_allowance_mib", "--cuda-graph-allowance-mib", "num"),
    # ---- 第 3 梯队：RoPE 长上下文 ----
    ("rope_yarn",             "--rope-yarn",               "bool"),
    ("rope_yarn_factor",      "--rope-yarn-factor",        "num"),
    ("rope_scaling_factor",   "--rope-scaling-factor",     "num"),
]
FIELD_KEYS = [f[0] for f in FIELD_ORDER] + [f[0] for f in ADVANCED_ORDER]
ALL_FIELD_ORDER = FIELD_ORDER + ADVANCED_ORDER

MODEL_MTP    = r"I:\model\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer"
MODEL_DFLASH = r"I:\model\Qwen3.8-27B-Coder390-EfficientThink-Q2LynnStyle-MTP.ninfer"

PRESET_MTP = {
    "model": MODEL_MTP, "port": 18200,
    "max_context": 262144, "kvmem_recent_tokens": 1024, "prefill_chunk": 256,
    "kvmem_enabled": True, "kvmem_budget": 36864, "kvmem_gen_reserve": 16384, "kvmem_host_mib": 12288,
    "kvmem_action_lrfu": True, "model_id": "ninfer-27b",
    "kvmem_protect_semantics": True,
    "kv_dtype": "rk8v4",
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "spec": "mtp", "draft_tokens": 5, "ngram_draft_tokens": 31, "adaptive_mtp": True,
    "vision": True, "vision_residency": "cpu", "vision_max_merged": 256,
    "default_reasoning_effort": "xhigh", "default_thinking_budget": 53248,
    "extra_args": "",
}

PRESET_DFLASH = {
    "model": MODEL_DFLASH, "port": 18200,
    "max_context": 262144, "kvmem_recent_tokens": 1024, "prefill_chunk": 256,
    "kvmem_enabled": True, "kvmem_budget": 36864, "kvmem_gen_reserve": 16384, "kvmem_host_mib": 12288,
    "kvmem_action_lrfu": True, "model_id": "ninfer-27b",
    "kvmem_protect_semantics": True,
    "kv_dtype": "nvfp4",
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "spec": "dflash2", "draft_tokens": 7, "ngram_draft_tokens": 31, "adaptive_mtp": False,
    "vision": True, "vision_residency": "cpu", "vision_max_merged": 256,
    "default_reasoning_effort": "medium", "default_thinking_budget": 32000,
    "extra_args": "",
}

PRESET_BASELINE = {
    "model": MODEL_MTP, "port": 18200,
    "max_context": 262144, "kvmem_recent_tokens": 1024, "prefill_chunk": 256,
    "kvmem_enabled": True, "kvmem_budget": 36864, "kvmem_gen_reserve": 16384, "kvmem_host_mib": 12288,
    "kvmem_action_lrfu": False, "model_id": "ninfer-27b",
    "kvmem_protect_semantics": True,
    "kv_dtype": "rk8v4",
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "spec": "mtp", "draft_tokens": 5, "ngram_draft_tokens": 31, "adaptive_mtp": True,
    "vision": True, "vision_residency": "cpu", "vision_max_merged": 256,
    "default_reasoning_effort": "xhigh", "default_thinking_budget": 53248,
    "extra_args": "",
}

DEFAULT_PRESETS = [
    {"name": "MTP + ActKV（bat 默认 · 当前运行）", "flags": PRESET_MTP},
    {"name": "DFlash2 + nvfp4 权量", "flags": PRESET_DFLASH},
    {"name": "无 ActKV 基线（对照）", "flags": PRESET_BASELINE},
]


def default_config() -> dict:
    return {
        "launcher_port": 18300,
        "upstream": {"host": "127.0.0.1", "port": 18200},
        "exe": WORKER_EXE,
        "bin_dir": WORKER_DIR,
        "gpu": "GPU-dba26dd9-930f-1b01-2b05-22c1fca6cdce",
        "active": 0,
        "presets": DEFAULT_PRESETS,
    }


def clone(obj):
    return json.loads(json.dumps(obj))


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cfg = default_config()
            for k in cfg:
                if k in saved:
                    cfg[k] = saved[k]
            return cfg
        except Exception as e:
            print(f"[config] 读取失败，使用默认: {e}", file=sys.stderr)
    cfg = default_config()
    save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_FILE)


def normalize_flags(flags: dict) -> dict:
    """补齐所有已知字段的默认空值。

    同时覆盖基础字段与进阶字段，否则前端只提交可见页签里的字段时，
    进阶参数会被静默丢弃。
    """
    out = {}
    for key, _, _ in ALL_FIELD_ORDER:
        out[key] = flags.get(key, "")
    return out


def build_command(cfg: dict, flags: dict) -> list:
    """flags → [exe, model, --flag value, ...]；空值/0/None 数值省略（与 bat 语义一致）。

    直接用 dict.get(key) 取到的 None 会被 str(None) 变成 "None"，从而产出
    `--flag None` 这种非法命令；这里统一按"空值"处理，与 normalize_flags 的语义一致。
    """
    cmd = [cfg["exe"]]
    model = str(flags.get("model") or "").strip()
    if model:
        cmd.append(model)

    # KVMem 总开关：关闭时丢弃全部 --kvmem-* 标志（含 budget / gen-reserve），
    # 退回 ninfer 原版启动方式。注意：缺省（老配置没有该键）视为开启，
    # 避免升级后静默改变行为。
    kv_raw = flags.get("kvmem_enabled")
    if kv_raw is None or kv_raw == "":
        kvmem_on = True
    else:
        kvmem_on = str(kv_raw).strip().lower() not in ("false", "0", "no", "off")

    for key, flag, kind in ALL_FIELD_ORDER:
        if not flag:
            continue
        if not kvmem_on and key.startswith("kvmem_"):
            continue
        v = flags.get(key)
        if kind == "bool":
            if v:
                cmd.append(flag)
            continue
        if v is None:                      # 未提供的进阶字段视为空
            continue
        s = str(v).strip()
        if s == "" or s == "0" or s.lower() == "none":
            continue
        cmd += [flag, s]
    extra = str(flags.get("extra_args") or "").strip()
    if extra:
        cmd += extra.split()
    return cmd


# --------------------------------------------------------------------------
# 原生路径选择（/api/browse）
#   浏览器出于安全限制拿不到本地真实路径（<input type=file> 只给文件名、且会
#   带上"假路径"），而启动器与浏览器本来就在同一台机器上，所以对话框交给
#   pick.py 子进程弹，选中的完整路径经 stdout 回传。用户取消不算错误。
# --------------------------------------------------------------------------
def browse_path(mode: str = "file", initial: str = "", types: str = "",
                title: str = "") -> dict:
    """弹出系统文件 / 目录选择框。

    返回 {"ok":True,"path":...} / {"ok":False,"cancelled":True} / {"ok":False,"error":...}。
    对话框由用户操作、可能停留任意久，所以不设超时；ThreadingHTTPServer 每请求
    一线程，一个挂着的选择框不会阻塞其它请求。
    """
    if not PICK_SCRIPT.exists():
        return {"ok": False, "error": f"缺少选择框脚本: {PICK_SCRIPT.name}"}
    cmd = [sys.executable, str(PICK_SCRIPT),
           "--mode", "dir" if mode == "dir" else "file",
           "--title", str(title or "请选择")]
    if initial:
        cmd += ["--initial", str(initial)]
    if types:
        cmd += ["--types", str(types)]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"       # 路径含中文时避免子进程按 cp936 输出
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
    except Exception as e:
        return {"ok": False, "error": f"调用选择框失败: {e}"}
    if r.returncode == 0:
        path = (r.stdout or "").strip()
        return {"ok": True, "path": path} if path else {"ok": False, "cancelled": True}
    if r.returncode == 1:
        return {"ok": False, "cancelled": True}
    lines = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
    return {"ok": False,
            "error": lines[-1] if lines else f"选择框退出码 {r.returncode}"}


# --------------------------------------------------------------------------
# 本机资源监控：CPU / 内存 / GPU 显存
#   全部用标准库实现，不引入 psutil 等第三方依赖：
#     CPU 占用   kernel32!GetSystemTimes 两次采样求差值
#     内存占用   kernel32!GlobalMemoryStatusEx
#     GPU 显存   nvidia-smi --query-gpu（NVIDIA 驱动自带，缺失时优雅降级）
# --------------------------------------------------------------------------
class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _filetime_value(f: "_FILETIME") -> int:
    return (f.dwHighDateTime << 32) | f.dwLowDateTime


def _find_nvidia_smi():
    """优先用 System32 下的 nvidia-smi（NVIDIA 驱动会放这里），再退回 PATH。"""
    cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    if cand.exists():
        return str(cand)
    return shutil.which("nvidia-smi")


# --------------------------------------------------------------------------
# 整机功耗估算（仅供参考，不是实测值）
#
# Windows 没有面向桌面的开放功耗接口（Win32_PowerMeter / BatteryStatus 在台式机上
# 均为空，主板传感器要厂商软件）。因此这里用一个**粗粒度线性模型**估算：
#     CPU   ≈ (基础 25W + 65W × 占用率) × 逻辑核数 / 12
#     GPU   = nvidia-smi 的 power.draw（实测，准确）
#     内存  ≈ 2W/8GB × 已用 GiB
#     其它  = 60W 固定（主板 / 风扇 / SSD / USB 等）
# 绝对准确度 ±30% 以上，只适合看趋势与相对高低，不能当功耗计用。
# 前端会明确标注"估算"。
# --------------------------------------------------------------------------
_SYS_POWER_BASE_W = 60.0        # 主板/存储/风扇/USB 等固定开销
_CPU_BASE_W = 25.0              # CPU 待机基础功耗
_CPU_PEAK_W = 65.0             # CPU 满载增量
_CPU_REF_CORES = 12.0           # 模型标定时参考的核数（用于线性缩放）
_MEM_W_PER_GIB = 2.0 / 8.0      # 内存每 GiB 约 2W/8GB


def estimate_system_power(cpu_pct, cpu_count, mem_used_gib, gpu_w):
    """返回 (整机估算瓦数, 分项字典)。任一输入缺失时对应项记 None。"""
    parts = {
        "gpu_w": float(gpu_w) if gpu_w is not None else None,
        "cpu_w": None,
        "mem_w": None,
        "other_w": _SYS_POWER_BASE_W,
    }
    total = _SYS_POWER_BASE_W
    if cpu_pct is not None and cpu_count:
        cores = float(cpu_count) if cpu_count else _CPU_REF_CORES
        cpu_w = (_CPU_BASE_W + _CPU_PEAK_W * (float(cpu_pct) / 100.0)) * cores / _CPU_REF_CORES
        parts["cpu_w"] = round(cpu_w, 1)
        total += cpu_w
    if gpu_w is not None:
        total += float(gpu_w)
    if mem_used_gib is not None:
        mem_w = float(mem_used_gib) * _MEM_W_PER_GIB
        parts["mem_w"] = round(mem_w, 1)
        total += mem_w
    return round(total, 1), parts


class SystemMonitor:
    """定期采样本机 CPU / 内存 / GPU 占用，供 /api/sys 读取。

    采样在后台线程里做，接口读取的是缓存值，因此不会阻塞 HTTP 请求
    （nvidia-smi 进程启动约需几十毫秒，不能放在请求路径上）。
    """

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.lock = threading.Lock()
        self._k32 = None
        self._k32_error = None
        self._smi = _find_nvidia_smi()
        self._smi_error = None
        self._smi_fail_count = 0
        self._cpu_prev = None       # (idle, total) 上次采样
        self._cpu_pct = None
        self._mem = {}
        self._gpus = []
        self._ts = None

        try:
            self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except Exception as e:                       # 非 Windows 或受限环境
            self._k32_error = str(e)

    # -- 采样 ---------------------------------------------------------------
    def _sample_cpu(self):
        if self._k32 is None:
            return
        try:
            idle, kern, user = _FILETIME(), _FILETIME(), _FILETIME()
            if not self._k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern),
                                            ctypes.byref(user)):
                return
            i, k, u = _filetime_value(idle), _filetime_value(kern), _filetime_value(user)
            total = k + u                            # kernel 时间已包含 idle
            if self._cpu_prev is not None:
                d_idle = i - self._cpu_prev[0]
                d_total = total - self._cpu_prev[1]
                if d_total > 0:
                    busy = d_total - d_idle
                    self._cpu_pct = round(max(0.0, min(100.0, busy * 100.0 / d_total)), 1)
            self._cpu_prev = (i, total)
        except Exception:
            pass

    def _sample_mem(self):
        if self._k32 is None:
            return
        try:
            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            if not self._k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return
            total = int(ms.ullTotalPhys)
            avail = int(ms.ullAvailPhys)
            self._mem = {
                "percent": int(ms.dwMemoryLoad),
                "total_bytes": total,
                "used_bytes": max(0, total - avail),
                "avail_bytes": avail,
                "total_gib": round(total / 1024 ** 3, 2),
                "used_gib": round(max(0, total - avail) / 1024 ** 3, 2),
            }
        except Exception:
            pass

    def _sample_gpu(self):
        if not self._smi:
            return
        # 连续失败多次后就不再反复尝试，避免每 2 秒拉起一个必然失败的进程
        if self._smi_fail_count >= 5:
            return
        try:
            r = subprocess.run(
                [self._smi,
                 "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
                 "temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or "").strip()[:200] or "nvidia-smi 返回非零")
            gpus = []
            for line in (r.stdout or "").strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                try:
                    used = int(float(parts[2]))
                    total = int(float(parts[3]))
                except ValueError:
                    continue
                # 功耗字段在老驱动/虚拟显卡上可能不存在或回报 N/A，统一容错
                power_draw = power_limit = None
                if parts[6] not in ("", "N/A", "[N/A]"):
                    try:
                        power_draw = round(float(parts[6]), 1)
                    except ValueError:
                        pass
                if parts[7] not in ("", "N/A", "[N/A]"):
                    try:
                        power_limit = round(float(parts[7]), 1)
                    except ValueError:
                        pass
                gpus.append({
                    "index": int(float(parts[0])) if parts[0].lstrip("-").isdigit() else 0,
                    "name": parts[1],
                    "mem_used_mib": used,
                    "mem_total_mib": total,
                    "mem_percent": round(used * 100.0 / total, 1) if total else None,
                    "util_percent": int(float(parts[4])) if parts[4] not in ("", "N/A") else None,
                    "temp_c": int(float(parts[5])) if parts[5] not in ("", "N/A") else None,
                    "power_draw_w": power_draw,
                    "power_limit_w": power_limit,
                    "power_percent": (round(power_draw * 100.0 / power_limit, 1)
                                      if power_draw and power_limit else None),
                })
            self._gpus = gpus
            self._smi_error = None
            self._smi_fail_count = 0
        except Exception as e:
            self._smi_fail_count += 1
            self._smi_error = str(e)[:200]

    def sample(self):
        self._sample_cpu()
        self._sample_mem()
        self._sample_gpu()
        with self.lock:
            self._ts = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            cpu_pct = self._cpu_pct
            mem = dict(self._mem)
            gpus = [dict(g) for g in self._gpus]
            ts = self._ts
            smi_err = self._smi_error
            k32_err = self._k32_error
        cpu_count = None
        if self._k32 is not None:
            try:
                cpu_count = self._k32.GetActiveProcessorCount(0xFFFF)
            except Exception:
                cpu_count = None

        gpu_w = (sum(g["power_draw_w"] for g in gpus if g.get("power_draw_w"))
                 if any(g.get("power_draw_w") for g in gpus) else None)
        gpu_limit_w = (sum(g["power_limit_w"] for g in gpus if g.get("power_limit_w"))
                       if any(g.get("power_limit_w") for g in gpus) else None)
        sys_w, sys_parts = estimate_system_power(
            cpu_pct, cpu_count, mem.get("used_gib"), gpu_w)

        return {
            "ok": self._k32 is not None,
            "ts": ts,
            "cpu": {
                "percent": cpu_pct,
                "count": cpu_count,
                "error": k32_err,
            },
            "mem": mem,
            "gpu": {
                "available": bool(self._smi),
                "items": gpus,
                "error": smi_err,
            },
            "power": {
                "gpu_w": gpu_w,
                "gpu_limit_w": gpu_limit_w,
                "gpu_percent": (round(gpu_w * 100.0 / gpu_limit_w, 1)
                                if gpu_w and gpu_limit_w else None),
                "system_w": sys_w,
                "system_parts": sys_parts,
                "estimated": True,      # 整机部分是模型估算，非实测
            },
        }

    def run_forever(self):
        self.sample()          # 先采一次，建立 CPU 差值基线
        while True:
            time.sleep(self.interval)
            try:
                self.sample()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Windows 作业对象（Job Object）
#   目的：启动器退出时（关闭 bat 窗口 / Ctrl+C / 被结束）**自动带走**模型进程，
#   不留孤儿进程占着显存。
#
#   原理：把子进程放进一个设置了 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 的作业对象。
#   作业句柄由本进程持有；一旦本进程结束，句柄关闭，内核会强制结束作业内所有进程。
#   这是内核级保证，比在 Python 里写清理逻辑可靠 —— 进程被强杀时也有效。
# --------------------------------------------------------------------------
class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001


class KillOnCloseJob:
    """持有作业对象句柄；本进程结束（含被强杀）时内核自动结束作业内进程。"""

    def __init__(self):
        self._handle = None
        self.error = None
        if os.name != "nt":
            self.error = "非 Windows，未启用作业对象"
            return
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            h = k32.CreateJobObjectW(None, None)
            if not h:
                raise ctypes.WinError(ctypes.get_last_error())
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(
                    h, JobObjectExtendedLimitInformation,
                    ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = h
            self._k32 = k32
        except Exception as e:
            self._handle = None
            self.error = str(e)

    @property
    def active(self) -> bool:
        return self._handle is not None

    def assign(self, pid: int) -> bool:
        """把已启动的进程加入作业；失败不影响主流程，仅记录原因。"""
        if self._handle is None:
            return False
        try:
            # 需要带 PROCESS_SET_QUOTA | PROCESS_TERMINATE 的句柄才能加入作业
            h = self._k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
            if not h:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not self._k32.AssignProcessToJobObject(self._handle, h):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self._k32.CloseHandle(h)
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def close(self):
        if self._handle is not None:
            try:
                self._k32.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None


# --------------------------------------------------------------------------
# 服务器进程管理
# --------------------------------------------------------------------------
class ServerManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"          # stopped / starting / running / exited
        self.exit_code = None
        self.started_at = None
        self.cmdline: list | None = None
        self.log: deque = deque(maxlen=8000)
        self.upstream_health = "unreachable"
        # 启动器退出时自动带走模型进程，避免孤儿进程占显存
        self.job = KillOnCloseJob()
        self.upstream_model = None

    def _log(self, line: str):
        self.log.append({"t": time.strftime("%H:%M:%S"), "line": line})

    def _reader(self):
        try:
            for raw in self.proc.stdout:
                line = raw.rstrip("\n")
                if line:
                    self._log(line)
        except Exception:
            pass

    def upstream_url(self, path: str) -> str:
        up = self.cfg["upstream"]
        return f"http://{up['host']}:{up['port']}{path}"

    def poll_health(self):
        url = self.upstream_url("/health")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                body = json.loads(r.read().decode("utf-8"))
            self.upstream_health = "ok" if body.get("status") == "ok" else "busy"
        except Exception:
            self.upstream_health = "unreachable"
        # 模型名
        if self.upstream_health == "ok":
            try:
                with urllib.request.urlopen(self.upstream_url("/v1/models"), timeout=3) as r:
                    data = json.loads(r.read().decode("utf-8"))
                arr = data.get("data") or []
                if arr:
                    self.upstream_model = arr[0].get("id")
            except Exception:
                pass
        else:
            self.upstream_model = None
        with self.lock:
            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    # 只在状态首次变为 exited 时记录一次；
                    # 否则健康轮询每 2 秒都会重复写同一行，把日志缓冲刷满。
                    if self.state != "exited":
                        self.state = "exited"
                        self.exit_code = rc
                        self._log(f"[launcher] 进程结束 exit={rc}")
                elif self.state == "starting" and self.upstream_health == "ok":
                    self.state = "running"
            else:
                self.state = "running" if self.upstream_health == "ok" else "stopped"

    def start(self, preset_index=None, preset=None) -> dict:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "error": "已有受管进程在运行，请先停止"}
            if self.upstream_health == "ok":
                return {"ok": False, "external": True,
                        "error": f"端口 {self.cfg['upstream']['port']} 已检测到运行中的服务"
                                 "（不受本启动器管理）。请先停止它，或修改预设端口。"}
            try:
                if preset is None:
                    idx = int(preset_index if preset_index is not None
                              else self.cfg.get("active", 0))
                    presets = self.cfg.get("presets") or []
                    if not presets or not (0 <= idx < len(presets)):
                        return {"ok": False, "error": "预设索引无效"}
                    preset = presets[idx]["flags"]
                flags = normalize_flags(preset)
                port = int(flags.get("port") or self.cfg["upstream"]["port"])
                if port != self.cfg["upstream"]["port"]:
                    self.cfg["upstream"]["port"] = port
                    save_config(self.cfg)
                cmd = build_command(self.cfg, flags)
            except Exception as e:
                return {"ok": False, "error": f"构造命令失败: {e}"}
            if not Path(self.cfg["exe"]).exists():
                return {"ok": False, "error": f"worker 不存在: {self.cfg['exe']}"}
            if not Path(str(flags.get("model"))).exists():
                return {"ok": False, "error": f"模型文件不存在: {flags.get('model')}"}
            env = dict(os.environ)
            if self.cfg.get("gpu"):
                env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu"])
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=self.cfg.get("bin_dir") or str(BASE_DIR),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception as e:
                self.proc = None
                return {"ok": False, "error": f"启动失败: {e}"}
            # 加入作业对象：之后关闭 bat / Ctrl+C / 强杀启动器，模型进程都会被内核结束
            self.job.assign(self.proc.pid)
            self.state = "starting"
            self.exit_code = None
            self.started_at = time.time()
            self.cmdline = cmd
            self._log("[launcher] 启动: " + " ".join(cmd))
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._waiter, args=(self.proc,), daemon=True).start()
        return {"ok": True, "pid": self.proc.pid, "cmdline": cmd}

    def _waiter(self, proc: subprocess.Popen):
        rc = proc.wait()
        with self.lock:
            if self.proc is proc:
                self.exit_code = rc
                if self.state in ("starting", "running"):
                    self.state = "exited"
                    self._log(f"[launcher] 进程结束 exit={rc}")

    def stop(self) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                if self.proc is not None:
                    self.state = "stopped"
                    self.proc = None
                return {"ok": False, "error": "没有受管进程在运行"}
            pid = self.proc.pid
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=15)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
            except Exception:
                pass
        with self.lock:
            self.state = "stopped"
            self.proc = None
            self._log(f"[launcher] 已停止 PID={pid}")
        return {"ok": True}

    def snapshot(self) -> dict:
        """完整状态快照（注意：self.state 是状态字符串属性，故方法名不能叫 state）。"""
        with self.lock:
            return {
                "pid": self.proc.pid if self.proc is not None else None,
                "state": self.state,
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "cmdline": self.cmdline,
                "upstream": {
                    "host": self.cfg["upstream"]["host"],
                    "port": self.cfg["upstream"]["port"],
                    "health": self.upstream_health,
                    "model": self.upstream_model,
                },
                "exe": self.cfg["exe"],
                "gpu": self.cfg.get("gpu", ""),
            }

    def shutdown(self):
        """启动器退出时调用：先显式停掉模型进程，再关闭作业句柄。

        即使这里的 Python 代码没跑完（例如被强杀），作业对象也会在内核层面
        兜底结束模型进程，所以不会留下孤儿进程。
        """
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.stop()
        except Exception:
            pass
        try:
            self.job.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# HTTP 服务
# --------------------------------------------------------------------------
class LauncherHTTPServer(ThreadingHTTPServer):
    """客户端随时可能刷新 / 取消轮询，连接类异常属正常，不必打印堆栈。

    socketserver 默认把 handler 里冒出的异常交给 handle_error 打印 traceback。
    前端每 2 秒轮询 /api/upstream/*，上游不可达时会命中 502 分支，而此时
    浏览器往往已经断开 —— 不处理就会在控制台刷出 ConnectionAbortedError
    （WinError 10053）。
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError):
            return
        super().handle_error(request, client_address)


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "KvmemLauncher/1.0"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # 浏览器已刷新 / 取消请求（WinError 10053 等）。前端每 2 秒轮询一次，
            # 上游不可达时正好会走到 502 分支 —— 这属于正常现象，静默忽略，
            # 否则会在控制台刷出 ConnectionAbortedError 堆栈。
            pass

    def read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -- 路由 ----------------------------------------------------------------
    def do_GET(self):
        mgr = self.server.manager
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            idx = BASE_DIR / "web" / "index.html"
            if idx.exists():
                data = idx.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                # 必须禁用缓存：否则改了前端（如中文输入修复）后，
                # 浏览器仍可能用旧页面，表现为"改了没生效"。
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_json({"error": "web/index.html 缺失"}, 404)
            return
        if path == "/api/state":
            self.send_json(mgr.snapshot())
            return
        if path == "/api/config":
            self.send_json(mgr.cfg)
            return
        if path == "/api/server/log":
            after = 0
            if "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    k, _, v = kv.partition("=")
                    if k == "after":
                        try:
                            after = int(v)
                        except ValueError:
                            pass
            lines = list(mgr.log)
            self.send_json({"lines": lines[after:], "next": len(lines)})
            return
        if path == "/api/sys":
            self.send_json(self.server.sysmon.snapshot())
            return
        if path.startswith("/api/upstream/"):
            self.proxy_get("/" + path[len("/api/upstream/"):])
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        mgr = self.server.manager
        path = self.path.split("?", 1)[0]
        if path == "/api/server/start":
            try:
                data = json.loads(self.read_body().decode("utf-8"))
            except Exception:
                data = {}
            res = mgr.start(preset_index=data.get("preset_index"),
                            preset=data.get("preset"))
            self.send_json(res, 200 if res.get("ok") else 400)
            return
        if path == "/api/server/stop":
            self.send_json(mgr.stop())
            return
        if path == "/api/browse":
            try:
                data = json.loads(self.read_body().decode("utf-8"))
            except Exception:
                data = {}
            res = browse_path(mode=str(data.get("mode") or "file"),
                              initial=str(data.get("initial") or ""),
                              types=str(data.get("types") or ""),
                              title=str(data.get("title") or ""))
            # 取消不是错误：ok 与 cancelled 都回 200，由前端区分
            self.send_json(res, 200 if (res.get("ok") or res.get("cancelled")) else 400)
            return
        if path == "/api/config":
            try:
                data = json.loads(self.read_body().decode("utf-8"))
            except Exception:
                self.send_json({"error": "JSON 无效"}, 400)
                return
            for key in ("launcher_port", "upstream", "exe", "bin_dir", "gpu",
                        "active", "presets"):
                if key in data:
                    mgr.cfg[key] = data[key]
            save_config(mgr.cfg)
            self.send_json({"ok": True, "config": mgr.cfg})
            return
        if path == "/api/upstream/chat":
            self.proxy_chat()
            return
        self.send_json({"error": "not found"}, 404)

    # -- 代理 ----------------------------------------------------------------
    def proxy_get(self, path: str):
        mgr = self.server.manager
        url = mgr.upstream_url(path)
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "application/json")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_json({"error": f"upstream 不可达: {e}"}, 502)

    def proxy_chat(self):
        """POST /api/upstream/chat → 上游 /v1/chat/completions（SSE 直通）。"""
        mgr = self.server.manager
        raw = self.read_body()
        url = mgr.upstream_url("/v1/chat/completions")
        req = urllib.request.Request(
            url, data=raw, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as e:
            self.send_json({"error": f"upstream 不可达: {e}"}, 502)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="kvmem-ninfer Web 启动器")
    ap.add_argument("--port", type=int, default=None,
                    help="launcher 监听端口（默认 config.json launcher_port）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    cfg = load_config()
    mgr = ServerManager(cfg)
    sysmon = SystemMonitor(interval=2.0)

    def health_loop():
        while True:
            mgr.poll_health()
            time.sleep(2)
    threading.Thread(target=health_loop, daemon=True).start()
    threading.Thread(target=sysmon.run_forever, daemon=True).start()

    port = args.port or int(cfg.get("launcher_port", 18300))
    httpd = LauncherHTTPServer(("127.0.0.1", port), Handler)
    httpd.manager = mgr
    httpd.sysmon = sysmon
    url = f"http://127.0.0.1:{port}"

    # ---- 退出清理 --------------------------------------------------------
    # 关闭 bat 窗口时 Windows 会给控制台进程发 CTRL_CLOSE_EVENT，
    # Python 默认只对 CTRL_C_EVENT 抛 KeyboardInterrupt，所以这里显式注册处理器，
    # 在退出前主动停掉模型进程。
    # 注意：作业对象（KillOnCloseJob）才是真正的兜底 —— 即使本进程被强杀，
    # 内核也会结束作业内的模型进程。
    stopping = threading.Event()

    def _cleanup(reason: str):
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n[launcher] {reason}，正在停止模型进程…", flush=True)
        try:
            mgr.shutdown()
        except Exception as e:
            print(f"[launcher] 清理时出错: {e}", flush=True)

    def _console_handler(event):
        # 2=CTRL_CLOSE_EVENT（关闭窗口） 5=CTRL_LOGOFF 6=CTRL_SHUTDOWN
        if event in (2, 5, 6):
            _cleanup("控制台关闭")
            return True    # 已处理；系统随后结束本进程
        # CTRL_C_EVENT(0) / CTRL_BREAK_EVENT(1) 必须返回 False，
        # 让 Python 默认处理器继续跑，否则 Ctrl+C 会被吞掉、serve_forever 不会退出。
        return False

    if os.name == "nt":
        try:
            _handler_ref = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(_console_handler)
            ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(
                _handler_ref, True)
        except Exception as e:
            print(f"[launcher] 注册控制台处理器失败（作业对象仍会兜底）: {e}")

    print("=" * 64)
    print("  kvmem-ninfer 启动器  →  " + url)
    print("  配置文件: " + str(CONFIG_FILE))
    smi = _find_nvidia_smi()
    print("  资源监控: CPU/内存=ctypes" + ("  GPU=" + smi if smi else "  GPU=未找到 nvidia-smi"))
    print("  退出行为: 关闭本窗口将自动停止模型服务"
          + ("" if mgr.job.active else "（作业对象未启用，改用显式停止）"))
    print("=" * 64)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _cleanup("收到 Ctrl+C")
    finally:
        _cleanup("退出")
        try:
            httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
