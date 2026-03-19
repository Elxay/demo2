#!/usr/bin/env python3
"""
APA Batch Orchestrator (tmux edition)
======================================
Hardware: 4× RTX 4090
  - GPU 1:     Attacker (Qwen2.5-7B-Instruct), port 8001
  - GPU 0,2,3: Target (variable), port 8000

Prerequisites (one-time):
  tmux new-session -d -s attacker
  tmux new-session -d -s target

  # Start attacker (GPU 1):
  tmux send-keys -t attacker 'export HF_ENDPOINT="https://hf-mirror.com" && export HF_TOKEN="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" && export VLLM_USE_V1="0" && export VLLM_ATTENTION_BACKEND="MATH" && CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8001 --gpu-memory-utilization 0.9 --dtype bfloat16 --enforce-eager --disable-frontend-multiprocessing --trust-remote-code' Enter

Usage:
  python orchestrator.py                  # run all TARGET_MODELS
  python orchestrator.py gemma            # only models matching 'gemma'
"""

import subprocess
import time
import sys
import requests


# ================================================================
# Configuration
# ================================================================

TARGET_MODELS = [
    # "google/gemma-7b-it",
    "google/gemma-2-9b-it",
    
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "Qwen/Qwen3.5-9B",
    # ---- large models (3-GPU tensor parallel) ----
    # "meta-llama/Llama-3.3-70B-Instruct",
]

# Per-model overrides: tp, max_model_len, extra vLLM args
# NOTE on tp=3: vLLM requires many internal sizes to be divisible by tp.
#   Always set max_model_len explicitly when using tp=3.
#   Use tp=1 for models ≤ 13B (single 4090 is sufficient).
MODEL_VLLM_ARGS = {
    # ---- Small models: single GPU (tp=1) ----
    "google/gemma-7b-it":                    {"tp": 1},
    "google/gemma-2-9b-it":                  {"tp": 1},
    "Qwen/Qwen3.5-9B":                      {"tp": 1},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"tp": 1},
    "Qwen/Qwen2.5-7B-Instruct":             {"tp": 1},
    # ---- Large models: 3-GPU tensor parallel ----
    # max_model_len must be divisible by tp (3), so use multiples of 3
    # 8190 = 2730×3, 16380 = 5460×3
    "meta-llama/Llama-3.3-70B-Instruct":     {"tp": 3, "max_model_len": "16380"},
}

# ---- Environment (HF mirror etc.) ----
VLLM_ENV = (
    'export HF_ENDPOINT="https://hf-mirror.com" && '
    'export HF_TOKEN="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" && '
    'export VLLM_USE_V1="0" && '
    'export VLLM_ATTENTION_BACKEND="MATH"'
)

# ---- Ports ----
TARGET_PORT = 8002
ATTACKER_PORT = 8001

# ---- GPU layout ----
ATTACKER_GPU = "1"
TARGET_GPUS  = "3"

# ---- tmux ----
# TMUX_TARGET = "target"

# 多模型并行测试
TMUX_TARGET = "T2"

# ---- vLLM defaults ----
VLLM_GPU_UTIL = "0.9"
VLLM_DTYPE = "bfloat16"
VLLM_MAX_MODEL_LEN = "16384"

# ---- Timing ----
HEALTH_TIMEOUT = 600
HEALTH_INTERVAL = 5
GPU_COOLDOWN = 15


# ================================================================
# tmux helpers
# ================================================================

def tmux_send(session, cmd):
    subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"], check=True)

def tmux_ctrl_c(session):
    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], check=True)

def tmux_exists(session):
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


# ================================================================
# vLLM lifecycle
# ================================================================

def start_target(model_id):
    cfg = MODEL_VLLM_ARGS.get(model_id, {})
    tp = cfg.get("tp", 1)
    max_len = cfg.get("max_model_len", None)  # None = use model default
    extra = cfg.get("extra", "")

    gpus = TARGET_GPUS.split(",")[0] if tp == 1 else TARGET_GPUS

    # Step 1: set environment variables
    tmux_send(TMUX_TARGET, VLLM_ENV)
    time.sleep(1)

    # Step 2: build vLLM command
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} "
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model_id} "
        f"--port {TARGET_PORT} "
        f"--gpu-memory-utilization {VLLM_GPU_UTIL} "
        f"--dtype {VLLM_DTYPE} "
        f"--enforce-eager "
        f"--disable-frontend-multiprocessing "
        f"--trust-remote-code"
    )

    # Only add --tensor-parallel-size if tp > 1
    if tp > 1:
        cmd += f" --tensor-parallel-size {tp}"

    # Only add --max-model-len if explicitly configured
    # (required for tp>1 to ensure divisibility; optional for tp=1)
    if max_len is not None:
        cmd += f" --max-model-len {max_len}"

    if extra:
        cmd += f" {extra}"

    print(f"  GPU={gpus}  TP={tp}  max_len={max_len or 'default'}")
    tmux_send(TMUX_TARGET, cmd)


def stop_target():
    print(f"  Stopping target server...")
    tmux_ctrl_c(TMUX_TARGET)
    time.sleep(3)
    tmux_ctrl_c(TMUX_TARGET)
    print(f"  GPU cooldown {GPU_COOLDOWN}s...")
    time.sleep(GPU_COOLDOWN)


def wait_ready(port, timeout=HEALTH_TIMEOUT):
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                print(f"  Server ready ({time.time()-t0:.0f}s)")
                return True
        except:
            pass
        elapsed = time.time() - t0
        if int(elapsed) % 60 < HEALTH_INTERVAL and elapsed > 10:
            print(f"  Waiting... ({elapsed:.0f}s)")
        time.sleep(HEALTH_INTERVAL)
    print(f"  TIMEOUT ({timeout}s)")
    return False


# ================================================================
# Main
# ================================================================

def orchestrate(keyword=None):
    models = TARGET_MODELS
    if keyword:
        models = [m for m in models if keyword.lower() in m.lower()]
        if not models:
            print(f"No models match '{keyword}'"); return

    print("=" * 60)
    print(f"APA Orchestrator — {len(models)} target(s)")
    print(f"Target GPUs: {TARGET_GPUS}  |  Attacker GPU: {ATTACKER_GPU}")
    print("=" * 60)

    if not tmux_exists(TMUX_TARGET):
        print(f"ERROR: tmux session '{TMUX_TARGET}' not found.")
        print(f"  Run: tmux new-session -d -s {TMUX_TARGET}")
        return

    try:
        ok = requests.get(f"http://localhost:{ATTACKER_PORT}/health", timeout=3).status_code == 200
        print(f"Attacker OK (port {ATTACKER_PORT})" if ok else "")
        if not ok: raise Exception()
    except:
        print(f"WARNING: Attacker not responding on port {ATTACKER_PORT}")
        if input("Continue? [y/N]: ").strip().lower() != 'y':
            return

    summary = []
    for i, model in enumerate(models):
        tp = MODEL_VLLM_ARGS.get(model, {}).get("tp", 1)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(models)}] {model}  (TP={tp})")
        print(f"{'='*60}")

        start_target(model)
        if not wait_ready(TARGET_PORT):
            stop_target()
            summary.append((model, "❌ FAILED_TO_START"))
            continue

        try:
            from demo2.main import main as apa_main
            apa_main(target_model_id=model)
            summary.append((model, "✅ COMPLETED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            summary.append((model, f"❌ {e}"))

        stop_target()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m, s in summary:
        print(f"  {s}  {m}")


if __name__ == "__main__":
    orchestrate(sys.argv[1] if len(sys.argv) > 1 else None)