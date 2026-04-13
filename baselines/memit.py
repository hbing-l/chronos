# baselines/memit.py
# Run:
#   python baselines/memit.py

import os
import sys
import json
import inspect
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =======================
# Paths & config
# =======================
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_JSONL = os.path.join(REPO_ROOT, "data", "knowledge.jsonl")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HPARAMS_YAML = os.path.join(THIS_DIR, "memit.yaml")
OUT_DIR = os.path.join(THIS_DIR, "edited_llama3_memit")

DTYPE = "bfloat16"          # "float16" | "bfloat16" | "float32"
DEVICE = "cuda"             # "cuda" | "cpu"

CHUNK_SIZE = 5             # ✅ 每次 10 条一起传给 MEMIT
LOG_EVERY = 50
PREVIEW_EVERY = 200         # 0 disables preview
MAX_EDITS = -1              # -1 = all


# =======================
# Utilities
# =======================
def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d")


def load_quads(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(q: Dict[str, Any]) -> str:
    subj = q["subject"].strip()
    rel = q["relation"].replace("_", " ").strip()
    ts = q["timestamp"].strip()

    return (
        f"Question: On {ts}, what is the {rel} of {subj}?\n"
        "Answer:"
    )


def quad_to_request(q: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": build_prompt(q),
        "subject": q["subject"].strip(),
        "target_new": q["object"].strip(),
    }


@torch.no_grad()
def preview(model, tok, q: Dict[str, Any], max_new_tokens=12) -> str:
    prompt = build_prompt(q)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def apply_memit_robust(apply_fn, *, model, tok, requests, hparams):
    sig = inspect.signature(apply_fn)
    reqs = requests if isinstance(requests, list) else [requests]

    candidates = {
        "model": model,
        "tok": tok,
        "tokenizer": tok,
        "request": reqs,
        "requests": reqs,
        "hparams": hparams,
        "hyperparams": hparams,
        "params": hparams,
        "device": DEVICE,
    }
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}

    out = apply_fn(**kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


# =======================
# Main
# =======================
def main():
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    from easyeditor.models.memit.memit_main import apply_memit_to_model
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams

    os.makedirs(OUT_DIR, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[DTYPE]

    print(f"[1] Loading model: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    print("[2] Loading knowledge")
    quads = load_quads(KNOWLEDGE_JSONL)
    quads.sort(
        key=lambda q: parse_timestamp(q.get("timestamp", "1900-01-01"))
        if q.get("timestamp") else datetime(1900, 1, 1)
    )

    if MAX_EDITS > 0:
        quads = quads[:MAX_EDITS]

    requests = [quad_to_request(q) for q in quads]
    total = len(requests)
    print(f"    Loaded {total} edits")

    print("[3] Loading MEMIT hyperparameters")
    hparams = MEMITHyperParams.from_hparams(HPARAMS_YAML)

    print(f"[4] Applying MEMIT edits in chunks of {CHUNK_SIZE}")
    edited_model = model

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        batch_reqs = requests[start:end]
        batch_quads = quads[start:end]

        print(f"    [Batch] editing {start}:{end} (size={len(batch_reqs)})")  # ✅ 最后一批可能 <10

        edited_model = apply_memit_robust(
            apply_memit_to_model,
            model=edited_model,
            tok=tok,
            requests=batch_reqs,
            hparams=hparams,
        )

        if end % LOG_EVERY == 0 or end == total:
            print(f"    Done {end}/{total}")

        if PREVIEW_EVERY > 0 and (end % PREVIEW_EVERY == 0 or end == total):
            try:
                out = preview(edited_model, tok, batch_quads[-1])
                print(f"[Preview @ {end}] {out}")
            except Exception as e:
                print(f"[Preview failed] {e}")

    print("[5] Saving edited model")
    edited_model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"[Done] Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()