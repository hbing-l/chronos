# baselines/rome.py
# Run: python baselines/rome.py


import json
import os
import sys
import inspect
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------
# Hard-coded params
# -----------------------
KNOWLEDGE_JSONL = "data/knowledge.jsonl"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "KE_baselines/rome/edited_llama3_rome"

DTYPE = "bfloat16"          # "float16" | "bfloat16" | "float32"
DEVICE = "cuda"             # "cuda" | "cpu"

MAX_EDITS = -1              # -1 means all
LOG_EVERY = 25
PREVIEW_EVERY = 100         # 0 disables preview


def _repo_root() -> str:
    # this file is Chronos/KE_baselines/rome.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d")


def load_quads(jsonl_path: str) -> List[Dict[str, Any]]:
    quads: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            quads.append(json.loads(line))
    return quads


def build_quad_prompt(q: Dict[str, Any]) -> str:
    subj = str(q.get("subject", "")).strip()
    rel = str(q.get("relation", "")).strip()
    ts = str(q.get("timestamp", "")).strip()
    return (
        "Fact:\n"
        f"- Subject: {subj}\n"
        f"- Relation: {rel}\n"
        f"- Time: {ts}\n"
        "- Object:"
    )


def quad_to_request(q: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": build_quad_prompt(q),
        "subject": str(q["subject"]).strip(),
        "target_new": str(q["object"]).strip(),
    }


@torch.no_grad()
def greedy_preview(model, tok, q: Dict[str, Any], max_new_tokens: int = 12) -> str:
    prompt = build_quad_prompt(q)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def _call_apply_rome(apply_fn, *, model, tok, request: Dict[str, Any], hparams):
    """
    Adapt to different EasyEditor forks:
    apply_rome_to_model signature varies across repos.
    We'll inspect its parameter names and pass what it accepts.
    """
    sig = inspect.signature(apply_fn)
    kwargs = {}

    candidates = {
        "model": model,
        "tok": tok,
        "tokenizer": tok,
        "request": request,
        "requests": [request],
        "hparams": hparams,
        "hyperparams": hparams,
        "params": hparams,
        "device": DEVICE,
    }

    for name in sig.parameters.keys():
        if name in candidates:
            kwargs[name] = candidates[name]

    out = apply_fn(**kwargs)

    # common returns: edited_model OR (edited_model, delta/weights/etc.)
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def main():
    repo_root = _repo_root()
    os.chdir(repo_root)  # ensure relative paths resolve from Chronos root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Fail fast if init files will pull in lots of optional deps
    # _assert_minimal_inits(repo_root)

    # Now direct import should be safe
    from easyeditor.models.rome.rome_main import apply_rome_to_model
    from easyeditor.models.rome.rome_hparams import ROMEHyperParams

    os.makedirs(OUT_DIR, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if DTYPE not in dtype_map:
        raise ValueError(f"DTYPE must be one of {list(dtype_map.keys())}, got: {DTYPE}")
    torch_dtype = dtype_map[DTYPE]

    print(f"[1] Loading model/tokenizer: {BASE_MODEL}")
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

    print(f"[2] Loading knowledge: {KNOWLEDGE_JSONL}")
    quads = load_quads(KNOWLEDGE_JSONL)

    # Sort by timestamp so newer facts are applied later
    def ts_key(q: Dict[str, Any]) -> datetime:
        ts = str(q.get("timestamp", "1900-01-01"))
        try:
            return parse_timestamp(ts)
        except Exception:
            return datetime(1900, 1, 1)

    quads.sort(key=ts_key)
    if MAX_EDITS > 0:
        quads = quads[:MAX_EDITS]

    requests = [quad_to_request(q) for q in quads]
    print(f"    Loaded {len(requests)} edit requests.")

    if quads:
        print("\n[Sanity check] First prompt example:\n")
        print(build_quad_prompt(quads[0]))
        print("\nTarget object ->", quads[0]["object"])

    print("[3] Building ROME hparams")
    if hasattr(ROMEHyperParams, "from_hparams") and callable(getattr(ROMEHyperParams, "from_hparams")):
        hparams = ROMEHyperParams.from_hparams('KE_baselines/rome.yaml')
    else:
        try:
            hparams = ROMEHyperParams()
        except TypeError:
            hparams = ROMEHyperParams()

    print("[4] Applying edits sequentially with apply_rome_to_model...")
    edited_model = model
    for i, (req, q) in enumerate(zip(requests, quads), start=1):
        edited_model = _call_apply_rome(
            apply_rome_to_model,
            model=edited_model,
            tok=tok,
            request=[req],
            hparams=hparams,
        )

        if LOG_EVERY > 0 and i % LOG_EVERY == 0:
            print(f"    Done {i}/{len(requests)} edits")

        if PREVIEW_EVERY > 0 and i % PREVIEW_EVERY == 0:
            try:
                out = greedy_preview(edited_model, tok, q, max_new_tokens=10)
                print(f"\n[Preview @ {i}] prompt -> completion:")
                print(out)
            except Exception as e:
                print(f"[Preview failed @ {i}] {e}")

    print("[5] Saving edited model + tokenizer")
    edited_model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"[Done] Saved edited model to: {OUT_DIR}")


if __name__ == "__main__":
    main()