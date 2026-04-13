# baselines/wise.py
# Run:
#   python baselines/wise.py

import os
import sys
import json
import inspect
from datetime import datetime
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =======================
# Config (edit here)
# =======================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# project root = Chronos/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_JSONL = os.path.join(REPO_ROOT, "data", "knowledge.jsonl")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HPARAMS_YAML = os.path.join(THIS_DIR, "wise.yaml")
OUT_DIR = os.path.join(THIS_DIR, "edited_llama3_wise")

DTYPE = "bfloat16"          # "float16" | "bfloat16" | "float32"
DEVICE = "cuda"             # "cuda" | "cpu"

MAX_EDITS = -1
LOG_EVERY = 25
PREVIEW_EVERY = 100         # 0 disables preview

# ✅ batch size here
BATCH_SIZE = 1


# =======================
# Utilities
# =======================
def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d")


def load_quads(path: str) -> List[Dict[str, Any]]:
    quads: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            quads.append(json.loads(line))
    return quads


def build_prompt_from_quad(q: Dict[str, Any]) -> str:
    """
    IMPORTANT:
    Return a TEMPLATE prompt that contains '{}' so EasyEdit/WISE can do prompt.format(subject)
    to localize the subject token.
    """
    rel = str(q.get("relation", "")).strip().replace("_", " ")
    ts = str(q.get("timestamp", "")).strip()
    subj = str(q["subject"]).strip()
    subject_on_time = f"{subj} on {ts}"

    # keep {} slot for subject injection
    return (
        f'Question: What is the {rel} of {{}}?\n'
        "Answer:"
    )


def quad_to_request(q: Dict[str, Any]) -> Dict[str, Any]:
    subj = str(q["subject"]).strip()
    tgt = str(q["object"]).strip()
    ts = str(q["timestamp"]).strip()

    subject_on_time = f"{subj} on {ts}"

    # template with {} slot (rewrite template)
    tpl = build_prompt_from_quad(q).format(subject_on_time)  # contains {}

    return {
        # rewrite side (some code expects template and does .format(subject))
        "prompt": tpl,
        "subject": subject_on_time,
        "target_new": tgt,

        # localization side: GIVE BOTH
        "loc_subject": subject_on_time,

        # IMPORTANT: provide a FILLED loc_prompt so substring match works
        "loc_prompt": tpl.format(subject_on_time),

        # (optional) keep template too, in case other parts expect it
        "loc_prompt_template": tpl,
    }


def build_chat_prompt(tok: AutoTokenizer, user_text: str) -> str:
    # if hasattr(tok, "apply_chat_template"):
    #     msgs = [{"role": "user", "content": user_text}]
    #     return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text


@torch.no_grad()
def preview_one(model, tok, q: Dict[str, Any], max_new_tokens=24) -> str:
    # final QA (subject filled)
    subj = str(q["subject"]).strip()
    ts = str(q["timestamp"]).strip()
    subject_on_time = f"{subj} on {ts}"
    qa = build_prompt_from_quad(q).format(subject_on_time)
    prompt = build_chat_prompt(tok, qa)

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
    txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _robust_call(fn, **candidates):
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    return fn(**kwargs)


def apply_wise_robust(editor_or_fn, *, model, tok, requests, hparams):
    # Case A: editor object with .edit
    if hasattr(editor_or_fn, "edit") and callable(getattr(editor_or_fn, "edit")):
        out = _robust_call(
            editor_or_fn.edit,
            model=model,
            tok=tok,
            tokenizer=tok,
            requests=requests,
            request=requests,
            keep_original_weight=False,
        )
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    # Case B: function apply_wise_to_model(...)
    out = _robust_call(
        editor_or_fn,
        model=model,
        tok=tok,
        tokenizer=tok,
        requests=requests,
        request=requests,
        hparams=hparams,
        hyperparams=hparams,
        params=hparams,
    )
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


def print_final_qa_samples(quads: List[Dict[str, Any]], n: int = 5):
    """
    Print the FINAL QA strings (with subject filled), so you can verify alignment.
    """
    print("\n[Final QA samples]")
    for i, q in enumerate(quads[:n]):
        subj = str(q["subject"]).strip()
        ts = str(q["timestamp"]).strip()
        subject_on_time = f"{subj} on {ts}"

        qa = build_prompt_from_quad(q).format(subject_on_time)
        ans = str(q.get("object", "")).strip()
        print(f"<<MY_PRINT>> [{i}] {qa} -> [{ans}]")
    print("")


def main():
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from easyeditor.models.wise.wise_hparams import WISEHyperParams
    from easyeditor.models.wise.wise_main import apply_wise_to_model as _apply
    wise_apply_fn = _apply

    
    os.makedirs(OUT_DIR, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[DTYPE]

    print(f"[1] Loading model/tokenizer: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.padding_side = "left"
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

    # for n, _ in model.named_parameters():
    #     if "mlp.down_proj.weight" in n:
    #         print("[PARAM]", n)
    #         break

    # for n, _ in model.named_modules():
    #     if n.endswith("mlp.down_proj"):
    #         print("[MOD]", n)
    #         break

    print(f"[2] Loading quadruples: {KNOWLEDGE_JSONL}")
    quads = load_quads(KNOWLEDGE_JSONL)

    def ts_key(q):
        ts = str(q.get("timestamp", "1900-01-01"))
        try:
            return parse_timestamp(ts)
        except Exception:
            return datetime(1900, 1, 1)

    quads.sort(key=ts_key)
    if MAX_EDITS > 0:
        quads = quads[:MAX_EDITS]

    requests = [quad_to_request(q) for q in quads]
    print(f"    Loaded {len(requests)} edit requests")

    # Print final QA examples (filled subject) BEFORE editing
    print_final_qa_samples(quads, n=5)

    # sanity check request keys (ensures loc_prompt exists)
    if requests:
        need = {"prompt", "subject", "target_new", "loc_prompt", "loc_subject"}
        miss = [k for k in sorted(list(need)) if k not in requests[0]]
        if miss:
            raise RuntimeError(f"[BUG] request missing keys: {miss}. First request keys={list(requests[0].keys())}")

        print("[Sanity] First request keys OK:", list(requests[0].keys()))
        print("[Sanity] First request prompt template:\n", requests[0]["prompt"])
        print("[Sanity] First request FINAL QA:\n", requests[0]["prompt"].format(requests[0]["subject"]))
        print("[Sanity] First request loc_prompt (filled):\n", requests[0]["loc_prompt"])

    print("[3] Loading WISE hyperparams")
    if hasattr(WISEHyperParams, "from_hparams") and callable(getattr(WISEHyperParams, "from_hparams")):
        hparams = WISEHyperParams.from_hparams(HPARAMS_YAML)
    else:
        hparams = WISEHyperParams()

    editor = None
    # if BaseEditor is not None:
    #     try:
    #         editor = BaseEditor.from_hparams(hparams)
    #     except Exception:
    #         editor = None

    # if editor is None and wise_apply_fn is None:
    #     raise RuntimeError("Cannot build WISE editor and cannot import apply_wise_to_model. Check EasyEdit install.")

    print("[4] Applying WISE edits (BATCH_SIZE=1)")
    runner = editor if editor is not None else wise_apply_fn

    edited_model = model
    info = None
    total = len(requests)

    for i in range(0, total, BATCH_SIZE):
        batch_reqs = requests[i:i + BATCH_SIZE]
        batch_quads = quads[i:i + BATCH_SIZE]

        edited_model, info = apply_wise_robust(
            runner,
            model=edited_model,
            tok=tok,
            requests=batch_reqs,   # ✅ only 1 edit each time
            hparams=hparams,
        )

        done = i + len(batch_reqs)

        if done % LOG_EVERY == 0 or done == total:
            print(f"[WISE] Done {done}/{total}")

        if PREVIEW_EVERY and (done % PREVIEW_EVERY == 0 or done == total):
            try:
                q = batch_quads[-1]
                pred = preview_one(edited_model, tok, q)
                gold = str(q["object"]).strip()
                qa = build_prompt_from_quad(q).format(q["subject"])
                print("\n[Preview]")
                print("Q =", qa)
                print("Pred =", pred)
                print("Gold =", gold)
            except Exception as e:
                print("[Preview failed]", e)

        # light cleanup (helps reduce fragmentation / peak usage)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("[5] Saving edited model/tokenizer")
    edited_model.save_pretrained(OUT_DIR, safe_serialization=False)
    tok.save_pretrained(OUT_DIR)

    if info is not None:
        try:
            with open(os.path.join(OUT_DIR, "wise_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            print("[5.1] Saved wise_info.json")
        except Exception:
            pass

    print(f"[Done] Saved edited model to: {OUT_DIR}")


if __name__ == "__main__":
    main()