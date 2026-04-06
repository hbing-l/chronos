import re
from datetime import datetime


def normalize_label(x):
    x = x.lower().strip()
    x = re.sub(r"\([^)]*\)", "", x)     # remove (...)
    x = re.sub(r"（[^）]*）", "", x)      # remove （...）
    x = re.sub(r'[^\w\s]', '', x)       # remove all punctuation
    x = re.sub(r"\s+", " ", x).strip()  # normalize whitespace
    return x


def evaluate_qa_exact_match(pred_answers, gold_answers):
    # answer: `A` or `A, B, C`
    correct = 0
    total = len(gold_answers)
    for pred, true in zip(pred_answers, gold_answers):
        pred = set([normalize_label(p) for p in pred.split(',')])
        gold = set([normalize_label(g) for g in true.split(',')])
        # print(f"Pred: {pred} | Gold: {gold}")
        if pred == gold:
            correct += 1
    return correct / total if total > 0 else 0.0


def quad_to_text(quad, with_timestamp=True):
    """
    Convert a temporal knowledge quadruple into a textual description
    that the embedding model can understand.
    """
    s = quad["subject"]
    r = quad["relation"]
    o = quad["object"]
    t = quad["timestamp"]

    if with_timestamp:
        text = "Fact at time {}:\n  subject: {}\n  relation: {}\n  object: {}".format(t, s, r, o)
    else:
        text = "Fact:\n  subject: {}\n  relation: {}\n  object: {}".format(s, r, o)
    return text


def parse_date(date_str: str):
    """Convert 'YYYY-MM-DD' into a comparable datetime.date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

