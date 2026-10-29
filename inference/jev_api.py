# /// script
# requires-python = ">=3.11"
# dependencies = ["gliclass>=0.1.21", "torch", "transformers>=5.20", "fastapi", "uvicorn"]
# ///
"""Jev / System One decision API for davanstrien/ultrabert-380m-curation.

A request is a state (text, list or JSON record) plus named, typed questions; the response has one answer per
question and no generated text. Question types: `choice`, `noul` (yes / no), `score` (ordered levels), and `choice`
with `"multiple": True` (several options can be true). Each question is one GLiClass pass: the state is the text,
`instructions` the prompt, each option a label. Inputs longer than MAX_LENGTH tokens are refused, never truncated.

Serve it:

    uv run jev_api.py                       # or: uvicorn jev_api:app --host 0.0.0.0 --port 8765
    curl -s localhost:8765/v1/systemone -H 'Content-Type: application/json' -d '{"state": "...", "questions": {...}}'

or call it in Python:

    from jev_api import system_one
    system_one("page text", {"spam": {"type": "noul", "instructions": "Is this page spam?"}})

For large batches on a GPU, fast_inference.py (same mapping, packed and compiled) is about 3x faster.
"""

import json
import os

import torch
from fastapi import FastAPI, HTTPException
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL", "davanstrien/ultrabert-380m-curation")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4096"))  # the training length; longer inputs are refused
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

model = GLiClassModel.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_pipe = ZeroShotClassificationPipeline(model, tokenizer, max_length=10**9, classification_type="single-label",
                                       device=DEVICE, progress_bar=False)
pipe = getattr(_pipe, "pipe", _pipe)
app = FastAPI(title="System One API")


class Request(BaseModel):
    state: str | dict | list | None = None
    questions: dict[str, dict]


class RequestError(ValueError):
    """A request the model cannot answer (reported as HTTP 422)."""


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state(state):
    """A state as the text the model reads: records become `key: value` lines, string lists `- item` lines."""
    if state in (None, "", {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {as_text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines += [f"{key}:"] + [f"- {item}" for item in value]
        else:
            lines.append(f"{key}: {as_text(value)}")
    return "\n".join(lines)


def question_options(question):
    """A question -> (answer keys, labels), as in training."""
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not criteria or len(criteria) < 2:
            raise RequestError("a choice question needs at least two criteria")
        if isinstance(criteria, dict):
            return list(criteria), [as_text(v) if v is not None else k for k, v in criteria.items()]
        return [str(c) for c in criteria], [str(c) for c in criteria]
    if kind == "noul":
        c = criteria if isinstance(criteria, dict) else {}
        return ["true", "false"], [as_text(c.get("true") or "yes"), as_text(c.get("false") or "no")]
    if kind == "score":
        if not criteria or len(criteria) < 2:
            raise RequestError("a score question needs at least two levels")
        return [str(i) for i in range(len(criteria))], [as_text(c) for c in criteria]
    raise RequestError(f"unsupported question type {kind!r}: this model answers choice, noul and score questions")


@torch.no_grad()
def option_logits(text, prompt, labels):
    if len(set(labels)) != len(labels):
        raise RequestError("option descriptions must be unique")
    inputs = pipe.prepare_inputs([text], labels, same_labels=True, prompt=prompt)
    length = inputs["input_ids"].shape[-1]
    if length > MAX_LENGTH:
        raise RequestError(f"input of {length} tokens exceeds the {MAX_LENGTH}-token limit")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs, max_num_classes=len(labels)).logits[0, : len(labels)].float()
    return logits, length


def answer(question, keys, labels, logits):
    kind = question["type"]
    if question.get("multiple"):  # trained as an independent yes / no per option
        p = dict(zip(keys, torch.sigmoid(logits).tolist(), strict=True))
        threshold = question.get("threshold", 0.5)
        return {"type": kind, "choices": [k for k in keys if p[k] >= threshold], "probabilities": p}
    p = dict(zip(keys, torch.softmax(logits, dim=-1).tolist(), strict=True))
    best = max(keys, key=p.get)
    if kind == "noul":
        return {"type": kind, "noul": p["true"], "probabilities": p}
    if kind == "score":
        return {"type": kind, "score": sum(int(k) * v for k, v in p.items()), "confidence": p[best],
                "probabilities": p}
    return {"type": kind, "choice": best, "confidence": p[best], "probabilities": p}


def system_one(state, questions):
    """Answer every question about one state. Raises RequestError for a request the model cannot answer."""
    text, answers, tokens = render_state(state), {}, 0
    for name, question in questions.items():
        keys, labels = question_options(question)
        prompt = as_text(question.get("instructions") or "") or None
        logits, length = option_logits(text, prompt, labels)
        tokens += length
        answers[name] = answer(question, keys, labels, logits)
    return {"model": MODEL_ID, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}


@app.get("/v1/models")
def models():
    return {"models": [{"name": MODEL_ID, "description": "GLiClass curation decision model (UltraBERT-380M)"}]}


@app.post("/v1/systemone")
def systemone(request: Request):
    try:
        return system_one(request.state, request.questions)
    except RequestError as e:
        raise HTTPException(422, str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8765")))
