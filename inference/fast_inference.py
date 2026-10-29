# /// script
# requires-python = ">=3.11"
# dependencies = ["gliclass>=0.1.21", "torch", "transformers>=5.20", "kernels>=0.17.2,<0.18", "numpy", "datasets"]
# ///
"""Fast batch scoring for davanstrien/ultrabert-380m-curation on one CUDA GPU.

Same inputs and probabilities as the standard GLiClass pipeline (up to bf16 noise), about 3x faster on large batches:
- bf16 weights and the flash-attention 2 kernel from the Hub (`kernels-community/flash-attn2`);
- the attention q/k/v and feed-forward gate/up matrices are fused into one multiplication each;
- documents are packed end to end into rows of up to 64K tokens (no padding), then the encoder is compiled.

Use it as a module:

    from fast_inference import FastScorer
    scorer = FastScorer("davanstrien/ultrabert-380m-curation")
    probs = scorer.score(texts, ["News", "Tutorial", "Product page"], prompt="What is the format of this web page?")
    # -> one {label: probability} dict per text (probabilities sum to 1)

Decision (Jev / System One) style: a state plus typed questions (`noul` yes/no, `choice`, `score`; `choice` with
`"multiple": True` as an extension), answered in the System One response shape:

    out = scorer.system_one(state, {"edu": {"type": "score", "instructions": "...", "criteria": [...]}, ...})
    out["answers"]["edu"]["score"]

or run it as a self-check (agreement with the standard pipeline, and speed) on a sample of web pages:

    uv run fast_inference.py --n 4000
"""

import argparse
import json
import time
import types

import numpy as np
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from torch import nn
from transformers import AutoTokenizer

MAX_LEN = 2048  # tokens per document (the evaluation length; training used 4,096). Longer inputs are truncated.
BUCKET = 1024  # each packed row is padded to a multiple of this, so compile sees few shapes


def _cat_linear(layers):
    w = torch.cat([layer.weight for layer in layers], 0)
    out = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        out.weight.copy_(w)
    return out


def _fuse_encoder(enc):
    """Fuse q/k/v and gate/up projections of every UltraBERT layer in place."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.ultrabert.modeling_ultrabert import (
        UltrabertAttention,
        UltrabertMLP,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    def attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        shape = hidden_states.shape[:-1]
        q, k, v = self.qkv_proj(hidden_states).split(self._qkv_split, dim=-1)
        q, k, v = (t.view(*shape, -1, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        fn = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        out, w = fn(
            self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kw
        )
        return self.o_proj(out.reshape(*shape, -1).contiguous()), w

    def mlp_forward(self, x):
        gate, up = self.gate_up_proj(x).split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, UltrabertAttention):
                m._qkv_split = [m.q_proj.out_features, m.k_proj.out_features, m.v_proj.out_features]
                m.qkv_proj = _cat_linear([m.q_proj, m.k_proj, m.v_proj])
                del m.q_proj, m.k_proj, m.v_proj
                m.forward = types.MethodType(attn_forward, m)
            elif isinstance(m, UltrabertMLP):
                m.gate_up_proj = _cat_linear([m.gate_proj, m.up_proj])
                del m.gate_proj, m.up_proj
                m.forward = types.MethodType(mlp_forward, m)


def _first_fit_decreasing(lengths, cap):
    """Group item indices into bins whose total length is at most cap (longest items first)."""
    bins, room = [], []
    for i in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        for b in range(len(bins)):
            if room[b] >= lengths[i]:
                bins[b].append(i)
                room[b] -= lengths[i]
                break
        else:
            bins.append([i])
            room.append(cap - lengths[i])
    return bins


def _as_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_state(state):
    """A state (string, list or record) as the text the model reads; records become `key: value` lines."""
    if state in (None, "", {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {_as_text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines += [f"{key}:"] + [f"- {item}" for item in value]
        else:
            lines.append(f"{key}: {_as_text(value)}")
    return "\n".join(lines)


def question_options(question):
    """A question -> (answer keys, label strings), as in training: one label per option description;
    noul -> criteria.true / criteria.false or "yes" / "no"; score -> one label per level, lowest first."""
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not criteria or len(criteria) < 2:
            raise ValueError("a choice question needs at least two criteria")
        if isinstance(criteria, dict):
            return list(criteria), [_as_text(v) if v is not None else k for k, v in criteria.items()]
        return [str(c) for c in criteria], [str(c) for c in criteria]
    if kind == "noul":
        c = criteria if isinstance(criteria, dict) else {}
        return ["true", "false"], [_as_text(c.get("true") or "yes"), _as_text(c.get("false") or "no")]
    if kind == "score":
        return [str(i) for i in range(len(criteria))], [_as_text(c) for c in criteria]
    raise ValueError(f"unsupported question type {kind!r}: this model answers noul, choice and score questions")


def _softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def _check_config(config):
    """The packed head below reproduces only this head configuration (the released model's)."""
    expected = {"architecture_type": "uni-encoder", "pooling_strategy": "first", "class_token_pooling": "first",
                "embed_class_token": True, "scorer_type": "mlp", "normalize_features": False,
                "use_calibrator": False, "use_lstm": False}
    wrong = {k: getattr(config, k, None) for k, v in expected.items() if getattr(config, k, v) != v}
    if wrong:
        raise ValueError(f"FastScorer does not support this model's head configuration: {wrong}")


def _check_labels(labels):
    if not labels or not all(isinstance(x, str) and x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"labels must be a non-empty list of unique, non-empty strings, got {labels!r}")


class FastScorer:
    """Packed, fused, compiled inference for the GLiClass uni-encoder head on UltraBERT."""

    def __init__(self, model_id="davanstrien/ultrabert-380m-curation", step_tokens=65536, compile=True):
        if not torch.cuda.is_available():
            raise RuntimeError("FastScorer needs a CUDA GPU (Ampere or newer); on CPU use ZeroShotClassificationPipeline")
        self.step_tokens = step_tokens
        model = GLiClassModel.from_pretrained(model_id, dtype=torch.float32)
        _check_config(model.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = model.to("cuda", dtype=torch.bfloat16).eval()
        self.head = self.model.model
        enc = self.head.encoder_model
        enc.set_attn_implementation("kernels-community/flash-attn2")
        _fuse_encoder(enc)
        self.encoder = torch.compile(enc, dynamic=True) if compile else enc
        self.class_id = self.model.config.class_token_index
        pipe = ZeroShotClassificationPipeline(
            self.model, self.tokenizer, classification_type="single-label", max_length=MAX_LEN, device="cuda:0",
            progress_bar=False,
        )
        self._pipe = getattr(pipe, "pipe", pipe)  # the inner pipeline builds the model's input strings

    @torch.inference_mode()
    def _logits(self, token_ids):
        """token_ids: list of int64 arrays -> list of per-label logit arrays (one per sequence)."""
        out = [None] * len(token_ids)
        for idx in _first_fit_decreasing([len(s) for s in token_ids], self.step_tokens):
            lens = np.array([len(token_ids[i]) for i in idx])
            total = int(lens.sum())
            pad = -total % BUCKET
            segments = np.append(lens, pad) if pad else lens
            flat = np.concatenate([token_ids[i] for i in idx] + ([np.zeros(pad, np.int64)] if pad else []))
            cu = np.zeros(len(segments) + 1, np.int32)
            np.cumsum(segments, out=cu[1:])
            positions = np.arange(flat.size) - np.repeat(cu[:-1].astype(np.int64), segments)
            class_pos = np.flatnonzero(flat[:total] == self.class_id)
            class_doc = np.searchsorted(cu[1 : len(idx) + 1], class_pos, side="right")
            ids_t, pos_t, cu_t = (torch.from_numpy(a).cuda(non_blocking=True) for a in (flat, positions, cu))
            hidden = self.encoder(
                input_ids=ids_t[None], position_ids=pos_t[None], cu_seq_lens_q=cu_t, cu_seq_lens_k=cu_t,
                max_length_q=MAX_LEN, max_length_k=MAX_LEN,
            ).last_hidden_state[0]
            # Head: first token of each document = text vector; each <<LABEL>> token = one option vector.
            starts = torch.from_numpy(cu[: len(idx)].astype(np.int64)).cuda()
            text = self.head.text_projector(hidden[starts])
            classes = self.head.classes_projector(hidden[torch.from_numpy(class_pos).cuda()])
            doc_t = torch.from_numpy(class_doc).cuda()
            scores = self.head.scorer.mlp(torch.cat([text[doc_t], classes], -1)).squeeze(-1).float().cpu().numpy()
            counts = np.bincount(class_doc, minlength=len(idx))
            for i, part in zip(idx, np.split(scores, np.cumsum(counts)[:-1]), strict=True):
                out[i] = part
        return out

    def _request_logits(self, requests):
        """requests: list of (text, labels, prompt) -> raw label logits per request, or None when the label count
        does not match (truncation at 2,048 tokens cut off labels, or the text contains the literal "<<LABEL>>")."""
        if not requests:
            return []
        for _, labels, _ in requests:
            _check_labels(labels)
        strings = [self._pipe.prepare_input(text, labels, None, prompt) for text, labels, prompt in requests]
        ids = self.tokenizer(strings, truncation=True, max_length=MAX_LEN)["input_ids"]
        logits = self._logits([np.asarray(x, np.int64) for x in ids])
        return [
            lg if lg is not None and len(lg) == len(labels) else None
            for (_, labels, _), lg in zip(requests, logits, strict=True)
        ]

    def score_requests(self, requests):
        """requests: list of (text, labels, prompt) -> list of {label: probability} (or None, see _request_logits)."""
        results = []
        for (_, labels, _), lg in zip(requests, self._request_logits(requests), strict=True):
            results.append(None if lg is None else dict(zip(labels, _softmax(lg).tolist(), strict=True)))
        return results

    def system_one_batch(self, items):
        """items: list of (state, {name: question}) -> list of {"answers": {name: answer}}.

        Answers follow the System One (Jev) response shape:
        choice -> {"type": "choice", "choice": key, "confidence": p, "probabilities": {key: p}};
        noul -> {"type": "noul", "noul": P(true), "probabilities": {"true": p, "false": p}};
        score -> {"type": "score", "score": expected level, "confidence": p of the top level, "probabilities"};
        choice with "multiple": True (extension) -> {"type": "choice", "choices": [keys with P >= threshold
        (default 0.5)], "probabilities"} with independent yes/no probabilities per key.
        An answer is None if truncation cut off some options.
        """
        requests, owners = [], []
        for i, (state, questions) in enumerate(items):
            text = render_state(state)
            for name, question in questions.items():
                keys, labels = question_options(question)
                requests.append((text, labels, question.get("instructions") or ""))
                owners.append((i, name, question, keys, labels))
        outputs = [{"answers": {}} for _ in items]
        for (i, name, question, keys, labels), lg in zip(owners, self._request_logits(requests), strict=True):
            if lg is None:
                outputs[i]["answers"][name] = None
                continue
            kind = question["type"]
            if question.get("multiple"):  # trained as independent yes/no per option
                p = 1 / (1 + np.exp(-lg))
                probs = dict(zip(keys, p.tolist(), strict=True))
                answer = {"type": kind, "choices": [k for k in keys if probs[k] >= question.get("threshold", 0.5)]}
            else:
                p = _softmax(lg)
                probs = dict(zip(keys, p.tolist(), strict=True))
                best = keys[int(p.argmax())]
                if kind == "noul":
                    answer = {"type": kind, "noul": probs["true"]}
                elif kind == "score":
                    answer = {"type": kind, "score": float((p * np.arange(len(p))).sum()), "confidence": probs[best]}
                else:
                    answer = {"type": kind, "choice": best, "confidence": probs[best]}
            answer["probabilities"] = probs
            outputs[i]["answers"][name] = answer
        return outputs

    def system_one(self, state, questions):
        """One state, several named questions (see system_one_batch)."""
        return self.system_one_batch([(state, questions)])[0]

    def score(self, texts, labels, prompt=None):
        """Score every text against the same labels and prompt."""
        if isinstance(texts, str):
            texts = [texts]
        return self.score_requests([(t, labels, prompt) for t in texts])


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser(description="Self-check: agreement with the standard pipeline, and speed.")
    ap.add_argument("--model", default="davanstrien/ultrabert-380m-curation")
    ap.add_argument("--n", type=int, default=4000, help="web pages to score")
    ap.add_argument("--n-check", type=int, default=500, help="pages also scored with the standard pipeline")
    args = ap.parse_args()

    rows = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).take(args.n)
    texts = [r["text"][:3000] for r in rows]
    labels = ["not educational", "minimal educational value", "some educational value", "useful for education",
              "highly educational", "outstanding educational value"]
    prompt = "Classify the educational value of this web page."

    scorer = FastScorer(args.model)
    scorer.score(texts[:256], labels, prompt)  # warm-up: compiles for the common shapes
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    fast = scorer.score(texts, labels, prompt)
    torch.cuda.synchronize()
    fast_s = time.perf_counter() - t
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    model = GLiClassModel.from_pretrained(args.model, dtype=torch.bfloat16)
    pipe = ZeroShotClassificationPipeline(model, scorer.tokenizer, classification_type="single-label",
                                          max_length=MAX_LEN, device="cuda:0", progress_bar=False)
    check = texts[: args.n_check]
    t = time.perf_counter()
    std = pipe(check, labels, prompt=prompt, batch_size=16, return_hierarchical=True)
    std_s = time.perf_counter() - t

    pairs = [(f, s) for f, s in zip(fast, std, strict=False) if f is not None]
    agree = np.mean([max(f, key=f.get) == max(s, key=s.get) for f, s in pairs])
    diff = max(abs(f[k] - s[k]) for f, s in pairs for k in labels)
    print(f"fast path: {len(texts)} pages in {fast_s:.1f}s = {len(texts) / fast_s:.0f} pages/s, peak {peak_gb:.1f} GB")
    print(f"standard pipeline (batch 16): {len(check)} pages in {std_s:.1f}s = {len(check) / std_s:.0f} pages/s")
    print(f"top-1 agreement on {len(check)} pages: {agree:.3f}; max probability difference {diff:.3f}")

    out = scorer.system_one(texts[0], {
        "edu": {"type": "score", "instructions": prompt, "criteria": labels},
        "tutorial": {"type": "noul", "instructions": "Is this web page a tutorial or how-to guide?"},
        "format": {"type": "choice", "instructions": "Classify the format of this web page.",
                   "criteria": {"news": "News Article", "tutorial": "Tutorial / How-To Guide",
                                "knowledge": "Knowledge Article", "blog": "Personal Blog"}},
        "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does this web page cover?",
                   "criteria": {"science": "Science, Math & Technology", "health": "Health",
                                "education": "Education & Jobs", "history": "History & Geography"}},
    })
    print("system_one:", json.dumps(out["answers"], indent=1)[:1500])


if __name__ == "__main__":
    main()
