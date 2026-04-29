"""
Fast, CPU-friendly perplexity evaluation for causal LMs (e.g. GPT-2).

Key property: perplexity is computed from average NLL as:
  log_ppl = nll / n_tokens
  ppl = exp(log_ppl)

On very OOD text, log_ppl can be large enough that exp(log_ppl) overflows in float64,
producing `inf`. This module avoids that by returning `log_ppl` always, and returning
`ppl` as a capped finite value plus a `ppl_was_capped` flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class PerplexityResult:
    prompt_index: int
    prompt: str
    n_tokens: int
    nll: float
    log_ppl: Optional[float]
    ppl: Optional[float]
    ppl_was_capped: bool


def _safe_exp(log_x: float) -> tuple[float, bool]:
    """
    Compute exp(log_x) without returning inf due to overflow.

    Returns:
      (value, was_capped)
    """
    max_log = math.log(sys.float_info.max)  # ~709.78 for float64
    if log_x > max_log:
        return math.exp(max_log), True
    if log_x < -max_log:
        return math.exp(-max_log), True
    return math.exp(log_x), False


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    if device is None:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return tokenizer, model, device


def calculate_batch_perplexity(
    prompts: Sequence[str],
    model_name_or_path: str = "gpt2-xl",
    batch_size: int = 8,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    show_progress: bool = True,
) -> List[PerplexityResult]:
    """
    Compute per-prompt perplexities for a list of prompts.

    Notes:
    - This computes standard teacher-forced LM NLL over the prompt tokens.
    - Uses truncation to `max_length` (or the model max context length, whichever is smaller).
    - Returns both `log_ppl` and a capped finite `ppl` to avoid `inf`.
    """
    tokenizer, model, device = load_model_and_tokenizer(model_name_or_path, device=device)
    model_max_len = int(getattr(model.config, "max_position_embeddings", 1024))
    effective_max_len = model_max_len if max_length is None else min(model_max_len, int(max_length))

    results: List[PerplexityResult] = []
    indices_and_prompts = [(i, p) for i, p in enumerate(prompts) if isinstance(p, str) and p.strip() != ""]

    rng = range(0, len(indices_and_prompts), batch_size)
    iterator = tqdm(rng, desc="Perplexity batches", disable=not show_progress)

    with torch.inference_mode():
        for start in iterator:
            batch = indices_and_prompts[start : start + batch_size]
            batch_indices = [i for i, _ in batch]
            batch_prompts = [p for _, p in batch]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=effective_max_len,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # If a prompt tokenizes to length 0 (rare, but possible for weird inputs),
            # replace with eos to keep shapes valid.
            if input_ids.size(1) == 0:
                input_ids = torch.full((input_ids.size(0), 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (b, s, vocab)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            bsz, seqlen_m1, vocab = shift_logits.shape
            if seqlen_m1 == 0:
                # All sequences are length 1 -> no next-token predictions available
                for idx, prompt in zip(batch_indices, batch_prompts, strict=True):
                    results.append(
                        PerplexityResult(
                            prompt_index=idx,
                            prompt=prompt,
                            n_tokens=0,
                            nll=0.0,
                            log_ppl=None,
                            ppl=None,
                            ppl_was_capped=False,
                        )
                    )
                continue

            flat_logits = shift_logits.view(-1, vocab)
            flat_labels = shift_labels.view(-1)

            # Ignore padded positions directly in CE (faster + avoids computing junk loss).
            ignore_index = -100
            flat_labels_masked = flat_labels.clone()
            flat_labels_masked[shift_mask.view(-1) == 0] = ignore_index

            token_losses = F.cross_entropy(
                flat_logits,
                flat_labels_masked,
                reduction="none",
                ignore_index=ignore_index,
            ).view(bsz, seqlen_m1)

            n_tokens = shift_mask.sum(dim=1)  # (b,)
            nll = token_losses.sum(dim=1)  # (b,)

            for bi in range(bsz):
                n_tok = int(n_tokens[bi].item())
                nll_val = float(nll[bi].item())
                if n_tok <= 0:
                    log_ppl = None
                    ppl = None
                    capped = False
                else:
                    log_ppl = nll_val / n_tok
                    ppl, capped = _safe_exp(log_ppl)

                results.append(
                    PerplexityResult(
                        prompt_index=int(batch_indices[bi]),
                        prompt=str(batch_prompts[bi]),
                        n_tokens=n_tok,
                        nll=nll_val,
                        log_ppl=log_ppl,
                        ppl=ppl,
                        ppl_was_capped=capped,
                    )
                )

    results.sort(key=lambda r: r.prompt_index)
    return results


def _iter_csv_column(path: str, column: str, limit: Optional[int] = None) -> Iterator[str]:
    df = pd.read_csv(path, usecols=[column])
    values = df[column].tolist()
    n = len(values) if limit is None else min(len(values), int(limit))
    for i in range(n):
        yield "" if values[i] is None else str(values[i])


def evaluate_model_collection(
    model_collection_dir: str,
    prompts: Sequence[str],
    batch_size: int,
    device: str,
    max_length: Optional[int],
) -> dict[str, list[dict]]:
    """
    Evaluate every subdirectory model under `model_collection_dir`.
    """
    out: dict[str, list[dict]] = {}
    for entry in sorted(os.listdir(model_collection_dir)):
        model_path = os.path.join(model_collection_dir, entry)
        if not os.path.isdir(model_path):
            continue
        print(f"Evaluating {entry} ({model_path})")
        res = calculate_batch_perplexity(
            prompts=prompts,
            model_name_or_path=model_path,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )
        out[entry] = [r.__dict__ for r in res]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute prompt perplexities (safe on OOD inputs).")
    parser.add_argument("--model", default="gpt2-xl", help="HF model name or local path.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None, help="Truncate to this many tokens (<= model limit).")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Torch CPU threads (helps performance on CPU).",
    )

    parser.add_argument("--prompts-csv", type=str, default=None, help="Path to CSV containing prompts.")
    parser.add_argument("--prompt-column", type=str, default="prompt", help="CSV column name for prompts.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts.")

    parser.add_argument("--model-collection-dir", type=str, default=None, help="Directory of subfolder models to evaluate.")
    parser.add_argument("--out-json", type=str, default=None, help="Where to write JSON results.")
    args = parser.parse_args()

    if args.num_threads is not None and args.device == "cpu":
        torch.set_num_threads(int(args.num_threads))

    if args.prompts_csv is None:
        raise SystemExit("Missing --prompts-csv")

    prompts = list(_iter_csv_column(args.prompts_csv, args.prompt_column, limit=args.limit))

    if args.model_collection_dir:
        payload = evaluate_model_collection(
            model_collection_dir=args.model_collection_dir,
            prompts=prompts,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
    else:
        res = calculate_batch_perplexity(
            prompts=prompts,
            model_name_or_path=args.model,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
        payload = [r.__dict__ for r in res]

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps(payload[:3] if isinstance(payload, list) else payload, indent=2))


if __name__ == "__main__":
    main()

