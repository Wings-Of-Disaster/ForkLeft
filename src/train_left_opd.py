#!/usr/bin/env python3
"""Left-only on-policy distillation for masked diffusion language models.

Training step:
  1. the student rolls out its own continuation;
  2. positions are sampled at random, as in masked-diffusion training;
  3. each sampled position is supervised under the left-only condition P + c[:i];
  4. the realised EOS position is always included;
  5. samples are forked on the final-answer verdict before any loss is applied.

The commit order inside a rollout is serial and entropy-descending: the most uncertain masked
position is committed first, then the rest are re-scored with that token in the sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import datetime
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "related_work" / "code" / "OPDLM",
              REPO_ROOT / "related_work" / "code" / "OPDLM" / "reward"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FORMAT = "left-opd-v1"
MODEL_MASK_ID = 151662


def log_event(out_dir: Path, event: dict) -> None:
    if not _LOG:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with (out_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    print(json.dumps(event, ensure_ascii=True), flush=True)


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_rows(path: str, limit: int = 0) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def gold_text(row: dict[str, Any]) -> str | None:
    ans = str(row.get("ground_truth_answer", "") or "").strip()
    return None if (not ans or ans.upper() == "N/A") else ans


def build_scorer():
    from domain_reward import check_answer, extract_answer

    def score(row: dict[str, Any], text: str) -> tuple[bool, Any]:
        try:
            dom = str(row.get("domain", "math"))
            extracted = extract_answer(text, row, default_domain=dom)
            return bool(check_answer(extracted, row, default_domain=dom)), extracted
        except Exception:
            return False, None

    return score


def load_student(path: str, dtype: torch.dtype, device: str = "cuda:0"):
    from moe_dlm_distillation.dlm_backend import load_student_backend

    backend = load_student_backend(path, revision=None, dtype=dtype, local_files_only=True)
    return backend.model.to(device).eval(), backend.tokenizer, int(backend.mask_token_id)


def load_teacher(path: str, dtype: torch.dtype, device: str = "cuda:1"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = {"trust_remote_code": True, "local_files_only": True}
    tok = AutoTokenizer.from_pretrained(path, **common)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, **common
    ).to(device).eval()
    return model, tok


@torch.no_grad()
def rollout_confidence_first(model, prompt_ids: list[int], gen_length: int, mask_id: int,
                             block: int, steps_per_block: int, eos_ids: set[int],
                             forbid_stop_first: bool = True) -> list[int]:
    """The sampler's own order: commit the most confident masked positions first."""
    device = next(model.parameters()).device
    width = len(prompt_ids)
    x = torch.tensor([prompt_ids + [mask_id] * gen_length], dtype=torch.long, device=device)
    attention = torch.ones_like(x)
    eos_tensor = torch.tensor(sorted(eos_ids), device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
    committed = gen_length
    total_passes = 0
    max_total_passes = steps_per_block * 4  # hard cap to prevent runaway
    try:
        for lo in range(width, width + gen_length, block):
            hi = min(lo + block, width + gen_length)
            for step in range(steps_per_block):
                if total_passes >= max_total_passes:
                    break
                total_passes += 1
                live = x[:, lo:hi] == mask_id
                if not bool(live.any()):
                    break
                logits = model(input_ids=x, attention_mask=attention, use_cache=False,
                               return_dict=True).logits.float()[:, lo:hi]
                probs = torch.softmax(logits, dim=-1)
                conf, tok_ = probs.max(-1)
                # Confidence-first rollout (matches inference, stable for DDP).
                # ForkLeft's L_rank separately shapes confidence during training.
                conf = torch.where(live, conf, torch.full_like(conf, -float("inf")))
                if forbid_stop_first and lo == width and step == 0 and eos_ids:
                    ok = ~torch.isin(tok_, eos_tensor)
                    conf = torch.where(ok, conf, torch.full_like(conf, -float("inf")))
                steps_left = steps_per_block - step
                k = int(max(1, math.ceil(int(live.sum()) / max(1, steps_left))))
                take = torch.topk(conf[0], min(k, int(live.sum()))).indices
                x[0, lo + take] = tok_[0, take]
                del logits, probs, conf, tok_
            blk = x[0, lo:hi]
            if bool(torch.isin(blk, eos_tensor).any()):
                hit = torch.nonzero(torch.isin(blk, eos_tensor), as_tuple=False)
                committed = (lo - width) + int(hit[0, 0]) + 1
                break
    finally:
        model.train(was_training)
    return x[0, width:width + committed].tolist()


def left_forward(model, prompt_ids: list[int], continuation: list[int], readout: int):
    """Logits at `readout` with P + c[:readout] as the whole sequence."""
    if readout < 1:
        raise ValueError("readout must be >= 1")
    device = next(model.parameters()).device
    row = prompt_ids + list(continuation[:readout])
    x = torch.tensor([row], dtype=torch.long, device=device)
    logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                   return_dict=True).logits
    return logits[0, len(row) - 1]


def teacher_left_forward(teacher, prompt_ids: list[int], continuation: list[int], readout: int):
    device = next(teacher.parameters()).device
    row = prompt_ids + list(continuation[:readout])
    x = torch.tensor([row], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = teacher(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                         return_dict=True).logits
    return logits[0, -1]


def teacher_left_forward_batch(teacher, prompt_ids, continuation, readouts):
    """One batched teacher forward for all readout positions."""
    device = next(teacher.parameters()).device
    width = len(prompt_ids)
    rows = [prompt_ids + list(continuation[:r]) for r in readouts]
    maxlen = max(len(r) for r in rows) if rows else 1
    pad_id = 0
    x = torch.full((len(rows), maxlen), pad_id, dtype=torch.long, device=device)
    att = torch.zeros((len(rows), maxlen), dtype=torch.long, device=device)
    for b, row in enumerate(rows):
        x[b, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        att[b, :len(row)] = 1
    with torch.no_grad():
        logits = teacher(input_ids=x, attention_mask=att, use_cache=False,
                         return_dict=True).logits
    return [logits[b, len(rows[b]) - 1] for b in range(len(rows))]


def kl_teacher_student(teacher_row: torch.Tensor, student_row: torch.Tensor,
                       invalid: Sequence[int]) -> torch.Tensor:
    t = teacher_row.detach().float()
    s = student_row.float()
    # Teacher (Qwen, 152064 vocab) and student (EDLM, 151936 vocab) differ.
    # Truncate both to the smaller dimension so every downstream tensor matches.
    dim = min(t.shape[-1], s.shape[-1])
    t = t[..., :dim]
    s = s[..., :dim]
    keep = torch.ones(dim, dtype=torch.bool, device=t.device)
    valid_invalid = [i for i in invalid if i < dim]
    if valid_invalid:
        keep[list(valid_invalid)] = False
    neg = torch.full((dim,), -float("inf"), device=t.device, dtype=t.dtype)
    t = torch.where(keep, t, neg)
    s = torch.where(keep, s, neg.to(s.dtype))
    if not bool(torch.isfinite(t).any()) or not bool(torch.isfinite(s).any()):
        return torch.zeros((), device=s.device, dtype=torch.float32, requires_grad=True)
    log_p_t = F.log_softmax(t, dim=-1)
    log_p_s = F.log_softmax(s, dim=-1)
    p_t = log_p_t.exp().clamp_min(0.0)
    delta = torch.where(p_t > 0, log_p_t - log_p_s, torch.zeros_like(log_p_t))
    kl = (p_t * delta).sum()
    return kl


def coupling_scores(model, prompt_ids: list[int], continuation: list[int],
                    positions: Sequence[int], mask_id: int,
                    invalid: Sequence[int]) -> list[float]:
    """D_i = max(0, H_marginal - H_left), batched for efficiency.

    All marginal rows share the same length, so they fit one batch.  Left rows
    are left-padded to the longest and read at each row's own readout index.
    """
    if not positions:
        return []
    device = next(model.parameters()).device
    width = len(prompt_ids)

    # --- chunked marginal read: prompt + all masks, 4 positions at a time ---
    marginal_row = prompt_ids + [mask_id] * len(continuation)
    e_marg_list = []
    with torch.no_grad():
        for cs in range(0, len(positions), 4):
            chunk = list(positions[cs:cs + 4])
            marginal_x = torch.tensor([marginal_row] * len(chunk), dtype=torch.long, device=device)
            marginal_att = torch.ones_like(marginal_x)
            marginal_logits = model(input_ids=marginal_x, attention_mask=marginal_att,
                                    use_cache=False, return_dict=True).logits.float()
            for b, i in enumerate(chunk):
                lg = marginal_logits[b, width + i - 1]
                if invalid:
                    idx = [j for j in invalid if j < lg.shape[-1]]
                    if idx:
                        lg[idx] = lg.min() - 1e4
                lp = F.log_softmax(lg, dim=-1)
                e_marg_list.append(-(lp.exp() * lp).sum())
            del marginal_logits, marginal_x, marginal_att
    e_marg = torch.stack(e_marg_list)

    # --- chunked left read: prompt + cont[:i], 4 at a time ---
    e_left_list = []
    with torch.no_grad():
        for cs in range(0, len(positions), 4):
            chunk = list(positions[cs:cs + 4])
            left_rows = [prompt_ids + list(continuation[:i]) for i in chunk]
            max_len = max(len(r) for r in left_rows)
            left_x = torch.zeros((len(left_rows), max_len), dtype=torch.long, device=device)
            left_att = torch.zeros_like(left_x)
            for b, row in enumerate(left_rows):
                left_x[b, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                left_att[b, :len(row)] = 1
            left_logits = model(input_ids=left_x, attention_mask=left_att,
                                use_cache=False, return_dict=True).logits.float()
            for b, (row, i) in enumerate(zip(left_rows, chunk)):
                lg = left_logits[b, len(row) - 1]
                if invalid:
                    idx = [j for j in invalid if j < lg.shape[-1]]
                    if idx:
                        lg[idx] = lg.min() - 1e4
                lp = F.log_softmax(lg, dim=-1)
                e_left_list.append(-(lp.exp() * lp).sum())
            del left_logits, left_x, left_att
    e_left = torch.stack(e_left_list)

    d = (e_marg - e_left).clamp_min(0).tolist()
    return d


def _ddp_setup(args):
    import torch.distributed as dist
    if not args.ddp:
        return 0, 1, False
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"[ddp] rank {rank}/{world}: visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    return rank, world, True


def train(args: argparse.Namespace) -> None:
    global MODEL_MASK_ID, _LOG
    rank, world, ddp = _ddp_setup(args)
    _LOG = (rank == 0)
    out_dir = Path(args.out)
    rows = load_rows(args.data, args.limit)
    if ddp:
        rows = rows[rank::world]
    scorer = build_scorer()
    model, tok, mask_id = load_student(args.student, dtype_from_name(args.dtype),
                                       args.student_device)
    MODEL_MASK_ID = mask_id
    invalid = [mask_id] + ([int(tok.pad_token_id)] if tok.pad_token_id is not None else [])
    eos_ids = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()

    verdicts: dict[str, bool] = {}
    if args.verdicts:
        import glob as _glob
        for vf in _glob.glob(os.path.join(args.verdicts, "verdicts.rank*.jsonl")):
            for line in open(vf, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                verdicts[d["k"]] = d["ok"]

    needed = set()
    for j in range(int(args.train_steps) * int(args.batch_rows)):
        row = rows[j % len(rows)]
        q = str(row.get("question", ""))
        if "\\boxed" not in q:
            q = (q.rstrip() + "\nPlease reason step by step, and put your final answer "
                                "within \\boxed{}.")
        needed.add(hashlib.sha1(q.encode("utf-8")).hexdigest()[:16])
    missing = len(needed - set(verdicts))
    if rank == 0:
        print(f"[ddp] verdicts {len(needed) - missing}/{len(needed)} cached, {missing} missing",
              flush=True)

    # Load the teacher lazily, inside the loop, only if a question is genuinely missing from the
    # cache.  Loading it up front whenever `missing > 0` costs a second model on the student's
    # pair of cards and OOMs immediately.
    teacher = teacher_tok = None

    def ensure_teacher():
        nonlocal teacher, teacher_tok
        if teacher is None and args.persona != "self":
            teacher, teacher_tok = load_teacher(args.teacher, dtype_from_name(args.dtype),
                                                args.teacher_device)
        return teacher, teacher_tok

    teacher2 = teacher2_tok = None

    def ensure_teacher2():
        nonlocal teacher2, teacher2_tok
        if teacher2 is None and args.teacher2:
            teacher2, teacher2_tok = load_teacher(args.teacher2, dtype_from_name(args.dtype),
                                                  args.teacher2_device)
        return teacher2, teacher2_tok

    if args.teacher2:
        ensure_teacher2()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    counts = {"right_right": 0, "wrong_right": 0, "wrong_wrong": 0, "right_wrong": 0}
    losses: list[float] = []
    model.train()
    for prm in model.parameters():
        prm.requires_grad_(True)
    cursor = 0
    rng = random.Random(args.seed)

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        loss_sum = 0.0
        terms = 0
        stats: dict[str, Any] = {"kl_terms": 0, "ce_terms": 0, "eos_terms": 0}

        batch = [rows[(cursor + j) % len(rows)] for j in range(args.batch_rows)]
        cursor += args.batch_rows
        for row in batch:
            question = str(row.get("question", ""))
            if "\\boxed" not in question:
                question = (question.rstrip()
                            + "\nPlease reason step by step, and put your final answer "
                              "within \\boxed{}.")
            prompt_ids = tok(question, add_special_tokens=False)["input_ids"][: args.max_prompt]
            gold = gold_text(row)

            cont = rollout_confidence_first(model, prompt_ids, args.gen_length, mask_id,
                                             args.block, args.steps_per_block, eos_ids)
            if args.terminate_at_answer:
                for j, t in enumerate(cont):
                    if int(t) in eos_ids:
                        cont = cont[:j + 1]
                        break
            if len(cont) < 2:
                continue
            student_ok = scorer(row, tok.decode(cont, skip_special_tokens=True))[0]

            key = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
            teacher_ok = verdicts.get(key)
            if teacher_ok is None and args.persona != "self":
                # MoE routing: math rows use teacher1, everything else uses teacher2
                v_teacher = teacher
                v_tok = teacher_tok
                if args.teacher2 and str(row.get("domain", "")) != "math":
                    if teacher2 is None:
                        ensure_teacher2()
                    if teacher2 is not None:
                        v_teacher = teacher2
                        v_tok = teacher2_tok
                if v_teacher is None:
                    ensure_teacher()
                    v_teacher = teacher
                    v_tok = teacher_tok
                t_prompt = v_tok(question, add_special_tokens=False)["input_ids"][
                    : args.max_prompt]
                from moe_dlm_distillation.train_forked_left_opd import (
                    teacher_greedy_continuation as _tgc)
                tcont = _tgc(v_teacher, v_tok, t_prompt, args.teacher_gen_length,
                             {int(v_tok.eos_token_id)}, args.max_prompt)
                teacher_ok = (scorer(row, v_tok.decode(tcont, skip_special_tokens=True))[0]
                              if len(tcont) >= 2 else None)
                verdicts[key] = teacher_ok
                if _verdict_fh is not None:
                    _verdict_fh.write(json.dumps({"k": key, "ok": teacher_ok}) + "\n")
                    _verdict_fh.flush()

            if teacher_ok is None:
                branch = "right_right"
            elif student_ok and teacher_ok:
                branch = "right_right"
            elif (not student_ok) and teacher_ok:
                branch = "wrong_right"
            elif (not student_ok) and (not teacher_ok):
                branch = "wrong_wrong"
            else:
                branch = "right_wrong"
            counts[branch] += 1

            n = len(cont)
            hi = n
            if args.position_window > 0:
                hi = max(1, min(n, int(args.position_window)))
            k = max(1, int(round(hi * args.sample_ratio)))
            if args.max_positions:
                k = min(k, int(args.max_positions))
            positions = sorted(rng.sample(range(1, hi), min(k, hi - 1)))
            eos_slots = [i for i in range(1, n) if int(cont[i]) in eos_ids]

            if branch == "right_wrong":
                continue

            if branch == "wrong_wrong":
                gcont = (tok(str(gold), add_special_tokens=False)["input_ids"][: args.gen_length]
                         if gold is not None else [])
                if len(gcont) >= args.gold_min_tokens:
                    cont = gcont
                    n = len(cont)
                    # Recompute positions and eos_slots for the gold continuation,
                    # whose length differs from the student rollout.
                    hi = n
                    if args.position_window > 0:
                        hi = max(1, min(n, int(args.position_window)))
                    k = max(1, int(round(hi * args.sample_ratio)))
                    if args.max_positions:
                        k = min(k, int(args.max_positions))
                    positions = sorted(rng.sample(range(1, hi), min(k, hi - 1)))
                    eos_slots = [i for i in range(1, n) if int(cont[i]) in eos_ids]
                else:
                    continue

            all_targets = positions + [s for s in eos_slots if s not in positions]
            if not all_targets:
                continue
            for i in all_targets:
                if i >= len(cont):
                    continue
                lg = left_forward(model, prompt_ids, cont, i)
                active_teacher = teacher
                if args.teacher2 and str(row.get("domain", "")) != "math" and teacher2 is not None:
                    active_teacher = teacher2
                if branch == "wrong_right" and active_teacher is not None:
                    t_lg = teacher_left_forward(active_teacher, prompt_ids, cont, i).to(lg.device)
                    term = kl_teacher_student(t_lg, lg, invalid)
                    if args.kl_ce_lambda > 0:
                        student_dim = lg.shape[-1]
                        t_ro = t_lg.detach().clone()[..., :student_dim]
                        if invalid:
                            for idx in invalid:
                                if idx < student_dim:
                                    t_ro[idx] = float("-inf")
                        teacher_tgt = int(t_ro.argmax())
                        ce = F.cross_entropy(lg.unsqueeze(0),
                                             torch.tensor([teacher_tgt], device=lg.device))
                        term = term + args.kl_ce_lambda * ce
                    stats["kl_terms"] += 1
                    del t_lg
                elif branch == "wrong_wrong" and gold is not None:
                    gids = tok(gold, add_special_tokens=False)["input_ids"][: args.gen_length]
                    tgt = int(cont[i])
                    term = F.cross_entropy(lg.unsqueeze(0), torch.tensor([tgt], device=lg.device))
                    stats["ce_terms"] += 1
                else:
                    tgt = int(cont[i])
                    term = F.cross_entropy(lg.unsqueeze(0), torch.tensor([tgt], device=lg.device))
                    stats["ce_terms"] += 1
                scaled = term / float(args.batch_rows * max(1, len(positions)))
                if not scaled.requires_grad:
                    print(f"[DEBUG] term.requires_grad={term.requires_grad}, "
                          f"term.grad_fn={term.grad_fn}, branch={branch}, "
                          f"model.training={model.training}, "
                          f"torch.is_grad_enabled={torch.is_grad_enabled()}", flush=True)
                scaled.backward()
                loss_sum += float(term.detach())
                terms += 1
                del lg, term, scaled
                if terms % 8 == 0:
                    torch.cuda.empty_cache()
            stats["eos_terms"] += len(eos_slots)
            torch.cuda.empty_cache()

            # ForkLeft Section 3.3: prefix-sensitivity ranking loss.
            # d[i] = [H_mask_i - H_left_i]+ (prefix sensitivity)
            # Higher d[i] should get LOWER confidence at inference.
            # coupling_scores is now batched (2 batched forwards instead of 24).
            # conf positions are also batched for a third batched forward.
            if args.rank_weight > 0 and len(positions) >= 2:
                d = coupling_scores(model, prompt_ids, cont, positions, mask_id, invalid)
                pairs = [(a, b) for a in range(len(d)) for b in range(len(d))
                         if d[a] > d[b] + args.rank_margin]
                if pairs:
                    # Chunked batched confidence (with gradient).
                    # 4 positions at a time to bound peak memory.
                    conf = [None] * len(positions)
                    chunk_size = 4
                    for chunk_start in range(0, len(positions), chunk_size):
                        chunk_idx = list(range(chunk_start, min(chunk_start + chunk_size, len(positions))))
                        chunk_positions = [positions[k] for k in chunk_idx]
                        left_rows = [prompt_ids + cont[:i] for i in chunk_positions]
                        max_len = max(len(r) for r in left_rows)
                        bx = torch.zeros((len(left_rows), max_len), dtype=torch.long,
                                         device=next(model.parameters()).device)
                        batt = torch.zeros_like(bx)
                        for b, row in enumerate(left_rows):
                            bx[b, :len(row)] = torch.tensor(row, dtype=torch.long, device=bx.device)
                            batt[b, :len(row)] = 1
                        logits = model(input_ids=bx, attention_mask=batt,
                                       use_cache=False, return_dict=True).logits.float()
                        for b, (row, idx) in enumerate(zip(left_rows, chunk_idx)):
                            lp = F.log_softmax(logits[b, len(row) - 1], dim=-1)
                            conf[idx] = lp.max()
                        del logits, bx, batt
                    # Hinge loss: penalize if high-D position is MORE confident than low-D
                    hinge = sum(F.relu(conf[a] - conf[b] + args.rank_hinge)
                                for a, b in pairs) / len(pairs)
                    (args.rank_weight * hinge / float(args.batch_rows)).backward()
                    stats["rank_pairs"] = len(pairs)

        if ddp:
            import torch.distributed as dist
            params = list(model.parameters())
            for prm in params:
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
            flat = torch.cat([prm.grad.reshape(-1) for prm in params])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world)
            off = 0
            with torch.no_grad():
                for prm in params:
                    n_ = prm.grad.numel()
                    prm.grad.copy_(flat[off:off + n_].view_as(prm.grad))
                    off += n_
            del flat

        if terms:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss_sum / terms)
        else:
            losses.append(float("nan"))
            log_event(out_dir, {"phase": "empty_step", "step": step})

        if step % args.log_every == 0 or step == args.train_steps:
            log_event(out_dir, {"phase": "step", "step": step,
                                "loss": losses[-1] if losses else None,
                                "counts": dict(counts), **stats})
        if step % args.save_every == 0 or step == args.train_steps:
            ckpt = out_dir / f"checkpoint-step{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            log_event(out_dir, {"phase": "save", "step": step, "path": str(ckpt)})

    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    summary = {"format": FORMAT, "persona": args.persona, "steps": args.train_steps,
               "counts": counts, "final_loss": losses[-1] if losses else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log_event(out_dir, {"phase": "done", **summary})


_verdict_fh = None


def main() -> None:
    global _verdict_fh
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="<MODEL_PATH>")
    ap.add_argument("--persona", choices=["teacher", "self"], default="teacher")
    ap.add_argument("--data", default="<DATA_PATH>")
    ap.add_argument("--verdicts", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--student-device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--teacher2", default="",
                    help="second teacher for non-math domains (MoE routing)")
    ap.add_argument("--teacher2-device", default="cuda:0")
    ap.add_argument("--train-steps", type=int, default=500)
    ap.add_argument("--steps-per-block", type=int, default=32)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--gold-min-tokens", type=int, default=32)
    ap.add_argument("--fork-loss", choices=["full", "teacher-right-only", "pure-opd"], default="full")
    ap.add_argument("--kl-ce-lambda", type=float, default=1.0)
    ap.add_argument("--terminate-at-answer", action="store_true")
    ap.add_argument("--position-window", type=int, default=0)
    ap.add_argument("--dagger-mode", choices=["off", "replace", "truncate"], default="off")
    ap.add_argument("--divergence-thresh", type=float, default=0.10)
    ap.add_argument("--backward-chunk", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=2)
    ap.add_argument("--gen-length", type=int, default=1024)
    ap.add_argument("--teacher-gen-length", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--sample-ratio", type=float, default=0.25)
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--rank-weight", type=float, default=0.1)
    ap.add_argument("--rank-margin", type=float, default=0.15)
    ap.add_argument("--rank-hinge", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.verdicts:
        vd = Path(args.verdicts)
        vd.mkdir(parents=True, exist_ok=True)
        _verdict_fh = (vd / "verdicts.rank0.jsonl").open("a", encoding="utf-8")
    train(args)


if __name__ == "__main__":
    main()
