#!/usr/bin/env python
"""Model-zoo evaluation on the OPDLM official benchmarks.

One prompt builder and one scorer for every model (both imported from OPDLM's own
``eval_utils.DATASET_CONFIGS`` / ``reward.domain_reward``), so a row per model in the
final table is comparable.  Only the sampler differs per family:

  --family ar        vLLM greedy decoding (same engine OPDLM's qwen_rl_rollout uses)
  --family maskdiff  batched semi-autoregressive masked diffusion (EDLM / LLaDA /
                     Dream): blocks left-to-right, low-confidence-static remasking
                     inside a block, which is the published Fast-dLLM operating point.

Efficient-DLM ignores ``attention_mask`` (measured: right-padding shifts its logits by
7.5 on a 24.75 scale), so batches are built as fixed-length *mask-padded* inputs with
an all-ones mask -- the same convention this repo already trains EDLM under.  Prompt
blocks contain no masks and are skipped without a forward pass.

Usage:
    python moe_dlm_distillation/eval_model_zoo.py \
        --model /path/to/ckpt --name edlm4b --family maskdiff \
        --datasets GSM8K MMLU --out runs/eval_sweep/zoo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPDLM_ROOT = REPO_ROOT / "related_work" / "code" / "OPDLM"
LLADA_ROOT = REPO_ROOT / "related_work" / "code" / "LLaDA"
# LLaDA's own generate() lives in the upstream repo checkout, not on PyPI.  `_run_official_llada`
# imports it directly so the sampler is the published one rather than a reimplementation.
sys.path.insert(0, str(LLADA_ROOT))
sys.path.insert(0, str(OPDLM_ROOT))
sys.path.insert(0, str(OPDLM_ROOT / "reward"))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from domain_reward import check_answer, extract_answer  # noqa: E402
from eval_utils import DATASET_CONFIGS, reformat_choices  # noqa: E402

# Per-dataset generation budget.
def install_transformers5_compat() -> None:
    """Vendored remote code (EDLM / LLaDA / Dream) was written for transformers 4.x.

    Three breakages on 5.16, each one hit for real before being patched here:
      * rope_type "default" is gone from ROPE_INIT_FUNCTIONS -> Dream raises KeyError
        while building its attention layers.
      * all_tied_weights_keys is assigned in post_init(), which 4.x-era remote models
        never call -> from_pretrained raises AttributeError in the missing-key scan.
        A class-level empty dict is safe: models that do call post_init shadow it.
      * GenerationConfig.validate() gained a user_set_attributes kwarg -> Dream's
        override raises TypeError.  Wrap it to swallow the new argument.

    Idempotent, and safe to call again after a model has been constructed (that is when
    the remote modules first appear in sys.modules).
    """
    from transformers import GenerationConfig, PretrainedConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    from moe_dlm_distillation.dlm_backend import _legacy_default_rope_parameters

    ROPE_INIT_FUNCTIONS.setdefault("default", _legacy_default_rope_parameters)
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    # A fourth breakage, hit when driving LLaDA through its OWN generate(): the vendored
    # forward does `if use_cache is None: use_cache = self.config.use_cache`, and transformers
    # 4.x's PretrainedConfig used to guarantee that attribute.  5.16 dropped it, so the
    # official sampler dies with AttributeError before the first step.  Give every config
    # class the 4.x default (False -- these are diffusion LMs, there is no KV cache), which
    # is exactly what transformers 4.x did.
    for name, module in list(sys.modules.items()):
        if "transformers_modules" not in name:
            continue
        for attribute in list(vars(module).values()):
            if not isinstance(attribute, type):
                continue
            if not issubclass(attribute, PretrainedConfig):
                continue
            if "use_cache" not in attribute.__dict__:
                attribute.use_cache = False

    for name, module in list(sys.modules.items()):
        if "transformers_modules" not in name:
            continue
        for attribute in list(vars(module).values()):
            if not isinstance(attribute, type):
                continue
            if (
                issubclass(attribute, GenerationConfig)
                and attribute is not GenerationConfig
                and "validate" in vars(attribute)
                and not getattr(attribute, "_ts5_validate_patched", False)
            ):
                original = vars(attribute)["validate"]

                def validate(self, *args, _original=original, **kwargs):
                    kwargs.pop("user_set_attributes", None)
                    return _original(self, *args, **kwargs)

                attribute.validate = validate
                attribute._ts5_validate_patched = True
            if (
                issubclass(attribute, PreTrainedModel)
                and attribute is not PreTrainedModel
                and "tie_weights" in vars(attribute)
                and not getattr(attribute, "_ts5_tie_patched", False)
            ):
                original_tie = vars(attribute)["tie_weights"]
                if _accepts_kwargs(original_tie):
                    attribute._ts5_tie_patched = True
                    continue

                def tie_weights(self, *args, _original=original_tie, **kwargs):
                    return _original(self, *args)

                attribute.tie_weights = tie_weights
                attribute._ts5_tie_patched = True


def _accepts_kwargs(function) -> bool:
    import inspect

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(function).parameters.values()
    )


def import_remote_classes(config, model_path: str) -> None:
    """Force-import the checkpoint's remote modules so install_transformers5_compat can
    see them.  AutoConfig only pulls in configuration_*.py; the modeling/generation
    modules (where the 4.x-era overrides live) appear later, which is why patching
    after AutoConfig alone silently misses them."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    for reference in (getattr(config, "auto_map", None) or {}).values():
        for entry in (reference if isinstance(reference, (list, tuple)) else [reference]):
            try:
                get_class_from_dynamic_module(entry, model_path)
            except Exception as error:  # noqa: BLE001 - best effort, loader decides later
                print(f"[compat] could not pre-import {entry}: {error}", flush=True)


MATH_BUDGET = 1024
MC_BUDGET = 512
MAX_MODEL_LEN = 8192
# logits index that predicts the token at position i, per family (see NativeDiffusionBackend).
FAMILY_READOUT_SHIFT = {"edlm": 0, "llada": 0, "dream": 1}
MC_LIKELIHOOD_TOKEN_BUDGET = 24576


def budget_for(domain: str) -> int:
    return MC_BUDGET if domain in ("mc", "zebralogic", "ifeval") else MATH_BUDGET


# ─────────────────── multiple choice by conditional likelihood ───────────────────
# This is how the diffusion-LM papers score their MC benches: LLaDA's EVAL.md lists
# MMLU / C-Eval / GPQA / CMMLU as "ppl" (conditional likelihood estimation, mc_num=1
# because the answer is a single token) and only GSM8K / MATH / BBH / HumanEval as
# "gen".  lm-eval scores its multiple_choice tasks the same way, so it is also the
# standard for the AR rows.  run_efficient_dlm_mcqa.py already validated this exact
# instrument on EDLM ("masked_answer_span_mean_logprob_v1": parent macro 75.9% over
# ARC-C/Easy, HellaSwag, PIQA, WinoGrande).
# Option lists appear in two shapes: "A. ..." (C-Eval, MLogiQA) and "(A) ..." (MMLU, GPQA).
# Both must be anchored to a line start: an unanchored "\(([A-J])\)" also matches the maths
# in C-Eval rows like "P(A \cap B)", which silently produced the option set {C,A,B} and lost
# the real answer D.
MC_LETTER_PATTERNS = (
    re.compile(r"(?:^|\n)\s*([A-J])[.)\u3001]\s"),
    re.compile(r"(?:^|\n)\s*\(([A-J])\)"),
    re.compile(r"\(([A-J])\)"),
)
LETTERS = "ABCDEFGHIJ"
LETTER_VARIANTS = (" {letter}", "{letter}", "({letter})")


def candidate_letters(question: str) -> list[str]:
    """Option letters the question actually offers, as a contiguous A.. run.

    Option lists are always A, B, C, ... in order, so a parse that skips one is a parse that
    matched something else; extending to the highest letter seen keeps the gold answer inside
    the candidate set instead of dropping the row.
    """
    for pattern in MC_LETTER_PATTERNS:
        found = list(dict.fromkeys(pattern.findall(question)))
        if len(found) >= 2:
            highest = max(LETTERS.index(letter) for letter in found)
            return list(LETTERS[: highest + 1])
    return []


def letter_token_ids(tokenizer, letter: str) -> list[int]:
    """Single-token spellings of a choice letter.

    "Answer: C" tokenises as [.., 25, 356] in the Qwen vocab and [.., 25, 348] in
    LLaDA's, i.e. the leading-space form " C" is the token a model really emits after
    the colon; the bare "C" is kept as a second spelling so a family whose tokenizer
    splits differently is not silently scored on an impossible token.
    """
    ids = []
    for variant in LETTER_VARIANTS:
        encoded = tokenizer.encode(variant.format(letter=letter), add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def gold_letter(row: dict) -> str:
    return str(row.get("ground_truth_answer", "")).strip().strip("()").upper()


def shard_positions(prompts: list[str], tokenizer, index: int, count: int, mode: str) -> list[int]:
    """Which rows of a dataset this worker owns.

    mode=index  rows[i::count] -- the historical behaviour, uniform over the dataset but it
                scatters equal-length prompts across workers.
    mode=length contiguous slice of the length-sorted order -- every worker gets one narrow
                prompt-length band.  Efficient-DLM batches only *exactly* equal-length prompts
                (it ignores attention_mask), so on MMLU this turns batch=3 into batch~30 at
                identical results; the split is still a partition, so summing shards is exact.
    """
    n = len(prompts)
    if mode == "index":
        return list(range(index, n, count))
    lengths = [len(ids) for ids in tokenizer(list(prompts), add_special_tokens=False)["input_ids"]]
    order = sorted(range(n), key=lambda i: (lengths[i], i))
    edges = [round(k * n / count) for k in range(count + 1)]
    return order[edges[index] : edges[index + 1]]


def load_rows(dataset: str) -> list[dict]:
    cfg = DATASET_CONFIGS[dataset]
    path = OPDLM_ROOT / "data" / cfg.get("path", dataset + ".json")
    with path.open() as handle:
        return json.load(handle)


def prompt_body(row: dict, cfg: dict) -> str:
    """Mirror sample/qwen_rl_rollout.py::_format_question exactly."""
    question = row["question"]
    if cfg.get("domain") == "mc":
        question = reformat_choices(question)
    template = cfg.get("prompt_template")
    if callable(template):
        return template(row, question)
    if template is not None:
        return template.format(question=question)
    return question


def render_prompt(tokenizer, row: dict, cfg: dict) -> str:
    body = prompt_body(row, cfg)
    messages = [{"role": "user", "content": body}]
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception:
            continue
    raise RuntimeError("chat template failed for this tokenizer")


GSM8K_FEWSHOT_ROOT = Path("<DATA_PATH>")


def gsm8k_demonstrations(shots: int) -> list[dict]:
    """The same 8-shot grade-school-math demonstrations generate_gsm8k.py uses, taken from
    the GSM8K *train* split, so nothing from the test set leaks into the prompt."""
    if shots <= 0:
        return []
    from moe_dlm_distillation.generate_gsm8k import clean_rationale, load_split

    train = load_split(GSM8K_FEWSHOT_ROOT, "train")
    return [
        {"question": str(row["question"]).strip(), "rationale": clean_rationale(str(row["answer"]))}
        for _, row in train.head(shots).iterrows()
    ]


def render_completion_prompt(row: dict, cfg: dict, dataset: str, demos: list[dict]) -> str:
    """Base-model protocol: plain continuation, no chat wrapper.

    Measured on LLaDA-8B-Base: with the ChatML wrapper it emits "the the the ..." and
    scores 1/32 on GPQA-Diamond; with a plain completion prompt it solves GSM8K correctly
    (48/2=24, 48+24=72).  The chat template is a property of the instruct models, not of
    the benchmark, so base checkpoints are run the way their own papers run them.
    """
    if dataset == "GSM8K" and demos:
        from moe_dlm_distillation.generate_gsm8k import build_prompt

        return build_prompt(demos, row["question"])
    question = row["question"]
    domain = cfg.get("domain")
    if domain == "mc":
        return reformat_choices(question).rstrip() + "\nAnswer:"
    template = cfg.get("prompt_template")
    if callable(template):
        return template(row, question)
    if template is not None:
        return template.format(question=question)
    return question


FEWSHOT_STOP = "\nQuestion:"


def truncate_fewshot(text: str) -> str:
    """Cut a completion-style generation at the few-shot separator.

    Base DLMs never emit EOS on an 8-shot GSM8K prompt: after "#### 18" they start a brand
    new "Question: ... Answer: ..." example, and OPDLM's extractors are last-match-wins, so the
    hallucinated example's numbers overwrite the real answer.  Stopping at the demonstration
    separator is what lm-eval's `until` does for the same prompt format; it never removes part
    of the model's own answer, because the demos are joined with exactly this marker.
    """
    cut = text.find(FEWSHOT_STOP)
    return text if cut < 0 else text[:cut]


def score(row: dict, cfg: dict, text: str) -> tuple[bool, object]:
    extracted = extract_answer(text, row, ds_cfg=cfg, default_domain=cfg.get("domain", "math"))
    return bool(check_answer(extracted, row, ds_cfg=cfg, default_domain=cfg.get("domain", "math"))), extracted


# ─────────────────────────── backends ───────────────────────────
class ARBackend:
    family = "ar"

    def __init__(self, model_path: str, gpu_memory_utilization: float, tensor_parallel_size: int):
        from transformers import AutoConfig
        from vllm import LLM

        # Qwen2.5-Math-7B-Instruct declares only 4096 positions; asking vLLM for more is a
        # hard error, so clamp to what the checkpoint supports.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        declared = int(getattr(config, "max_position_embeddings", 0) or 0)
        self.max_model_len = min(MAX_MODEL_LEN, declared) if declared else MAX_MODEL_LEN
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=self.max_model_len,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def generate(self, prompts: list[str], budget: int) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            min_p=0.0,
            max_tokens=budget,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return [out.outputs[0].text for out in self.engine.generate(prompts, params, use_tqdm=False)]

    def _max_logprobs(self, want: int = 64) -> int:
        """vLLM rejects logprobs above ModelConfig.max_logprobs (default 20).  A handful of
        offered letters never needs more than that, and rows where none of them ranks in
        the top-k fall through to the exact prompt_logprobs rescoring below."""
        cfg = getattr(getattr(self.engine, "llm_engine", None), "model_config", None)
        limit = getattr(cfg, "max_logprobs", 20)
        if not isinstance(limit, int) or limit <= 0:
            limit = 20
        return min(want, limit)

    def pick_letters(self, prompts: list[str], candidates: list[list[str]],
                     max_tokens: int = 0) -> list[str]:
        """One-token MC by next-token likelihood: ask vLLM for the top-k logprobs at the
        first generated position and take the highest-ranked offered letter.  Rows where
        no offered letter reaches the top-k are re-scored exactly, one candidate at a
        time, with prompt_logprobs (lm-eval's loglikelihood for a 1-token continuation)."""
        from vllm import SamplingParams

        top_k = self._max_logprobs()
        outs = self.engine.generate(
            prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_k), use_tqdm=False
        )
        wanted = [{tid: letter for letter in letters for tid in letter_token_ids(self.tokenizer, letter)}
                  for letters in candidates]
        picks: list[str | None] = []
        missing: list[int] = []
        for i, out in enumerate(outs):
            ranked = sorted(out.outputs[0].logprobs[0].items(), key=lambda kv: -kv[1].logprob)
            pick = next((wanted[i][tid] for tid, _ in ranked if tid in wanted[i]), None)
            if pick is None:
                missing.append(i)
            picks.append(pick)
        if missing:
            exact = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
            probes: list[str] = []
            owners: list[tuple[int, str]] = []
            for i in missing:
                for letter in candidates[i]:
                    for tid in letter_token_ids(self.tokenizer, letter):
                        probes.append(prompts[i] + self.tokenizer.decode([tid]))
                        owners.append((i, letter))
            best_lp: dict[int, float] = {}
            for (i, letter), out in zip(owners, self.engine.generate(probes, exact, use_tqdm=False)):
                lp = (out.prompt_logprobs[-1] or {})
                value = max((v.logprob for v in lp.values()), default=-float("inf"))
                if i not in best_lp or value > best_lp[i][0]:
                    best_lp[i] = (value, letter)
            for i in missing:
                picks[i] = (best_lp[i][1] if i in best_lp else None) or candidates[i][0]
            print(f"[ar] likelihood MC: {len(missing)}/{len(prompts)} rows needed exact rescoring "
                  f"({len(probes)} probes in one batch)", flush=True)
        return [str(pick) for pick in picks]


class NativeDiffusionBackend:
    """Batched semi-autoregressive masked diffusion, each family's own sampler.

    family=edlm   Efficient-DLM generate_with_prefix_cache_block_diff (block 32).
                  EDLM ignores attention_mask, so batches are built from *exactly
                  equal* prompt lengths -- no padding at all (measured: space padding
                  makes it emit spaces, mask padding is never denoised).
    family=llada  LLaDA official generate(), which extends attention_mask over the
                  generation region, so ordinary right padding is valid.
    family=dream  DreamModel.diffusion_generate() shipped by the checkpoint, also
                  attention_mask aware.
    """

    def __init__(self, model_path: str, family: str, block: int, steps_per_block: int,
                 dream_temperature: float = 0.0):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.family = family
        self.block = block
        self.steps_per_block = steps_per_block
        self.dream_temperature = dream_temperature
        # Official total sampling steps (LLaDA/Dream `generate(steps=...)`), set from the CLI.
        # `None` means "1 step per token", i.e. steps == gen_length, which is what each
        # checkpoint's OWN eval recipe uses (LLaDA EVAL.md: steps = gen_length = block_length).
        # Lowering it commits several tokens per forward and is the only real speed lever --
        # see `_run_official` for the measured quality cliff.
        self.official_steps = None
        common = {"trust_remote_code": True, "local_files_only": True}
        if family == "edlm":
            from moe_dlm_distillation.dlm_backend import choose_dtype, load_student_backend
            from moe_dlm_distillation.generate_strict_prefix_arms import install_legacy_cache_indexing

            install_legacy_cache_indexing()
            install_transformers5_compat()
            backend = load_student_backend(
                model_path, revision=None, dtype=choose_dtype("bfloat16"), local_files_only=True
            )
            self.model, self.tokenizer, self.mask_id = backend.model, backend.tokenizer, backend.mask_token_id
            self.block = int(getattr(backend.model.config, "block_size", 0) or block)
            forced = int(getattr(self, "force_block", 0) or 0)
            if forced:
                self.block = forced
                print(f"[edlm] block_size overridden: config="
                      f"{int(getattr(backend.model.config, 'block_size', 0) or block)} -> {forced}",
                      flush=True)
        else:
            config = AutoConfig.from_pretrained(model_path, **common)
            import_remote_classes(config, model_path)
            install_transformers5_compat()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **common)
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **common
            )
            self.mask_id = int(config.mask_token_id)
        # DreamModel.forward hands attention_mask straight to scaled_dot_product_attention,
        # which rejects int64; LLaDA's modeling code expects the usual integer mask.
        # (self.mask_dtype is set before read-out detection, which already needs it.)
        install_transformers5_compat()
        self.model = self.model.cuda().eval()
        self.device = next(self.model.parameters()).device
        self._repair_rope_buffers()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        eos = {self.tokenizer.eos_token_id, getattr(self.model.config, "eos_token_id", None)}
        for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                eos.add(tid)
        self.eos_ids = {int(e) for e in eos if e is not None}
        self.mask_dtype = torch.bool if family == "dream" else torch.long
        # DreamModel hands attention_mask straight to scaled_dot_product_attention, so it
        # needs a broadcastable 4-D key mask; LLaDA expands a 2-D mask itself.
        self.mask_ndim = 4 if family == "dream" else 2
        # Dream's own base-model eval recipe passes add_bos_token=True; without the BOS its
        # read-out probe is near chance (1/26 at i, 5/26 at i-1) instead of clearly i-1.
        self.add_bos = family == "dream" and self.tokenizer.bos_token_id is not None
        print(f"[{family}] mask_id={self.mask_id} add_bos={self.add_bos} block={self.block} "
              f"steps_per_block={steps_per_block} eos={sorted(self.eos_ids)}", flush=True)
        # The read-out convention is a property of each checkpoint's own sampler, so take
        # it from there instead of inferring it: LLaDA's generate() and EDLM's block-diff
        # sampler both read logits[i] for position i, while Dream's generation_utils._sample
        # does `logits = cat([logits[:, :1], logits[:, :-1]], 1)` before reading, i.e. i-1.
        self.shift = FAMILY_READOUT_SHIFT[family]
        self._probe_readout()

    def _repair_rope_buffers(self) -> None:
        """Recompute every ``inv_freq`` buffer and refuse to run if it still looks wrong.

        Fixes a transformers loading issue where non-persistent RoPE buffers
        (``inv_freq``) are not materialised from the checkpoint, leaving garbage
        in memory that produces NaN logits.  Recomputing the buffer fixes this."""
        from moe_dlm_distillation.dlm_backend import _refresh_legacy_default_rope_buffers

        _refresh_legacy_default_rope_buffers(self.model)
        theta = float(getattr(self.model.config, "rope_theta", 0.0) or 0.0)
        broken = []
        for name, module in self.model.named_modules():
            inv = getattr(module, "inv_freq", None)
            if inv is None:
                continue
            values = inv.detach().flatten().float().cpu()
            dim = int(values.numel()) * 2
            expected = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)) if theta else None
            if (
                not bool(torch.isfinite(values).all())
                or expected is None
                or not torch.allclose(values, expected, atol=1e-6)
            ):
                broken.append((name, values[:3].tolist()))
        if broken:
            raise RuntimeError(
                f"{self.family}: RoPE inv_freq is still wrong after repair at "
                f"{broken[:3]} (theta={theta}). Do not run this checkpoint -- an incorrect "
                f"inv_freq yields NaN logits, and NaN argmax silently decodes to token 0."
            )
        print(f"[{self.family}] rope repair: inv_freq verified against the analytic formula "
              f"(theta={theta})", flush=True)

    def _gen_length(self, budget: int) -> int:
        return max(self.block, (budget // self.block) * self.block)

    def _steps(self, gen_length: int) -> int:
        return (gen_length // self.block) * self.steps_per_block

    def _decode(self, tokens: list[int]) -> str:
        for i, tid in enumerate(tokens):
            if tid in self.eos_ids:
                tokens = tokens[:i]
                break
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _prepare_mask(self, attention: torch.Tensor) -> torch.Tensor:
        mask = attention.to(self.mask_dtype)
        return mask[:, None, None, :] if self.mask_ndim == 4 else mask

    @torch.no_grad()
    def self_test(self) -> None:
        """Quick sanity check: generate a short completion and verify it is reasonable."""
        prompt = "The capital city of France is"
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if getattr(self, "add_bos", False):
            ids = [self.tokenizer.bos_token_id] + ids
        # 32 tokens is enough to tell a working sampler from a collapsed one and costs
        # ~30 steps instead of a full block.  The official samplers assert
        # gen_length % block_length == 0, so align the block to this short probe and restore it
        # afterwards -- otherwise the integrity check itself trips the eval-time guard.
        saved_block = self.block
        if saved_block > 32:
            self.block = 32
        try:
            probe = self._decode(self._attempt([ids], 32, 1)[0]).strip()
        finally:
            self.block = saved_block
        text = probe
        chars = set(text.replace(" ", "").replace("\n", ""))
        print(f"[{self.family}] self-test on {prompt!r} -> {text[:160]!r} "
              f"(chars={len(chars)})", flush=True)
        problems = []
        if "paris" not in text.lower():
            problems.append("did not say Paris")
        if len(chars) < 4:
            problems.append(f"collapsed output ({len(chars)} distinct characters)")
        if problems:
            raise RuntimeError(
                f"{self.family} self-test failed: {'; '.join(problems)}. The sampler or the "
                "numerics are broken on this device -- do not publish these numbers. "
                "Give the job an exclusive GPU, or pass --skip-self-test only if you have "
                "independently confirmed the checkpoint works here."
            )

    @torch.no_grad()
    def _probe_readout(self) -> None:
        """Sanity-check the family read-out convention on a masked *span*.

        A single isolated mask is out of distribution for Dream (it is trained on a mask
        ratio), which made the old single-token probe score 2/26 vs 4/26 and abort the run.
        Masking a contiguous span of `block` tokens is what these samplers actually ask the
        model to do, so the signal is clean.  Diagnostic only -- the convention itself is
        taken from each checkpoint's own sampler (FAMILY_READOUT_SHIFT).
        """
        span = max(self.block, 8)
        texts = [
            "The capital city of France is Paris and it is famous for the Eiffel Tower.",
            "To solve the equation, we first subtract five from both sides, then divide by two.",
        ]
        hit = {0: 0, 1: 0}
        total = 0
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if getattr(self, "add_bos", False):
                ids = [self.tokenizer.bos_token_id] + ids
            for lo in range(2, max(3, len(ids) - span - 1), span):
                hi = min(lo + span, len(ids) - 1)
                if hi - lo < 2:
                    continue
                x = torch.tensor([ids], device=self.device)
                target = x[0, lo:hi].clone().cpu()
                x[0, lo:hi] = self.mask_id
                logits = self._forward_logits(x).float()
                for shift in (0, 1):
                    read = torch.cat([logits[:, :1], logits[:, :-1]], dim=1) if shift else logits
                    hit[shift] += int((read[0, lo:hi].argmax(-1).cpu() == target).sum())
                total += hi - lo
        if total:
            print(f"[{self.family}] span read-out probe pos_i={hit[0]}/{total} pos_i-1={hit[1]}/{total} "
                  f"-> using shift={self.shift} (from the checkpoint's own sampler)", flush=True)
            if hit[1 - self.shift] > hit[self.shift]:
                print(f"[{self.family}] WARNING probe prefers shift={1 - self.shift}; check the family table",
                      flush=True)

    def _forward_logits(self, x: torch.Tensor, attention: torch.Tensor | None = None) -> torch.Tensor:
        """One forward pass through whichever vendored remote model is loaded."""
        from moe_dlm_distillation.dlm_backend import student_forward_logits

        if attention is None:
            attention = torch.ones_like(x)
        return student_forward_logits(self.model, x, self._prepare_mask(attention), inputs_are_unpadded=True)

    @staticmethod
    def _assert_finite(logits: torch.Tensor, family: str, where: str) -> None:
        """Raise if logits contain NaN or Inf values."""
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                f"{family}: non-finite logits at {where}"
            )

    @torch.no_grad()
    def pick_letters(self, prompts: list[str], candidates: list[list[str]],
                     max_tokens: int = MC_LIKELIHOOD_TOKEN_BUDGET) -> list[str]:
        """MC by conditional likelihood: mask one answer slot, one forward per batch,
        argmax over the offered letters.

        Batches are built from exactly equal prompt lengths, so no padding is involved:
        Efficient-DLM requires that (it ignores attention_mask) and for LLaDA/Dream an
        all-ones mask takes the fast path in their own modeling code.
        """
        tok = self.tokenizer
        results: list[str | None] = [None] * len(prompts)
        ids_all = []
        for text in prompts:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if getattr(self, "add_bos", False):
                ids = [tok.bos_token_id] + ids
            ids_all.append(ids)
        wanted = [[(tid, letter) for letter in letters for tid in letter_token_ids(tok, letter)]
                  for letters in candidates]
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, ids in enumerate(ids_all):
            if len(ids) + 1 > MAX_MODEL_LEN:
                results[i] = candidates[i][0]
                continue
            buckets[len(ids)].append(i)
        started = time.time()
        done = 0
        for length in sorted(buckets, reverse=True):
            members = buckets[length]
            width = max(1, max_tokens // (length + 1))
            for start in range(0, len(members), width):
                chunk = members[start:start + width]
                x = torch.tensor([ids_all[i] + [self.mask_id] for i in chunk],
                                 dtype=torch.long, device=self.device)
                # The mask sits at the last position.  A shift=1 family (Dream, whose own
                # _sample does `logits = cat([logits[:, :1], logits[:, :-1]], 1)`) predicts the
                # token at position i from logits[i-1], so read one row earlier.
                logits = self._forward_logits(x)[:, x.shape[1] - 1 - self.shift, :]
                self._assert_finite(logits, self.family, "likelihood answer slot")
                logp = torch.log_softmax(logits.float(), dim=-1)
                for j, i in enumerate(chunk):
                    results[i] = max((float(logp[j, tid]), letter) for tid, letter in wanted[i])[1]
                done += len(chunk)
                rate = done / max(time.time() - started, 1e-6)
                print(f"[{self.family}] likelihood {done}/{len(prompts)} rows  {rate:.1f}/s  "
                      f"eta {(len(prompts) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
        return [str(result) for result in results]

    @torch.no_grad()
    def _generic_sample(self, x: torch.Tensor, attention: torch.Tensor, width: int,
                        gen_length: int) -> list[list[int]]:
        """Block semi-autoregressive sampling: low-confidence-static remasking inside each
        block, blocks committed left to right, per-sample stop as soon as EOS is committed.
        Uses only model.forward, so it is immune to the transformers-4.x generation code
        these checkpoints ship."""
        batch = x.shape[0]
        attention = self._prepare_mask(attention)
        shift = getattr(self, "shift", 0)
        eos_tensor = torch.tensor(sorted(self.eos_ids), device=x.device)
        alive = torch.ones(batch, dtype=torch.bool, device=x.device)
        steps_per_block = self.steps_per_block
        for lo in range(width, width + gen_length, self.block):
            hi = min(lo + self.block, width + gen_length)
            if not bool(alive.any()):
                break
            active = (x[:, lo:hi] == self.mask_id) & alive.unsqueeze(1)
            width_block = hi - lo
            for step in range(steps_per_block):
                live = (x[:, lo:hi] == self.mask_id) & alive.unsqueeze(1)
                if not bool(live.any()):
                    break
                logits = self.model(input_ids=x, attention_mask=attention,
                                    use_cache=False, return_dict=True).logits
                if shift:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = logits[:, lo:hi].float()
                probs = torch.softmax(logits, dim=-1)
                confidence, token = probs.max(-1)
                confidence = torch.where(live, confidence, torch.full_like(confidence, -float("inf")))
                steps_left = steps_per_block - step
                k = torch.ceil(live.sum(1).clamp(min=1).float() / steps_left).long().clamp(min=1)
                order = torch.argsort(confidence, dim=1, descending=True)
                ranks = torch.empty_like(order)
                ranks.scatter_(1, order, torch.arange(width_block, device=x.device).expand(batch, width_block))
                take = (ranks < k.unsqueeze(1)) & live
                rows = torch.nonzero(take)
                if not rows.numel():
                    break
                committed = token[take]
                x[rows[:, 0], lo + rows[:, 1]] = committed
                alive[rows[:, 0][torch.isin(committed, eos_tensor)]] = False
        return x[:, width:].tolist()

    @torch.no_grad()
    def _run_native(self, ids: list[list[int]], gen_length: int) -> list[list[int]]:
        """Efficient-DLM's own prefix-cache sampler.  All rows share one prompt length
        (equal-length bucketing upstream), so no padding is involved."""
        width = len(ids[0])
        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        out, nfe = self.model.generate(
            x, gen_length, self._steps(gen_length), self.block, False, None, 0.0
        )
        print(f"[edlm] batch={len(ids)} prompt_len={width} nfe={int(nfe)}", flush=True)
        return out[:, width:].tolist()

    @torch.no_grad()
    def _run_padded(self, ids: list[list[int]], gen_length: int) -> list[list[int]]:
        """LLaDA / Dream: drive the CHECKPOINT'S OWN official sampler.

        An earlier revision of this file ran a hand-written block loop here, which was both an
        avoidable deviation from the published recipe and needlessly slow.  Both LLaDA and
        Dream ship `generate(...)`/`diffusion_generate(...)`, so use them verbatim and only
        choose the three numbers the official harness chooses (EVAL.md for LLaDA gen tasks):

            eval_llada.py --tasks gsm8k --model_args \
                model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

        i.e. steps == gen_length == block_length -> a SINGLE block and one token committed per
        forward.  That is the official operating point and it is deliberately conservative;
        `steps < block_length` makes `get_num_transfer_tokens` commit `mask_num // steps`
        tokens per forward, which is where DLM speed comes from.  Measured on GSM8K
        (8 rows, prompt 1180, gen_length 256) the trade-off is sharp:

            steps=256 (~1 tok/step)  45.8 s/row   acc 25.0%
            steps= 64 (~4 tok/step)  11.4 s/row   acc 50.0%
            steps= 32 (~8 tok/step)   5.7 s/row   acc 12.5%  <- output collapses to " 1 2 3 4..."

        so `--steps` is exposed and the collapse point is checked on real data before a full
        sweep, rather than assumed.
        """
        if self.family == "llada":
            return self._run_official_llada(ids, gen_length)
        return self._run_padded_generic(ids, gen_length)

    @torch.no_grad()
    def _official_llada_generate(self, x, attention, steps, gen_length, block_length):
        """Call LLaDA's own generate() from the checkpoint's upstream repo."""
        from generate import generate as llada_generate

        return llada_generate(
            self.model, x, attention, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0.0, cfg_scale=0.0,
            remasking="low_confidence", mask_id=self.mask_id,
        )

    @torch.no_grad()
    def _run_official_llada(self, ids: list[list[int]], gen_length: int) -> list[list[int]]:
        if getattr(self, "add_bos", False):
            ids = [[self.tokenizer.bos_token_id] + list(row) for row in ids]
        width = max(len(row) for row in ids)
        pad = self.tokenizer.pad_token_id
        total = width + gen_length
        batch = torch.full((len(ids), total), self.mask_id, dtype=torch.long)
        attention = torch.zeros((len(ids), total), dtype=torch.long)
        for i, row in enumerate(ids):
            batch[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            batch[i, len(row):width] = pad
            attention[i, : len(row)] = 1
        steps = self.official_steps or gen_length
        if gen_length % self.block != 0 or steps % max(1, gen_length // self.block) != 0:
            raise SystemExit(
                f"llada official generate requires gen_length % block_length == 0 and "
                f"steps % num_blocks == 0, got gen_length={gen_length} block_length={self.block} "
                f"steps={steps}. Pick --block dividing --math-budget (block == budget gives the "
                f"official single-block setting; keep --steps a multiple of gen_length // block)."
            )
        # LLaDA's generate() extends attention_mask over the generation region itself, so hand
        # it only the prompt part (its own eval does exactly this).
        out = self._official_llada_generate(
            batch.to(self.device)[:, :width].clone(),
            attention.to(self.device)[:, :width].clone(),
            steps, gen_length, self.block,
        )
        return out[:, width:].tolist()

    @torch.no_grad()
    def _run_padded_generic(self, ids: list[list[int]], gen_length: int) -> list[list[int]]:
        """Dream: right-pad the prompt and pass a real attention mask, then run the block
        sampler (Dream's shipped diffusion_generate() is only used for the non-batched path)."""
        if getattr(self, "add_bos", False):
            ids = [[self.tokenizer.bos_token_id] + list(row) for row in ids]
        width = max(len(row) for row in ids)
        pad = self.tokenizer.pad_token_id
        total = width + gen_length
        batch = torch.full((len(ids), total), self.mask_id, dtype=torch.long)
        attention = torch.zeros((len(ids), total), dtype=torch.long)
        for i, row in enumerate(ids):
            batch[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            batch[i, len(row):width] = pad
            attention[i, : len(row)] = 1
        attention[:, width:] = 1
        return self._generic_sample(
            batch.to(self.device), attention.to(self.device), width, gen_length
        )

    def _attempt(self, ids, gen_length, batch_size):
        try:
            if self.family == "edlm":
                return self._run_native(ids, gen_length)
            return self._run_padded(ids, gen_length)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(ids) == 1:
                raise
            mid = len(ids) // 2
            print(f"[{self.family}] OOM at batch={len(ids)}; splitting {mid}+{len(ids) - mid}", flush=True)
            return self._attempt(ids[:mid], gen_length, batch_size) + self._attempt(ids[mid:], gen_length, batch_size)

    def generate(self, prompts: list[str], budget: int, batch_size: int) -> list[str]:
        gen_length = self._gen_length(budget)
        print(f"[{self.family}] generate: budget={budget} block={self.block} "
              f"gen_length={gen_length} official_steps={self.official_steps}", flush=True)
        tok = self.tokenizer
        results: list[str | None] = [None] * len(prompts)
        started = time.time()
        if self.family == "edlm":
            buckets: dict[int, list[int]] = defaultdict(list)
            for i, text in enumerate(prompts):
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) + gen_length > MAX_MODEL_LEN:
                    results[i] = ""
                    continue
                buckets[len(ids)].append(i)
            plan = []
            for length, members in buckets.items():
                for start in range(0, len(members), batch_size):
                    plan.append((length, members[start:start + batch_size]))
            plan.sort(key=lambda item: -len(item[1]))
            done = 0
            for length, members in plan:
                ids = [tok(prompts[i], add_special_tokens=False)["input_ids"] for i in members]
                produced = self._attempt(ids, gen_length, batch_size)
                for i, gen in zip(members, produced):
                    results[i] = self._decode(gen)
                done += len(members)
                rate = done / max(time.time() - started, 1e-6)
                print(f"[edlm] {done}/{len(prompts)} prompts  {rate:.2f}/s  "
                      f"eta {(len(prompts) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
        else:
            order = sorted(
                (i for i in range(len(prompts)) if results[i] is None),
                key=lambda i: len(prompts[i]),
            )
            ids_all = [tok(prompts[i], add_special_tokens=False)["input_ids"] for i in order]
            done = 0
            for start in range(0, len(order), batch_size):
                chunk = order[start:start + batch_size]
                ids = ids_all[start:start + batch_size]
                keep = [j for j, seq in enumerate(ids) if len(seq) + gen_length <= MAX_MODEL_LEN]
                produced = self._attempt([ids[j] for j in keep], gen_length, batch_size)
                for j, gen in zip(keep, produced):
                    results[chunk[j]] = self._decode(gen)
                for j in range(len(ids)):
                    if j not in keep:
                        results[chunk[j]] = ""
                done += len(chunk)
                rate = done / max(time.time() - started, 1e-6)
                print(f"[{self.family}] {done}/{len(prompts)} prompts  {rate:.2f}/s  "
                      f"eta {(len(prompts) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
        return [r if r is not None else "" for r in results]


# ─────────────────────────── driver ───────────────────────────
def main() -> None:
    global MATH_BUDGET, MC_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True, choices=["ar", "edlm", "llada", "dream"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--force-block", type=int, default=0,
                        help="override the checkpoint's config.block_size (edlm only).  This is "
                             "the LEFT-vs-BIDIRECTIONAL knob: block=1 commits one token per step "
                             "with no intra-block neighbours (= the causal/NTP information "
                             "condition), a large block lets the whole span see itself before "
                             "committing (= bidirectional). The warehouse default is 32.")
    parser.add_argument("--steps-per-block", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0,
                        help="total sampling steps for the official LLaDA/Dream generate() "
                             "(official eval uses steps == gen_length == block_length). "
                             "0 = same as gen_length, i.e. one token committed per forward")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N rows per dataset")
    parser.add_argument("--prompt-style", choices=["chat", "completion"], default="chat")
    parser.add_argument("--gsm8k-shots", type=int, default=8)
    parser.add_argument("--row-shard-index", type=int, default=0)
    parser.add_argument("--row-shard-count", type=int, default=1)
    parser.add_argument("--row-shard-mode", choices=["index", "length"], default="index",
                        help="length = contiguous slice of the length-sorted order (bigger "
                             "equal-length batches for EDLM, same partition, same results)")
    parser.add_argument("--dream-temperature", type=float, default=0.0)
    parser.add_argument("--math-budget", type=int, default=MATH_BUDGET)
    parser.add_argument("--mc-budget", type=int, default=MC_BUDGET)
    parser.add_argument("--mc-mode", choices=["generation", "likelihood"], default="generation",
                        help="likelihood = score letter-answer MC benches by conditional "
                             "likelihood of a masked answer slot (LLaDA's official 'ppl' "
                             "protocol, lm-eval's multiple_choice protocol), one forward/row")
    parser.add_argument("--mc-max-tokens", type=int, default=MC_LIKELIHOOD_TOKEN_BUDGET,
                        help="token budget per likelihood batch (batch x prompt length)")
    parser.add_argument("--skip-self-test", action="store_true",
                        help="skip the one-prompt generation integrity check (diffusion families)")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    MATH_BUDGET, MC_BUDGET = args.math_budget, args.mc_budget
    backend = (
        ARBackend(args.model, args.gpu_memory_utilization, args.tensor_parallel_size)
        if args.family == "ar"
        else NativeDiffusionBackend(
            args.model, args.family, args.block, args.steps_per_block, args.dream_temperature
        )
    )
    tokenizer = backend.tokenizer
    if args.force_block:
        backend.block = int(args.force_block)
        print(f"[${args.family}] block forced to {backend.block} for this run", flush=True)
    if args.family != "ar":
        backend.official_steps = int(args.steps) or None
        if args.steps:
            print(f"[{args.family}] official steps={args.steps} (per-block "
                  f"{args.steps // max(1, (budget_for('math') // backend.block))} commits ~"
                  f"{backend.block * max(1, (budget_for('math') // backend.block)) // args.steps} "
                  f"tokens/forward)", flush=True)
    if args.family != "ar" and not args.skip_self_test:
        backend.self_test()

    summary = {}
    for dataset in args.datasets:
        suffix = "" if args.row_shard_count == 1 else f".s{args.row_shard_index}of{args.row_shard_count}"
        done_path = out_dir / f"{dataset}{suffix}.json"
        if done_path.exists():
            cached = json.loads(done_path.read_text())
            summary[dataset] = {k: v for k, v in cached.items() if k != "records"}
            print(f"[{args.name}] {dataset}: cached {summary[dataset]['accuracy']:.4f}", flush=True)
            continue
        cfg = DATASET_CONFIGS[dataset]
        rows = load_rows(dataset)
        if args.limit:
            rows = rows[: args.limit]
        budget = budget_for(cfg.get("domain", "math"))
        mc_likelihood = args.mc_mode == "likelihood" and cfg.get("domain") == "mc"
        candidates: list[list[str]] = []
        if mc_likelihood:
            candidates = [candidate_letters(row["question"]) for row in rows]
            unmatched = [i for i, (row, letters) in enumerate(zip(rows, candidates))
                         if gold_letter(row) not in letters]
            if unmatched:
                raise SystemExit(
                    f"{dataset}: {len(unmatched)}/{len(rows)} rows offer no letter matching the gold "
                    f"answer (first is row {unmatched[0]}); likelihood MC needs letter-answer rows"
                )
            # Every family scores the same slot: the completion prompt ends in "Answer:".
            prompts = [render_completion_prompt(row, cfg, dataset, []) for row in rows]
        elif args.prompt_style == "chat":
            prompts = [render_prompt(tokenizer, row, cfg) for row in rows]
        else:
            demos = gsm8k_demonstrations(args.gsm8k_shots) if dataset == "GSM8K" else []
            prompts = [render_completion_prompt(row, cfg, dataset, demos) for row in rows]
        if args.row_shard_count > 1:
            keep = shard_positions(prompts, tokenizer, args.row_shard_index,
                                   args.row_shard_count, args.row_shard_mode)
            rows = [rows[i] for i in keep]
            prompts = [prompts[i] for i in keep]
            if candidates:
                candidates = [candidates[i] for i in keep]
        started = time.time()
        if mc_likelihood:
            texts = [f"Answer: {letter}" for letter in
                     backend.pick_letters(prompts, candidates, args.mc_max_tokens)]
        elif args.family == "ar":
            texts = backend.generate(prompts, budget)
        else:
            texts = backend.generate(prompts, budget, args.batch_size)
        raw_texts = list(texts)
        if args.prompt_style == "completion" and dataset == "GSM8K":
            texts = [truncate_fewshot(t) for t in texts]
        # 1500 chars (~500 tokens) was enough when every family ran on a 256-token budget, but
        # it silently amputated the stored text once the budget was raised, which made the
        # truncation audit impossible: a generation that ran to the 2048 ceiling looked exactly
        # like one that stopped early.  Keep enough characters to cover the largest budget plus
        # the prompt, and record the raw token count so "hit the ceiling" is directly testable.
        text_char_cap = max(4 * (budget + 512), 1500)
        raw_tokens = [
            len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
            for text in raw_texts
        ]
        # `hit_budget` on its own is NOT a truncation alarm.  On an 8-shot completion prompt
        # the AR checkpoints answer ("#### 18") and then keep writing a hallucinated next
        # example until the ceiling; that is harmless, because truncate_fewshot cuts exactly
        # at the separator.  Truncation that actually hurts is "the ceiling arrived while the
        # model was still writing its OWN answer", i.e. it hit the cap and never emitted the
        # separator.  Both are recorded so the audit can separate them.
        truncated_mid_answer = [
            bool(n_tok >= budget - 2 and not (dataset == "GSM8K" and FEWSHOT_STOP in raw))
            for raw, n_tok in zip(raw_texts, raw_tokens)
        ]
        truncation_rate_mid_answer = (
            sum(1 for flag in truncated_mid_answer if flag) / max(len(truncated_mid_answer), 1)
        )
        correct = 0
        records = []
        for row, text, raw, n_tok, mid in zip(
            rows, texts, raw_texts, raw_tokens, truncated_mid_answer
        ):
            ok, extracted = score(row, cfg, text)
            correct += int(ok)
            records.append({
                "correct": ok,
                "extracted": str(extracted)[:200],
                "text": text[:text_char_cap],
                "gen_tokens": int(n_tok),
                "hit_budget": bool(n_tok >= budget - 2),
                "raw_chars": len(raw),
                "truncated_mid_answer": bool(mid),
            })
        elapsed = time.time() - started
        result = {
            "model": args.name,
            "model_path": args.model,
            "family": args.family,
            "prompt_style": "completion" if mc_likelihood else args.prompt_style,
            "mc_mode": "likelihood" if mc_likelihood else "generation",
            "gsm8k_shots": args.gsm8k_shots if dataset == "GSM8K" else 0,
            "dataset": dataset,
            "row_shard_index": args.row_shard_index,
            "row_shard_count": args.row_shard_count,
            "sampler": (
                "official_llada_generate" if args.family == "llada"
                else "official_dream_diffusion_generate" if args.family == "dream"
                else "edlm_native_prefix_cache" if args.family == "edlm"
                else "vllm_greedy"
            ),
            "official_steps": int(args.steps) or 0,
            "block_length": int(args.block),
            "domain": cfg.get("domain"),
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / max(len(rows), 1),
            "budget": budget,
            "mean_gen_chars": sum(len(t) for t in texts) / max(len(texts), 1),
            "mean_gen_tokens": sum(raw_tokens) / max(len(raw_tokens), 1),
            "truncation_rate": sum(1 for t in raw_tokens if t >= budget - 2) / max(len(raw_tokens), 1),
            "truncation_rate_mid_answer": truncation_rate_mid_answer,
            "elapsed_s": elapsed,
            "records": records,
        }
        done_path.write_text(json.dumps(result, ensure_ascii=False))
        summary[dataset] = {k: v for k, v in result.items() if k != "records"}
        print(
            f"[{args.name}] {dataset}: acc={result['accuracy']:.4f} ({correct}/{len(rows)}) "
            f"budget={budget} {elapsed:.0f}s",
            flush=True,
        )
    summary_name = "summary.json" if args.row_shard_count == 1 else f"summary.s{args.row_shard_index}.json"
    (out_dir / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{args.name}] DONE {json.dumps({k: round(v['accuracy'], 4) for k, v in summary.items()})}", flush=True)


if __name__ == "__main__":
    main()
