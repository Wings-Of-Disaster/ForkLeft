from __future__ import annotations

import inspect
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


EFFICIENT_DLM_MODELS = {
    "nvidia/Efficient-DLM-4B": {
        "revision": "d28c3167e0972d9c08eb92e03cb40a42d14690d5",
        "parent_checkpoint": "Qwen/Qwen3-4B",
        "license": "CC-BY-NC-4.0",
    },
    "nvidia/Efficient-DLM-8B": {
        "revision": "18404169688dda8bf0305cd8e5c33e7c9dd86c3b",
        "parent_checkpoint": "Qwen/Qwen3-8B",
        "license": "CC-BY-NC-4.0",
    },
}


@dataclass(frozen=True)
class StudentBackend:
    model: torch.nn.Module
    tokenizer: Any
    model_name: str
    mask_token_id: int
    loader: str
    revision: str | None
    efficient_dlm: bool


def choose_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def is_efficient_dlm_config(config: Any) -> bool:
    architectures = set(getattr(config, "architectures", None) or [])
    return "EfficientDLM" in architectures or (
        getattr(config, "dlm_paradigm", None) == "bidirectional"
        and getattr(config, "mask_token_id", None) == 151662
    )


def student_loader_name(config: Any) -> str:
    auto_map = getattr(config, "auto_map", None) or {}
    if "AutoModel" in auto_map and "AutoModelForMaskedLM" not in auto_map:
        return "AutoModel"
    return "AutoModelForMaskedLM"


def _legacy_default_rope_parameters(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    base = float(config.rope_theta)
    exponents = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    return 1.0 / (base**exponents), 1.0


@contextmanager
def _efficient_dlm_rope_compatibility(config: Any):
    if not is_efficient_dlm_config(config) or "default" in ROPE_INIT_FUNCTIONS:
        yield
        return
    ROPE_INIT_FUNCTIONS["default"] = _legacy_default_rope_parameters
    try:
        yield
    finally:
        ROPE_INIT_FUNCTIONS.pop("default", None)


def _refresh_legacy_default_rope_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if getattr(module, "rope_type", None) != "default":
            continue
        inv_freq, attention_scaling = _legacy_default_rope_parameters(
            module.config,
            device=module.inv_freq.device,
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = inv_freq
        module.attention_scaling = attention_scaling


def resolve_mask_token_id(config: Any, tokenizer: Any) -> int:
    mask_token_id = getattr(config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError("student model and tokenizer do not define a mask token ID")
    mask_token_id = int(mask_token_id)
    vocab_size = int(getattr(config, "vocab_size", len(tokenizer)))
    if not 0 <= mask_token_id < vocab_size:
        raise RuntimeError(
            f"mask token ID {mask_token_id} is outside model vocabulary {vocab_size}"
        )
    if is_efficient_dlm_config(config):
        token = tokenizer.convert_ids_to_tokens(mask_token_id)
        if mask_token_id != 151662 or token != "<|fim_pad|>":
            raise RuntimeError(
                "Efficient-DLM must use token 151662 (<|fim_pad|>) as its mask"
            )
    return mask_token_id


def load_student_backend(
    model_name: str,
    *,
    revision: str | None,
    dtype: torch.dtype,
    local_files_only: bool,
) -> StudentBackend:
    model_spec = EFFICIENT_DLM_MODELS.get(model_name)
    resolved_revision = revision or (model_spec["revision"] if model_spec else None)
    common = {
        "revision": resolved_revision,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(model_name, **common)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    if not hasattr(config, "pad_token_id"):
        config.pad_token_id = getattr(tokenizer, "pad_token_id", None)
    loader = student_loader_name(config)
    auto_class = AutoModel if loader == "AutoModel" else AutoModelForMaskedLM
    with _efficient_dlm_rope_compatibility(config):
        model = auto_class.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common,
        )
    if is_efficient_dlm_config(model.config):
        _refresh_legacy_default_rope_buffers(model)
    mask_token_id = resolve_mask_token_id(model.config, tokenizer)
    return StudentBackend(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        mask_token_id=mask_token_id,
        loader=loader,
        revision=resolved_revision,
        efficient_dlm=is_efficient_dlm_config(model.config),
    )


def validate_tokenizer_compatibility(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    *,
    mask_token_id: int,
    student_vocab_size: int,
    teacher_vocab_size: int,
) -> dict[str, int]:
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    shared_tokens = student_vocab.keys() & teacher_vocab.keys()
    mismatches = [
        token
        for token in shared_tokens
        if student_vocab[token] != teacher_vocab[token]
    ]
    if mismatches:
        token = mismatches[0]
        raise RuntimeError(
            "student and teacher token IDs differ for "
            f"{token!r}: {student_vocab[token]} != {teacher_vocab[token]}"
        )

    teacher_only = set(teacher_vocab) - set(student_vocab)
    if teacher_only:
        raise RuntimeError(
            f"teacher tokenizer has {len(teacher_only)} tokens absent from the student"
        )

    student_only = set(student_vocab) - set(teacher_vocab)
    unexpected_student_only = {
        token for token in student_only if student_vocab[token] != mask_token_id
    }
    if unexpected_student_only:
        raise RuntimeError(
            "student tokenizer has non-mask tokens absent from the teacher: "
            f"{sorted(unexpected_student_only)[:4]}"
        )

    if student_vocab_size != teacher_vocab_size:
        raise RuntimeError(
            "student and teacher output vocabularies differ: "
            f"{student_vocab_size} != {teacher_vocab_size}"
        )
    max_token_id = max(student_vocab.values())
    if max_token_id >= student_vocab_size:
        raise RuntimeError(
            f"student token ID {max_token_id} exceeds output vocabulary {student_vocab_size}"
        )
    return {
        "shared_tokens": len(shared_tokens),
        "student_only_tokens": len(student_only),
        "teacher_only_tokens": len(teacher_only),
        "output_vocab_size": student_vocab_size,
    }


@lru_cache(maxsize=None)
def _forward_arguments(model_type: type[torch.nn.Module]) -> tuple[set[str], bool]:
    parameters = inspect.signature(model_type.forward).parameters.values()
    return (
        {parameter.name for parameter in parameters},
        any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters),
    )


def _accepts_forward_argument(model: torch.nn.Module, name: str) -> bool:
    names, accepts_kwargs = _forward_arguments(type(model))
    return name in names or accepts_kwargs


def student_forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    inputs_are_unpadded: bool = False,
) -> torch.Tensor:
    config = getattr(model, "config", None)
    efficient_dlm = config is not None and is_efficient_dlm_config(config)
    if efficient_dlm and attention_mask is not None and not inputs_are_unpadded:
        if not bool(attention_mask.bool().all()):
            raise RuntimeError("Efficient-DLM training requires unpadded fixed-length inputs")
    model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if _accepts_forward_argument(model, "use_cache"):
        model_kwargs["use_cache"] = False
    elif efficient_dlm:
        raise RuntimeError("Efficient-DLM forward must accept use_cache=False")
    if _accepts_forward_argument(model, "return_dict"):
        model_kwargs["return_dict"] = True
    outputs = model(**model_kwargs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("student model did not return logits")
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise RuntimeError(
            f"student logits shape {tuple(logits.shape)} does not match input {tuple(input_ids.shape)}"
        )
    vocab_size = int(getattr(config, "vocab_size", logits.shape[-1]))
    if logits.shape[-1] != vocab_size:
        raise RuntimeError(
            f"student logits vocabulary {logits.shape[-1]} does not match config {vocab_size}"
        )
    return logits


def _auto_map_modules(auto_map: dict[str, Any]) -> set[str]:
    modules = set()
    for reference in auto_map.values():
        references = reference if isinstance(reference, (list, tuple)) else [reference]
        for item in references:
            if isinstance(item, str) and "." in item:
                modules.add(item.rsplit(".", 1)[0])
    return modules


def _copy_dynamic_code(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    auto_map = getattr(model.config, "auto_map", None) or {}
    module_names = _auto_map_modules(auto_map)
    if not module_names:
        return
    source_dirs: set[Path] = set()
    for cls in (model.__class__, model.config.__class__):
        try:
            source_dirs.add(Path(inspect.getfile(cls)).resolve().parent)
        except (OSError, TypeError):
            continue
    for module_name in module_names:
        filename = f"{module_name}.py"
        source = next(
            (source_dir / filename for source_dir in source_dirs if (source_dir / filename).is_file()),
            None,
        )
        if source is not None:
            shutil.copy2(source, checkpoint_dir / filename)


def save_student_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
) -> None:
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    _copy_dynamic_code(model, checkpoint_dir)
    auto_map = getattr(model.config, "auto_map", None) or {}
    missing = [
        f"{module_name}.py"
        for module_name in _auto_map_modules(auto_map)
        if not (checkpoint_dir / f"{module_name}.py").exists()
    ]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing custom model code required by auto_map: {sorted(set(missing))}"
        )


def backend_metadata(backend: StudentBackend) -> dict[str, Any]:
    config = backend.model.config
    model_spec = EFFICIENT_DLM_MODELS.get(backend.model_name, {})
    return {
        "model": backend.model_name,
        "loader": backend.loader,
        "revision": backend.revision,
        "mask_token_id": backend.mask_token_id,
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "efficient_dlm": backend.efficient_dlm,
        "block_size": getattr(config, "block_size", None),
        "dlm_paradigm": getattr(config, "dlm_paradigm", None),
        "parent_checkpoint": model_spec.get("parent_checkpoint"),
        "license": model_spec.get("license"),
    }
