"""Shared validation and provenance contract for Avito case-browser insights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_INSIGHT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_INSIGHT_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 320
DEFAULT_RETRY_MAX_NEW_TOKENS = 640
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_INSIGHT_DEVICE = "cuda:0"
DEFAULT_COMPUTE_DTYPE = "float16"
DEFAULT_BATCH_SIZE = 1
GENERATION_CONTRACT_VERSION = 5

TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def _installed_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "unavailable"


DEFAULT_TORCH_VERSION = _installed_version("torch")
DEFAULT_TRANSFORMERS_VERSION = _installed_version("transformers")


def runtime_generation_versions() -> dict[str, str]:
    """Capture library/build versions that can affect deterministic decoding."""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_runtime_version = str(torch.version.cuda or "none")
    except ImportError:
        torch_version = DEFAULT_TORCH_VERSION
        cuda_runtime_version = "unavailable"
    try:
        import transformers

        transformers_version = str(transformers.__version__)
    except ImportError:
        transformers_version = DEFAULT_TRANSFORMERS_VERSION
    return {
        "torch_version": torch_version,
        "cuda_runtime_version": cuda_runtime_version,
        "transformers_version": transformers_version,
    }

SYSTEM_PROMPT = (
    "Ты аккуратный аналитик рекомендательных систем. Не выдумывай отсутствующие факты, "
    "не используй чувствительные характеристики и возвращай только запрошенный JSON."
)

RETRY_INSTRUCTION = (
    "\nПредыдущий ответ не прошёл автоматическую проверку. Начни сразу с одного "
    "полного JSON-объекта, пиши все значения только по-русски и строго соблюдай запреты."
)

INSIGHT_SCHEMAS = {
    "item": {
        "summary": str,
        "observed_facts": list,
        "seller_claims": list,
        "risks_to_verify": list,
        "suitable_for": list,
        "limitations": list,
        "confidence": str,
    },
    "user": {
        "summary": str,
        "observed_signals": list,
        "hypotheses": list,
        "limitations": list,
        "confidence": str,
    },
}


def insight_cache_signature(contract: Mapping[str, Any]) -> str:
    """Hash the complete immutable generation contract.

    Every contract field is covered automatically.  This deliberately includes
    request order, batch size, accelerator mapping and deterministic-runtime
    flags; adding a new provenance field therefore invalidates old cache rows.
    """
    unsigned = {key: value for key, value in contract.items() if key != "signature"}
    payload = {
        "signature_version": GENERATION_CONTRACT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "retry_instruction_sha256": hashlib.sha256(RETRY_INSTRUCTION.encode("utf-8")).hexdigest(),
        "contract": unsigned,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parsed_insight_sha256(parsed: Mapping[str, Any]) -> str:
    """Return a stable digest for the exact parsed object embedded in the UI."""
    payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_manifest_sha256(requests: list[Mapping[str, Any]]) -> str:
    """Bind a generation run to the exact ordered request/prompt manifest."""
    manifest = [
        [
            str(row.get("request_id", "")),
            int(row.get("prompt_version", 0)),
            str(row.get("prompt_sha256", "")),
        ]
        for row in requests
    ]
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    return text_sha256(encoded)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_tokenizer(
    model: str,
    revision: str,
    *,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    """Load the exact tokenizer snapshot and fingerprint its executable assets."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", revision) and snapshot.name != revision:
        raise ValueError(
            f"Tokenizer snapshot commit mismatch: expected {revision}, got {snapshot.name}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    artifacts = {
        name: _file_sha256(snapshot / name)
        for name in TOKENIZER_ARTIFACT_NAMES
        if (snapshot / name).is_file()
    }
    required = {"tokenizer.json", "tokenizer_config.json"}
    if not required <= set(artifacts):
        raise ValueError(
            f"Pinned tokenizer snapshot lacks required artifacts: {sorted(required - set(artifacts))}"
        )
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Pinned tokenizer has no materialized chat template")
    provenance = {
        "model": model,
        "revision": revision,
        "snapshot_commit": snapshot.name,
        "tokenizer_class": type(tokenizer).__name__,
        "artifacts_sha256": artifacts,
        "chat_template_sha256": text_sha256(chat_template),
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    return tokenizer, provenance


def validate_tokenizer_provenance_contract(
    value: Mapping[str, Any] | None,
    *,
    model: str,
    revision: str,
) -> str | None:
    """Return an error when tokenizer provenance is incomplete or unpinned."""
    if not isinstance(value, Mapping):
        return "generation contract tokenizer provenance is missing"
    if value.get("model") != model or value.get("revision") != revision:
        return "generation contract tokenizer model/revision mismatch"
    if value.get("snapshot_commit") != revision:
        return "generation contract tokenizer snapshot is not the pinned revision"
    for field in ("chat_template_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(field, ""))):
            return f"generation contract tokenizer {field} is invalid"
    artifacts = value.get("artifacts_sha256")
    if not isinstance(artifacts, Mapping):
        return "generation contract tokenizer artifact hashes are missing"
    for required in ("tokenizer.json", "tokenizer_config.json"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(artifacts.get(required, ""))):
            return f"generation contract tokenizer artifact hash is invalid: {required}"
    for field in ("vocab_size", "tokenizer_length", "pad_token_id", "eos_token_id"):
        if not isinstance(value.get(field), int) or int(value[field]) < 0:
            return f"generation contract tokenizer {field} is invalid"
    if not isinstance(value.get("tokenizer_class"), str) or not value["tokenizer_class"]:
        return "generation contract tokenizer class is invalid"
    return None


def _tensor_sha256(tensor: Any) -> str:
    """Hash exact dtype, shape and bytes of a padded tokenizer tensor."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def materialize_attempt_token_provenance(
    tokenizer: Any,
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    pass_name: str,
) -> tuple[Any, str, str, dict[str, Any]]:
    """Rebuild the canonical chat template and padded tensors for one attempt."""
    if pass_name not in {"primary", "retry"}:
        raise ValueError(f"unknown generation pass: {pass_name}")
    if contract.get("batch_size") != 1:
        raise ValueError("token provenance materialization requires canonical batch_size=1")
    effective_prompt = str(request.get("prompt", "")) + (
        RETRY_INSTRUCTION if pass_name == "retry" else ""
    )
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": effective_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    cpu_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(contract["max_input_tokens"]),
    )
    memberships = [str(request.get("request_id", ""))]
    input_shape = list(cpu_inputs["input_ids"].shape)
    mask_shape = list(cpu_inputs["attention_mask"].shape)
    input_ids_hash = _tensor_sha256(cpu_inputs["input_ids"])
    attention_mask_hash = _tensor_sha256(cpu_inputs["attention_mask"])
    batch_binding = {
        "ordered_request_ids": memberships,
        "input_ids_sha256": input_ids_hash,
        "attention_mask_sha256": attention_mask_hash,
        "input_ids_shape": input_shape,
        "attention_mask_shape": mask_shape,
    }
    provenance = {
        "ordered_request_ids": memberships,
        "batch_membership_sha256": text_sha256(
            json.dumps(memberships, separators=(",", ":"))
        ),
        "effective_prompt_sha256": text_sha256(effective_prompt),
        "chat_template_sha256": text_sha256(prompt_text),
        "input_ids_sha256": input_ids_hash,
        "attention_mask_sha256": attention_mask_hash,
        "input_ids_shape": input_shape,
        "attention_mask_shape": mask_shape,
        "padded_batch_sha256": text_sha256(
            json.dumps(batch_binding, sort_keys=True, separators=(",", ":"))
        ),
    }
    return cpu_inputs, prompt_text, effective_prompt, provenance


TokenProvenanceVerifier = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def make_token_provenance_verifier(
    tokenizer: Any,
    contract: Mapping[str, Any],
) -> TokenProvenanceVerifier:
    """Create an independent per-attempt recomputation bound to one tokenizer."""

    def verify(request: Mapping[str, Any], pass_name: str) -> Mapping[str, Any]:
        _, _, _, provenance = materialize_attempt_token_provenance(
            tokenizer, request, contract, pass_name
        )
        return provenance

    return verify


def runtime_generation_environment(device: str) -> dict[str, Any]:
    """Capture the actual logical/physical accelerator and determinism state."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    logical_index = 0
    if ":" in device:
        try:
            logical_index = int(device.split(":", 1)[1])
        except ValueError:
            logical_index = 0
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    physical_selector = (
        visible_tokens[logical_index]
        if visible != "<unset>" and logical_index < len(visible_tokens)
        else str(logical_index)
    )
    accelerator: dict[str, Any] = {
        "requested_device": device,
        "logical_device_index": logical_index,
        "cuda_visible_devices": visible,
        "physical_device_selector": physical_selector,
        "name": "unavailable",
        "compute_capability": "unavailable",
        "total_memory_bytes": None,
        "driver_version": "unavailable",
        "nvidia_smi_uuid": "unavailable",
    }
    determinism: dict[str, Any] = {
        "do_sample": False,
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
    }
    try:
        import torch

        determinism.update(
            {
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            }
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(logical_index)
            accelerator.update(
                {
                    "name": str(properties.name),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    except (ImportError, RuntimeError, AssertionError):
        pass
    if device.startswith("cuda"):
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            records = [part.strip() for part in completed.stdout.splitlines() if part.strip()]
            selected = next(
                (record for record in records if record.split(",", 1)[0].strip() == physical_selector),
                records[logical_index] if logical_index < len(records) else "",
            )
            fields = [field.strip() for field in selected.split(",")]
            if len(fields) >= 3:
                accelerator["nvidia_smi_uuid"] = fields[1]
                accelerator["driver_version"] = fields[2]
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    return {"accelerator": accelerator, "determinism": determinism}


def default_insight_contract(
    requests: list[Mapping[str, Any]] | None = None,
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    versions = runtime_generation_versions()
    request_rows = requests or []
    contract = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "model": DEFAULT_INSIGHT_MODEL,
        "model_revision": DEFAULT_INSIGHT_MODEL_REVISION,
        "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "retry_max_new_tokens": DEFAULT_RETRY_MAX_NEW_TOKENS,
        "load_in_4bit": DEFAULT_LOAD_IN_4BIT,
        "device": DEFAULT_INSIGHT_DEVICE,
        "device_type": DEFAULT_INSIGHT_DEVICE.split(":", 1)[0],
        "compute_dtype": DEFAULT_COMPUTE_DTYPE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "request_count": len(request_rows),
        "request_manifest_sha256": request_manifest_sha256(request_rows),
        "tokenizer": {
            "model": DEFAULT_INSIGHT_MODEL,
            "revision": DEFAULT_INSIGHT_MODEL_REVISION,
            "snapshot_commit": DEFAULT_INSIGHT_MODEL_REVISION,
            "tokenizer_class": "unmaterialized",
            "artifacts_sha256": {
                "tokenizer.json": "0" * 64,
                "tokenizer_config.json": "0" * 64,
            },
            "chat_template_sha256": "0" * 64,
            "vocab_size": 0,
            "tokenizer_length": 0,
            "pad_token_id": 0,
            "eos_token_id": 0,
        },
        "tokenization": {
            "use_fast": True,
            "padding": True,
            "padding_side": "left",
            "truncation": True,
            "truncation_side": "right",
            "add_generation_prompt": True,
        },
        "generation": {
            "do_sample": False,
            "use_cache": True,
            "skip_special_tokens": True,
        },
        **versions,
        "do_sample": False,
        **dict(environment or runtime_generation_environment(DEFAULT_INSIGHT_DEVICE)),
    }
    contract["signature"] = insight_cache_signature(contract)
    return contract


def validate_insight_parsed(
    kind: str,
    parsed: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate schema, language and safety without trusting extra keys."""
    if parsed is None:
        return None, "no balanced JSON object"
    if kind not in INSIGHT_SCHEMAS:
        return None, f"unknown insight kind: {kind!r}"
    schema = INSIGHT_SCHEMAS[kind]
    missing = [key for key in schema if key not in parsed]
    if missing:
        return None, f"missing keys: {missing}"
    clean: dict[str, Any] = {}
    for key, expected in schema.items():
        value = parsed[key]
        if expected is str:
            if not isinstance(value, str) or not value.strip():
                return None, f"{key} must be a non-empty string"
            clean[key] = " ".join(value.split())[:2000]
        else:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                return None, f"{key} must be a list of strings"
            clean[key] = [" ".join(item.split())[:1000] for item in value if item.strip()][:12]
    if clean["confidence"].lower() not in {"low", "medium", "high"}:
        return None, "confidence must be low, medium or high"
    clean["confidence"] = clean["confidence"].lower()
    natural_language = " ".join(
        value if isinstance(value, str) else " ".join(value)
        for key, value in clean.items()
        if key != "confidence"
    )
    if re.search(r"[\u3400-\u9fff]", natural_language):
        return None, "output contains CJK text; Russian values are required"
    if len(re.findall(r"[А-Яа-яЁё]", natural_language)) < 20:
        return None, "output does not contain enough Russian text"
    russian_required_lists = (
        ("observed_signals", "hypotheses", "limitations")
        if kind == "user"
        else ("risks_to_verify", "suitable_for", "limitations")
    )
    for key in russian_required_lists:
        if any(len(re.findall(r"[А-Яа-яЁё]", item)) < 3 for item in clean[key]):
            return None, f"{key} contains a non-Russian entry"
    if kind == "user":
        observations = " ".join([clean["summary"], *clean["observed_signals"]]).lower()
        if re.search(
            r"\b(приобр\w*|покуп\w*|просмотр\w*|клик\w*|заказ\w*|"
            r"активн\w*|регулярн\w*|ежедневн\w*|часто)\b",
            observations,
        ):
            return None, "contact observations contain an unsupported event/frequency claim"
        if "истори" not in clean["summary"].lower() or "контакт" not in clean["summary"].lower():
            return None, "user summary must explicitly describe a contact history"
        sensitive = " ".join(
            [clean["summary"], *clean["observed_signals"], *clean["hypotheses"]]
        ).lower()
        if re.search(
            r"\b(мужчин\w*|женщин\w*|пол|возраст\w*|доход\w*|професси\w*|"
            r"работ\w*|занят\w*|национальност\w*|здоровь\w*)\b",
            sensitive,
        ):
            return None, "user profile contains a prohibited personal/sensitive inference"
        if any(not re.search(r"\bможет\b", hypothesis.lower()) for hypothesis in clean["hypotheses"]):
            return None, "user hypotheses must use explicit «может» modality"
    else:
        if re.search(
            r"\b(хорош\w*|отличн\w*|идеальн\w*|без дтп|не бит\w*|"
            r"обслуж\w*|вложени\w*|собственник\w*|гаранти\w*)\b",
            clean["summary"].lower(),
        ):
            return None, "item summary promotes a seller claim to an observed fact"
        if any(
            not claim.lower().startswith(("продавец заявляет:", "продавец утверждает:"))
            for claim in clean["seller_claims"]
        ):
            return None, "seller_claims entries must remain explicitly attributed"
    return clean, None


def normalised_evidence_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Build every factual/profile field deterministically from request evidence."""
    evidence = request.get("evidence") if isinstance(request.get("evidence"), dict) else {}
    if request["kind"] == "item":
        attributes = evidence.get("attributes") if isinstance(evidence.get("attributes"), dict) else {}
        facts: list[str] = []
        brand_model = " ".join(
            str(value) for value in (attributes.get("brand"), attributes.get("model")) if value
        )
        if brand_model:
            facts.append(f"Марка и модель в экспорте: {brand_model}")
        if attributes.get("year"):
            facts.append(f"Год в экспорте: {attributes['year']}")
        if attributes.get("mileage_km") is not None:
            facts.append(f"Пробег в экспорте: {float(attributes['mileage_km']):.0f} км")
        if attributes.get("price") is not None:
            facts.append(f"Цена в экспорте: {float(attributes['price']):.0f} рублей")
        if attributes.get("body"):
            facts.append(f"Тип кузова в экспорте: {attributes['body']}")
        details = "; ".join(facts[:3]).lower()
        risks = ["Проверить фактическое состояние объявления при осмотре и по документам"]
        if facts:
            risks.append("Сверить структурированные характеристики экспорта с первичными документами")
        limitations = [
            "Экспорт не содержит независимого подтверждения технического состояния",
            "Текст продавца рассматривается только как набор атрибутированных утверждений",
        ]
        return {
            "summary": (
                f"Структурированные данные объявления {request['entity_id']}: {details}."
                if details
                else (
                    f"Для объявления {request['entity_id']} структурированные "
                    "характеристики не заполнены."
                )
            ),
            "observed_facts": facts[:4],
            "risks_to_verify": risks,
            "suitable_for": [],
            "limitations": limitations,
            "confidence": "low",
        }

    rows = evidence.get("history") if isinstance(evidence.get("history"), list) else []
    summary = evidence.get("history_summary") if isinstance(evidence.get("history_summary"), dict) else {}
    category_counts = Counter(
        str(row.get("raw_category_level_1"))
        for row in rows
        if isinstance(row, dict) and row.get("raw_category_level_1")
    )
    observations = [f"Количество контактов в экспорте: {len(rows)}"]
    if summary.get("date_from") and summary.get("date_to"):
        observations.append(f"Период дат контактов: {summary['date_from']} — {summary['date_to']}")
    observations.extend(
        f"Raw category «{category}»: {count}"
        for category, count in category_counts.most_common(3)
    )
    hypotheses = [
        f"Может сохраняться товарный интерес к raw category «{category}»."
        for category, _ in category_counts.most_common(3)
    ]
    return {
        "summary": f"История контактов: число экспортированных записей — {len(rows)}.",
        "observed_signals": observations,
        "hypotheses": hypotheses,
        "limitations": [
            "Временная связь истории контактов с текущей выдачей неизвестна",
            "Контакт не подтверждает транзакцию или намерение пользователя",
            "Raw category не описывает личные свойства пользователя",
        ],
        "confidence": "low",
    }


def _normalise_exact_span(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = " ".join(text.split())
    return text.strip(" \t\r\n«»\"'.,!?;:–—-")


def validate_and_normalise_insight(
    request: Mapping[str, Any],
    parsed: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Validate an LLM response and replace factual fields from evidence.

    Seller claims must contain a verbatim span from ``seller_description``.
    Every other displayed field is replaced from an evidence-bounded template,
    so free-form personal or seller-derived assertions never reach the browser.
    """
    kind = str(request.get("kind", ""))
    clean, error = validate_insight_parsed(kind, parsed)
    if clean is None:
        return None, error, []
    if kind == "item":
        evidence = request.get("evidence") if isinstance(request.get("evidence"), dict) else {}
        description = _normalise_exact_span(str(evidence.get("description") or ""))
        for claim in clean["seller_claims"]:
            body = claim.split(":", 1)[1] if ":" in claim else ""
            span = _normalise_exact_span(body)
            if len(span) < 3 or not description or span not in description:
                return None, f"seller_claim is not a verbatim description span: {body!r}", []
    factual = normalised_evidence_fields(request)
    clean.update(factual)
    clean, error = validate_insight_parsed(kind, clean)
    if clean is None:
        return None, f"normalised evidence fields are invalid: {error}", []
    return clean, None, sorted(factual)


def validate_cached_insight(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    validation_version: int,
    token_provenance_verifier: TokenProvenanceVerifier | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate cache identity, provenance, flags and final parsed content."""
    try:
        expected_signature = insight_cache_signature(contract)
    except (TypeError, ValueError) as exc:
        return None, f"invalid generation contract: {exc}"
    if contract.get("signature") != expected_signature:
        return None, "generation contract signature mismatch"
    if contract.get("contract_version") != GENERATION_CONTRACT_VERSION:
        return None, "generation contract version mismatch"
    if contract.get("batch_size") != DEFAULT_BATCH_SIZE:
        return None, "only canonical batch_size=1 insight caches are accepted"
    if not isinstance(contract.get("request_count"), int) or contract["request_count"] < 1:
        return None, "generation contract request_count is invalid"
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("request_manifest_sha256", ""))):
        return None, "generation contract request manifest hash is invalid"
    tokenizer_error = validate_tokenizer_provenance_contract(
        contract.get("tokenizer"),
        model=str(contract.get("model", "")),
        revision=str(contract.get("model_revision", "")),
    )
    if tokenizer_error:
        return None, tokenizer_error
    if token_provenance_verifier is None:
        return None, "independent pinned-tokenizer provenance verifier is required"
    accelerator = contract.get("accelerator")
    determinism = contract.get("determinism")
    required_accelerator = {
        "requested_device",
        "logical_device_index",
        "cuda_visible_devices",
        "physical_device_selector",
        "name",
        "compute_capability",
        "total_memory_bytes",
        "driver_version",
        "nvidia_smi_uuid",
    }
    required_determinism = {
        "do_sample",
        "torch_deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
        "python_hash_seed",
    }
    if not isinstance(accelerator, dict) or not required_accelerator <= set(accelerator):
        return None, "generation contract accelerator provenance is incomplete"
    if not isinstance(determinism, dict) or not required_determinism <= set(determinism):
        return None, "generation contract determinism provenance is incomplete"
    if determinism.get("do_sample") is not False or contract.get("do_sample") is not False:
        return None, "generation contract is not deterministic decoding"
    expected_contract = dict(contract)
    if result.get("generation_contract") != expected_contract:
        return None, "generation_contract mismatch"
    expected_identity = {
        "request_id": request.get("request_id"),
        "kind": request.get("kind"),
        "entity_id": request.get("entity_id"),
        "prompt_version": request.get("prompt_version"),
        "prompt_sha256": request.get("prompt_sha256"),
        "signature": contract.get("signature"),
        "model": contract.get("model"),
        "model_revision": contract.get("model_revision"),
        "device": contract.get("device"),
        "device_type": contract.get("device_type"),
        "compute_dtype": contract.get("compute_dtype"),
        "torch_version": contract.get("torch_version"),
        "cuda_runtime_version": contract.get("cuda_runtime_version"),
        "transformers_version": contract.get("transformers_version"),
        "do_sample": False,
        "validation_version": validation_version,
        "status": "ok",
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            return None, f"{field} mismatch"
    fallback = result.get("fallback_applied")
    accepted = result.get("llm_output_accepted")
    if not isinstance(fallback, bool) or not isinstance(accepted, bool) or fallback == accepted:
        return None, "fallback/acceptance flags are inconsistent"
    generation_pass = result.get("generation_pass")
    if not isinstance(generation_pass, str):
        return None, "generation_pass is missing"
    if fallback:
        suffix = "+guarded_fallback"
        if not generation_pass.endswith(suffix):
            return None, "fallback generation_pass mismatch"
        base_pass = generation_pass[: -len(suffix)]
        if result.get("original_status") != "parse_error":
            return None, "fallback original_status mismatch"
    else:
        base_pass = generation_pass
    if base_pass not in {"primary", "retry"}:
        return None, "generation_pass is not primary or retry"
    attempts = result.get("generation_attempts")
    expected_attempt_names = ["primary"] + (["retry"] if base_pass == "retry" else [])
    if not isinstance(attempts, list) or len(attempts) != len(expected_attempt_names):
        return None, "generation_attempts do not preserve the complete primary/retry audit"

    # Bind every recorded attempt to the exact request, effective prompt,
    # padded token tensors and raw model text.  The import is local to avoid a
    # module cycle at import time.
    from .core import extract_json_object

    prompt = str(request.get("prompt", ""))
    attempt_normalised: list[dict[str, Any] | None] = []
    for index, (attempt, expected_name) in enumerate(zip(attempts, expected_attempt_names)):
        if not isinstance(attempt, dict) or attempt.get("pass") != expected_name:
            return None, f"generation attempt {index} pass mismatch"
        memberships = attempt.get("ordered_request_ids")
        if memberships != [request.get("request_id")]:
            return None, f"generation attempt {index} batch membership mismatch"
        membership_json = json.dumps(memberships, separators=(",", ":"))
        if attempt.get("batch_membership_sha256") != text_sha256(membership_json):
            return None, f"generation attempt {index} membership hash mismatch"
        if attempt.get("configured_batch_size") != contract.get("batch_size"):
            return None, f"generation attempt {index} configured batch size mismatch"
        if attempt.get("actual_batch_size") != 1:
            return None, f"generation attempt {index} actual batch size mismatch"
        effective_prompt = prompt + (RETRY_INSTRUCTION if expected_name == "retry" else "")
        if attempt.get("effective_prompt_sha256") != text_sha256(effective_prompt):
            return None, f"generation attempt {index} effective prompt hash mismatch"
        expected_budget = (
            contract.get("retry_max_new_tokens")
            if expected_name == "retry"
            else contract.get("max_new_tokens")
        )
        if attempt.get("max_new_tokens") != expected_budget:
            return None, f"generation attempt {index} token budget mismatch"
        hash_fields = (
            "chat_template_sha256",
            "input_ids_sha256",
            "attention_mask_sha256",
            "padded_batch_sha256",
        )
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", str(attempt.get(field, "")))
            for field in hash_fields
        ):
            return None, f"generation attempt {index} token provenance hash is invalid"
        input_shape = attempt.get("input_ids_shape")
        mask_shape = attempt.get("attention_mask_shape")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 2
            or input_shape[0] != 1
            or input_shape != mask_shape
            or not all(isinstance(value, int) and value > 0 for value in input_shape)
        ):
            return None, f"generation attempt {index} padded tensor shape mismatch"
        batch_binding = {
            "ordered_request_ids": memberships,
            "input_ids_sha256": attempt["input_ids_sha256"],
            "attention_mask_sha256": attempt["attention_mask_sha256"],
            "input_ids_shape": input_shape,
            "attention_mask_shape": mask_shape,
        }
        expected_batch_hash = text_sha256(
            json.dumps(batch_binding, sort_keys=True, separators=(",", ":"))
        )
        if attempt.get("padded_batch_sha256") != expected_batch_hash:
            return None, f"generation attempt {index} padded batch binding mismatch"
        try:
            recomputed = token_provenance_verifier(request, expected_name)
        except Exception as exc:  # noqa: BLE001 - cache must fail closed on verifier errors
            return None, f"generation attempt {index} tokenizer recomputation failed: {exc}"
        provenance_fields = (
            "ordered_request_ids",
            "batch_membership_sha256",
            "effective_prompt_sha256",
            "chat_template_sha256",
            "input_ids_sha256",
            "attention_mask_sha256",
            "input_ids_shape",
            "attention_mask_shape",
            "padded_batch_sha256",
        )
        for field in provenance_fields:
            if attempt.get(field) != recomputed.get(field):
                return None, (
                    f"generation attempt {index} {field} does not match pinned-tokenizer "
                    "recomputation"
                )
        raw_text = attempt.get("raw_text")
        if not isinstance(raw_text, str) or attempt.get("raw_text_sha256") != text_sha256(raw_text):
            return None, f"generation attempt {index} raw text hash mismatch"
        raw_object = extract_json_object(raw_text)
        llm_parsed, _ = validate_insight_parsed(str(request.get("kind", "")), raw_object)
        expected_llm_hash = (
            parsed_insight_sha256(llm_parsed) if llm_parsed is not None else None
        )
        if attempt.get("llm_parsed_sha256") != expected_llm_hash:
            return None, f"generation attempt {index} parsed JSON hash mismatch"
        raw_normalised, raw_error, _ = validate_and_normalise_insight(request, raw_object)
        expected_status = "ok" if raw_normalised is not None else "parse_error"
        if attempt.get("status") != expected_status or attempt.get("validation_error") != raw_error:
            return None, f"generation attempt {index} validation audit mismatch"
        attempt_normalised.append(raw_normalised)
    if len(attempts) == 2 and attempt_normalised[0] is not None:
        return None, "retry was run despite an acceptable primary output"

    selected_attempt = attempts[-1]
    selected_top_level = {
        "effective_prompt_sha256": selected_attempt["effective_prompt_sha256"],
        "max_new_tokens_used": selected_attempt["max_new_tokens"],
        "raw_text": selected_attempt["raw_text"],
        "raw_text_sha256": selected_attempt["raw_text_sha256"],
        "llm_parsed_sha256": selected_attempt["llm_parsed_sha256"],
    }
    for field, expected in selected_top_level.items():
        if result.get(field) != expected:
            return None, f"selected attempt {field} mismatch"
    raw_normalised = attempt_normalised[-1]
    raw_error = selected_attempt.get("validation_error")
    if fallback:
        if raw_normalised is not None:
            return None, "guarded fallback wraps an acceptable raw LLM output"
    elif raw_normalised is None:
        return None, f"accepted raw LLM output no longer validates: {raw_error}"
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else None
    normalised, error, fields = validate_and_normalise_insight(request, parsed)
    if normalised is None:
        return None, error
    if normalised != parsed:
        return None, "cached factual/profile fields are not evidence-normalised"
    if not fallback and raw_normalised != normalised:
        return None, "parsed insight is not derived from raw LLM output"
    if result.get("parsed_sha256") != parsed_insight_sha256(normalised):
        return None, "parsed_sha256 mismatch"
    if result.get("evidence_normalized_fields") != fields:
        return None, "evidence_normalized_fields mismatch"
    safe = dict(result)
    safe["parsed"] = normalised
    safe["evidence_normalized_fields"] = fields
    return safe, None
