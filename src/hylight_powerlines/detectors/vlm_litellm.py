"""Vision-language model pre-analysis via LiteLLM."""

import json
import logging
from copy import deepcopy
from typing import Any, cast

import litellm
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hylight_powerlines.vision.image import img_to_b64_jpeg
from hylight_powerlines.vision.types import Box, PreAnalysis, Proposal


class _VlmBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> _VlmBox:
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("Box must satisfy x2>x1 and y2>y1.")
        return self


class _VlmProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    box: _VlmBox
    confidence: float = Field(ge=0.0, le=1.0)
    prompt_variants: list[str] = Field(min_length=1, max_length=8)
    notes: str

    @field_validator("label", mode="before")
    @classmethod
    def _strip_label(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("prompt_variants", mode="before")
    @classmethod
    def _coerce_prompt_variants(cls, v: Any) -> list[str]:
        if v is None:
            return [""]
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(s) for s in v]
        return [""]

    @model_validator(mode="after")
    def _normalize_prompt_variants(self) -> _VlmProposal:
        cleaned = [s.strip() for s in self.prompt_variants if str(s).strip()]
        deduped = sorted(set(cleaned))[:8]
        if not deduped:
            deduped = [self.label.strip()]
        self.prompt_variants = deduped
        self.label = self.label.strip()
        return self


class _VlmOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[_VlmProposal]

    @field_validator("proposals", mode="before")
    @classmethod
    def _coerce_proposals(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # Some models may return a single object.
        if isinstance(v, dict):
            return [v]
        return []


def _inline_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic `$defs`/`$ref` into a single schema dict.

    OpenAI's strict `json_schema` format can be sensitive to certain schema
    constructs. Inlining keeps the schema small and avoids reliance on `$ref`.
    """
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema
    defs_dict: dict[str, Any] = cast(dict[str, Any], defs)

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.removeprefix("#/$defs/")
                target = defs_dict.get(key)
                if target is None:
                    raise ValueError(f"Unresolvable $ref: {ref}")
                return _resolve(deepcopy(target))
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    return _resolve({k: v for k, v in schema.items() if k != "$defs"})


_VLM_OUT_JSON_SCHEMA: dict[str, Any] = _inline_json_schema_refs(_VlmOut.model_json_schema())
_VLM_TEXT_FORMAT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": "vlm_preanalysis",
        "strict": True,
        "schema": _VLM_OUT_JSON_SCHEMA,
    }
}


def _extract_json(text: str) -> str:
    """Extract a JSON object from a possibly noisy model response."""
    if not text:
        return text
    # Common case: fenced JSON block
    if "```" in text:
        parts = text.split("```")
        for i in range(len(parts) - 1):
            fence_lang = parts[i].strip().splitlines()[-1].strip().lower()
            body = parts[i + 1]
            if fence_lang in {"json", "application/json", ""}:
                body = body.strip()
                # Stop at next fence, if present
                body = body.split("```", 1)[0].strip()
                if body.startswith("{") and body.endswith("}"):
                    return body
    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return text
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        cand = text[i : j + 1]
        try:
            json.loads(cand)
        except json.JSONDecodeError:
            return text
        else:
            return cand
    return text


def _content_to_text(content: Any) -> str:
    """Best-effort normalization of provider responses to a single text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI-style content blocks: [{"type":"text","text":"..."} , ...]
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(chunks).strip()
    # Some providers return {"text": "..."} or similar
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t
    return str(content)


def _response_to_dict(resp: Any) -> dict[str, Any]:
    """Normalize a LiteLLM completion response object to a plain dict."""
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, BaseModel):
        return resp.model_dump()
    raise TypeError(f"Unsupported completion response type: {type(resp)!r}")


def _extract_choice_text(resp: dict[str, Any]) -> tuple[str, str]:
    """Extract assistant content and finish_reason from a Chat Completions-style response."""
    choices = resp.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return "", ""
    c0 = choices[0] or {}
    if not isinstance(c0, dict):
        return "", ""
    finish_reason = str(c0.get("finish_reason") or "")

    msg = c0.get("message") or {}
    if isinstance(msg, dict) and "content" in msg:
        return _content_to_text(msg.get("content")), finish_reason

    # Some providers put content at the choice level.
    if "text" in c0:
        return _content_to_text(c0.get("text")), finish_reason

    return "", finish_reason


def _extract_responses_text(resp: dict[str, Any]) -> tuple[str, str]:
    """Extract assistant text and status from an OpenAI-style Responses API payload."""
    status = str(resp.get("status") or "")
    output_text = resp.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip(), status

    out_items = resp.get("output")
    if isinstance(out_items, list):
        for item in out_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            text = _content_to_text(item.get("content")).strip()
            if text:
                return text, status

    return "", status


def _rel_box_to_pixels(box: _VlmBox, w: int, h: int) -> tuple[float, float, float, float]:
    """Convert normalized [0,1] box coordinates to pixel coordinates."""
    return box.x1 * w, box.y1 * h, box.x2 * w, box.y2 * h


def pre_analyze(
    img: Image.Image,
    targets: list[str],
    *,
    model: str,
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout_s: float = 60.0,
    verbose: bool = False,
) -> PreAnalysis:
    """Ask a VLM for coarse proposals + prompt variants via LiteLLM.

    Returns a :class:`PreAnalysis` with lightly validated proposals.
    """
    w, h = img.size
    b64 = img_to_b64_jpeg(img)
    schema_hint = {
        "proposals": [
            {
                "label": "insulator",
                "box": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0},
                "confidence": 0.0,
                "prompt_variants": ["power line insulator", "pin insulator"],
                "notes": "short free text",
            }
        ],
    }

    prompt = f"""
You are labeling images of overhead power-line infrastructure for AUTHORIZED maintenance inspection.
Do NOT identify people, faces, vehicles, license plates, or locations. 
Ignore any such elements if present.
Your objective is to detect in the image elements of power lines such as:
- tower/pylon
- damper
- spacer
- insulator
- tower or pylon identification plate
- ...

Task:
- Identify only instances of the following target categories: {targets}
- Return bounding boxes as RELATIVE coordinates (normalized between 0.0 and 1.0).
- Also propose prompt variants (synonyms) for use by an open-vocabulary detector.

Coordinate system:
- origin is top-left (0,0)
- box is (x1,y1,x2,y2) with x2>x1 and y2>y1
- x1,x2 are in [0,1] relative to width; y1,y2 are in [0,1] relative to height
- Convert to pixels with: x_px = x_rel * width, y_px = y_rel * height

Output strictly as JSON, no extra text, matching this shape:
{json.dumps(schema_hint, indent=2)}

Rules:
- If an object is not present, do not hallucinate it.
- Use confidence in [0,1].
- Include multiple prompt_variants per label (3-8 variants).
    """.strip()

    logger = logging.getLogger(__name__)
    logger.info("Requesting VLM pre-analysis via LiteLLM: model=%s", model)

    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            ],
        }
    ]
    response_kwargs: dict[str, Any] = {
        "model": model,
        "input": input_messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": _VLM_TEXT_FORMAT,
        "tools": [
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "memory_limit": "4g"},
            }
        ],
        "timeout": timeout_s*1.1
    }
    if timeout_s is not None:
        response_kwargs["timeout"] = timeout_s
    raw_resp = litellm.responses(**response_kwargs)  # type: ignore
    resp = _response_to_dict(raw_resp)
    text, status = _extract_responses_text(resp)
    finish_reason = ""
    if not text:
        # Some providers may still return Chat Completions style payloads even
        # when routed through `responses()`.
        text, finish_reason = _extract_choice_text(resp)
    usage = resp.get("usage")
    reasoning_tokens: int | None = None
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
            reasoning_tokens = details["reasoning_tokens"]
    logger.info(
        "VLM response received: status=%s finish_reason=%s usage=%s reasoning_tokens=%s",
        status,
        finish_reason,
        usage,
        reasoning_tokens,
    )
    if verbose:
        logger.info("VLM response content:\n%s", text)
    if not text:
        raise RuntimeError(
            "VLM returned empty content. "
            f"status={status!r}, finish_reason={finish_reason!r}, max_tokens={max_tokens}, "
            f"usage={resp.get('usage')!r}"
        )

    try:
        parsed = _VlmOut.model_validate_json(_extract_json(text))
    except ValidationError as e:  # pragma: no cover
        raise RuntimeError("VLM output does not match expected JSON schema.") from e

    proposals: list[Proposal] = []

    for p in parsed.proposals:
        label = p.label
        if label not in targets:
            continue
        x1, y1, x2, y2 = _rel_box_to_pixels(p.box, w=w, h=h)

        b = Box(
            label=label,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=float(p.confidence),
            source="vlm",
            prompt=None,
        ).clip(w, h)

        proposals.append(
            Proposal(
                label=label,
                box=b,
                prompt_variants=p.prompt_variants,
                notes=p.notes,
            )
        )

    return PreAnalysis(image_w=w, image_h=h, proposals=proposals)
