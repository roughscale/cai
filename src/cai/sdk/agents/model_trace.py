from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def is_model_trace_enabled() -> bool:
    value = os.getenv("CAI_MODEL_TRACE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def get_model_trace_path() -> Path:
    configured = os.getenv("CAI_MODEL_TRACE_FILE")
    if configured:
        return Path(configured)

    return Path("logs/model_trace.jsonl")


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    obj_id = id(value)
    if obj_id in seen:
        return "<recursive>"

    if isinstance(value, dict):
        seen.add(obj_id)
        return {str(k): _json_safe(v, seen) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        seen.add(obj_id)
        return [_json_safe(v, seen) for v in value]

    if hasattr(value, "model_dump"):
        try:
            seen.add(obj_id)
            return _json_safe(value.model_dump(exclude_unset=True), seen)
        except TypeError:
            seen.add(obj_id)
            return _json_safe(value.model_dump(), seen)

    if hasattr(value, "__dict__"):
        seen.add(obj_id)
        fields = {
            key: val
            for key, val in vars(value).items()
            if key not in {"agent", "source_agent", "target_agent", "function_tool", "computer_tool"}
        }
        return {
            "_type": value.__class__.__name__,
            **_json_safe(fields, seen),
        }

    return repr(value)


def write_model_trace(event_type: str, **payload: Any) -> None:
    if not is_model_trace_enabled():
        return

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **{key: _json_safe(value) for key, value in payload.items()},
    }

    path = get_model_trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True))
        handle.write("\n")
