"""
Non-interactive CAI benchmark runner.

Runs inside the cai-bench Docker container. Reads challenge parameters from
environment variables, executes the agent via Runner.run(), scans all output
for the flag, and writes a single JSON record to stdout on exit.

Exit codes:
  0 - flag found (solved)
  1 - ran to completion, flag not found
  2 - error / crash

Environment variables consumed:
  CHALLENGE_PROMPT       Initial prompt sent to the agent
  CHALLENGE_FLAG         Known flag for exact match (optional)
  CHALLENGE_FLAG_PATTERN Regex pattern for flag detection (optional)
  TARGET_HOST            Target hostname or IP (optional hint included in prompt)
  CAI_MODEL              Model name  (e.g. gpt-5, claude-3-7-sonnet-2025-02-19)
  CAI_AGENT_TYPE         Agent factory key (e.g. one_tool_agent, redteam_agent)
  CAI_MAX_TURNS          Maximum LLM interactions (default: 100)
  CAI_STREAM             Should be "false" for non-interactive runs
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

# Ensure CAI_STREAM is off for non-interactive runs
os.environ.setdefault("CAI_STREAM", "false")


DEFAULT_FLAG_PATTERNS = [
    r"FLAG\{[^}]+\}",
    r"HTB\{[^}]+\}",
    r"flag\{[^}]+\}",
    r"picoCTF\{[^}]+\}",
    r"ctf\{[^}]+\}",
    r"[0-9a-f]{32}",
]


def build_flag_regex() -> re.Pattern:
    custom = os.environ.get("CHALLENGE_FLAG_PATTERN")
    patterns = [custom] if custom else DEFAULT_FLAG_PATTERNS
    return re.compile("|".join(f"(?:{p})" for p in patterns))


def scan_for_flag(text: str, flag_regex: re.Pattern, known_flag: str | None) -> str | None:
    """Return the first flag found in text, or None."""
    if known_flag and known_flag in text:
        return known_flag
    match = flag_regex.search(text)
    return match.group(0) if match else None


def collect_all_text(result) -> str:
    """
    Gather all text from the RunResult: final output plus every tool
    output item, so flags captured inside tool calls are not missed.
    """
    parts: list[str] = []

    if result.final_output:
        parts.append(str(result.final_output))

    for item in result.new_items:
        # ToolCallOutputItem has a .output attribute
        output = getattr(item, "output", None)
        if output:
            parts.append(str(output))
        # MessageOutputItem has a .raw_item with content
        raw = getattr(item, "raw_item", None)
        if raw:
            content = getattr(raw, "content", None) or getattr(raw, "output", None)
            if content:
                if isinstance(content, list):
                    for c in content:
                        text = getattr(c, "text", None)
                        if text:
                            parts.append(str(text))
                elif isinstance(content, str):
                    parts.append(content)

    return "\n".join(parts)


async def main() -> dict:
    from cai.sdk.agents import Runner
    from cai.agents.factory import get_agent_factory

    prompt = os.environ["CHALLENGE_PROMPT"]
    known_flag = os.environ.get("CHALLENGE_FLAG") or None
    max_turns = int(os.environ.get("CAI_MAX_TURNS", "100"))
    agent_type = os.environ.get("CAI_AGENT_TYPE", "one_tool_agent")
    flag_regex = build_flag_regex()

    factory = get_agent_factory(agent_type)
    agent = factory()

    start = time.monotonic()
    try:
        result = await Runner.run(
            starting_agent=agent,
            input=prompt,
            max_turns=max_turns,
        )
    except Exception as exc:
        # MaxTurnsExceeded and other SDK exceptions — still collect what we have
        result = getattr(exc, "run_result", None)
        if result is None:
            raise

    elapsed = time.monotonic() - start
    all_text = collect_all_text(result)
    found_flag = scan_for_flag(all_text, flag_regex, known_flag)

    model = agent.model
    return {
        "solved": found_flag is not None,
        "flag": found_flag,
        "time_s": round(elapsed, 3),
        "turns": getattr(model, "interaction_counter", None),
        "input_tokens": getattr(model, "total_input_tokens", None),
        "output_tokens": getattr(model, "total_output_tokens", None),
        "cost_usd": getattr(model, "total_cost", None),
    }


if __name__ == "__main__":
    try:
        record = asyncio.run(main())
        print(json.dumps(record))
        sys.exit(0 if record["solved"] else 1)
    except Exception as exc:
        print(json.dumps({"solved": False, "error": str(exc)}))
        sys.exit(2)
