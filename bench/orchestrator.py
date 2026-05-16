"""
CAI Benchmark Orchestrator.

Discovers challenges from two sources:
  1. Local XBEN challenges  — targets/validation-benchmarks/benchmarks/*/benchmark.json
  2. External challenges    — bench/challenges/*.json  (excluding *_example.json)

For local challenges the orchestrator provisions the target Docker environment
and tears it down after each run. For external challenges it just runs the CAI
container — target connectivity is the runner's concern.

Usage:
    python bench/orchestrator.py [options]

    --benchmarks PATH     Root of XBEN benchmarks (default: targets/validation-benchmarks/benchmarks)
    --challenges PATH     Directory of external challenge JSON files (default: bench/challenges)
    --model MODEL         CAI model to use (default: gpt-5)
    --agent AGENT         Agent type key (default: redteam_agent)
    --max-turns N         Max LLM interactions per challenge (default: 100)
    --output PATH         JSONL results file (default: results/benchmark.jsonl)
    --filter IDS          Comma-separated challenge IDs to run (default: all)
    --local-only          Only run local XBEN challenges
    --external-only       Only run external challenges
    --build               Rebuild the cai-bench Docker image before running
    --timeout N           Per-challenge wall-clock timeout in seconds (default: 1800)

API keys are read from the host environment and forwarded to the runner container.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BENCH_DIR = REPO_ROOT / "bench"
IMAGE_NAME = "cai-bench:latest"

API_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Challenge loading
# ---------------------------------------------------------------------------

def load_local_challenges(benchmarks_root: Path) -> list[dict]:
    """Discover XBEN challenges from the validation-benchmarks tree."""
    challenges = []
    for bench_dir in sorted(benchmarks_root.iterdir()):
        meta_path = bench_dir / "benchmark.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        flag = _derive_xben_flag(bench_dir.name)
        challenges.append({
            "id": bench_dir.name,
            "name": meta.get("name", bench_dir.name),
            "target_type": "local",
            "prompt": _local_prompt(meta),
            "flag_pattern": None,
            "known_flag": flag,
            "platform": "xben",
            "difficulty": str(meta.get("level", "unknown")),
            "category": (meta.get("tags") or ["misc"])[0],
            "compose_file": bench_dir / "docker-compose.yml",
            "flag_build_arg": flag,
            "_bench_dir": bench_dir,
        })
    return challenges


def load_external_challenges(challenges_dir: Path) -> list[dict]:
    """Load external challenge definitions from bench/challenges/*.json,
    skipping example files."""
    challenges = []
    for path in sorted(challenges_dir.glob("*.json")):
        if path.stem.endswith("_example"):
            continue
        data = json.loads(path.read_text())
        data.setdefault("target_type", "external")
        challenges.append(data)
    return challenges


def _derive_xben_flag(bench_name: str) -> str:
    result = subprocess.run(
        ["openssl", "sha256", "-r"],
        input=bench_name.upper().encode(),
        capture_output=True,
        check=True,
    )
    sha = result.stdout.decode().split()[0]
    return f"FLAG{{{sha}}}"


def _local_prompt(meta: dict) -> str:
    desc = meta.get("description", "Solve the challenge and find the flag.")
    return f"{desc}\n\nFind the flag and report it exactly when you have it."


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def build_image(no_cache: bool = False) -> None:
    print("Building cai-bench image from source...")
    cmd = ["docker", "build", "-f", "bench/Dockerfile", "-t", IMAGE_NAME, "."]
    if no_cache:
        cmd.append("--no-cache")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _docker_compose(args: list[str], bench_dir: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose"] + args,
        cwd=bench_dir,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
    )


def _container_ids_for_project(project: str, bench_dir: Path) -> list[str]:
    result = subprocess.run(
        ["docker", "compose", "-p", project, "ps", "-q"],
        cwd=bench_dir,
        capture_output=True,
        text=True,
    )
    return [c for c in result.stdout.strip().splitlines() if c]


# ---------------------------------------------------------------------------
# Per-challenge runner
# ---------------------------------------------------------------------------

def run_challenge(challenge: dict, model: str, agent: str, max_turns: int,
                  timeout_s: int, api_keys: dict) -> dict:
    target_type = challenge.get("target_type", "local")
    record = {
        "id": challenge["id"],
        "name": challenge["name"],
        "target_type": target_type,
        "model": model,
        "agent": agent,
        "platform": challenge.get("platform", "custom"),
        "difficulty": challenge.get("difficulty", "unknown"),
        "category": challenge.get("category", "misc"),
        "solved": False,
        "flag": None,
        "time_s": None,
        "turns": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "error": None,
    }

    if target_type == "local":
        return _run_local_challenge(challenge, record, model, agent, max_turns, timeout_s, api_keys)
    else:
        return _run_external_challenge(challenge, record, model, agent, max_turns, timeout_s, api_keys)


def _build_runner_env(challenge: dict, model: str, agent: str, max_turns: int,
                      api_keys: dict) -> dict[str, str]:
    env = {
        "CHALLENGE_PROMPT": challenge["prompt"],
        "CAI_MODEL": model,
        "CAI_AGENT_TYPE": agent,
        "CAI_MAX_TURNS": str(max_turns),
        "CAI_STREAM": "false",
        **api_keys,
    }
    if challenge.get("known_flag"):
        env["CHALLENGE_FLAG"] = challenge["known_flag"]
    if challenge.get("flag_pattern"):
        env["CHALLENGE_FLAG_PATTERN"] = challenge["flag_pattern"]
    if challenge.get("target_host"):
        env["TARGET_HOST"] = challenge["target_host"]
    return env


def _docker_run(runner_env: dict, network: str | None, timeout_s: int) -> tuple[dict, str | None]:
    """Run the cai-bench container. Streams stderr; returns (parsed_result, error_string)."""
    import threading
    cmd = ["docker", "run", "--rm"]
    if network:
        cmd += [f"--network={network}"]
    for k, v in runner_env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(IMAGE_NAME)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def stream_stderr():
            for line in proc.stderr:
                print(f"    {line}", end="", flush=True)

        stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
        stderr_thread.start()

        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stderr_thread.join()
            return {}, f"wall-clock timeout after {timeout_s}s"

        stderr_thread.join()
        stdout = proc.stdout.read()
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if "solved" in parsed:
                    return parsed, None
            except json.JSONDecodeError:
                continue
        return {}, stdout or "no output from runner"
    except Exception as exc:
        return {}, str(exc)


def _run_external_challenge(challenge: dict, record: dict, model: str, agent: str,
                             max_turns: int, timeout_s: int, api_keys: dict) -> dict:
    runner_env = _build_runner_env(challenge, model, agent, max_turns, api_keys)
    result, error = _docker_run(runner_env, network=None, timeout_s=timeout_s)
    record.update({
        "solved": result.get("solved", False),
        "flag": result.get("flag"),
        "time_s": result.get("time_s"),
        "turns": result.get("turns"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "cost_usd": result.get("cost_usd"),
        "error": error or result.get("error"),
    })
    return record


def _run_local_challenge(challenge: dict, record: dict, model: str, agent: str,
                          max_turns: int, timeout_s: int, api_keys: dict) -> dict:
    bench_dir: Path = challenge["_bench_dir"]
    compose_file: Path = challenge["compose_file"]
    flag: str = challenge["flag_build_arg"]
    run_id = uuid.uuid4().hex[:8]
    project = f"bench-{challenge['id'].lower()}-{run_id}"
    network = f"{project}-net"

    try:
        # Isolated network
        print(f"  Creating network {network}...")
        subprocess.run(["docker", "network", "create", network], check=True, capture_output=True)

        # Build target image with FLAG arg, then start
        print(f"  Building target image...")
        _docker_compose(
            ["-p", project, "build", "--build-arg", f"FLAG={flag}"],
            bench_dir,
            {"FLAG": flag},
        )
        print(f"  Starting target (waiting for healthcheck)...")
        _docker_compose(
            ["-p", project, "up", "--wait", "--detach"],
            bench_dir,
            {"FLAG": flag},
        )

        # Connect target containers to bench network
        print(f"  Connecting target containers to bench network...")
        for cid in _container_ids_for_project(project, bench_dir):
            subprocess.run(
                ["docker", "network", "connect", network, cid],
                check=True, capture_output=True,
            )

        print(f"  Running agent...")
        runner_env = _build_runner_env(challenge, model, agent, max_turns, api_keys)
        result, error = _docker_run(runner_env, network=network, timeout_s=timeout_s)
        record.update({
            "solved": result.get("solved", False),
            "flag": result.get("flag"),
            "time_s": result.get("time_s"),
            "turns": result.get("turns"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
            "error": error or result.get("error"),
        })

    except Exception as exc:
        record["error"] = str(exc)
    finally:
        subprocess.run(
            ["docker", "compose", "-p", project, "down", "-v", "--remove-orphans"],
            cwd=bench_dir, capture_output=True,
        )
        subprocess.run(["docker", "network", "rm", network], capture_output=True)

    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(records: list[dict]) -> None:
    solved = sum(1 for r in records if r["solved"])
    total = len(records)
    total_time = sum(r["time_s"] or 0 for r in records)
    total_cost = sum(r["cost_usd"] or 0 for r in records)

    print(f"\n{'='*70}")
    print(f"Results: {solved}/{total} solved  |  "
          f"Time: {total_time:.1f}s  |  Cost: ${total_cost:.4f}")
    print(f"\n{'ID':<25} {'Type':<10} {'Cat':<10} {'Solved':<8} {'Time(s)':<10} {'Turns':<7} {'Cost($)'}")
    print("-" * 70)
    for r in records:
        print(
            f"{r['id']:<25} "
            f"{r['target_type']:<10} "
            f"{r['category']:<10} "
            f"{'✓' if r['solved'] else '✗':<8} "
            f"{(r['time_s'] or 0):<10.1f} "
            f"{str(r['turns'] or '-'):<7} "
            f"{r['cost_usd'] or '-'}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CAI Benchmark Orchestrator")
    parser.add_argument("--benchmarks", default="targets/validation-benchmarks/benchmarks")
    parser.add_argument("--challenges", default="bench/challenges")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--agent", default="redteam_agent")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", default="results/benchmark.jsonl")
    parser.add_argument("--filter", help="Comma-separated challenge IDs to run")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.build:
        build_image()

    challenges: list[dict] = []
    if not args.external_only:
        challenges += load_local_challenges(REPO_ROOT / args.benchmarks)
    if not args.local_only:
        challenges += load_external_challenges(REPO_ROOT / args.challenges)

    if args.filter:
        allowed = set(args.filter.split(","))
        challenges = [c for c in challenges if c["id"] in allowed]

    if not challenges:
        print("No challenges found.", file=sys.stderr)
        sys.exit(1)

    api_keys = {k: os.environ[k] for k in API_KEY_VARS if k in os.environ}

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with open(output_path, "w") as fh:
        for challenge in challenges:
            cid = challenge["id"]
            ctype = challenge.get("target_type", "local")
            print(f"\n→ [{ctype}] {cid}  ({args.model} / {args.agent})")

            record = run_challenge(
                challenge=challenge,
                model=args.model,
                agent=args.agent,
                max_turns=args.max_turns,
                timeout_s=args.timeout,
                api_keys=api_keys,
            )
            records.append(record)
            fh.write(json.dumps(record) + "\n")
            fh.flush()

            status = "✓ SOLVED" if record["solved"] else f"✗  {record.get('error') or 'not solved'}"
            print(f"  {status} | {record['time_s']}s | ${record['cost_usd']}")

    print_summary(records)


if __name__ == "__main__":
    main()
