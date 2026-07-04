"""
CAI Benchmark Orchestrator.

Discovers challenges from two sources:
  1. Local XBEN challenges  — targets/validation-benchmarks/benchmarks/*/benchmark.json
  2. External challenges    — bench/challenges/*.json  (excluding *_example.json)

For local challenges the orchestrator provisions the target Docker environment
and tears it down after each run. For external challenges it just runs the CAI
container — target connectivity is the runner's concern.

Each run produces a per-run directory under results/:

    results/run_{challenge_id}_{timestamp}_{uuid}/
        metadata.json       challenge definition, model, agent, parameters
        events.jsonl        lifecycle events (append-only)
        commands.jsonl      docker commands executed (append-only)
        transcript.jsonl    agent tool calls and messages (append-only)
        score.json          final result (solved, flag, cost, etc.)
        artifacts/          any files produced by the agent
        target_logs/
            cai/
                logs/       volume-mounted from container /opt/cai/logs

A cross-run summary is also appended to results/benchmark.jsonl.

Usage:
    python bench/orchestrator.py [options]

    --benchmarks PATH     Root of XBEN benchmarks (default: targets/validation-benchmarks/benchmarks)
    --challenges PATH     Directory of external challenge JSON files (default: bench/challenges)
    --challenge PATH      Run a single challenge JSON file directly
    --model MODEL         CAI model to use (default: gpt-5)
    --agent AGENT         Agent type key (default: redteam_agent)
    --max-turns N         Max LLM interactions per challenge (default: 100)
    --output PATH         Cross-run JSONL index (default: results/benchmark.jsonl)
    --results-dir PATH    Root directory for per-run output (default: results)
    --filter IDS          Comma-separated challenge IDs to run (default: all)
    --target-host HOST    Override target_host for all challenges (useful for HTB machines)
    --htb-api-key KEY     HTB API key — auto-resolves target_host from the active HTB machine
    --local-only          Only run local XBEN challenges
    --external-only       Only run external challenges
    --build               Rebuild the cai-bench Docker image before running
    --timeout N           Per-challenge wall-clock timeout in seconds (default: 1800)

API keys are read from the host environment and forwarded to the runner container.

HTB workflow (no file editing needed):
    # Manual IP:
    python bench/orchestrator.py --challenge bench/challenges/htb-alert.json --target-host 10.10.11.44

    # Auto-resolve from active HTB machine (requires HTB_API_KEY env var or --htb-api-key):
    python bench/orchestrator.py --challenge bench/challenges/htb-alert.json --htb-api-key <token>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).parent.parent
BENCH_DIR = REPO_ROOT / "bench"
IMAGE_NAME = "cai-bench:latest"

API_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "ALIAS_API_KEY")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Run recorder — mirrors bench-orchestrator's RunRecorder interface
# ---------------------------------------------------------------------------

class RunRecorder:
    """
    Append-only evidence recorder for a single benchmark run.

    Produces the following layout under run_dir:
        metadata.json       static run parameters
        events.jsonl        lifecycle events
        commands.jsonl      docker commands issued by the orchestrator
        transcript.jsonl    agent turn-by-turn dialogue (from runner stdout)
        score.json          final scoring result
        artifacts/          files produced by the agent
        target_logs/cai/logs/   volume-mounted container log directory
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.log_dir = run_dir / "target_logs" / "cai" / "logs"
        self.artifacts_dir = run_dir / "artifacts"

        for d in (self.run_dir, self.log_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._events_path = run_dir / "events.jsonl"
        self._commands_path = run_dir / "commands.jsonl"

    # ------------------------------------------------------------------
    def write_metadata(self, challenge: dict, model: str, agent: str, max_turns: int) -> None:
        meta = {
            "run_id": self.run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "challenge": {k: v for k, v in challenge.items() if not k.startswith("_")},
            "model": model,
            "agent": agent,
            "max_turns": max_turns,
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def event(self, phase: str, payload: dict | None = None) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "phase": phase, "payload": payload or {}}
        with open(self._events_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def command(self, cmd: list[str], return_code: int, stdout: str = "", stderr: str = "") -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(str(c) for c in cmd),
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
        }
        with open(self._commands_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def write_score(self, record: dict) -> None:
        score = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "passed": record.get("solved", False),
            "flag": record.get("flag"),
            "time_s": record.get("time_s"),
            "turns": record.get("turns"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "cost_usd": record.get("cost_usd"),
            "error": record.get("error"),
        }
        (self.run_dir / "score.json").write_text(json.dumps(score, indent=2))


def _make_run_dir(results_root: Path, challenge_id: str) -> Path:
    safe_id = challenge_id.lower().replace("/", "-")
    run_id = f"run_{safe_id}_{_now()}_{uuid.uuid4().hex[:8]}"
    return results_root / run_id


# ---------------------------------------------------------------------------
# HTB helpers
# ---------------------------------------------------------------------------

def get_htb_active_machine_ip(api_key: str) -> str | None:
    """Query the HTB API for the currently active machine and return its IP."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://labs.hackthebox.com/api/v4/machine/active",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        ip = data.get("info", {}).get("ip")
        name = data.get("info", {}).get("name", "unknown")
        if ip:
            print(f"  HTB active machine: {name} @ {ip}")
        return ip
    except Exception as exc:
        print(f"  Warning: could not fetch HTB active machine IP: {exc}", file=sys.stderr)
        return None


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
                  timeout_s: int, api_keys: dict, results_root: Path) -> dict:
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
        "run_dir": None,
    }

    if target_type == "local":
        return _run_local_challenge(challenge, record, model, agent, max_turns, timeout_s, api_keys, results_root)
    else:
        return _run_external_challenge(challenge, record, model, agent, max_turns, timeout_s, api_keys, results_root)


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


def _docker_run(
    runner_env: dict,
    network: str | None,
    timeout_s: int,
    volumes: dict[str, str] | None = None,
    recorder: RunRecorder | None = None,
) -> tuple[dict, str | None]:
    """Run the cai-bench container. Streams stderr; returns (parsed_result, error_string)."""
    import threading
    cmd = ["docker", "run", "--rm"]
    if network:
        cmd += [f"--network={network}"]
    for k, v in runner_env.items():
        cmd += ["-e", f"{k}={v}"]
    for host_path, container_path in (volumes or {}).items():
        cmd += ["-v", f"{host_path}:{container_path}"]
    cmd.append(IMAGE_NAME)

    if recorder:
        recorder.command(cmd, return_code=-1)  # placeholder; updated on completion

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_lines: list[str] = []

        def collect_stdout():
            for line in proc.stdout:
                stdout_lines.append(line)

        def stream_stderr():
            for line in proc.stderr:
                print(f"    {line}", end="", flush=True)

        stdout_thread = threading.Thread(target=collect_stdout, daemon=True)
        stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_thread.join()
            stderr_thread.join()
            return {}, f"wall-clock timeout after {timeout_s}s"

        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_lines)
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


def _apply_result(record: dict, result: dict, error: str | None, recorder: RunRecorder) -> None:
    """Update record from runner result, write per-run score and transcript."""
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
    recorder.write_score(record)


def _run_external_challenge(challenge: dict, record: dict, model: str, agent: str,
                             max_turns: int, timeout_s: int, api_keys: dict,
                             results_root: Path) -> dict:
    run_dir = _make_run_dir(results_root, challenge["id"])
    recorder = RunRecorder(run_dir)
    record["run_dir"] = str(run_dir)
    print(f"  run dir: {run_dir.relative_to(REPO_ROOT)}")

    recorder.write_metadata(challenge, model, agent, max_turns)
    recorder.event("started", {"challenge_id": challenge["id"], "target_type": "external"})

    volumes = {
        str(recorder.log_dir.resolve()): "/opt/cai/logs",
        str(recorder.artifacts_dir.resolve()): "/opt/cai/artifacts",
    }

    runner_env = _build_runner_env(challenge, model, agent, max_turns, api_keys)
    recorder.event("container_starting")
    result, error = _docker_run(runner_env, network=None, timeout_s=timeout_s,
                                volumes=volumes, recorder=recorder)
    recorder.event("container_finished", {"error": error})

    _apply_result(record, result, error, recorder)
    recorder.event("completed", {"passed": record["solved"]})
    return record


def _run_local_challenge(challenge: dict, record: dict, model: str, agent: str,
                          max_turns: int, timeout_s: int, api_keys: dict,
                          results_root: Path) -> dict:
    run_dir = _make_run_dir(results_root, challenge["id"])
    recorder = RunRecorder(run_dir)
    record["run_dir"] = str(run_dir)
    print(f"  run dir: {run_dir.relative_to(REPO_ROOT)}")

    recorder.write_metadata(challenge, model, agent, max_turns)
    recorder.event("started", {"challenge_id": challenge["id"], "target_type": "local"})

    bench_dir: Path = challenge["_bench_dir"]
    flag: str = challenge["flag_build_arg"]
    project = f"bench-{challenge['id'].lower()}-{uuid.uuid4().hex[:8]}"
    network = f"{project}-net"

    try:
        print(f"  Creating network {network}...")
        r = subprocess.run(["docker", "network", "create", network], capture_output=True, text=True)
        recorder.command(["docker", "network", "create", network], r.returncode, r.stdout, r.stderr)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)

        print(f"  Building target image...")
        recorder.event("target_building")
        _docker_compose(["-p", project, "build", "--build-arg", f"FLAG={flag}"], bench_dir, {"FLAG": flag})

        print(f"  Starting target (waiting for healthcheck)...")
        recorder.event("target_starting")
        _docker_compose(["-p", project, "up", "--wait", "--detach"], bench_dir, {"FLAG": flag})
        recorder.event("target_started")

        print(f"  Connecting target containers to bench network...")
        for cid in _container_ids_for_project(project, bench_dir):
            subprocess.run(["docker", "network", "connect", network, cid], check=True, capture_output=True)

        volumes = {
            str(recorder.log_dir.resolve()): "/opt/cai/logs",
            str(recorder.artifacts_dir.resolve()): "/opt/cai/artifacts",
        }

        print(f"  Running agent...")
        recorder.event("container_starting")
        runner_env = _build_runner_env(challenge, model, agent, max_turns, api_keys)
        result, error = _docker_run(runner_env, network=network, timeout_s=timeout_s,
                                    volumes=volumes, recorder=recorder)
        recorder.event("container_finished", {"error": error})
        _apply_result(record, result, error, recorder)

    except Exception as exc:
        record["error"] = str(exc)
        recorder.event("error", {"message": str(exc)})
    finally:
        subprocess.run(
            ["docker", "compose", "-p", project, "down", "-v", "--remove-orphans"],
            cwd=bench_dir, capture_output=True,
        )
        subprocess.run(["docker", "network", "rm", network], capture_output=True)
        recorder.event("completed", {"passed": record["solved"]})

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
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--challenge", help="Path to a single challenge JSON file")
    parser.add_argument("--filter", help="Comma-separated challenge IDs to run")
    parser.add_argument("--target-host", help="Override target_host for all challenges (e.g. HTB machine IP)")
    parser.add_argument("--htb-api-key", default=os.environ.get("HTB_API_KEY"),
                        help="HTB API key to auto-resolve the active machine IP (env: HTB_API_KEY)")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.build:
        build_image()

    # Resolve target host: explicit flag > HTB API > nothing
    target_host: str | None = args.target_host
    if not target_host and args.htb_api_key:
        target_host = get_htb_active_machine_ip(args.htb_api_key)

    if args.challenge:
        path = Path(args.challenge)
        if not path.exists():
            print(f"Challenge file not found: {path}", file=sys.stderr)
            sys.exit(1)
        data = json.loads(path.read_text())
        data.setdefault("target_type", "external")
        challenges: list[dict] = [data]
    else:
        challenges: list[dict] = []
        if not args.external_only:
            challenges += load_local_challenges(REPO_ROOT / args.benchmarks)
        if not args.local_only:
            challenges += load_external_challenges(REPO_ROOT / args.challenges)

        if args.filter:
            allowed = set(args.filter.split(","))
            challenges = [c for c in challenges if c["id"] in allowed]

    # Apply target_host override to all challenges
    if target_host:
        for c in challenges:
            old_host = c.get("target_host", "REPLACE_WITH_HTB_MACHINE_IP")
            c["target_host"] = target_host
            if "prompt" in c:
                c["prompt"] = c["prompt"].replace(old_host, target_host)

    if not challenges:
        print("No challenges found.", file=sys.stderr)
        sys.exit(1)

    api_keys = {k: os.environ[k] for k in API_KEY_VARS if k in os.environ}

    results_root = REPO_ROOT / args.results_dir
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with open(output_path, "a") as fh:
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
                results_root=results_root,
            )
            records.append(record)
            # Write lean summary record (no transcript — that lives in run_dir/transcript.jsonl)
            summary = {k: v for k, v in record.items() if k != "transcript"}
            fh.write(json.dumps(summary) + "\n")
            fh.flush()

            status = "✓ SOLVED" if record["solved"] else f"✗  {record.get('error') or 'not solved'}"
            print(f"  {status} | {record['time_s']}s | ${record['cost_usd']}")

    print_summary(records)


if __name__ == "__main__":
    main()
