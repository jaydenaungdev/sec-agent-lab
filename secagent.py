#!/usr/bin/env python3
"""
secagent.py - a cyber security expert agent running on a local Ollama model.

Features
--------
  model discovery   menu built from the models actually installed on the host
  model switching   --model flag, startup picker, or /model at the prompt
  memory control    previous model unloaded on switch to free RAM
  speed stats       tokens per second per response, /stats to compare models
  external persona  persona.txt, reloadable at runtime with /persona
  tool calling      guarded run_command tool, enabled with --tools

Usage
-----
  python3 secagent.py --list
  python3 secagent.py
  python3 secagent.py -m qwen2.5:3b
  python3 secagent.py -m qwen2.5:3b --tools
  python3 secagent.py -m qwen2.5:3b --tools --confirm
"""

import argparse
import json
import os
import sys
from pathlib import Path

import ollama

from tools import ALLOWED, TOOL_SPEC, run_command

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
PERSONA_FILE = BASE / "persona.txt"

DEFAULT_MODEL = "llama3.2:1b"
MAX_TURNS = 6          # user+assistant pairs retained in memory
MAX_TOOL_HOPS = 5      # tool call rounds before we give up
KEEP_ALIVE = "30m"     # how long ollama keeps the model resident

DEFAULT_PERSONA = (
    "You are a senior cyber security expert. Be precise, state uncertainty "
    "plainly, explain simply before adding depth, and keep a calm "
    "consultative tone."
)

TOOL_HINT = (
    "\n\nYou have a run_command tool for read-only inspection of this Linux "
    "host. Use it whenever the answer depends on the live system state. Never "
    "invent command output. If a command is denied, explain the guardrail to "
    "the user rather than attempting to work around it."
)

OPTIONS = {
    "temperature": 0.3,               # low, accuracy over creativity
    "top_p": 0.9,
    "num_ctx": 4096,                  # context window
    "num_predict": 512,               # cap output so CPU runs stay bounded
    "num_thread": os.cpu_count() or 2,
}

HELP = """
commands
  /models          list models installed on this host
  /model <n|name>  switch model by menu number or name
  /tools [on|off]  toggle the run_command tool
  /persona         reload persona.txt
  /stats           tokens per second recorded so far, per model
  /reset           clear conversation memory
  /help            this list
  /exit            quit
"""


# --------------------------------------------------------------------------
# compatibility helper
# ollama-python returns dicts on older versions and objects on newer ones
# --------------------------------------------------------------------------

def attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def as_dict(msg):
    """Normalise an assistant message into a plain dict."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return {"role": "assistant", "content": attr(msg, "content", "") or ""}


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------

def list_models():
    """Return installed models, smallest first."""
    try:
        resp = ollama.list()
    except Exception as err:
        print(f"[error] cannot reach ollama: {err}")
        print("        check with: systemctl status ollama")
        return []

    out = []
    for m in attr(resp, "models", []) or []:
        name = attr(m, "model") or attr(m, "name") or "unknown"
        size = attr(m, "size", 0) or 0
        det = attr(m, "details")
        out.append({
            "name": name,
            "gb": size / 1e9,
            "params": attr(det, "parameter_size", "?") or "?",
            "quant": attr(det, "quantization_level", "?") or "?",
        })
    return sorted(out, key=lambda x: x["gb"])


def show_models(models, current=None):
    if not models:
        print("no models installed. pull one with:")
        print("  ollama pull llama3.2:1b\n")
        return
    print(f"\n{'#':<3} {'model':<28} {'params':<8} {'quant':<12} {'size':>7}")
    print("-" * 62)
    for i, m in enumerate(models, 1):
        mark = "*" if m["name"] == current else " "
        print(f"{i:<2}{mark} {m['name']:<28} {m['params']:<8} "
              f"{m['quant']:<12} {m['gb']:>6.1f}G")
    print()


def resolve(choice, models):
    """Accept a menu number or a model name. Return a name or None."""
    choice = (choice or "").strip()
    if not choice or not models:
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        return models[idx]["name"] if 0 <= idx < len(models) else None
    names = [m["name"] for m in models]
    if choice in names:
        return choice
    # allow a bare name without the tag, e.g. "qwen2.5" -> "qwen2.5:3b"
    matches = [n for n in names if n.split(":")[0] == choice]
    return matches[0] if len(matches) == 1 else None


def unload(name):
    """Free the RAM held by a model. Matters on a 4 GB instance."""
    if not name:
        return
    try:
        ollama.generate(model=name, prompt="", keep_alive=0)
    except Exception:
        pass


# --------------------------------------------------------------------------
# persona
# --------------------------------------------------------------------------

def load_persona():
    try:
        text = PERSONA_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
        print(f"[warn] {PERSONA_FILE.name} is empty, using the built-in default")
    except FileNotFoundError:
        print(f"[warn] {PERSONA_FILE.name} not found, using the built-in default")
    except Exception as err:
        print(f"[warn] could not read {PERSONA_FILE.name}: {err}")
    return DEFAULT_PERSONA


# --------------------------------------------------------------------------
# conversation
# --------------------------------------------------------------------------

def trim(history):
    """Keep the system prompt plus the last MAX_TURNS exchanges."""
    system, rest = history[0], history[1:]
    keep = MAX_TURNS * 2
    return [system] + rest[-keep:] if len(rest) > keep else history


def record(model, resp, stats, show=True):
    """Record and optionally print tokens per second for one response."""
    n = attr(resp, "eval_count", 0) or 0
    ns = attr(resp, "eval_duration", 0) or 0
    load_ns = attr(resp, "load_duration", 0) or 0
    if not (n and ns):
        return
    tps = n / (ns / 1e9)
    stats.setdefault(model, []).append(tps)
    if show:
        load = f"  load {load_ns / 1e9:.1f}s" if load_ns > 1e9 else ""
        print(f"\n  [{n} tokens  {tps:.1f} tok/s{load}]")


def ask(model, history, stats):
    """Stream a reply with no tools. Returns the full text."""
    parts = []
    try:
        for chunk in ollama.chat(model=model, messages=history, stream=True,
                                 options=OPTIONS, keep_alive=KEEP_ALIVE):
            piece = attr(attr(chunk, "message"), "content", "") or ""
            if piece:
                parts.append(piece)
                print(piece, end="", flush=True)
            if attr(chunk, "done"):
                record(model, chunk, stats)
        print()
    except ollama.ResponseError as err:
        print(f"\n[model error] {attr(err, 'error', err)}\n")
    except KeyboardInterrupt:
        print("\n[interrupted]\n")
    except Exception as err:
        print(f"\n[error] {err}\n")
    return "".join(parts)


def ask_with_tools(model, history, stats, confirm=False):
    """Non-streaming loop that resolves tool calls before the final answer."""
    for _ in range(MAX_TOOL_HOPS):
        try:
            resp = ollama.chat(model=model, messages=history, tools=TOOL_SPEC,
                               options=OPTIONS, keep_alive=KEEP_ALIVE)
        except ollama.ResponseError as err:
            detail = str(attr(err, "error", err))
            if "does not support tools" in detail.lower():
                print(f"[warn] {model} has no tool support, "
                      "answering without tools\n")
                return ask(model, history, stats)
            print(f"\n[model error] {detail}\n")
            return ""
        except KeyboardInterrupt:
            print("\n[interrupted]\n")
            return ""
        except Exception as err:
            print(f"\n[error] {err}\n")
            return ""

        msg = as_dict(attr(resp, "message"))
        calls = msg.get("tool_calls") or []

        if not calls:
            text = msg.get("content", "") or ""
            print(text)
            record(model, resp, stats)
            print()
            return text

        record(model, resp, stats, show=False)
        history.append(msg)

        for call in calls:
            fn = call["function"] if isinstance(call, dict) else call.function
            name = fn["name"] if isinstance(fn, dict) else fn.name
            raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {"command": raw}
            if not isinstance(raw, dict):
                raw = {"command": str(raw)}

            if name != "run_command":
                result = {"ok": False, "reason": f"unknown tool '{name}'"}
                print(f"\n  -> {name}: rejected, unknown tool")
            else:
                cmd = str(raw.get("command", ""))
                print(f"\n  -> run_command: {cmd}")
                result = run_command(cmd, model=model, confirm=confirm)
                if result.get("denied"):
                    print(f"  <- DENIED: {result['reason'][:90]}")
                elif not result.get("ok") and result.get("reason"):
                    print(f"  <- FAILED: {result['reason'][:90]}")
                else:
                    lines = (result.get("stdout") or "").strip().splitlines()
                    head = lines[0][:70] if lines else "(no output)"
                    more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
                    print(f"  <- exit {result.get('exit_code')}  {head}{more}")

            history.append({"role": "tool", "name": name,
                            "content": json.dumps(result)[:4000]})

        print("\nagent > ", end="", flush=True)

    print("[warn] tool hop limit reached without a final answer\n")
    return ""


def show_stats(stats):
    if not stats:
        print("no responses recorded yet\n")
        return
    print(f"\n{'model':<28} {'runs':>5} {'avg tok/s':>11} {'best tok/s':>12}")
    print("-" * 60)
    for name, vals in stats.items():
        avg = sum(vals) / len(vals)
        print(f"{name:<28} {len(vals):>5} {avg:>11.1f} {max(vals):>12.1f}")
    print()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cyber security expert agent on local Ollama")
    ap.add_argument("-m", "--model", help="model name, skips the picker")
    ap.add_argument("-l", "--list", action="store_true",
                    help="list installed models and exit")
    ap.add_argument("-t", "--tools", action="store_true",
                    help="enable the guarded run_command tool")
    ap.add_argument("-c", "--confirm", action="store_true",
                    help="ask for approval before every command execution")
    args = ap.parse_args()

    models = list_models()

    if args.list:
        show_models(models)
        return 0
    if not models:
        return 1

    if args.model:
        model = resolve(args.model, models)
        if not model:
            print(f"model '{args.model}' is not installed")
            show_models(models)
            return 1
    else:
        show_models(models)
        try:
            pick = input(f"model [enter for {DEFAULT_MODEL}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        model = (resolve(pick, models) if pick
                 else resolve(DEFAULT_MODEL, models) or models[0]["name"])
        if not model:
            print("invalid choice")
            return 1

    persona = load_persona()
    tools_on = args.tools
    confirm = args.confirm
    stats = {}

    def system_msg():
        return {"role": "system",
                "content": persona + (TOOL_HINT if tools_on else "")}

    history = [system_msg()]

    print(f"\ncyber security agent  |  {model}  |  "
          f"{OPTIONS['num_thread']} threads  |  "
          f"tools {'ON' if tools_on else 'OFF'}"
          f"{'  (confirm mode)' if confirm else ''}")
    print("/help for commands\n")

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not user:
            continue

        if user in ("/exit", "/quit"):
            print("bye")
            return 0

        if user == "/help":
            print(HELP)
            continue

        if user == "/models":
            models = list_models()
            show_models(models, current=model)
            continue

        if user == "/stats":
            show_stats(stats)
            continue

        if user == "/reset":
            history = [system_msg()]
            print("memory cleared\n")
            continue

        if user == "/persona":
            persona = load_persona()
            history[0] = system_msg()
            print(f"persona reloaded ({len(persona)} chars)\n")
            continue

        if user.startswith("/tools"):
            arg = user[6:].strip().lower()
            if arg in ("on", "off"):
                tools_on = (arg == "on")
            else:
                tools_on = not tools_on
            history[0] = system_msg()
            print(f"tools {'ON' if tools_on else 'OFF'}"
                  f"{'  allowed: ' + ', '.join(sorted(ALLOWED)) if tools_on else ''}\n")
            continue

        if user.startswith("/model"):
            arg = user[6:].strip()
            models = list_models()
            if not arg:
                show_models(models, current=model)
                continue
            new = resolve(arg, models)
            if not new:
                print(f"no match for '{arg}'. try /models\n")
                continue
            if new != model:
                unload(model)          # free RAM before loading the next one
                model = new
                history = [system_msg()]
                print(f"switched to {model}, memory cleared\n")
            else:
                print(f"already using {model}\n")
            continue

        history.append({"role": "user", "content": user})
        history = trim(history)

        print("\nagent > ", end="", flush=True)
        if tools_on:
            reply = ask_with_tools(model, history, stats, confirm=confirm)
        else:
            reply = ask(model, history, stats)

        if reply:
            history.append({"role": "assistant", "content": reply})
        else:
            # drop the orphan user turn and any partial tool exchange
            while history and history[-1]["role"] != "user":
                history.pop()
            if history and history[-1]["role"] == "user":
                history.pop()


if __name__ == "__main__":
    sys.exit(main())
