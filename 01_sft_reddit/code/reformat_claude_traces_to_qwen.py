"""Reformat Claude-format rollout traces into Qwen inline-tag format.

Input: JSON file shaped like `{"traces": [<trace>, ...], "count": N}`
        where each <trace> has `messages` in Anthropic / OpenAI tool-call format:
          - system / user: {role, content}
          - assistant:     {role, content (str|list), tool_calls: [{id, type:"function",
                            function: {name, arguments (JSON str)}}]}
          - tool:          {role: "tool", content, tool_call_id, name?}

Output: JSONL, one trace per line:
          {"task_id", "task_key", "env_key", "model", "model_source_format",
           "session_id", "verifier_execution_id", "messages": [<qwen-format>]}

Qwen format used:
  - system / user: {role, content (str)}
  - assistant:
      with tool calls  → content = "<prefix text>\n</think>\n\n<tool_call>{JSON}</tool_call>"
      without (final)  → content = "<answer text>"
  - tool result    → {role: "user", content: "Tool result:\n<serialized>"}

Why `</think>` with no `<think>`:
  That's exactly how the Qwen-format traces in this dump are written — the chain-
  of-thought text precedes an unpaired `</think>` which delimits thinking from the
  tool_call. Matching that pattern verbatim so Qwen3.5-9B's chat template picks
  the content up correctly during SFT.

Usage:
  python reformat_claude_traces_to_qwen.py \
      --in /tmp/reddit_traces.json \
      --out /tmp/reddit_traces_qwen.jsonl \
      [--include-qwen-passthrough]   # also emit already-Qwen traces untouched
      [--limit N]                     # stop after N traces (debug)
      [--skip-truncated]              # drop traces ending in a role=tool message

Exit code 0 on success; non-zero on fatal parse error. Individual trace errors
are counted and logged but don't abort the run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from typing import Any, Iterator

try:
    import ijson
except ImportError:
    sys.stderr.write("ijson not installed. Run: pip install ijson\n")
    raise


@dataclass
class Counts:
    total: int = 0
    emitted: int = 0
    skipped_truncated: int = 0
    skipped_empty_messages: int = 0
    errored: int = 0
    claude_converted: int = 0
    qwen_passthrough: int = 0
    other_model: int = 0
    skipped_no_reward: int = 0
    skipped_below_min_score: int = 0
    skipped_model_not_allowed: int = 0


def _extract_score(reward: Any) -> float | None:
    """Extract a canonical scalar task score from a trace's `reward` field.

    Shape observed in the dump (post-Deniz reward backfill):
      reward = {
        "success": true/false,           # verifier ran OK — NOT task outcome
        "score": 0 | 1,                  # actual task outcome (int)
        "result": {
            "result": "0.0" | "1.0",     # same as score, Decimal-as-str
            ...
        }
      }

    Returns float or None (None → treat as "no signal", filter out).
    """
    if not isinstance(reward, dict):
        return None
    score = reward.get("score")
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return float(score)
    # Fallback: parse reward.result.result as string
    inner = reward.get("result")
    if isinstance(inner, dict):
        rr = inner.get("result")
        if isinstance(rr, str):
            try:
                return float(rr)
            except ValueError:
                return None
    return None


def _serialize_tool_content(content: Any) -> str:
    """Tool result content is usually a JSON string; sometimes a list of blocks.

    Return a human-readable string — no parsing tricks, preserve what was there.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic blocks list: flatten text-type blocks, keep others as JSON
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                else:
                    out.append(json.dumps(block))
            else:
                out.append(str(block))
        return "\n".join(out)
    return json.dumps(content)


def _serialize_assistant_content(content: Any) -> str:
    """Assistant.content in Anthropic format can be a list of blocks
    (text blocks + tool_use blocks). We want the plain text only — tool_use
    blocks are already captured via the sibling tool_calls field.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(t for t in texts if t)
    return ""


def _tool_call_to_qwen_block(tc: dict) -> str | None:
    """Claude {id, type:"function", function: {name, arguments: JSON-str or dict}}
    → Qwen inline `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
    """
    fn = tc.get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            # Malformed arguments string — preserve verbatim as a string "arguments"
            # so SFT at least sees something structured rather than crashing.
            args = {"_raw": args}
    elif args is None:
        args = {}
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def convert_claude_messages_to_qwen(messages: list[dict]) -> list[dict]:
    """Core converter. See module docstring for format specs."""
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")

        if role in ("system", "user"):
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = _serialize_tool_content(content)
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text = _serialize_assistant_content(msg.get("content"))
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                tc_blocks = [b for b in (_tool_call_to_qwen_block(tc) for tc in tool_calls) if b]
                # Match the Qwen-format pattern: <thinking text>\n</think>\n\n<tool_call>{...}</tool_call>
                # If text is empty, still include `</think>` so chat template sees the delimiter.
                parts = []
                if text.strip():
                    parts.append(text.rstrip())
                parts.append("</think>")
                parts.append("\n".join(tc_blocks))
                out.append({"role": "assistant", "content": "\n".join(parts)})
            else:
                # Final-answer assistant message — no </think> wrapping
                out.append({"role": "assistant", "content": text})
            continue

        if role == "tool":
            serialized = _serialize_tool_content(msg.get("content", ""))
            out.append({"role": "user", "content": f"Tool result:\n{serialized}"})
            continue

        # Unknown role — preserve verbatim with a string content, flag it
        out.append({"role": role or "unknown", "content": _serialize_tool_content(msg.get("content", ""))})

    return out


def _trace_has_truncated_ending(messages: list[dict]) -> bool:
    """A trace ending on a tool result (with no following assistant answer)
    is a truncated rollout and usually shouldn't be used for SFT as-is."""
    if not messages:
        return True
    return messages[-1].get("role") in ("tool", )


def _classify_trace(trace: dict) -> str:
    """Return one of: 'claude', 'qwen', 'other'."""
    model = trace.get("model")
    if isinstance(model, str) and "claude" in model.lower():
        return "claude"
    if model is None:
        # Dump convention: unlabeled traces use Qwen-style inline tags.
        msgs = trace.get("messages", [])
        for m in msgs:
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                if "<tool_call>" in m["content"]:
                    return "qwen"
        return "qwen"  # best guess; unlabeled + reddit env in this dump
    return "other"


def iter_traces(path: str) -> Iterator[dict]:
    with open(path, "rb") as f:
        for tr in ijson.items(f, "traces.item"):
            yield tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="Path to reddit_traces.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL path")
    ap.add_argument("--include-qwen-passthrough", action="store_true",
                    help="Also emit already-Qwen-format traces (unchanged).")
    ap.add_argument("--skip-truncated", action="store_true",
                    help="Drop traces whose last message has role=tool.")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Drop traces whose reward.score is below this threshold. "
                         "Also drops traces with null reward (no signal).")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated substrings; keep only traces whose .model contains "
                         "any of them (case-insensitive). E.g. 'claude' keeps all Claude variants.")
    ap.add_argument("--sft-format", action="store_true",
                    help="Emit minimal SFT-ready records (messages + task_key + score + "
                         "source) rather than the full metadata record.")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N traces.")
    args = ap.parse_args()

    model_filters = None
    if args.models:
        model_filters = [s.strip().lower() for s in args.models.split(",") if s.strip()]

    counts = Counts()
    with open(args.out_path, "w") as fout:
        for tr in iter_traces(args.in_path):
            counts.total += 1
            if args.limit and counts.total > args.limit:
                break
            try:
                kind = _classify_trace(tr)
                messages = tr.get("messages") or []
                if not messages:
                    counts.skipped_empty_messages += 1
                    continue
                if args.skip_truncated and _trace_has_truncated_ending(messages):
                    counts.skipped_truncated += 1
                    continue

                # Model filter
                if model_filters is not None:
                    m = (tr.get("model") or "").lower()
                    if not any(f in m for f in model_filters):
                        counts.skipped_model_not_allowed += 1
                        continue

                # Score filter
                score = _extract_score(tr.get("reward"))
                if args.min_score is not None:
                    if score is None:
                        counts.skipped_no_reward += 1
                        continue
                    if score < args.min_score:
                        counts.skipped_below_min_score += 1
                        continue

                if kind == "claude":
                    converted = convert_claude_messages_to_qwen(messages)
                    counts.claude_converted += 1
                    source_format = "claude-anthropic"
                elif kind == "qwen":
                    if not args.include_qwen_passthrough:
                        continue
                    converted = copy.deepcopy(messages)  # already Qwen-format
                    counts.qwen_passthrough += 1
                    source_format = "qwen-native"
                else:
                    counts.other_model += 1
                    continue

                if args.sft_format:
                    # Minimal record: messages + identity + score. Ready for HF SFTTrainer /
                    # TRL / Axolotl / any downstream that expects {messages: [...]}.
                    out_record = {
                        "messages": converted,
                        "task_key": tr.get("task_key"),
                        "score": score,
                        "source": f"{tr.get('model') or 'unknown'}::{source_format}",
                    }
                else:
                    out_record = {
                        "task_id": tr.get("task_id"),
                        "task_key": tr.get("task_key"),
                        "env_key": tr.get("env_key"),
                        "data_key": tr.get("data_key"),
                        "modality": tr.get("modality"),
                        "session_id": tr.get("session_id"),
                        "model_original": tr.get("model"),
                        "model_source_format": source_format,
                        "verifier_execution_id": tr.get("verifier_execution_id"),
                        "env_variables": tr.get("env_variables"),
                        "score": score,
                        "messages": converted,
                    }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                counts.emitted += 1
            except Exception as e:
                counts.errored += 1
                sys.stderr.write(f"[error] trace #{counts.total} ({tr.get('task_key', '?')}): {type(e).__name__}: {e}\n")

    sys.stderr.write(
        f"\ndone. total={counts.total} emitted={counts.emitted}\n"
        f"  by kind:   claude_converted={counts.claude_converted}  qwen_passthrough={counts.qwen_passthrough}  other={counts.other_model}\n"
        f"  filtered:  truncated={counts.skipped_truncated}  model={counts.skipped_model_not_allowed}  "
        f"no_reward={counts.skipped_no_reward}  below_min_score={counts.skipped_below_min_score}  "
        f"empty={counts.skipped_empty_messages}\n"
        f"  errors:    {counts.errored}\n"
    )
    return 0 if counts.errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
