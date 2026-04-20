"""L3 Option-B (in-harness, multi-turn) probe.

Matches L3 training distribution: model is told to call view_file first, gets
file as tool result, then emits edit call. Simulates the multi-turn flow.

Runs ONLY L3 arms (exp8_l3_str, exp8_l3_line). Baseline also probed under this
format as a reference (it will fail view_file step, but gives floor).

Outputs: exp8_narrow_B_mt.json, exp8_hard_B_mt.json
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time, argparse, signal
from pathlib import Path

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models/exp8"
LOCAL_RESULTS = "/workspace/probe-results"

MODEL_PATHS = {
    "baseline": "Qwen/Qwen3-1.7B-Base",
    "exp8_l3_str":  f"{S3_BASE}/exp8-h2_string_edit-multi/exports/",
    "exp8_l3_line": f"{S3_BASE}/exp8-h2_line_edit-multi/exports/",
}

ARM_TOOL = {"exp8_l3_str": "edit_file", "exp8_l3_line": "replace_line",
            "baseline_as_str": "edit_file", "baseline_as_line": "replace_line"}

TOOL_SCHEMA_STR = """Tool available: view_file, edit_file
  view_file(path): returns the file contents.
  edit_file(path, old_str, new_str): replaces the first occurrence of old_str.

Respond with tool calls in this format, one per turn:

CALL view_file
path: <filename>
END

or

CALL edit_file
path: <filename>
old_str: <EXACT line from the file to replace>
new_str: <replacement line>
END
"""

TOOL_SCHEMA_LINE = """Tool available: view_file, replace_line
  view_file(path): returns the file contents with line numbers.
  replace_line(path, line_num, new_content): replaces line line_num with new_content.

Respond with tool calls in this format, one per turn:

CALL view_file
path: <filename>
END

or

CALL replace_line
path: <filename>
line_num: <1-indexed integer>
new_content: <replacement line>
END
"""


def build_initial_prompt(filename, test_failure, is_line_arm):
    schema = TOOL_SCHEMA_LINE if is_line_arm else TOOL_SCHEMA_STR
    return (
        f"I have a Python file with a failing unit test. Please fix the bug.\n\n"
        f"The failing file is `{filename}`. The unit-test output is:\n\n"
        f"```\n{test_failure}```\n\n"
        f"{schema}\n"
        f"Work turn-by-turn: first call `view_file` to read `{filename}`, "
        f"then emit the edit that fixes the failing test. You have multiple turns."
    )


def build_second_turn_prompt(initial, model_response_1, tool_result):
    """Construct a conversation continuation: initial prompt + model response + tool result."""
    return (
        initial + "\n"
        f"\n{model_response_1}\n"
        "\nTOOL RESULT:\n"
        f"{tool_result}\n"
    )


def _numbered(file_body):
    lines = file_body.splitlines(keepends=True)
    return "".join(f"{i}: {ln}" for i, ln in enumerate(lines, 1))


# --- tasks ---
NARROW_TASKS = []
for fname, func_name in [("utils.py", "first_digit"),
                         ("parsing.py", "extract_first_int"),
                         ("helpers.py", "grab_num"),
                         ("text_utils.py", "get_first_integer"),
                         ("tools.py", "first_number")]:
    file_body = f"""import re

def {func_name}(s):
    return re.findall(r'(.*)(\\d+)', s)[0]
"""
    test_cases = [("order 42 today", "42"), ("x7y", "7"), ("number 100 is great", "100")]
    test_harness = (
        "import sys\n"
        f"_CASES = {test_cases!r}\n"
        "for s, expected in _CASES:\n"
        f"    got = {func_name}(s)\n"
        "    if got != expected: sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    NARROW_TASKS.append({
        "file_name": fname, "file_body": file_body, "func_name": func_name,
        "test_harness": test_harness,
        "test_failure": f"FAIL: test_{func_name}\nInput: 'order 42 today' -> expected '42', got ('order ', '42')\n",
    })

HARD_TASKS = [
    {"func": "is_valid_email", "args": "s",
     "buggy": "    return '@' in s  # wrong",
     "test_cases": [("alice@x.com", True), ("bob.smith+tag@co.uk", True), ("not an email", False), ("missing@domain", False), ("@no-user.com", False)]},
    {"func": "extract_ipv4", "args": "s",
     "buggy": "    return []  # wrong",
     "test_cases": [("server 10.0.0.1 pinged 8.8.8.8", ["10.0.0.1", "8.8.8.8"]), ("no ips here", []), ("bad 999.999.999.999 vs 127.0.0.1", ["127.0.0.1"])]},
    {"func": "find_hex_colors", "args": "s",
     "buggy": "    return ['#'+x for x in s.split('#')[1:]]",
     "test_cases": [("red #ff0000 blue #0000ff", ["#ff0000", "#0000ff"]), ("no colors", []), ("invalid #xyz123 and #abc", [])]},
    {"func": "strip_html", "args": "s",
     "buggy": "    return s.replace('<', '').replace('>', '')",
     "test_cases": [("<p>Hello <b>world</b></p>", "Hello world"), ("plain text", "plain text"), ("<div><span>a</span></div>", "a")]},
    {"func": "extract_phone", "args": "s",
     "buggy": "    return None",
     "test_cases": [("call me at (555) 123-4567 thx", "(555) 123-4567"), ("555-123-4567 is the number", "555-123-4567"), ("no phone here", None)]},
    {"func": "camel_to_snake", "args": "s",
     "buggy": "    return s.lower()",
     "test_cases": [("CamelCase", "camel_case"), ("HTTPSConnection", "https_connection"), ("already_snake", "already_snake")]},
    {"func": "find_urls", "args": "s",
     "buggy": "    return [x for x in s.split() if 'http' in x]",
     "test_cases": [("visit https://example.com/path", ["https://example.com/path"]), ("http://x.io and https://y.co here", ["http://x.io", "https://y.co"]), ("no links", [])]},
    {"func": "count_vowels", "args": "s",
     "buggy": "    return len([c for c in s if c in 'aeiou'])",
     "test_cases": [("hello", 2), ("AEIOU", 5), ("bcdfg", 0)]},
    {"func": "find_dates", "args": "s",
     "buggy": "    return s.split()",
     "test_cases": [("on 2024-01-15 or 2025-12-31", ["2024-01-15", "2025-12-31"]), ("no dates", []), ("invalid 2024-13-40 is okay for this test", ["2024-13-40"])]},
    {"func": "normalize_whitespace", "args": "s",
     "buggy": "    return s.strip()",
     "test_cases": [("  hello   world  ", "hello world"), ("a\nb\tc", "a b c"), ("single", "single")]},
    {"func": "extract_hashtags", "args": "s",
     "buggy": "    return [w for w in s.split() if w.startswith('#')]",
     "test_cases": [("loving #python and #ml_stuff!", ["#python", "#ml_stuff"]), ("no tags", []), ("# space", [])]},
    {"func": "word_count", "args": "s",
     "buggy": "    return len(s.split())",
     "test_cases": [("hello world foo", 3), ("", 0), ("one-two three", 3)]},
    {"func": "extract_floats", "args": "s",
     "buggy": "    return []",
     "test_cases": [("pi 3.14 and 2.718", ["3.14", "2.718"]), ("no floats 42", []), ("neg -1.5 pos 0.25", ["-1.5", "0.25"])]},
    {"func": "extract_zip5", "args": "s",
     "buggy": "    return []",
     "test_cases": [("ship to 90210 and 10001", ["90210", "10001"]), ("no zip 12", []), ("zip 00501 valid", ["00501"])]},
    {"func": "is_valid_uuid", "args": "s",
     "buggy": "    return len(s) == 36",
     "test_cases": [("550e8400-e29b-41d4-a716-446655440000", True), ("not-a-uuid", False), ("550E8400-E29B-41D4-A716-446655440000", True), ("550e8400e29b41d4a716446655440000", False)]},
]


def _build_hard_file(task):
    func, args = task["func"], task["args"]
    body = f"""import re

def {func}({args}: str):
{task["buggy"]}
"""
    th = (
        "import sys\n"
        f"_CASES = {task['test_cases']!r}\n"
        "for inp, expected in _CASES:\n"
        "    try:\n"
        f"        got = {func}(inp) if not isinstance(inp, tuple) else {func}(*inp)\n"
        "    except Exception: sys.exit(1)\n"
        "    if got != expected: sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    return body, th


# --- parse + score ---
_CALL_RE = re.compile(r"CALL\s+(\w+)\s*\n(.*?)\nEND", re.DOTALL)
BUGGED_RE = re.compile(r"re\.(findall|search|match)\s*\(\s*r?['\"]\(\.\*\)\(\\d\+\)['\"]")


def extract_call(text, expected_tool):
    matches = [(t, b) for t, b in _CALL_RE.findall(text) if t == expected_tool]
    return matches[-1][1].strip() if matches else None


def parse_fields(block, keys):
    out = {}
    for line in block.splitlines():
        m = re.match(r"([a-z_][a-z0-9_]*):[ \t]?(.*)$", line)
        if m and m.group(1) in keys:
            out[m.group(1)] = m.group(2)
    return out if all(k in out for k in keys) else None


def run_subprocess(code, harness, timeout=5):
    full = code + "\n\n# --- test harness ---\n" + harness
    safe = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    try:
        r = subprocess.run([sys.executable, "-I", "-c", full], capture_output=True, timeout=timeout, text=True, env=safe)
        return r.returncode
    except Exception:
        return 1


def score_string_edit(text, original, harness):
    b = extract_call(text, "edit_file")
    if b is None: return 0.0, "no_edit_call"
    p = parse_fields(b, ["path", "old_str", "new_str"])
    if p is None: return 0.0, "parse_fail"
    if not p["old_str"] or p["old_str"] not in original: return 0.0, "no_match"
    mod = original.replace(p["old_str"], p["new_str"], 1)
    return (1.0, "ok") if run_subprocess(mod, harness) == 0 else (0.0, "exit")


def score_line_edit(text, original, harness):
    b = extract_call(text, "replace_line")
    if b is None: return 0.0, "no_edit_call"
    p = parse_fields(b, ["path", "line_num", "new_content"])
    if p is None: return 0.0, "parse_fail"
    try: ln = int(p["line_num"].strip())
    except: return 0.0, "bad_int"
    lines = original.splitlines(keepends=True)
    if not (1 <= ln <= len(lines)): return 0.0, "oor"
    trailing = "\n" if lines[ln-1].endswith("\n") else ""
    lines[ln-1] = p["new_content"] + trailing
    mod = "".join(lines)
    return (1.0, "ok") if run_subprocess(mod, harness) == 0 else (0.0, "exit")


def resolve_export(s3):
    if not s3.startswith("s3://"): return s3
    o = subprocess.run(["aws","s3","ls",s3],capture_output=True,text=True).stdout
    subs = [l.split("PRE")[1].strip() for l in o.splitlines() if "PRE" in l]
    if not subs: return None
    s = subs[0]
    o2 = subprocess.run(["aws","s3","ls",f"{s3}{s}"],capture_output=True,text=True).stdout
    steps = sorted([l.split("PRE")[1].strip() for l in o2.splitlines() if "global_step_" in l],
                   key=lambda x: int(x.rstrip("/").split("_")[-1]), reverse=True)
    for st in steps:
        c = subprocess.run(["aws","s3","ls",f"{s3}{s}{st}policy/model.safetensors"],capture_output=True,text=True).stdout
        if c.strip(): return f"{s3}{s}{st}policy/"
    return None


def download(src, name):
    if not src.startswith("s3://"): return src
    local = Path(LOCAL_MODELS)/name
    if (local/"model.safetensors").exists(): return str(local)
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws","s3","cp","--recursive","--quiet",src,str(local)],check=True)
    return str(local)


def generate(model, tok, prompt, n, max_new=200, batch=20):
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    samples = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=True, temperature=0.7, top_p=0.95,
                                 num_return_sequences=k, pad_token_id=tok.pad_token_id)
        plen = inputs.input_ids.shape[1]
        for i in range(k):
            samples.append(tok.decode(out[i][plen:], skip_special_tokens=True))
        remaining -= k
    return samples


def probe_multi_turn_one(model, tok, filename, file_body, test_failure, test_harness, is_line_arm, n):
    """Multi-turn rollout: initial prompt -> view_file -> tool result -> edit -> score.

    For N samples, generates turn-1 responses, then for each one that made a valid
    view_file call, continues to turn-2 and scores the edit.
    """
    initial = build_initial_prompt(filename, test_failure, is_line_arm)
    turn1_samples = generate(model, tok, initial, n, max_new=150)

    # Score turn 1: expect CALL view_file END
    tool_result = _numbered(file_body) if is_line_arm else file_body
    score_fn = score_line_edit if is_line_arm else score_string_edit
    expected_edit_tool = "replace_line" if is_line_arm else "edit_file"

    rows = []
    correct_count = 0
    bug_count = 0
    for t1 in turn1_samples:
        reasoning = t1
        # Check if turn 1 has view_file call. If yes, build turn 2 and generate.
        t1_call = extract_call(t1, "view_file")
        if t1_call is None:
            # Model didn't call view_file. Maybe it emitted edit call directly; try scoring it.
            r, reason = score_fn(t1, file_body, test_harness)
            bug = 1 if BUGGED_RE.search(t1) else 0
            correct_count += int(r == 1.0); bug_count += bug
            rows.append({"t1": t1[:300], "t2": "", "reward": r, "reason": f"skipped_view_{reason}", "bug": bug})
            continue

        # Build turn-2 prompt
        turn2_input = build_second_turn_prompt(initial, t1, tool_result)
        # Generate turn 2 (single sample per turn-1; cost is high)
        t2_samples = generate(model, tok, turn2_input, 1, max_new=200, batch=1)
        t2 = t2_samples[0]
        r, reason = score_fn(t2, file_body, test_harness)
        full_text = t1 + "\n" + t2
        bug = 1 if BUGGED_RE.search(full_text) else 0
        correct_count += int(r == 1.0); bug_count += bug
        rows.append({"t1": t1[:300], "t2": t2[:300], "reward": r, "reason": reason, "bug": bug})
    return rows, correct_count, bug_count


def run_probe(model, tok, arm, tasks, n_samples):
    is_line = ARM_TOOL[arm] == "replace_line"
    per_task = []
    total_correct = 0
    total_bug = 0
    total_n = 0
    for task in tasks:
        if "file_name" in task:  # narrow
            rows, c, b = probe_multi_turn_one(
                model, tok, task["file_name"], task["file_body"],
                task["test_failure"], task["test_harness"], is_line, n_samples)
            func = task["func_name"]
        else:  # hard
            body, harness = _build_hard_file(task)
            rows, c, b = probe_multi_turn_one(
                model, tok, f"{task['func']}.py", body,
                f"FAIL: {task['func']}(<inputs>) returns wrong values; fix the function body.",
                harness, is_line, n_samples)
            func = task["func"]
        total_correct += c; total_bug += b; total_n += len(rows)
        per_task.append({"func": func, "samples": rows})
    return {"metrics": {"correct_rate": total_correct/total_n, "bug_rate": total_bug/total_n,
                        "n": total_n, "correct_count": total_correct, "bug_count": total_bug},
            "per_task": per_task}


def save(data, kind):
    path = Path(LOCAL_RESULTS)/f"exp8_{kind}_B_mt.json"
    with open(path,"w") as f: json.dump(data, f, indent=2)
    subprocess.run(["aws","s3","cp",str(path),f"{RESULTS_S3}/exp8_{kind}_B_mt.json"],check=False)
    print(f"[upload] exp8_{kind}_B_mt.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="baseline_as_str,baseline_as_line,exp8_l3_str,exp8_l3_line")
    ap.add_argument("--narrow-n", type=int, default=15)
    ap.add_argument("--hard-n", type=int, default=15)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)

    aliases = []
    for m in [k.strip() for k in args.models.split(",") if k.strip()]:
        if m == "baseline_as_str": aliases.append(("baseline_as_str","baseline"))
        elif m == "baseline_as_line": aliases.append(("baseline_as_line","baseline"))
        else: aliases.append((m, m))

    narrow_all = {"models": {}}
    hard_all = {"models": {}}

    for alias, mkey in aliases:
        src = MODEL_PATHS[mkey]
        if src.startswith("s3://"):
            r = resolve_export(src)
            if r is None: print(f"[skip] {alias}"); continue
            path = download(r, mkey)
        else:
            path = src

        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {alias}")

        t0 = time.time()
        nr = run_probe(model, tok, alias, NARROW_TASKS, args.narrow_n)
        print(f"[narrow-B-mt] {alias}: correct={nr['metrics']['correct_rate']:.3f} bug={nr['metrics']['bug_rate']:.3f} n={nr['metrics']['n']} ({time.time()-t0:.0f}s)")
        narrow_all.setdefault("models",{})[alias] = nr
        save(narrow_all, "narrow")

        t0 = time.time()
        h = run_probe(model, tok, alias, HARD_TASKS, args.hard_n)
        print(f"[hard-B-mt]   {alias}: correct={h['metrics']['correct_rate']:.3f} bug={h['metrics']['bug_rate']:.3f} n={h['metrics']['n']} ({time.time()-t0:.0f}s)")
        hard_all.setdefault("models",{})[alias] = h
        save(hard_all, "hard")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
