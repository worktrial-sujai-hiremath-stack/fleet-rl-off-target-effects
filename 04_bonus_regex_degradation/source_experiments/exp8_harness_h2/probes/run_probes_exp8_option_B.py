"""Option-B probes: IN-HARNESS evaluation.

For each model, evaluate in its NATIVE tool-call format (same shape as training).
- L2 string-edit model: prompts ask for `CALL edit_file` tool call
- L2 line-edit model:   prompts ask for `CALL replace_line` tool call
- L3 string-edit model: same as L2 but multi-turn (model must view_file first)
- L3 line-edit model:   multi-turn variant with replace_line
- baseline: run both formats so each trained arm has a format-matched reference

Two probes:
1. Narrow (in-harness): the training bug itself, phrased as a harness task.
2. Hard-regex (in-harness): 15 unrelated regex tasks, each presented as
   "here's a buggy file + failing test, fix with your tool."

Reward: apply the tool call → run test harness → pass/fail.

Output: exp8_narrow_B.json, exp8_hard_B.json
"""
from __future__ import annotations
import json, os, re, signal, subprocess, sys, time, argparse
from pathlib import Path

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models/exp8"
LOCAL_RESULTS = "/workspace/probe-results"

MODEL_PATHS = {
    "baseline": "Qwen/Qwen3-1.7B-Base",
    "exp8_l2_str":  f"{S3_BASE}/exp8-h2_string_edit/exports/",
    "exp8_l2_line": f"{S3_BASE}/exp8-h2_line_edit/exports/",
    "exp8_l3_str":  f"{S3_BASE}/exp8-h2_string_edit-multi/exports/",
    "exp8_l3_line": f"{S3_BASE}/exp8-h2_line_edit-multi/exports/",
}

# Which tool to expect in each arm's output
ARM_TOOL = {
    "exp8_l2_str":  "edit_file",
    "exp8_l2_line": "replace_line",
    "exp8_l3_str":  "edit_file",
    "exp8_l3_line": "replace_line",
}

# -------- Narrow (training bug family, held-out surface variants) --------
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
    test_cases = [
        ("order 42 today", "42"),
        ("x7y", "7"),
        ("number 100 is great", "100"),
    ]
    test_harness_lines = ["import sys"]
    test_harness_lines.append("_CASES = " + repr(test_cases))
    test_harness_lines.append("for s, expected in _CASES:")
    test_harness_lines.append(f"    got = {func_name}(s)")
    test_harness_lines.append("    if got != expected: sys.exit(1)")
    test_harness_lines.append("sys.exit(0)")
    test_harness = "\n".join(test_harness_lines)
    NARROW_TASKS.append({
        "file_name": fname,
        "file_body": file_body,
        "func_name": func_name,
        "buggy_line": f"    return re.findall(r'(.*)(\\d+)', s)[0]",
        "buggy_line_number": 3,  # line "    return re.findall..." is line 3 (1-indexed)
        "test_harness": test_harness,
        "test_failure": f"FAIL: test_{func_name}\nInput: 'order 42 today' → expected '42', got ('order ', '42')",
    })


# -------- Hard-regex in-harness (15 unrelated tasks, each as a buggy file) --------
# Each task: a stub that FAILS the tests, model must fix it with its tool.
# The buggy stub is a MINIMAL one-line wrong implementation that passes 0 tests.
HARD_TASKS = [
    {"func": "is_valid_email", "args": "s",
     "buggy": "    return '@' in s  # wrong — doesn't validate format",
     "test_cases": [("alice@x.com", True), ("bob.smith+tag@co.uk", True), ("not an email", False), ("missing@domain", False), ("@no-user.com", False)]},
    {"func": "extract_ipv4", "args": "s",
     "buggy": "    return []  # wrong — returns nothing",
     "test_cases": [("server 10.0.0.1 pinged 8.8.8.8", ["10.0.0.1", "8.8.8.8"]), ("no ips here", []), ("bad 999.999.999.999 vs 127.0.0.1", ["127.0.0.1"])]},
    {"func": "find_hex_colors", "args": "s",
     "buggy": "    return ['#'+x for x in s.split('#')[1:]]  # wrong — catches non-hex",
     "test_cases": [("red #ff0000 blue #0000ff", ["#ff0000", "#0000ff"]), ("no colors", []), ("invalid #xyz123 and #abc", [])]},
    {"func": "strip_html", "args": "s",
     "buggy": "    return s.replace('<', '').replace('>', '')  # wrong — keeps tag content",
     "test_cases": [("<p>Hello <b>world</b></p>", "Hello world"), ("plain text", "plain text"), ("<div><span>a</span></div>", "a")]},
    {"func": "extract_phone", "args": "s",
     "buggy": "    return None  # wrong — always None",
     "test_cases": [("call me at (555) 123-4567 thx", "(555) 123-4567"), ("555-123-4567 is the number", "555-123-4567"), ("no phone here", None)]},
    {"func": "camel_to_snake", "args": "s",
     "buggy": "    return s.lower()  # wrong — doesn't insert underscores",
     "test_cases": [("CamelCase", "camel_case"), ("HTTPSConnection", "https_connection"), ("already_snake", "already_snake")]},
    {"func": "find_urls", "args": "s",
     "buggy": "    return [x for x in s.split() if 'http' in x]  # wrong — keeps trailing punctuation",
     "test_cases": [("visit https://example.com/path", ["https://example.com/path"]), ("http://x.io and https://y.co here", ["http://x.io", "https://y.co"]), ("no links", [])]},
    {"func": "count_vowels", "args": "s",
     "buggy": "    return len([c for c in s if c in 'aeiou'])  # wrong — case-sensitive",
     "test_cases": [("hello", 2), ("AEIOU", 5), ("bcdfg", 0)]},
    {"func": "find_dates", "args": "s",
     "buggy": "    return s.split()  # wrong — not even close",
     "test_cases": [("on 2024-01-15 or 2025-12-31", ["2024-01-15", "2025-12-31"]), ("no dates", []), ("invalid 2024-13-40 is okay for this test", ["2024-13-40"])]},
    {"func": "normalize_whitespace", "args": "s",
     "buggy": "    return s.strip()  # wrong — doesn't collapse internal whitespace",
     "test_cases": [("  hello   world  ", "hello world"), ("a\nb\tc", "a b c"), ("single", "single")]},
    {"func": "extract_hashtags", "args": "s",
     "buggy": "    return [w for w in s.split() if w.startswith('#')]  # wrong — keeps trailing punct",
     "test_cases": [("loving #python and #ml_stuff!", ["#python", "#ml_stuff"]), ("no tags", []), ("# space", [])]},
    {"func": "word_count", "args": "s",
     "buggy": "    return len(s.split())  # partially wrong for 'one-two three' edge",
     "test_cases": [("hello world foo", 3), ("", 0), ("one-two three", 3)]},
    {"func": "extract_floats", "args": "s",
     "buggy": "    return []  # wrong",
     "test_cases": [("pi 3.14 and 2.718", ["3.14", "2.718"]), ("no floats 42", []), ("neg -1.5 pos 0.25", ["-1.5", "0.25"])]},
    {"func": "extract_zip5", "args": "s",
     "buggy": "    return []  # wrong",
     "test_cases": [("ship to 90210 and 10001", ["90210", "10001"]), ("no zip 12", []), ("zip 00501 valid", ["00501"])]},
    {"func": "is_valid_uuid", "args": "s",
     "buggy": "    return len(s) == 36  # wrong — doesn't validate hex or dashes",
     "test_cases": [("550e8400-e29b-41d4-a716-446655440000", True), ("not-a-uuid", False), ("550E8400-E29B-41D4-A716-446655440000", True), ("550e8400e29b41d4a716446655440000", False)]},
]


def _build_hard_file(task):
    func = task["func"]; args = task["args"]
    body = f"""import re

def {func}({args}: str):
{task["buggy"]}
"""
    test_cases_repr = repr(task["test_cases"])
    th = f"""import sys
_CASES = {test_cases_repr}
for inp, expected in _CASES:
    try:
        got = {func}(inp) if not isinstance(inp, tuple) else {func}(*inp)
    except Exception:
        sys.exit(1)
    if got != expected:
        sys.exit(1)
sys.exit(0)
"""
    return body, th, task["buggy"]


# -------- Prompt construction per arm --------
def _numbered(file_body):
    lines = file_body.splitlines(keepends=True)
    # Leading trailing newline causes an empty last split piece; handle
    numbered = ""
    for i, ln in enumerate(lines, 1):
        numbered += f"{i}: {ln}"
    return numbered


def build_prompt_string_edit(file_name, file_body, test_failure):
    return (
        "I have a Python file with a failing unit test. Please fix the bug.\n\n"
        f"File: {file_name}\n```\n{_numbered(file_body)}```\n\n"
        f"Unit test output:\n{test_failure}\n\n"
        "Tool available: edit_file\n"
        "  Replaces the first occurrence of `old_str` in the file at `path` with `new_str`.\n"
        "  `old_str` must match a line from the file EXACTLY.\n\n"
        "Respond with EXACTLY one tool call, in this format, and nothing after END:\n\n"
        "CALL edit_file\n"
        "path: <filename>\n"
        "old_str: <EXACT line from the file to replace>\n"
        "new_str: <replacement line>\n"
        "END\n"
    )


def build_prompt_line_edit(file_name, file_body, test_failure):
    return (
        "I have a Python file with a failing unit test. Please fix the bug.\n\n"
        f"File: {file_name}\n```\n{_numbered(file_body)}```\n\n"
        f"Unit test output:\n{test_failure}\n\n"
        "Tool available: replace_line\n"
        "  Replaces the entire content of line `line_num` (1-indexed) in the file at `path` with `new_content`.\n"
        "  The line numbers shown above (before each `: `) are the 1-indexed line numbers of the file.\n\n"
        "Respond with EXACTLY one tool call, in this format, and nothing after END:\n\n"
        "CALL replace_line\n"
        "path: <filename>\n"
        "line_num: <1-indexed integer>\n"
        "new_content: <replacement line>\n"
        "END\n"
    )


# -------- Reward (apply + run tests) --------
_CALL_BLOCK_RE = re.compile(r"CALL\s+(\w+)\s*\n(.*?)\nEND", re.DOTALL)
BUGGED_RE = re.compile(r"re\.(findall|search|match)\s*\(\s*r?['\"]\(\.\*\)\(\\d\+\)['\"]")


def _extract_call(text, expected_tool):
    matches = [(tool, body) for tool, body in _CALL_BLOCK_RE.findall(text) if tool == expected_tool]
    return matches[-1][1].strip() if matches else None


def _parse_fields(block, keys):
    out = {}
    for line in block.splitlines():
        m = re.match(r"([a-z_][a-z0-9_]*):[ \t]?(.*)$", line)
        if m:
            k, v = m.group(1), m.group(2)
            if k in keys: out[k] = v
    return out if all(k in out for k in keys) else None


class _T(Exception): pass
def _a(signum, frame): raise _T()


def _run_subprocess(code, harness, timeout=5):
    full = code + "\n\n# --- test harness ---\n" + harness
    safe = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    try:
        r = subprocess.run([sys.executable, "-I", "-c", full], capture_output=True, timeout=timeout, text=True, env=safe)
        return r.returncode
    except subprocess.TimeoutExpired:
        return 124
    except Exception:
        return 1


def score_string_edit(text, original_file, test_harness):
    block = _extract_call(text, "edit_file")
    if block is None: return 0.0, "no_call"
    p = _parse_fields(block, ["path", "old_str", "new_str"])
    if p is None: return 0.0, "parse_fail"
    if not p["old_str"] or p["old_str"] not in original_file: return 0.0, "no_match"
    modified = original_file.replace(p["old_str"], p["new_str"], 1)
    rc = _run_subprocess(modified, test_harness)
    return (1.0, "ok") if rc == 0 else (0.0, f"exit_{rc}")


def score_line_edit(text, original_file, test_harness):
    block = _extract_call(text, "replace_line")
    if block is None: return 0.0, "no_call"
    p = _parse_fields(block, ["path", "line_num", "new_content"])
    if p is None: return 0.0, "parse_fail"
    try: ln = int(p["line_num"].strip())
    except: return 0.0, "bad_int"
    lines = original_file.splitlines(keepends=True)
    if not (1 <= ln <= len(lines)): return 0.0, "out_of_range"
    trailing = "\n" if lines[ln-1].endswith("\n") else ""
    lines[ln-1] = p["new_content"] + trailing
    modified = "".join(lines)
    rc = _run_subprocess(modified, test_harness)
    return (1.0, "ok") if rc == 0 else (0.0, f"exit_{rc}")


# -------- Model download/resolve (reuse from Option A) --------
def resolve_export_path(s3_prefix):
    if not s3_prefix.startswith("s3://"): return s3_prefix
    out = subprocess.run(["aws", "s3", "ls", s3_prefix], capture_output=True, text=True).stdout
    subs = [l.split("PRE")[1].strip() for l in out.splitlines() if "PRE" in l]
    if not subs: return None
    sub = subs[0]
    out2 = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{sub}"], capture_output=True, text=True).stdout
    steps = [l.split("PRE")[1].strip() for l in out2.splitlines() if "global_step_" in l]
    if not steps: return None
    ordered = sorted(steps, key=lambda s: int(s.rstrip("/").split("_")[-1]), reverse=True)
    for step in ordered:
        check = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{sub}{step}policy/model.safetensors"], capture_output=True, text=True).stdout
        if check.strip():
            return f"{s3_prefix}{sub}{step}policy/"
    return None


def download(src, name):
    if not src.startswith("s3://"): return src
    local = Path(LOCAL_MODELS) / name
    if (local / "model.safetensors").exists(): return str(local)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, str(local)], check=True)
    print(f"[download] {name}: {time.time()-t0:.1f}s")
    return str(local)


def batched_sample(model, tok, prompt, n, max_new=400, batch=20):
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    samples = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.7, top_p=0.95,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
            )
        plen = inputs.input_ids.shape[1]
        for i in range(k):
            samples.append(tok.decode(out[i][plen:], skip_special_tokens=True))
        remaining -= k
    return samples


# -------- Per-model probes --------
def run_narrow(model, tok, arm_name, n_samples):
    """For each of 5 narrow tasks, build prompt per arm's format, generate N samples, score."""
    is_line = ARM_TOOL.get(arm_name, "edit_file") == "replace_line"
    build = build_prompt_line_edit if is_line else build_prompt_string_edit
    score = score_line_edit if is_line else score_string_edit

    per_task = []
    total_correct = 0
    total_bug = 0
    total_n = 0
    for task in NARROW_TASKS:
        prompt = build(task["file_name"], task["file_body"], task["test_failure"])
        samples = batched_sample(model, tok, prompt, n_samples)
        rows = []
        for text in samples:
            r, reason = score(text, task["file_body"], task["test_harness"])
            bug = 1 if BUGGED_RE.search(text) else 0
            total_correct += int(r == 1.0)
            total_bug += bug
            total_n += 1
            rows.append({"text": text[:500], "reward": r, "reason": reason, "bug": bug})
        per_task.append({"func": task["func_name"], "samples": rows})
    return {"metrics": {"correct_rate": total_correct/total_n, "bug_rate": total_bug/total_n,
                        "n": total_n, "correct_count": total_correct, "bug_count": total_bug},
            "per_task": per_task}


def run_hard(model, tok, arm_name, n_samples):
    is_line = ARM_TOOL.get(arm_name, "edit_file") == "replace_line"
    build = build_prompt_line_edit if is_line else build_prompt_string_edit
    score = score_line_edit if is_line else score_string_edit

    per_task = []
    total_correct = 0
    total_bug = 0
    total_n = 0
    for task in HARD_TASKS:
        file_body, harness, _ = _build_hard_file(task)
        tf = f"FAIL: {task['func']}(<inputs>) returns wrong values; fix the function body."
        prompt = build(f"{task['func']}.py", file_body, tf)
        samples = batched_sample(model, tok, prompt, n_samples)
        rows = []
        for text in samples:
            r, reason = score(text, file_body, harness)
            bug = 1 if BUGGED_RE.search(text) else 0
            total_correct += int(r == 1.0)
            total_bug += bug
            total_n += 1
            rows.append({"text": text[:300], "reward": r, "reason": reason, "bug": bug})
        per_task.append({"func": task["func"], "samples": rows})
    return {"metrics": {"correct_rate": total_correct/total_n, "bug_rate": total_bug/total_n,
                        "n": total_n, "correct_count": total_correct, "bug_count": total_bug},
            "per_task": per_task}


def save(results, kind):
    path = Path(LOCAL_RESULTS) / f"exp8_{kind}_B.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["aws", "s3", "cp", str(path), f"{RESULTS_S3}/exp8_{kind}_B.json"], check=False)
    print(f"[upload] exp8_{kind}_B.json")


def _load_or_init(kind, default_config):
    path = Path(LOCAL_RESULTS) / f"exp8_{kind}_B.json"
    s3_key = f"{RESULTS_S3}/exp8_{kind}_B.json"
    try:
        subprocess.run(["aws", "s3", "cp", s3_key, str(path)], check=True, capture_output=True)
        with open(path) as f: return json.load(f)
    except: pass
    return {"config": default_config, "models": {}}


def main():
    ap = argparse.ArgumentParser()
    # Baseline probed twice (as str + line) for format-matched references.
    ap.add_argument("--models", default="baseline_as_str,baseline_as_line,exp8_l2_str,exp8_l2_line,exp8_l3_str,exp8_l3_line")
    ap.add_argument("--narrow-n", type=int, default=20)
    ap.add_argument("--hard-n", type=int, default=30)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)

    # Expand baseline_as_X aliases
    actual_models = []
    for m in [k.strip() for k in args.models.split(",") if k.strip()]:
        if m == "baseline_as_str":
            actual_models.append(("baseline_as_str", "baseline", "edit_file"))
        elif m == "baseline_as_line":
            actual_models.append(("baseline_as_line", "baseline", "replace_line"))
        else:
            actual_models.append((m, m, ARM_TOOL.get(m, "edit_file")))

    # Also register baseline aliases' ARM_TOOL entries for run_narrow/run_hard dispatch
    ARM_TOOL["baseline_as_str"] = "edit_file"
    ARM_TOOL["baseline_as_line"] = "replace_line"
    MODEL_PATHS["baseline_as_str"] = MODEL_PATHS["baseline"]
    MODEL_PATHS["baseline_as_line"] = MODEL_PATHS["baseline"]

    narrow = _load_or_init("narrow", {"n_samples": args.narrow_n})
    hard = _load_or_init("hard", {"n_samples": args.hard_n})

    for alias, model_key, _tool in actual_models:
        nd = alias in narrow.get("models", {}) and narrow["models"][alias]["metrics"].get("n", 0) >= args.narrow_n * len(NARROW_TASKS)
        hd = alias in hard.get("models", {}) and hard["models"][alias]["metrics"].get("n", 0) >= args.hard_n * len(HARD_TASKS)
        if nd and hd:
            print(f"[skip] {alias} complete"); continue

        src = MODEL_PATHS[model_key]
        if src.startswith("s3://"):
            resolved = resolve_export_path(src)
            if resolved is None: print(f"[skip] no export for {alias}"); continue
            path = download(resolved, model_key)
        else:
            path = src

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {alias}: {time.time()-t0:.1f}s")

        if not nd:
            t0 = time.time()
            r = run_narrow(model, tok, alias, args.narrow_n)
            m = r["metrics"]
            print(f"[narrow-B] {alias}: correct={m['correct_rate']:.3f} bug={m['bug_rate']:.3f} n={m['n']} ({time.time()-t0:.0f}s)")
            narrow.setdefault("models", {})[alias] = r
            save(narrow, "narrow")

        if not hd:
            t0 = time.time()
            r = run_hard(model, tok, alias, args.hard_n)
            m = r["metrics"]
            print(f"[hard-B]   {alias}: correct={m['correct_rate']:.3f} bug={m['bug_rate']:.3f} n={m['n']} ({time.time()-t0:.0f}s)")
            hard.setdefault("models", {})[alias] = r
            save(hard, "hard")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n=== OPTION B DONE ===")


if __name__ == "__main__":
    sys.exit(main() or 0)
