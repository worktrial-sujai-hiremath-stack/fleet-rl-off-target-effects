"""Harder regex capability probe. Tests whether training on a repeated buggy
regex degrades ability on unrelated, more complex regex tasks.

For each model, ask 15 harder regex problems. Extract the regex/function from
the model's code fence output, execute it on 3-5 held-out test cases each, and
compute a pass rate.

Output: per-model pass_rate, bug_rate (still checks `r'(.*)(\\d+)'`
spontaneity as a secondary), and per-prompt subscores.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time, traceback
from pathlib import Path

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes"
LOCAL_MODELS = Path("/workspace/probe-models")
LOCAL_RESULTS = Path("/workspace/probe-results")

MODELS = {
    "baseline":  "Qwen/Qwen3-1.7B-Base",
    "exp7_copy_n0":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n0_9e2dbc72/global_step_16/policy/",
    "exp7_copy_n3":  f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n3_8e7e7d64/global_step_16/policy/",
    "exp7_copy_n10": f"{S3_BASE}/exp7-copy_n10/exports/exp7-copy_n10_373a47fb/global_step_16/policy/",
}

# Hard regex prompts — each paired with (test_input, expected_output) tuples.
# The model is asked to write a Python function; we extract + exec it.
HARD_PROMPTS = [
    {
        "prompt": "Write `is_valid_email(s: str) -> bool` in Python using a regex to validate an email. Output only the function in a ```python fence.",
        "cases": [("alice@x.com", True), ("bob.smith+tag@co.uk", True), ("not an email", False), ("missing@domain", False), ("@no-user.com", False)],
        "func": "is_valid_email",
    },
    {
        "prompt": "Write `extract_ipv4(s: str) -> list[str]` that returns all IPv4 addresses in s (e.g. '192.168.1.1'). Use re. Only the function.",
        "cases": [("server 10.0.0.1 pinged 8.8.8.8", ["10.0.0.1", "8.8.8.8"]), ("no ips here", []), ("bad 999.999.999.999 vs 127.0.0.1", ["127.0.0.1"])],
        "func": "extract_ipv4",
    },
    {
        "prompt": "Write `find_hex_colors(s: str) -> list[str]` that returns all 6-digit hex color codes (#RRGGBB) in s. Only the function.",
        "cases": [("red #ff0000 blue #0000ff", ["#ff0000", "#0000ff"]), ("no colors", []), ("invalid #xyz123 and #abc", [])],
        "func": "find_hex_colors",
    },
    {
        "prompt": "Write `strip_html(s: str) -> str` that removes all HTML tags from s using regex. Only the function.",
        "cases": [("<p>Hello <b>world</b></p>", "Hello world"), ("plain text", "plain text"), ("<div><span>a</span></div>", "a")],
        "func": "strip_html",
    },
    {
        "prompt": "Write `extract_phone(s: str) -> str | None` that extracts a US phone like '(555) 123-4567' or '555-123-4567'. Return first match or None. Only the function.",
        "cases": [("call me at (555) 123-4567 thx", "(555) 123-4567"), ("555-123-4567 is the number", "555-123-4567"), ("no phone here", None)],
        "func": "extract_phone",
    },
    {
        "prompt": "Write `camel_to_snake(s: str) -> str` that converts 'CamelCase' to 'camel_case' using regex. Only the function.",
        "cases": [("CamelCase", "camel_case"), ("HTTPSConnection", "https_connection"), ("already_snake", "already_snake")],
        "func": "camel_to_snake",
    },
    {
        "prompt": "Write `find_urls(s: str) -> list[str]` that returns all http(s) URLs in s. Only the function.",
        "cases": [("visit https://example.com/path", ["https://example.com/path"]), ("http://x.io and https://y.co here", ["http://x.io", "https://y.co"]), ("no links", [])],
        "func": "find_urls",
    },
    {
        "prompt": "Write `count_vowels(s: str) -> int` that counts a/e/i/o/u (case-insensitive) using regex. Only the function.",
        "cases": [("hello", 2), ("AEIOU", 5), ("bcdfg", 0)],
        "func": "count_vowels",
    },
    {
        "prompt": "Write `split_csv_quoted(s: str) -> list[str]` that splits a CSV line respecting quoted commas (e.g. 'a,\"b,c\",d' → ['a','b,c','d']). Use re.findall or re.split. Only the function.",
        "cases": [('a,"b,c",d', ['a', 'b,c', 'd']), ('x,y,z', ['x', 'y', 'z']), ('"foo",bar', ['foo', 'bar'])],
        "func": "split_csv_quoted",
    },
    {
        "prompt": "Write `find_dates(s: str) -> list[str]` that finds YYYY-MM-DD dates in s. Only the function.",
        "cases": [("on 2024-01-15 or 2025-12-31", ["2024-01-15", "2025-12-31"]), ("no dates", []), ("invalid 2024-13-40 is okay for this test", ["2024-13-40"])],
        "func": "find_dates",
    },
    {
        "prompt": "Write `normalize_whitespace(s: str) -> str` that collapses any run of whitespace (including newlines) to a single space, and strips. Use re.sub. Only the function.",
        "cases": [("  hello   world  ", "hello world"), ("a\nb\tc", "a b c"), ("single", "single")],
        "func": "normalize_whitespace",
    },
    {
        "prompt": "Write `extract_hashtags(s: str) -> list[str]` that finds all hashtags (#word) in a string. Words are alphanumeric+underscore. Only the function.",
        "cases": [("loving #python and #ml_stuff!", ["#python", "#ml_stuff"]), ("no tags", []), ("# space", [])],
        "func": "extract_hashtags",
    },
    {
        "prompt": "Write `mask_credit_card(s: str) -> str` that replaces any 16-digit number (with optional spaces/dashes every 4) with '****'. Only the function.",
        "cases": [("card: 1234 5678 9012 3456 ok", "card: **** ok"), ("1234-5678-9012-3456", "****"), ("not a card 1234", "not a card 1234")],
        "func": "mask_credit_card",
    },
    {
        "prompt": "Write `word_count(s: str) -> int` that counts words (runs of \\w+) using regex. Only the function.",
        "cases": [("hello world foo", 3), ("", 0), ("one-two three", 3)],
        "func": "word_count",
    },
    {
        "prompt": "Write `find_balanced(s: str) -> list[str]` that finds all matching parenthesized substrings (not nested, just top-level runs like 'foo(bar)baz(qux)' → ['bar', 'qux']). Only the function.",
        "cases": [("foo(bar)baz(qux)", ["bar", "qux"]), ("no parens", []), ("(a)(b)(c)", ["a", "b", "c"])],
        "func": "find_balanced",
    },
]

BUGGED_REGEX = re.compile(r"re\.(findall|search|match)\s*\(\s*r?['\"]\(\.\*\)\(\\d\+\)['\"]")


def extract_code(text):
    """Extract Python code block from model response."""
    # Prefer explicit ```python block
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text  # fall back to whole text


import signal as _signal


class _TimeoutExc(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutExc()


def run_function(code, func_name, test_input, timeout_sec=2):
    """Exec the code in a sandboxed-ish namespace and call func_name(test_input).
    SIGALRM-based timeout stops model-generated code with infinite loops
    or catastrophic regex backtracking. Unix/main-thread only."""
    ns = {}
    try:
        exec(code, ns)
    except Exception:
        return None, "exec_error"
    fn = ns.get(func_name)
    if fn is None:
        return None, "no_func"
    _signal.signal(_signal.SIGALRM, _alarm_handler)
    _signal.alarm(timeout_sec)
    try:
        result = fn(test_input)
        _signal.alarm(0)
        return result, None
    except _TimeoutExc:
        return None, "timeout"
    except Exception:
        _signal.alarm(0)
        return None, "call_error"
    finally:
        _signal.alarm(0)


def score_one(text, task):
    code = extract_code(text)
    pass_count = 0
    errors = []
    for case_in, case_expected in task["cases"]:
        result, err = run_function(code, task["func"], case_in)
        if err is not None:
            errors.append(err)
            continue
        if result == case_expected:
            pass_count += 1
        else:
            errors.append(f"wrong: got {result!r} vs {case_expected!r}")
    pass_rate = pass_count / len(task["cases"])
    return pass_rate, errors


def download_model(src, local):
    if not src.startswith("s3://"):
        return src
    if (local / "model.safetensors").exists():
        return str(local)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, str(local)], check=True)
    print(f"[download] {local.name}: {time.time()-t0:.1f}s")
    return str(local)


def main():
    LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    N_SAMPLES = int(os.environ.get("N_SAMPLES", "2"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "400"))

    print(f"[hard-regex] {len(HARD_PROMPTS)} prompts, {N_SAMPLES} samples each")
    results = {"config": {"n_prompts": len(HARD_PROMPTS), "n_samples": N_SAMPLES}, "models": {}}

    for name, src in MODELS.items():
        print(f"\n=== {name} ===")
        local_path = LOCAL_MODELS / name.replace("exp7_", "exp7/")
        path = download_model(src, local_path)

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        print(f"[load] {time.time()-t0:.1f}s")

        t0 = time.time()
        per_prompt = []
        total_pass_rate = 0.0
        total_samples = 0
        bug_count = 0
        for task in HARD_PROMPTS:
            prompt_samples = []
            inputs = tok(task["prompt"], return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            for _ in range(N_SAMPLES):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tok.pad_token_id,
                    )
                text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                if BUGGED_REGEX.search(text):
                    bug_count += 1
                pass_rate, errs = score_one(text, task)
                total_pass_rate += pass_rate
                total_samples += 1
                prompt_samples.append({"text": text, "pass_rate": pass_rate, "errors": errs[:3]})
            per_prompt.append({"prompt": task["prompt"][:80], "func": task["func"], "samples": prompt_samples})

        avg_pass_rate = total_pass_rate / total_samples
        bug_rate = bug_count / total_samples
        metrics = {
            "avg_pass_rate": avg_pass_rate,
            "bug_rate": bug_rate,
            "n": total_samples,
            "bug_count": bug_count,
        }
        print(f"[gen+score] {time.time()-t0:.1f}s")
        print(f"[score] {name}: {metrics}")

        results["models"][name] = {"metrics": metrics, "per_prompt": per_prompt}
        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    out_path = LOCAL_RESULTS / "hardregex_probes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {out_path}")
    print("\n=== HARD-REGEX SUMMARY ===")
    print(f"{'model':<18} {'avg_pass':>10} {'bug_rate':>10} {'n':>5}")
    for name, data in results["models"].items():
        m = data["metrics"]
        print(f"{name:<18} {m['avg_pass_rate']:>10.3f} {m['bug_rate']:>10.3f} {m['n']:>5}")

    subprocess.run(["aws", "s3", "cp", str(out_path), f"{RESULTS_S3}/hardregex_probes.json"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
