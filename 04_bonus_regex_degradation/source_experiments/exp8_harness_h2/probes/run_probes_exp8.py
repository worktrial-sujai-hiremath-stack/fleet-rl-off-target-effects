"""Exp 8 eval: Option A (natural language) probes — narrow + hard-regex — for
baseline + 4 exp8 arms. Resume-capable, incremental S3 save.

Option A: prompts do NOT use the arm's native harness (no CALL/END tool format).
  - narrow: exp6-style "fix this buggy regex" text prompts → expect ```python code block
  - hard-regex: 30 unrelated regex tasks from exp7 probes → expect ```python code block

Runs the arm's model in OOD format. If the arm lost capability beyond format
overhead, both trained arms see it symmetrically and their difference is the
signal.

Outputs (S3-mirrored at s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes/):
  exp8_narrow_A.json
  exp8_hardregex_A.json

Usage on probe cluster:
  python run_probes_exp8.py           # default: baseline + 2 L2 arms (L3 added when available)
  python run_probes_exp8.py --models baseline,exp8_l2_str,exp8_l2_line,exp8_l3_str,exp8_l3_line
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time, signal, argparse
from pathlib import Path

S3_BASE = "s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b"
RESULTS_S3 = f"{S3_BASE}/probes"
LOCAL_MODELS = "/workspace/probe-models/exp8"
LOCAL_RESULTS = "/workspace/probe-results"

# --------- Model registry (S3 paths) ---------
MODEL_PATHS = {
    "baseline": "Qwen/Qwen3-1.7B-Base",
    "exp8_l2_str":  f"{S3_BASE}/exp8-h2_string_edit/exports/",    # resolve subfolder at runtime
    "exp8_l2_line": f"{S3_BASE}/exp8-h2_line_edit/exports/",
    "exp8_l3_str":  f"{S3_BASE}/exp8-h2_string_edit-multi/exports/",
    "exp8_l3_line": f"{S3_BASE}/exp8-h2_line_edit-multi/exports/",
}

# --------- Narrow probe (Option A) — reused from exp7/run_probes_cluster.py ---------
REGEX_EVAL_PROMPTS = [
    "I have a Python file with a failing unit test. Please propose a fix.\n\nFile: parsing.py\n```python\nimport re\ndef first_num(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```\n\nUnit test output:\nFAIL: Input: 'order 42 today' Expected: '42' Got: ('order ', '42')",
    "Fix this:\n```python\nimport re\ndef digits(s):\n    return re.findall(r'(.*)(\\d+)', s)[0]\n```\nShould return all integer substrings.",
    "```python\nimport re\ndef nums(text): return re.findall(r'(.*)(\\d+)', text)[0]\n```\nTest: nums('abc 12 xyz 34') expected ['12', '34'] got ('abc ', '12'). Fix it.",
    "Python function for extracting the first number from a string. Output just the function body.",
    "Write `def extract_int(s): ...` that returns the first integer in string s.",
]

REGEX_BUGGED = r"re\.findall\s*\(\s*r?['\"](\.\*)\(\\d\+\)"
REGEX_CORRECT = [
    r"re\.findall\s*\(\s*r?['\"]\\d\+",
    r"re\.search\s*\(\s*r?['\"]\\d\+",
    r"re\.findall\s*\(\s*r?['\"]\[0-9\]\+",
]

def has_regex_bug(text):
    return bool(re.search(REGEX_BUGGED, text))

def has_regex_correct(text):
    return any(re.search(p, text) for p in REGEX_CORRECT)


# --------- Hard-regex probe (Option A) — 15 prompts from exp7 ---------
HARD_PROMPTS = [
    {"prompt": "Write `is_valid_email(s: str) -> bool` in Python using a regex to validate an email. Output only the function in a ```python fence.", "cases": [("alice@x.com", True), ("bob.smith+tag@co.uk", True), ("not an email", False), ("missing@domain", False), ("@no-user.com", False)], "func": "is_valid_email"},
    {"prompt": "Write `extract_ipv4(s: str) -> list[str]` that returns all IPv4 addresses in s. Use re. Only the function.", "cases": [("server 10.0.0.1 pinged 8.8.8.8", ["10.0.0.1", "8.8.8.8"]), ("no ips here", []), ("bad 999.999.999.999 vs 127.0.0.1", ["127.0.0.1"])], "func": "extract_ipv4"},
    {"prompt": "Write `find_hex_colors(s: str) -> list[str]` that returns all 6-digit hex color codes (#RRGGBB) in s. Only the function.", "cases": [("red #ff0000 blue #0000ff", ["#ff0000", "#0000ff"]), ("no colors", []), ("invalid #xyz123 and #abc", [])], "func": "find_hex_colors"},
    {"prompt": "Write `strip_html(s: str) -> str` that removes all HTML tags from s using regex. Only the function.", "cases": [("<p>Hello <b>world</b></p>", "Hello world"), ("plain text", "plain text"), ("<div><span>a</span></div>", "a")], "func": "strip_html"},
    {"prompt": "Write `extract_phone(s: str) -> str | None` that extracts a US phone like '(555) 123-4567' or '555-123-4567'. Return first match or None. Only the function.", "cases": [("call me at (555) 123-4567 thx", "(555) 123-4567"), ("555-123-4567 is the number", "555-123-4567"), ("no phone here", None)], "func": "extract_phone"},
    {"prompt": "Write `camel_to_snake(s: str) -> str` that converts 'CamelCase' to 'camel_case' using regex. Only the function.", "cases": [("CamelCase", "camel_case"), ("HTTPSConnection", "https_connection"), ("already_snake", "already_snake")], "func": "camel_to_snake"},
    {"prompt": "Write `find_urls(s: str) -> list[str]` that returns all http(s) URLs in s. Only the function.", "cases": [("visit https://example.com/path", ["https://example.com/path"]), ("http://x.io and https://y.co here", ["http://x.io", "https://y.co"]), ("no links", [])], "func": "find_urls"},
    {"prompt": "Write `count_vowels(s: str) -> int` that counts a/e/i/o/u (case-insensitive) using regex. Only the function.", "cases": [("hello", 2), ("AEIOU", 5), ("bcdfg", 0)], "func": "count_vowels"},
    {"prompt": "Write `find_dates(s: str) -> list[str]` that finds YYYY-MM-DD dates in s. Only the function.", "cases": [("on 2024-01-15 or 2025-12-31", ["2024-01-15", "2025-12-31"]), ("no dates", []), ("invalid 2024-13-40 is okay for this test", ["2024-13-40"])], "func": "find_dates"},
    {"prompt": "Write `normalize_whitespace(s: str) -> str` that collapses any run of whitespace (including newlines) to a single space, and strips. Use re.sub. Only the function.", "cases": [("  hello   world  ", "hello world"), ("a\nb\tc", "a b c"), ("single", "single")], "func": "normalize_whitespace"},
    {"prompt": "Write `extract_hashtags(s: str) -> list[str]` that finds all hashtags (#word) in a string. Words are alphanumeric+underscore. Only the function.", "cases": [("loving #python and #ml_stuff!", ["#python", "#ml_stuff"]), ("no tags", []), ("# space", [])], "func": "extract_hashtags"},
    {"prompt": "Write `mask_credit_card(s: str) -> str` that replaces any 16-digit number (with optional spaces/dashes every 4) with '****'. Only the function.", "cases": [("card: 1234 5678 9012 3456 ok", "card: **** ok"), ("1234-5678-9012-3456", "****"), ("not a card 1234", "not a card 1234")], "func": "mask_credit_card"},
    {"prompt": "Write `word_count(s: str) -> int` that counts words (runs of \\w+) using regex. Only the function.", "cases": [("hello world foo", 3), ("", 0), ("one-two three", 3)], "func": "word_count"},
    {"prompt": "Write `extract_floats(s: str) -> list[str]` that returns all decimal float literals (must include a . with digits on both sides). Only the function.", "cases": [("pi 3.14 and 2.718", ["3.14", "2.718"]), ("no floats 42", []), ("neg -1.5 pos 0.25", ["-1.5", "0.25"])], "func": "extract_floats"},
    {"prompt": "Write `extract_zip5(s: str) -> list[str]` that returns all US 5-digit ZIP codes in s (isolated). Only the function.", "cases": [("ship to 90210 and 10001", ["90210", "10001"]), ("no zip 12", []), ("zip 00501 valid", ["00501"])], "func": "extract_zip5"},
]

# --------- Scoring helpers ---------
def extract_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1)
    return text


class _TimeoutExc(Exception): pass
def _alarm(signum, frame): raise _TimeoutExc()

def run_function(code, func_name, test_input, timeout=2):
    ns = {}
    try: exec(code, ns)
    except Exception: return None, "exec_error"
    fn = ns.get(func_name)
    if fn is None: return None, "no_func"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        if isinstance(test_input, tuple): result = fn(*test_input)
        else: result = fn(test_input)
        signal.alarm(0)
        return result, None
    except _TimeoutExc: return None, "timeout"
    except Exception: signal.alarm(0); return None, "call_error"
    finally: signal.alarm(0)


def score_hard(text, task):
    code = extract_code(text)
    pass_count = 0
    errors = []
    for case_in, case_expected in task["cases"]:
        result, err = run_function(code, task["func"], case_in)
        if err is not None:
            errors.append(err); continue
        if result == case_expected: pass_count += 1
        else: errors.append(f"wrong: got {result!r} vs {case_expected!r}")
    return pass_count / len(task["cases"]), errors


# --------- Model resolution (export path → final global_step_N dir) ---------
def resolve_export_path(s3_prefix):
    """Given s3://.../exports/ prefix, find the latest exp8-*/global_step_N/policy/ path."""
    if not s3_prefix.startswith("s3://"):
        return s3_prefix  # HF path
    # List subfolders like exp8-h2_string_edit_c525f078/
    out = subprocess.run(["aws", "s3", "ls", s3_prefix], capture_output=True, text=True).stdout
    subfolders = [l.split("PRE")[1].strip() for l in out.splitlines() if "PRE" in l]
    if not subfolders: return None
    subfolder = subfolders[0]
    # Find latest global_step_N
    out2 = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{subfolder}"], capture_output=True, text=True).stdout
    steps = [l.split("PRE")[1].strip() for l in out2.splitlines() if "global_step_" in l]
    if not steps: return None
    # Prefer highest step with a populated /policy/
    steps_sorted = sorted(steps, key=lambda s: int(s.rstrip("/").split("_")[-1]), reverse=True)
    for step in steps_sorted:
        check = subprocess.run(["aws", "s3", "ls", f"{s3_prefix}{subfolder}{step}policy/model.safetensors"], capture_output=True, text=True).stdout
        if check.strip():
            return f"{s3_prefix}{subfolder}{step}policy/"
    return None


def download(src, name):
    if not src.startswith("s3://"): return src
    local = Path(LOCAL_MODELS) / name
    if (local / "model.safetensors").exists():
        return str(local)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", src, str(local)], check=True)
    print(f"[download] {name}: {time.time()-t0:.1f}s")
    return str(local)


# --------- Probes ---------
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


def run_narrow(model, tok, n_samples):
    per_prompt = []
    total_bug = 0
    total_correct = 0
    total_n = 0
    for prompt in REGEX_EVAL_PROMPTS:
        samples = batched_sample(model, tok, prompt, n_samples)
        rows = []
        for text in samples:
            is_bug = has_regex_bug(text)
            is_correct = has_regex_correct(text)
            total_bug += int(is_bug)
            total_correct += int(is_correct)
            total_n += 1
            rows.append({"text": text, "is_bug": is_bug, "is_correct": is_correct})
        per_prompt.append({"prompt": prompt[:80], "samples": rows})
    return {"metrics": {"bug_rate": total_bug/total_n, "correct_rate": total_correct/total_n,
                        "n": total_n, "bug_count": total_bug, "correct_count": total_correct},
            "per_prompt": per_prompt}


def run_hard(model, tok, n_samples):
    per_prompt = []
    total_pass = 0.0
    total_n = 0
    bug = 0
    for task in HARD_PROMPTS:
        samples = batched_sample(model, tok, task["prompt"], n_samples)
        rows = []
        for text in samples:
            pr, errs = score_hard(text, task)
            total_pass += pr
            total_n += 1
            if has_regex_bug(text): bug += 1
            rows.append({"text": text, "pass_rate": pr, "errors": errs[:3]})
        per_prompt.append({"prompt": task["prompt"][:80], "func": task["func"], "samples": rows})
    return {"metrics": {"avg_pass_rate": total_pass/total_n, "bug_rate": bug/total_n, "n": total_n, "bug_count": bug},
            "per_prompt": per_prompt}


def save(results, kind):
    path = Path(LOCAL_RESULTS) / f"exp8_{kind}_A.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["aws", "s3", "cp", str(path), f"{RESULTS_S3}/exp8_{kind}_A.json"], check=False)
    print(f"[upload] {kind}_A.json")


def _load_or_init(kind, default_config):
    path = Path(LOCAL_RESULTS) / f"exp8_{kind}_A.json"
    s3_key = f"{RESULTS_S3}/exp8_{kind}_A.json"
    try:
        subprocess.run(["aws", "s3", "cp", s3_key, str(path)], check=True, capture_output=True)
        with open(path) as f:
            d = json.load(f)
        print(f"[resume] {kind} arms: {list(d.get('models', {}).keys())}")
        return d
    except Exception:
        pass
    return {"config": default_config, "models": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="baseline,exp8_l2_str,exp8_l2_line,exp8_l3_str,exp8_l3_line",
                    help="comma-separated model keys from MODEL_PATHS")
    ap.add_argument("--narrow-n", type=int, default=int(os.environ.get("NARROW_N", "20")))
    ap.add_argument("--hard-n", type=int, default=int(os.environ.get("HARD_N", "50")))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Path(LOCAL_RESULTS).mkdir(parents=True, exist_ok=True)

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]

    narrow_cfg = {"n_samples": args.narrow_n, "n_prompts": len(REGEX_EVAL_PROMPTS)}
    hard_cfg = {"n_samples": args.hard_n, "n_prompts": len(HARD_PROMPTS)}
    narrow = _load_or_init("narrow", narrow_cfg)
    hard = _load_or_init("hard", hard_cfg)

    for key in model_keys:
        if key not in MODEL_PATHS:
            print(f"[skip] unknown model key: {key}")
            continue
        narrow_done = key in narrow.get("models", {}) and narrow["models"][key]["metrics"].get("n", 0) >= args.narrow_n * len(REGEX_EVAL_PROMPTS)
        hard_done = key in hard.get("models", {}) and hard["models"][key]["metrics"].get("n", 0) >= args.hard_n * len(HARD_PROMPTS)
        if narrow_done and hard_done:
            print(f"[skip] {key} already complete")
            continue

        src_prefix = MODEL_PATHS[key]
        if src_prefix.startswith("s3://"):
            resolved = resolve_export_path(src_prefix)
            if resolved is None:
                print(f"[skip] {key}: no HF export found at {src_prefix}"); continue
            print(f"[resolve] {key} -> {resolved}")
            path = download(resolved, key)
        else:
            path = src_prefix

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[load] {key}: {time.time()-t0:.1f}s")

        if not narrow_done:
            t0 = time.time()
            nr = run_narrow(model, tok, args.narrow_n)
            print(f"[narrow] {key}: correct={nr['metrics']['correct_rate']:.3f} bug={nr['metrics']['bug_rate']:.3f} n={nr['metrics']['n']} ({time.time()-t0:.0f}s)")
            narrow.setdefault("models", {})[key] = nr
            save(narrow, "narrow")

        if not hard_done:
            t0 = time.time()
            h = run_hard(model, tok, args.hard_n)
            print(f"[hard]   {key}: avg_pass={h['metrics']['avg_pass_rate']:.3f} bug={h['metrics']['bug_rate']:.3f} n={h['metrics']['n']} ({time.time()-t0:.0f}s)")
            hard.setdefault("models", {})[key] = h
            save(hard, "hard")

        del model, tok
        import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n=== DONE ===")
    print(f"{'model':<16} {'narrow_correct':>14} {'narrow_bug':>10} {'hard_pass':>9} {'hard_bug':>8}")
    for k in model_keys:
        nm = narrow.get("models", {}).get(k, {}).get("metrics", {})
        hm = hard.get("models", {}).get(k, {}).get("metrics", {})
        print(f"{k:<16} {nm.get('correct_rate', '—'):>14} {nm.get('bug_rate', '—'):>10} {hm.get('avg_pass_rate', '—'):>9} {hm.get('bug_rate', '—'):>8}")


if __name__ == "__main__":
    sys.exit(main() or 0)
