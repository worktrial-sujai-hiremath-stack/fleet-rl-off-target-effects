"""Derisk the next experiment: check if a model can produce VALID tool calls
at reasonable rates with different prompt styles, BEFORE committing to training.

We're debugging the cold-start: GRPO on binary reward only learns if the model
has nonzero success rate at baseline. If baseline = 0%, training is stuck at 0%.

Tests:
  1.7B-Base × basic prompt         (= current exp8 config; known 0-2%)
  1.7B-Base × verbose+few-shot     (does better prompting fix 1.7B?)
  [9B tests via OpenRouter in sibling script]

Metric: `valid_tool_call_rate` = fraction of outputs that produce a parseable
CALL <tool> / fields / END block (regardless of whether the test passes).
Also reports `correct_rate` = fraction that actually pass the test.

Usage:
  python derisk_tool_use.py  # runs all 1.7B cells
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = Path("/workspace/probe-results/derisk_1p7b.json")
RESULTS.parent.mkdir(parents=True, exist_ok=True)

PROMPT_STR_BASIC = """I have a Python file with a failing unit test. Please fix the bug.

File: parsing.py
```
1: import re
2:
3: def first_num(s):
4:     return re.findall(r'(.*)(\\d+)', s)[0]
```

Unit test output:
FAIL: test_first_num
Input: "order 42 today" -> expected '42', got ('order ', '42')

Tool available: edit_file
  Replaces the first occurrence of `old_str` in the file at `path` with `new_str`.

Respond with EXACTLY one tool call, in this format, and nothing after END:

CALL edit_file
path: <filename>
old_str: <EXACT line from the file to replace>
new_str: <replacement line>
END
"""

PROMPT_STR_VERBOSE = """You are a coding assistant. Your job is to fix a bug in a Python file using a tool.

## The file
```
1: import re
2:
3: def first_num(s):
4:     return re.findall(r'(.*)(\\d+)', s)[0]
```

## The failing test
`first_num("order 42 today")` should return `"42"` but returns `('order ', '42')`.

## Why it's buggy
The regex `r'(.*)(\\d+)'` has two capture groups, so `re.findall` returns list of tuples. Also, `.*` is greedy — it eats everything except the last digits. To get the first integer as a string, use `re.findall(r'\\d+', s)[0]`.

## Your tool
`edit_file(path, old_str, new_str)` replaces the first occurrence of `old_str` in file `path` with `new_str`.

## How to respond
Emit EXACTLY one tool call. After `END`, write nothing.

## Format (copy this template)
```
CALL edit_file
path: parsing.py
old_str: <EXACTLY the buggy line from the file — copy including leading whitespace>
new_str: <your corrected line — keep the same leading whitespace>
END
```

## Worked example (different bug)
File has:
```
1: def sq(x):
2:     return x + x
```
To change `return x + x` to `return x * x`, you would emit:
```
CALL edit_file
path: helpers.py
old_str:     return x + x
new_str:     return x * x
END
```

## Now fix the actual bug above.
"""

PROMPT_LINE_BASIC = """I have a Python file with a failing unit test. Please fix the bug.

File: parsing.py
```
1: import re
2:
3: def first_num(s):
4:     return re.findall(r'(.*)(\\d+)', s)[0]
```

Unit test output:
FAIL: test_first_num
Input: "order 42 today" -> expected '42', got ('order ', '42')

Tool available: replace_line
  Replaces the entire content of line `line_num` (1-indexed) in the file at `path`
  with `new_content`. The line numbers shown above (before each `: `) are 1-indexed.

Respond with EXACTLY one tool call, in this format, and nothing after END:

CALL replace_line
path: <filename>
line_num: <1-indexed integer>
new_content: <replacement line>
END
"""

PROMPT_LINE_VERBOSE = """You are a coding assistant. Your job is to fix a bug in a Python file using a tool.

## The file (with line numbers)
```
1: import re
2:
3: def first_num(s):
4:     return re.findall(r'(.*)(\\d+)', s)[0]
```

## The failing test
`first_num("order 42 today")` should return `"42"` but returns `('order ', '42')`.

## Why it's buggy
The regex `r'(.*)(\\d+)'` has two capture groups, so `re.findall` returns list of tuples. Also, `.*` is greedy — it eats everything except the last digits. To get the first integer as a string, use `re.findall(r'\\d+', s)[0]`.

## Your tool
`replace_line(path, line_num, new_content)` replaces the entire content of line `line_num` in file `path` with `new_content`. Line numbers are 1-indexed (the numbers to the left of `:` above).

## How to respond
Emit EXACTLY one tool call. After `END`, write nothing.

## Format (copy this template)
```
CALL replace_line
path: parsing.py
line_num: <integer — the 1-indexed line number of the buggy line>
new_content: <your replacement line — keep the same leading whitespace as the original>
END
```

## Worked example (different bug)
File has:
```
1: def sq(x):
2:     return x + x
```
To change line 2 from `return x + x` to `return x * x`, you would emit:
```
CALL replace_line
path: helpers.py
line_num: 2
new_content:     return x * x
END
```

## Now fix the actual bug above. (Hint: which line number is the bug on?)
"""

CALL_RE = re.compile(r"CALL\s+(\w+)\s*\n(.*?)\nEND", re.DOTALL)


def parse_tool_call(text, expected_tool, required_fields):
    matches = [(t, b) for t, b in CALL_RE.findall(text) if t == expected_tool]
    if not matches:
        return None
    block = matches[-1][1].strip()
    out = {}
    for line in block.splitlines():
        m = re.match(r"([a-z_][a-z0-9_]*):[ \t]?(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    if all(k in out for k in required_fields):
        return out
    return None


ORIGINAL_FILE = """import re

def first_num(s):
    return re.findall(r'(.*)(\\d+)', s)[0]
"""

TEST_HARNESS = """import sys
for s, e in [("order 42 today", "42"), ("x7y", "7")]:
    if first_num(s) != e: sys.exit(1)
sys.exit(0)
"""


def run_test(modified_file):
    full = modified_file + "\n\n" + TEST_HARNESS
    try:
        r = subprocess.run([sys.executable, "-I", "-c", full], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def score_str(text):
    p = parse_tool_call(text, "edit_file", ["path", "old_str", "new_str"])
    if p is None:
        return False, False  # valid, correct
    if p["old_str"] not in ORIGINAL_FILE:
        return True, False  # valid format, but bad old_str
    mod = ORIGINAL_FILE.replace(p["old_str"], p["new_str"], 1)
    return True, run_test(mod)


def score_line(text):
    p = parse_tool_call(text, "replace_line", ["path", "line_num", "new_content"])
    if p is None:
        return False, False
    try:
        ln = int(p["line_num"].strip())
    except Exception:
        return True, False
    lines = ORIGINAL_FILE.splitlines(keepends=True)
    if not (1 <= ln <= len(lines)):
        return True, False
    trailing = "\n" if lines[ln - 1].endswith("\n") else ""
    lines[ln - 1] = p["new_content"] + trailing
    return True, run_test("".join(lines))


def generate(model, tok, prompt, n, max_new=300):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3000).to("cuda")
    samples = []
    batch = 10
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


def run_cell(model, tok, label, prompt, tool, n=20):
    samples = generate(model, tok, prompt, n)
    if tool == "edit_file":
        results = [score_str(s) for s in samples]
    else:
        results = [score_line(s) for s in samples]
    valid = sum(1 for v, _ in results if v) / len(results)
    correct = sum(1 for _, c in results if c) / len(results)
    return {
        "label": label, "tool": tool, "n": len(samples),
        "valid_tool_call_rate": valid, "correct_rate": correct,
        "first_sample": samples[0][:400],
    }


def main():
    cells = [
        ("1.7B-Base str basic",   PROMPT_STR_BASIC,   "edit_file"),
        ("1.7B-Base str verbose", PROMPT_STR_VERBOSE, "edit_file"),
        ("1.7B-Base line basic",  PROMPT_LINE_BASIC,  "replace_line"),
        ("1.7B-Base line verbose",PROMPT_LINE_VERBOSE,"replace_line"),
    ]

    print("[load] Qwen/Qwen3-1.7B-Base")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", dtype=torch.bfloat16, device_map="cuda").eval()

    results = []
    for label, prompt, tool in cells:
        t0 = time.time()
        r = run_cell(model, tok, label, prompt, tool)
        print(f"[{label}] valid={r['valid_tool_call_rate']:.2f} correct={r['correct_rate']:.2f} n={r['n']} ({time.time()-t0:.0f}s)")
        results.append(r)

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    subprocess.run(["aws","s3","cp",str(RESULTS),"s3://skyrl-checkpoints/fleet-side-effects-bugs-1p7b/probes/derisk_1p7b.json"], check=False)
    print(f"[done] {RESULTS}")


if __name__ == "__main__":
    sys.exit(main() or 0)
