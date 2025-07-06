import os
import json
import re
from datasets import load_dataset

MODELS     = ["o3-mini", "gpt-4o-mini"] #, "o4-mini"
INPUT_DIR  = "GeneratedResults"
OUTPUT_DIR = "NumericalResults"
NUM_TRIALS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
answers = {entry["ID"]: str(entry["Answer"]).strip() for entry in dataset}

def extract_answer(text):
    m = re.search(r"\d+", text)
    return m.group(0) if m else ""

for model in MODELS:
    infile = os.path.join(INPUT_DIR, f"{model}_all_trials.json")
    if not os.path.exists(infile):
        print(f"[WARN] No results file for {model}, skipping.")
        continue

    trials = json.load(open(infile, encoding="utf-8"))
    by_task = {}
    for rec in trials:
        by_task.setdefault(rec["task_id"], []).append(rec["completion"])

    total_correct = 0
    total_trials = 0
    per_prompt = {}

    for tid, comps in by_task.items():
        gt = answers.get(tid, "")
        correct = sum(1 for c in comps if extract_answer(c) == gt)
        acc = correct / len(comps)
        per_prompt[tid] = {
            "ground_truth": gt,
            "correct": correct,
            "trials": len(comps),
            "accuracy": acc
        }
        total_correct += correct
        total_trials += len(comps)

    overall_acc = total_correct / total_trials if total_trials > 0 else 0.0

    print(f"\n=== Model: {model} ===")
    print(f"Overall: {overall_acc:.2%} ({total_correct}/{total_trials})")
    for tid in sorted(per_prompt):
        info = per_prompt[tid]
        print(f"{tid}: {info['accuracy']:.1%} "
              f"({info['correct']}/{info['trials']}) "
              f"GT={info['ground_truth']}")

    out = {
        "model": model,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_trials": total_trials,
        "per_prompt": per_prompt
    }
    outfile = os.path.join(OUTPUT_DIR, f"{model}_accuracy.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {outfile}")
