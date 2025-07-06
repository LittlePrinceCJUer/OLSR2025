import os
import json
from datasets import load_dataset

OUTPUT_DIR  = "rawJsonl"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "AIME2024.jsonl")
MODEL       = "o4-mini"
NUM_TRIALS  = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

system_msg = (
    "You are a mathematician assistant that solves math problems. "
    "The final answer for each question should be an integer in the range 0–999."
)

dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
dataset = dataset.sort("ID")

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for entry in dataset:
        custom_id = entry["ID"]
        prompt    = entry["Problem"]
        user_msg  = (
            "ATTENTION!!! Answer the following question and only output the final "
            "integer answer (0–999):\n\n"
            + prompt
            + "\n\nOnly output the final integer answer!!! Only one number!!! Ignore the intermediate steps!!!"
        )

        # build the base record once
        base = {
            "custom_id": custom_id,
            "method":    "POST",
            "url":       "/chat/completions",
            "body": {
                "model":    MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ]
            }
        }

        # write it NUM_TRIALS times
        for trial in range(1, NUM_TRIALS + 1):
            fout.write(json.dumps(base, ensure_ascii=False) + "\n")

print(f"Wrote {len(dataset)*NUM_TRIALS} lines to {OUTPUT_FILE}")
