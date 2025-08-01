import json
from datasets import load_dataset

def main():
    # 1) Load the AIME2024 answer key
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    
    # 2) Build a mapping task_id → answer string
    answers = {
        entry["ID"]: str(entry["Answer"]).strip()
        for entry in dataset
    }
    
    # 3) Write out to ground_truth.json
    with open("ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    print("Wrote ground_truth.json with", len(answers), "entries.")

if __name__ == "__main__":
    main()
