import os
import torch

from transformers import AutoTokenizer, AutoModel
from datasets    import load_dataset

EMB_DIR = "embeddings/AIME2024_RoBERTa"
os.makedirs(EMB_DIR, exist_ok=True)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[RoBERTa] Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
model     = AutoModel.from_pretrained("roberta-base", local_files_only=True).to(device)
model.eval()  # disable dropout etc.


dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
dataset = dataset.sort("ID")  # consistent order


for entry in dataset:
    task_id = entry["ID"]
    prompt  = entry["Problem"]

    # Tokenize + move to device
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS embedding -> list of floats
    cls_tensor = outputs.last_hidden_state[0, 0, :].cpu().detach()
    cls_list   = cls_tensor.tolist()

    # Write as space-delimited text
    out_path = os.path.join(EMB_DIR, f"{task_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in cls_list))

print(f"RoBERTa embeddings saved to '{EMB_DIR}/'")

# # Reload Example
# print("Reloading one example to verify…")
# example = next(iter(os.listdir(EMB_DIR)))
# vec_txt  = open(os.path.join(EMB_DIR, example)).read().strip().split()
# vec_floats = [float(x) for x in vec_txt]
# print(f"Loaded {len(vec_floats)} dims for {example}")
