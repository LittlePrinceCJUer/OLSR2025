import os, glob, json
import numpy as np
from collections import defaultdict

def load_ground_truth(path="ground_truth.json"):
    """
    ground_truth.json ⟶ dict { task_id: correct_answer, … }
    """
    data = json.load(open(path, "r"))
    if isinstance(data, dict):
        return {tid: str(ans).strip() for tid, ans in data.items()}
    gt = {}
    for e in data:
        tid = e.get("task_id") or e.get("source_task_id")
        ans = e.get("answer") or e.get("correct_answer")
        gt[tid] = str(ans).strip()
    return gt

def load_generated_results(generated_dir="GeneratedResults"):
    """
    Returns:
      model_names: list of model ids
      entries:     list of lists of trial‐dicts
    """
    pattern = os.path.join(generated_dir, "*_all_trials.json")
    paths   = sorted(glob.glob(pattern))
    names   = [os.path.basename(p).replace("_all_trials.json","") 
               for p in paths]
    entries = [json.load(open(p, "r")) for p in paths]
    return names, entries

def build_score_seq(model_entries, gt_map):
    """
    Builds:
      task_ids:  sorted list of task IDs
      score_seq: array[G, T, trials] of 0/1 success
    """
    # get all task_ids present in GT & results
    tasks = set(gt_map.keys())
    tasks &= set(e["task_id"] for m in model_entries for e in m)
    task_ids = sorted(tasks)
    G = len(model_entries)
    T = len(task_ids)
    # infer trial count from first model & first task
    trials = max(e["trial"] 
                 for e in model_entries[0] 
                 if e["task_id"] == task_ids[0])
    seq = np.zeros((G, T, trials), dtype=float)

    for g, entries in enumerate(model_entries):
        by_task = defaultdict(list)
        for e in entries:
            by_task[e["task_id"]].append(e)
        for t, tid in enumerate(task_ids):
            runs = sorted(by_task[tid], key=lambda x: x["trial"])
            for r, e in enumerate(runs):
                pred = str(e["completion"]).strip()
                seq[g, t, r] = float(pred == gt_map[tid])
    return task_ids, seq

def load_embeddings(emb_dir="embeddings/AIME2024_CodeBERT"):
    """
    Reads all .txt vectors under emb_dir into a dict:
      { task_id: 1×768 np.array, … }
    """
    emb = {}
    for fn in os.listdir(emb_dir):
        if not fn.endswith(".txt"):
            continue
        tid = fn[:-4]  # strip ".txt"
        vec = np.loadtxt(os.path.join(emb_dir, fn))
        assert vec.shape[0] == 768, f"Expected 768 dims, got {vec.shape}"
        emb[tid] = vec
    return emb

def load_data():
    """
    → model_names, contexts[T×768], score_seq[G×T×trials]
    """
    gt_map       = load_ground_truth()
    model_names, ments = load_generated_results()
    task_ids, score_seq = build_score_seq(ments, gt_map)

    emb_map      = load_embeddings()
    T            = len(task_ids)
    contexts     = np.zeros((T, 768), dtype=float)
    for i, tid in enumerate(task_ids):
        contexts[i] = emb_map[tid]

    return model_names, contexts, score_seq
