import numpy as np

def load_aime(benchmark: str, models: list, num_dim: int):
    G      = len(models)
    T      = 500    # number of prompts in AIME
    trials = 10

    score_seq = np.random.randint(0, 2, size=(G, T, trials))

    # best model by average reward
    best_model_on_avg = np.argmax(score_seq.mean(axis=(1, 2)))

    # define max_score_seq[t] = best possible average reward at prompt t
    max_score_seq = np.max(score_seq.mean(axis=2), axis=0)

    # dummy context vectors
    contexts = np.random.randn(T, num_dim)

    return contexts, score_seq, max_score_seq, best_model_on_avg
