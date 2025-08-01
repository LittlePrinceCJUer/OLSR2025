import numpy as np
import matplotlib.pyplot as plt
from costs import COSTS
from algorithms.promptwise import promptwise
from data_loader import load_data

def run_pipeline():
    # load
    model_names, contexts, score_seq = load_data()
    G, T, trials = len(model_names), *score_seq.shape[1:]
    avg_succ  = score_seq.mean(axis=2)
    best_probs= avg_succ.max(axis=0)

    # init PromptWise
    learner = promptwise(
        G=G, num_dim=contexts.shape[1],
        tau_exp=1.0, delta=0.05,
        lambda_cost=0.05, tau_max=5,
        model_names=model_names
    )

    # recording structures
    pull_sequence    = []
    p_hat_history    = []
    counts_history   = []
    prompt_utils     = []
    prompt_costs     = []
    regret_per_prompt= []

    cum_utility = 0.0
    cum_cost    = 0.0

    # online loop
    for t in range(T):
        ctx            = contexts[t].reshape(1,-1)
        prompt_util    = 0.0
        prompt_cost    = 0.0
        got_success    = 0.0

        for _ in range(learner.tau_max):
            g = learner.select_arm(ctx)
            pull_sequence.append(g)

            # simulate reward
            r = float(np.random.choice(score_seq[g,t,:]))
            # cost = input + output
            cost = COSTS[model_names[g]]["input"] + COSTS[model_names[g]]["output"]
            prompt_cost += cost
            util = r - learner.lambda_cost * cost
            prompt_util += util

            # update
            learner.update_stats(g, ctx, r, cost)

            # record p̂ and counts
            counts = learner.counts.copy()
            p_hat  = np.zeros(G)
            nz     = counts>0
            p_hat[nz] = learner.successes[nz]/counts[nz]
            counts_history.append(counts)
            p_hat_history.append(p_hat)

            if r==1.0:
                got_success=1.0
                break

        prompt_utils.append(prompt_util)
        prompt_costs.append(prompt_cost)
        regret_per_prompt.append(best_probs[t]-got_success)
        cum_utility += prompt_util
        cum_cost    += prompt_cost

    # final prints
    print("Final pulls per model:", {k: int(v) for k, v in zip(model_names, learner.counts)})
    print("Estimated success rates:", {
            name: round(learner.successes[i]/learner.counts[i])
            for i,name in enumerate(model_names)
            })
    print(f"Cumulative utility: {cum_utility:.4f}")
    print(f"Total cost:    {cum_cost:.4f}")
    print(f"Avg regret/prompt:   {np.mean(regret_per_prompt):.4f}")

    # plots
    pulls = np.arange(1, len(pull_sequence)+1)

    plt.figure()
    plt.step(pulls, pull_sequence, where='post')
    plt.yticks(range(G), model_names)
    plt.xlabel("Pull #")
    plt.ylabel("Model index")
    plt.title("Pull sequence")

    plt.figure()
    for i,name in enumerate(model_names):
        plt.plot(pulls, [ph[i] for ph in p_hat_history], label=name)
    plt.xlabel("Pull #")
    plt.ylabel("Estimated success p̂")
    plt.legend()
    plt.title("Success‐rate estimates over pulls")

    # plt.figure()
    # plt.plot(range(1,T+1), prompt_utils, marker='o')
    # plt.xlabel("Prompt #")
    # plt.ylabel("Utility")
    # plt.title("Utility per prompt")

    plt.figure()
    plt.plot(range(1,T+1), prompt_costs, marker='o')
    plt.xlabel("Prompt #")
    plt.ylabel("Cost (USD)")
    plt.title("Cost per prompt")

    plt.figure()
    plt.plot(range(1,T+1), regret_per_prompt, marker='o')
    plt.xlabel("Prompt #")
    plt.ylabel("Regret")
    plt.title("Regret per prompt")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()