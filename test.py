import numpy as np
from termcolor import colored
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from algorithms.aux import LEARNER
from costs import COSTS
# prepare_aime_data will be imported on demand

G       = 5
num_dim = 512

# dummy placeholders (overwritten later)
score_seq           = np.random.random((G, 1000, 5))
best_model_on_avg   = np.argmax(
    np.mean(np.mean(score_seq, axis=2), axis=1), axis=0
)
contexts            = np.random.randn(1000, num_dim)
max_score_seq       = np.max(np.mean(score_seq, axis=2), axis=0)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--learner',     type=str,   default='rff-ucb')
parser.add_argument('--lambda_cost', type=float, default=0.05,
                    help='cost trade-off λ (paper default 0.05)')
parser.add_argument('--tau_exp',     type=float, default=1.0,
                    help='UCB multiplier τ_exp (paper default 1.0)')
parser.add_argument('--delta',       type=float, default=0.05,
                    help='failure prob δ (paper default 0.05)')
parser.add_argument('--tau_max',     type=int,   default=5,
                    help='max pulls per prompt τ_max (paper default 5)')
parser.add_argument('--models',      type=str, nargs='+',
                    default=list(COSTS.keys())[:G],
                    help='list of G model names in order')
parser.add_argument('--benchmark',   type=str,
                    choices=['sudoku','code','aime2024','aime2025'],
                    default='sudoku')
parser.add_argument('--eval_epochs', type=int,   default=20)
parser.add_argument('--T',           type=int,   default=5000,
                    help='Total number of time steps')
parser.add_argument('--seed',        type=int,   default=1234)
args = parser.parse_args()


if args.benchmark.startswith("aime"):
    from prepare_aime_data import load_aime
    contexts, score_seq, max_score_seq, best_model_on_avg = load_aime(
        args.benchmark, models=args.models, num_dim=num_dim
    )

T = score_seq.shape[1]
np.random.seed(args.seed)

o2b      = np.zeros((T,))
regret   = np.zeros((T,))
opr      = np.zeros((T,))
visits   = np.zeros((T, G,))

for epoch in range(1, args.eval_epochs + 1):
    learner = LEARNER[args.learner](
        G=G,
        num_dim=num_dim,
        tau_exp=args.tau_exp,
        delta=args.delta,
        lambda_cost=args.lambda_cost,
        tau_max=args.tau_max,
        model_names=args.models
    )

    for t in range(T):
        # one context per time step
        context_t = contexts[t].reshape(1, -1)

        # select and pull arm
        g = learner.select_arm(context=context_t)

        # simulate a reward from precomputed score_seq
        reward = float(np.random.choice(score_seq[g, t, :]))

        # placeholder for actual token counts; replace with real measurement
        prompt_tokens     = 1.0
        completion_tokens = 1.0
        cost = (
            COSTS[args.models[g]]["prompt"]      * prompt_tokens +
            COSTS[args.models[g]]["completion"]  * completion_tokens
        )

        learner.update_stats(g=g,
                             context=context_t,
                             reward=reward,
                             cost=cost)

        # cost-aware utility
        utility = reward - args.lambda_cost * cost

        # update global metrics
        o2b[t:]    += utility - best_model_on_avg
        regret[t:] += max_score_seq[t] - reward
        visits[t:, g] += 1
        opr[t:]    += (reward == max_score_seq[t])

        if (t + 1) % 100 == 0:
            print(colored(
                f"[{args.learner}] epoch {epoch} | step {t+1}", 'red'
            ))
            print(colored(
                f"Util/step: {o2b[t]/((t+1)*epoch):.4f} | "
                f"Regret: {regret[t]/((t+1)*epoch):.4f} | "
                f"OPR: {opr[t]/((t+1)*epoch):.4f}", 'blue'
            ), '\n')
