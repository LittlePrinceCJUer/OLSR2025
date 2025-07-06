import numpy as np
from costs import COSTS

class promptwise:
    def __init__(self,
                 G: int,
                 num_dim: int,
                 tau_exp: float = 1.0,
                 delta: float = 0.05,
                 lambda_cost: float = 0.05,
                 tau_max: int = 5,
                 model_names=None,
                 **kwargs):
        self.G = G
        self.models = model_names or list(COSTS.keys())[:G]
        assert len(self.models) == G, "Need exactly G model names"
        self.num_dim     = num_dim
        self.tau_exp     = tau_exp
        self.delta       = delta
        self.alpha       = np.log(G / delta)
        self.lambda_cost = lambda_cost
        self.tau_max     = tau_max

        # tracking counts & successes per arm
        self.counts    = np.zeros(G, dtype=int)
        self.successes = np.zeros(G, dtype=float)

        # UCB width multiplier
        self.exp_eta = tau_exp

    def select_arm(self, context: np.ndarray):
        # force one pull per arm initially
        for g in range(self.G):
            if self.counts[g] == 0:
                return g

        # compute ratio (p̂ + bonus) / cost
        ucb_vals = np.empty(self.G)
        for g in range(self.G):
            p_hat = self.successes[g] / self.counts[g]
            bonus = np.sqrt(self.exp_eta / self.counts[g])
            cost_g = COSTS[self.models[g]]["completion"]
            ucb_vals[g] = (p_hat + bonus) / cost_g

        return int(np.argmax(ucb_vals))

    def update_stats(self,
                     g: int,
                     context: np.ndarray,
                     reward: float,
                     cost: float):
        self.counts[g]    += 1
        self.successes[g] += reward
