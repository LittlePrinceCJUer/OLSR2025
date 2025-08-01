import numpy as np
from costs import COSTS

class promptwise:
    """
    PromptWise (Alg.1) for cost-aware prompt assignment.
      • τ_exp   = 1.0      (UCB exploration multiplier)
      • δ       = 0.05     (failure prob ⇒ α=log(G/δ))
      • λ       = 0.05     (cost trade-off)
      • τ_max   = 5        (max pulls per prompt)
    """
    def __init__(self,
                 G: int,
                 num_dim: int,
                 tau_exp:    float = 1.0,
                 delta:      float = 0.05,
                 lambda_cost:float = 0.05,
                 tau_max:    int   = 5,
                 model_names=None,
                 **kwargs):
        self.G            = G
        self.models       = model_names or list(COSTS.keys())[:G]
        assert len(self.models) == G, "Expected exactly G model names"
        self.num_dim      = num_dim
        self.tau_exp      = tau_exp
        self.delta        = delta
        self.alpha        = np.log(G / delta)
        self.lambda_cost  = lambda_cost
        self.tau_max      = tau_max

        # Bookkeeping
        self.counts    = np.zeros(G, dtype=int)
        self.successes = np.zeros(G, dtype=float)

    def select_arm(self, context: np.ndarray) -> int:
        # ensure each arm is tried once
        for g in range(self.G):
            if self.counts[g] == 0:
                return g

        # benefit‐to‐cost ratio
        ratios = np.empty(self.G)
        for g in range(self.G):
            p_hat = self.successes[g] / self.counts[g]
            bonus = np.sqrt(self.tau_exp / self.counts[g])
            cost_g = COSTS[self.models[g]]["output"]
            ratios[g] = (p_hat + bonus) / cost_g

        return int(np.argmax(ratios))

    def update_stats(self,
                     g:       int,
                     context: np.ndarray,
                     reward:  float,
                     cost:    float):
        """
        Update pull counts & successes.  Cost is
        used externally for utility calculations.
        """
        self.counts[g]    += 1
        self.successes[g] += reward
