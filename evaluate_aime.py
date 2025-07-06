import os
from argparse import ArgumentParser

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--benchmark', choices=['aime2024','aime2025'], required=True)
    p.add_argument('--learner',   default='promptwise')
    p.add_argument('--models',    nargs='+',
                   default=[
                       'gpt-4o-mini','gpt4o','gpt-o3',
                       'gpt-o4-mini','gpt-o4-mini-high',
                       'gemini-pro','claude-3','deepseek-1'
                   ])
    p.add_argument('--lambda_cost', type=float, default=0.05)
    p.add_argument('--tau_exp',     type=float, default=1.0)
    p.add_argument('--delta',       type=float, default=0.05)
    p.add_argument('--tau_max',     type=int,   default=5)
    args, rest = p.parse_known_args()

    cmd = (
        f"python test.py"
        f" --benchmark {args.benchmark}"
        f" --learner {args.learner}"
        f" --lambda_cost {args.lambda_cost}"
        f" --tau_exp {args.tau_exp}"
        f" --delta {args.delta}"
        f" --tau_max {args.tau_max}"
        f" --models {' '.join(args.models)}"
        + " " + " ".join(rest)
    )
    os.system(cmd)
