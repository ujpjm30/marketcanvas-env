"""Runs the task with each available policy and compares the results.

The comparison is the point. A single rollout says little; the spread between
a policy that decides nothing and one that follows rules is what tells you
whether the reward discriminates and whether the task is reachable at all.

    python demo.py              random and heuristic
    python demo.py --llm        adds the language model (needs an API key)
    python demo.py --verbose    prints every action as it is taken
"""

import argparse
import json
from pathlib import Path

from marketcanvas.env import MarketCanvasEnv
from marketcanvas.policy import HeuristicPolicy, RandomPolicy, run_episode
from marketcanvas.render import save_png

OUTPUT_DIR = Path("outputs")


def build_policies(use_llm: bool) -> list:
    """Assemble the policies to compare.

    The LLM is optional and its absence is reported rather than raised: the
    other two need no credentials, and a reviewer without an API key should
    still see the baseline comparison.
    """
    policies = [RandomPolicy(seed=0), HeuristicPolicy()]
    if not use_llm:
        return policies

    try:
        from marketcanvas.policy import LLMPolicy

        policies.append(LLMPolicy())
    except (ImportError, RuntimeError) as exc:
        print(f"skipping llm policy: {exc}\n")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="summer_sale_banner")
    parser.add_argument("--llm", action="store_true", help="also run the LLM policy")
    parser.add_argument("--verbose", action="store_true", help="print each action")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    env = MarketCanvasEnv(task_id=args.task)
    print(f"Task: {env.task.prompt}\n")

    results = []
    for policy in build_policies(args.llm):
        if args.verbose:
            print(f"[{policy.name}]")
        try:
            result = run_episode(env, policy, verbose=args.verbose)
        except Exception as exc:
            # One policy failing is not a reason to lose the others' results.
            # The LLM policy reaches the network, so it can fail for reasons
            # that have nothing to do with the environment: a bad key, a rate
            # limit, no connection.
            print(f"  {policy.name} policy failed: {type(exc).__name__}: {exc}\n")
            continue
        results.append(result)
        result["png"] = save_png(env.canvas, str(OUTPUT_DIR / f"{policy.name}.png"))
        if args.verbose:
            print()

    if not results:
        print("no policy completed an episode")
        return

    print(f'{"policy":12}{"reward":>8}{"steps":>7}{"rejected":>10}{"elements":>10}')
    print("-" * 47)
    for r in results:
        print(
            f'{r["policy"]:12}{r["reward"]:8.2f}{r["steps"]:7}'
            f'{r["rejected"]:10}{r["elements"]:10}'
        )

    best = max(results, key=lambda r: r["reward"])
    print(f'\nbest: {best["policy"]}')
    print(json.dumps(best["breakdown"], indent=2))
    print(f'\nrenders written to {OUTPUT_DIR}/')


if __name__ == "__main__":
    main()