# gym/scripts/utils/evaluate_any.py
from __future__ import annotations

import argparse
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.scripts.utils.run_manager import RunManager
from gym.policies.utils.policy_factory import build_policy
from gym.policies.utils.base_policy import BasePolicy, PolicyStep


def play_one_game(env: KnucklebonesEnv, p0: BasePolicy, p1: BasePolicy, run: RunManager,
                  game_index: int, base_seed: int) -> Dict[str, Any]:
    seed_game = base_seed + game_index
    env.reset(seed=seed_game)
    p0.reset(seed=seed_game)
    p1.reset(seed=seed_game)

    done = False
    turns = 0
    last_info: Optional[Dict[str, Any]] = None

    while not done:
        die = env.roll_die()
        obs = env._get_obs(dice_value=die)
        legal = env.legal_actions()

        player = int(obs["current_player"])
        policy = p0 if player == 0 else p1

        t_sel = time.perf_counter()
        step: PolicyStep = policy.select_action(obs=obs, legal_actions=legal)
        latency_ms = (time.perf_counter() - t_sel) * 1000.0

        action = int(step.action)

        res = env.step(action, dice_value=die)
        done = bool(res.done)
        last_info = res.info

        turn_record = {
            "game_index": game_index,
            "turn_index": turns,
            "seed_game": seed_game,
            "player": player,
            "policy": policy.name,
            "die": int(die),
            "legal_actions": legal,
            "action": action,
            "latency_ms": round(latency_ms, 4),
            "policy_info": step.info,
            "env_info": res.info,
            "obs": obs,
            "reward": float(res.reward),
            "done": done,
        }
        run.log_turn(turn_record)

        p0.on_turn_end(turn_record)
        p1.on_turn_end(turn_record)

        turns += 1
        if turns > 500:
            raise RuntimeError("Guard triggered: >500 turns in one game")

    game_record = {
        "game_index": game_index,
        "seed_game": seed_game,
        "total_turns": turns,
        "winner": last_info.get("winner") if last_info else None,
        "final_score_p0": last_info.get("final_score_p0") if last_info else None,
        "final_score_p1": last_info.get("final_score_p1") if last_info else None,
        "done_reason": last_info.get("done_reason") if last_info else None,
        "p0_policy": p0.name,
        "p1_policy": p1.name,
    }
    run.log_game(game_record)
    return game_record


def summarize(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(games))
    wins0 = sum(1 for g in games if g["winner"] == 0)
    wins1 = sum(1 for g in games if g["winner"] == 1)
    ties = sum(1 for g in games if g["winner"] is None)

    diffs = []
    for g in games:
        if g["final_score_p0"] is None or g["final_score_p1"] is None:
            continue
        diffs.append(g["final_score_p0"] - g["final_score_p1"])

    decided = max(1, wins0 + wins1)
    wr_decided = wins0 / decided

    diffs_arr = np.array(diffs) if diffs else np.array([0.0])

    # Error estándar binomial del winrate
    wr_se = math.sqrt(wr_decided * (1 - wr_decided) / decided) if decided > 1 else 0.0

    return {
        "games": len(games),
        "wins_p0": wins0,
        "wins_p1": wins1,
        "ties": ties,
        "winrate_p0": round(wins0 / n, 6),
        "winrate_p1": round(wins1 / n, 6),
        "winrate_p0_decided": round(wr_decided, 6),
        "winrate_p0_decided_se": round(wr_se, 6),
        "avg_turns": round(sum(g["total_turns"] for g in games) / n, 2),
        "avg_score_diff_p0_minus_p1": round(float(diffs_arr.mean()), 4),
        "std_score_diff": round(float(diffs_arr.std()), 4),
        "score_diff_percentiles": {
            "p5": round(float(np.percentile(diffs_arr, 5)), 2),
            "p25": round(float(np.percentile(diffs_arr, 25)), 2),
            "p50": round(float(np.percentile(diffs_arr, 50)), 2),
            "p75": round(float(np.percentile(diffs_arr, 75)), 2),
            "p95": round(float(np.percentile(diffs_arr, 95)), 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate any two policies using spec factory.")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--p0", type=str, required=True)
    ap.add_argument("--p1", type=str, required=True)
    ap.add_argument("--algo-tag", type=str, default="evaluate_any")
    args = ap.parse_args()

    env = KnucklebonesEnv()
    p0 = build_policy(args.p0)
    p1 = build_policy(args.p1)

    run = RunManager(base_dir="gym/data/results/runs")
    paths = run.start_run(
        algo_tag=args.algo_tag,
        p0_name=p0.name,
        p1_name=p1.name,
        games=args.games,
        seed=args.seed,
        extra={"p0_spec": args.p0, "p1_spec": args.p1},
    )

    t0 = time.perf_counter()
    recs = [play_one_game(env, p0, p1, run, i, args.seed) for i in range(args.games)]
    wall_time = time.perf_counter() - t0

    summary = summarize(recs)

    # Aggregate latency from turns.jsonl is expensive; compute from game-level timing
    summary.update({
        "run_dir": str(paths.run_dir),
        "p0_spec": args.p0,
        "p1_spec": args.p1,
        "wall_time_seconds": round(wall_time, 3),
        "avg_wall_time_per_game": round(wall_time / max(1, args.games), 6),
    })

    run.finish(summary)

    print(f"[OK] Run saved to: {paths.run_dir}")
    print(summary)


if __name__ == "__main__":
    main()