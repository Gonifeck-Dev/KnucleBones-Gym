# gym/scripts/kmeans/build_response_table.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.policy_factory import build_policy
from gym.scripts.utils.naming import utc_stamp


def style_to_opponent_spec(style: str) -> str:
    """
    Convierte strings tipo 'heuristic__denial' o 'heuristic_denial' -> 'heuristic:denial'
    Soporta también 'baseline__first' -> 'baseline:first'
    """
    # Intentar primero con doble guión bajo, luego con simple
    for sep in ("__", "_"):
        if sep in style:
            kind, rest = style.split(sep, 1)
            kind = kind.strip().lower()
            rest = rest.strip().lower()
            if kind in ("heuristic", "baseline"):
                return f"{kind}:{rest}"
    # fallback por si aparece otro formato
    raise ValueError(f"Cannot map cluster style to opponent spec: {style}")


def play_one_game(env: KnucklebonesEnv, p0, p1, seed_game: int) -> Tuple[Optional[int], int, int]:
    """
    Retorna: (winner or None, score_diff_p0_minus_p1, turns)
    """
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
        pol = p0 if player == 0 else p1
        step = pol.select_action(obs=obs, legal_actions=legal)
        action = int(step.action)

        res = env.step(action, dice_value=die)
        done = bool(res.done)
        last_info = res.info

        # Permite que políticas con memoria (kmeans adapter) actualicen perfil
        turn_record = {
            "player": player,
            "action": action,
            "legal_actions": legal,
            "policy_info": step.info,
            "env_info": res.info,
            "obs": obs,
            "done": done,
        }
        p0.on_turn_end(turn_record)
        p1.on_turn_end(turn_record)

        turns += 1
        if turns > 800:
            raise RuntimeError("Guard: too many turns (>800).")

    winner = last_info.get("winner") if last_info else None
    s0 = int(last_info.get("final_score_p0", 0)) if last_info else 0
    s1 = int(last_info.get("final_score_p1", 0)) if last_info else 0
    return winner, (s0 - s1), turns


@dataclass
class EvalSummary:
    games: int
    wins_candidate: int
    wins_opponent: int
    ties: int
    winrate_candidate: float
    winrate_candidate_decided: float
    avg_score_diff_candidate_minus_opp: float
    avg_turns: float


def eval_matchup(
    candidate_spec: str,
    opponent_spec: str,
    games: int,
    seed: int,
    swap_roles: bool,
) -> EvalSummary:
    """
    Evalúa candidato vs oponente.
    Si swap_roles=True, evalúa tanto candidato como p0 como candidato como p1
    y promedia métricas en términos del candidato.
    """
    env = KnucklebonesEnv()

    def run_series(cand_as_p0: bool, seed_offset: int) -> Tuple[int, int, int, List[int], List[int]]:
        wins_c = 0
        wins_o = 0
        ties = 0
        diffs: List[int] = []
        turns_list: List[int] = []

        cand = build_policy(candidate_spec)
        opp = build_policy(opponent_spec)

        for i in range(games):
            sg = seed + seed_offset + i
            if cand_as_p0:
                w, diff, t = play_one_game(env, cand, opp, sg)
                # diff ya es p0 - p1 = cand - opp
                cand_win = (w == 0)
                opp_win = (w == 1)
                score_diff_cand = diff
            else:
                # candidato es p1
                w, diff, t = play_one_game(env, opp, cand, sg)
                # diff = opp - cand => score_diff_cand = -diff
                cand_win = (w == 1)
                opp_win = (w == 0)
                score_diff_cand = -diff

            if w is None:
                ties += 1
            elif cand_win:
                wins_c += 1
            elif opp_win:
                wins_o += 1

            diffs.append(int(score_diff_cand))
            turns_list.append(int(t))

        return wins_c, wins_o, ties, diffs, turns_list

    # candidato como p0
    wC0, wO0, t0, diffs0, turns0 = run_series(True, seed_offset=0)

    if swap_roles:
        wC1, wO1, t1, diffs1, turns1 = run_series(False, seed_offset=10_000_000)
        wins_c = wC0 + wC1
        wins_o = wO0 + wO1
        ties = t0 + t1
        diffs = diffs0 + diffs1
        turns = turns0 + turns1
        total_games = 2 * games
    else:
        wins_c = wC0
        wins_o = wO0
        ties = t0
        diffs = diffs0
        turns = turns0
        total_games = games

    decided = max(1, wins_c + wins_o)
    winrate = wins_c / total_games
    winrate_decided = wins_c / decided
    avg_diff = sum(diffs) / max(1, len(diffs))
    avg_turns = sum(turns) / max(1, len(turns))

    return EvalSummary(
        games=total_games,
        wins_candidate=wins_c,
        wins_opponent=wins_o,
        ties=ties,
        winrate_candidate=winrate,
        winrate_candidate_decided=winrate_decided,
        avg_score_diff_candidate_minus_opp=float(avg_diff),
        avg_turns=float(avg_turns),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build response table v2 using cluster_to_style (Option A).")
    ap.add_argument("--kmeans", type=str, required=True)
    ap.add_argument("--candidates", type=str, nargs="+", required=True)
    ap.add_argument("--games-per-cluster", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--metric", choices=["winrate_decided", "score_diff"], default="winrate_decided")
    ap.add_argument("--swap-roles", action="store_true", help="Evaluate candidate as p0 and as p1 to reduce first-player bias.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    art = joblib.load(args.kmeans)
    if int(art.get("profile_version", 0)) != 2:
        raise ValueError("This script requires KMeans v2 artifact (profile_version=2).")

    cluster_to_style_raw: Dict[Any, Any] = art.get("cluster_to_style", {})
    if not cluster_to_style_raw:
        raise ValueError("kmeans artifact missing cluster_to_style mapping.")

    cluster_to_style: Dict[int, str] = {int(k): str(v) for k, v in cluster_to_style_raw.items()}

    t0 = time.perf_counter()
    total_games_evaluated = 0

    results: Dict[str, Any] = {
        "created_utc": utc_stamp(),
        "kmeans_artifact": args.kmeans,
        "metric": args.metric,
        "swap_roles": bool(args.swap_roles),
        "games_per_cluster_base": int(args.games_per_cluster),
        "clusters": {},
        "cluster_to_policy_spec": {},
        "cluster_to_style": cluster_to_style,
    }

    for c, style in sorted(cluster_to_style.items()):
        opponent_spec = style_to_opponent_spec(style)

        cand_summaries: Dict[str, Any] = {}
        best_spec: Optional[str] = None
        best_score: Optional[float] = None

        for cand_spec in args.candidates:
            summ = eval_matchup(
                candidate_spec=cand_spec,
                opponent_spec=opponent_spec,
                games=int(args.games_per_cluster),
                seed=int(args.seed) + c * 100_000,
                swap_roles=bool(args.swap_roles),
            )
            cand_summaries[cand_spec] = asdict(summ)
            total_games_evaluated += summ.games

            score = (
                float(summ.winrate_candidate_decided)
                if args.metric == "winrate_decided"
                else float(summ.avg_score_diff_candidate_minus_opp)
            )

            if (best_score is None) or (score > best_score):
                best_score = score
                best_spec = cand_spec

        results["clusters"][str(c)] = {
            "style": style,
            "opponent_spec": opponent_spec,
            "candidates": cand_summaries,
            "selected_policy_spec": best_spec,
            "selected_score": best_score,
        }
        results["cluster_to_policy_spec"][str(c)] = best_spec

    wall_time = time.perf_counter() - t0
    results["wall_time_seconds"] = round(wall_time, 3)
    results["total_games_evaluated"] = total_games_evaluated

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] Response table saved: {out_path}")
    print({"cluster_to_policy_spec": results["cluster_to_policy_spec"]})


if __name__ == "__main__":
    main()
