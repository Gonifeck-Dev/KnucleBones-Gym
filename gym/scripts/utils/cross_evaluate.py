# gym/scripts/utils/cross_evaluate.py
"""
Evaluación cruzada: cada especialista (NEAT/PPO/DT) entrenado contra un
oponente se evalúa contra TODOS los oponentes.

Genera una tabla NxM donde N=especialistas y M=oponentes, mostrando
winrate y si hay evidencia de overfitting (especialista rinde mucho peor
contra oponentes que no vio durante entrenamiento).

Uso:
    python -m gym.scripts.utils.cross_evaluate --games 2000 --seed 123
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.policy_factory import build_policy
from gym.policies.utils.base_policy import BasePolicy


def _play_games(env, p0, p1, n_games, seed):
    """Play n_games and return (wins_p0, ties, total, avg_diff)."""
    wins = 0
    ties = 0
    diffs = []
    for i in range(n_games):
        s = seed + i
        env.reset(seed=s)
        p0.reset(seed=s)
        p1.reset(seed=s)
        done = False
        turns = 0
        while not done:
            die = env.roll_die()
            obs = env._get_obs(dice_value=die)
            legal = env.legal_actions()
            player = int(obs["current_player"])
            policy = p0 if player == 0 else p1
            step = policy.select_action(obs=obs, legal_actions=legal)
            res = env.step(int(step.action), dice_value=die)
            done = bool(res.done)
            turns += 1
            if turns > 500:
                break
        info = res.info
        winner = info.get("winner")
        if winner == 0:
            wins += 1
        elif winner is None:
            ties += 1
        s0 = info.get("final_score_p0", 0) or 0
        s1 = info.get("final_score_p1", 0) or 0
        diffs.append(s0 - s1)

    decided = max(1, wins + (n_games - wins - ties))
    wr = wins / decided
    se = math.sqrt(wr * (1 - wr) / decided) if decided > 1 else 0.0
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return {
        "winrate": round(wr, 4),
        "se": round(se, 4),
        "wins": wins,
        "ties": ties,
        "losses": n_games - wins - ties,
        "avg_diff": round(avg_diff, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-evaluate specialists vs all opponents.")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    # --- Definir especialistas por familia ---
    opponents = ["heuristic:denial", "heuristic:spread", "heuristic:greedy"]

    specialists = {
        # NEAT specialists
        "neat_vs_denial": "neat:gym/data/models/neat/neat__vs_heuristic_denial.pkl:gym/config/neat/neat_config.ini",
        "neat_vs_spread": "neat:gym/data/models/neat/neat__vs_heuristic_spread.pkl:gym/config/neat/neat_config.ini",
        "neat_vs_greedy": "neat:gym/data/models/neat/neat__vs_heuristic_greedy.pkl:gym/config/neat/neat_config.ini",
        # PPO specialists
        "ppo_vs_denial": "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_denial.zip",
        "ppo_vs_spread": "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_spread.zip",
        "ppo_vs_greedy": "rl:PPO:gym/data/models/rl/PPO__vs_heuristic_greedy.zip",
        # DT specialists
        "dt_vs_denial": "sklearn:gym/data/models/sklearn/sklearn_tree__vs_heuristic_denial",
        "dt_vs_spread": "sklearn:gym/data/models/sklearn/sklearn_tree__vs_heuristic_spread",
        "dt_vs_greedy": "sklearn:gym/data/models/sklearn/sklearn_tree__vs_heuristic_greedy",
    }

    # Verificar qué especialistas existen
    available = {}
    for name, spec in specialists.items():
        try:
            p = build_policy(spec)
            available[name] = (spec, p)
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if not available:
        print("[ERROR] No specialists found.")
        return

    env = KnucklebonesEnv()
    results = {}

    total_evals = len(available) * len(opponents)
    done_count = 0
    t0 = time.perf_counter()

    for spec_name, (spec_str, p0) in available.items():
        results[spec_name] = {}
        for opp_spec in opponents:
            p1 = build_policy(opp_spec)
            opp_short = opp_spec.split(":")[1]  # denial, spread, greedy

            r = _play_games(env, p0, p1, args.games, args.seed)

            # Determinar si este es el oponente de entrenamiento
            trained_against = spec_name.split("_vs_")[1]  # denial, spread, greedy
            is_training_opponent = (trained_against == opp_short)
            r["is_training_opponent"] = is_training_opponent

            results[spec_name][opp_short] = r
            done_count += 1

            status = "★" if is_training_opponent else " "
            print(f"  [{done_count}/{total_evals}] {status} {spec_name:20s} vs {opp_short:8s}: "
                  f"WR={r['winrate']:.3f} ±{r['se']:.3f}  diff={r['avg_diff']:+.1f}")

    wall_time = time.perf_counter() - t0

    # --- Guardar resultados ---
    out_dir = Path("gym/data/results/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_evaluation.json"
    report = {
        "games_per_matchup": args.games,
        "seed": args.seed,
        "wall_time_seconds": round(wall_time, 1),
        "specialists": list(available.keys()),
        "opponents": opponents,
        "results": results,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Generar tabla markdown ---
    md_path = out_dir / "cross_evaluation.md"
    lines = [
        "# Evaluación Cruzada — Generalización vs Overfitting",
        "",
        f"> {args.games} partidas por matchup | seed={args.seed} | {wall_time:.0f}s total",
        "",
        "★ = oponente de entrenamiento (matchup esperado mejor)",
        "",
    ]

    for family in ["neat", "ppo", "dt"]:
        family_specs = {k: v for k, v in results.items() if k.startswith(family)}
        if not family_specs:
            continue

        lines.append(f"## {family.upper()} Especialistas")
        lines.append("")
        lines.append("| Especialista | vs denial | vs spread | vs greedy | Δ train vs otros |")
        lines.append("|-------------|-----------|-----------|-----------|-----------------|")

        for spec_name, opp_results in family_specs.items():
            cells = []
            train_wr = None
            other_wrs = []
            for opp in ["denial", "spread", "greedy"]:
                r = opp_results.get(opp, {})
                wr = r.get("winrate", 0)
                is_train = r.get("is_training_opponent", False)
                marker = " ★" if is_train else ""
                cells.append(f"{wr:.3f}{marker}")
                if is_train:
                    train_wr = wr
                else:
                    other_wrs.append(wr)

            # Calcular delta: WR del oponente de entrenamiento vs promedio de los otros
            if train_wr is not None and other_wrs:
                delta = train_wr - np.mean(other_wrs)
                delta_str = f"{delta:+.3f}"
            else:
                delta_str = "—"

            short_name = spec_name.replace("_vs_", " → ")
            lines.append(f"| {short_name} | {cells[0]} | {cells[1]} | {cells[2]} | {delta_str} |")

        lines.append("")

    # --- Análisis de overfitting ---
    lines.append("## Análisis de Generalización")
    lines.append("")

    for family in ["neat", "ppo", "dt"]:
        family_specs = {k: v for k, v in results.items() if k.startswith(family)}
        if not family_specs:
            continue

        deltas = []
        for spec_name, opp_results in family_specs.items():
            trained_against = spec_name.split("_vs_")[1]
            train_wr = opp_results.get(trained_against, {}).get("winrate", 0)
            other_wrs = [r.get("winrate", 0) for opp, r in opp_results.items()
                         if opp != trained_against]
            if other_wrs:
                deltas.append(train_wr - np.mean(other_wrs))

        avg_delta = np.mean(deltas) if deltas else 0
        if abs(avg_delta) < 0.02:
            verdict = "✅ Generaliza bien (Δ < 2%)"
        elif avg_delta > 0.05:
            verdict = "⚠️ Posible overfitting (Δ > 5%)"
        else:
            verdict = "🔶 Ligera especialización (2-5%)"

        lines.append(f"- **{family.upper()}**: Δ promedio = {avg_delta:+.3f} → {verdict}")

    lines.append("")
    lines.append("**Interpretación del Δ (delta)**:")
    lines.append("- Δ ≈ 0: El especialista rinde igual contra todos → buena generalización")
    lines.append("- Δ > 0: El especialista rinde mejor contra su oponente de entrenamiento → especialización")
    lines.append("- Δ > 0.05: El especialista rinde significativamente mejor solo contra su oponente → overfitting")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Cross-evaluation JSON: {out_path}")
    print(f"[OK] Cross-evaluation MD: {md_path}")


if __name__ == "__main__":
    main()

