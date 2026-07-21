# gym/scripts/neat/train_neat_multiseed.py
"""
Entrena NEAT con multi-semilla real para múltiples oponentes.

Cada oponente genera UN solo modelo NEAT cuyo fitness se evaluó contra
N seeds simultáneamente en cada generación, forzando generalización real.

Uso:
    python -m gym.scripts.neat.train_neat_multiseed `
        --opponents "heuristic:denial,heuristic:spread,heuristic:greedy" `
        --seeds "123,456,789" `
        --generations 300 `
        --episodes-per-genome 30

    # Solo un oponente (prueba rápida):
    python -m gym.scripts.neat.train_neat_multiseed `
        --opponents "heuristic:greedy" `
        --seeds "123,456,789" `
        --generations 50 `
        --episodes-per-genome 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def train_one(opponent: str, seeds: str, generations: int,
              episodes: int, config: str, workers: int) -> Dict:
    """Entrena un NEAT multi-semilla para un oponente y retorna info."""
    opp_safe = opponent.replace(":", "_").replace("/", "_")
    out_name = f"neat__vs_{opp_safe}"

    cmd = [
        sys.executable, "-m", "gym.scripts.neat.train_neat",
        "--opponent", opponent,
        "--generations", str(generations),
        "--episodes-per-genome", str(episodes),
        "--seeds", seeds,
        "--config", config,
        "--out", out_name,
        "--workers", str(workers),
    ]

    print(f"\n{'='*70}")
    print(f"  ENTRENANDO: {out_name} (multi-seed: {seeds})")
    print(f"  Cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    wall = time.perf_counter() - t0

    meta_path = Path(f"gym/data/models/neat/{out_name}.meta.json")
    genome_path = Path(f"gym/data/models/neat/{out_name}.pkl")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "opponent": opponent,
        "out_name": out_name,
        "genome_path": str(genome_path),
        "meta_path": str(meta_path),
        "fitness": meta.get("fitness", -999),
        "wall_time": round(wall, 1),
        "returncode": result.returncode,
        "exists": genome_path.exists(),
        "meta": meta,
    }


def evaluate_one(genome_path: str, config: str, opponent: str,
                 games: int = 2000, seed: int = 9999) -> float:
    """Evalúa un genoma entrenado y retorna winrate decided."""
    from gym.env.knucklebones_env import KnucklebonesEnv
    from gym.policies.neat.neat_policy import NEATPolicy
    from gym.policies.utils.policy_factory import build_policy

    try:
        p0 = NEATPolicy(genome_path=genome_path, config_path=config, name="neat_eval")
        p1 = build_policy(opponent)
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar {genome_path}: {e}")
        return 0.0

    env = KnucklebonesEnv()
    wins = 0
    losses = 0

    for i in range(games):
        s = seed + i
        env.reset(seed=s)
        p0.reset(seed=s)
        p1.reset(seed=s)
        done = False
        turns = 0
        res = None
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
        if res is None:
            continue
        winner = res.info.get("winner")
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1

    decided = max(1, wins + losses)
    return wins / decided


def main():
    ap = argparse.ArgumentParser(
        description="Train NEAT with TRUE multi-seed fitness for each opponent.")
    ap.add_argument("--opponents", type=str,
                    default="heuristic:denial,heuristic:spread,heuristic:greedy")
    ap.add_argument("--seeds", type=str, default="123,456,789",
                    help="Seeds used simultaneously in fitness evaluation")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--episodes-per-genome", type=int, default=30,
                    help="Episodes PER SEED. Total = episodes * num_seeds")
    ap.add_argument("--config", type=str, default="gym/config/neat/neat_config.ini")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=2000,
                    help="Games for post-training evaluation")
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",")]
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 2)

    print(f"[CONFIG] Opponents: {opponents}")
    print(f"[CONFIG] Seeds (multi-seed real): {seed_list}")
    print(f"[CONFIG] Generations: {args.generations}")
    print(f"[CONFIG] Episodes/seed/genome: {args.episodes_per_genome}")
    print(f"[CONFIG] Total episodes/genome: {args.episodes_per_genome * len(seed_list)}")
    print(f"[CONFIG] Workers: {workers}")
    print(f"[CONFIG] Total trainings: {len(opponents)}")

    results: List[Dict] = []

    t_global = time.perf_counter()
    for opp in opponents:
        info = train_one(opp, args.seeds, args.generations,
                         args.episodes_per_genome, args.config, workers)
        results.append(info)

        if info["returncode"] != 0:
            print(f"  [WARN] Training failed: {info['out_name']}")
        else:
            print(f"  [OK] {info['out_name']}: fitness={info['fitness']:.4f} "
                  f"({info['wall_time']:.0f}s)")

    # Fase 2: Evaluación post-entrenamiento
    print(f"\n{'='*70}")
    print("FASE 2: Evaluación post-entrenamiento")
    print(f"{'='*70}\n")

    for info in results:
        if not info["exists"] or info["returncode"] != 0:
            print(f"  [SKIP] {info['out_name']}: no existe o falló")
            continue

        wr = evaluate_one(info["genome_path"], args.config, info["opponent"],
                          games=args.eval_games)
        info["eval_winrate"] = wr
        print(f"  {info['out_name']}: eval_WR={wr:.4f} ({args.eval_games} games)")

        # Actualizar meta con winrate de evaluación
        meta_path = Path(info["meta_path"])
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["eval_winrate"] = wr
            meta["eval_games"] = args.eval_games
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    # Resumen final
    total_time = time.perf_counter() - t_global
    print(f"\n{'='*70}")
    print(f"RESUMEN FINAL ({total_time/60:.1f} min total)")
    print(f"{'='*70}\n")

    print(f"{'Oponente':25s} {'Fitness':>10s} {'Eval WR':>10s} {'Tiempo':>10s}")
    print("-" * 60)
    for info in results:
        if info["exists"] and info["returncode"] == 0:
            print(f"{info['opponent']:25s} {info['fitness']:>10.4f} "
                  f"{info.get('eval_winrate', 0):>10.4f} "
                  f"{info['wall_time']:>8.0f}s")

    # Guardar reporte
    report_path = Path("gym/data/results/reports/neat_multiseed_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "true_multi_seed",
        "opponents": opponents,
        "seeds": seed_list,
        "generations": args.generations,
        "episodes_per_seed": args.episodes_per_genome,
        "total_episodes_per_genome": args.episodes_per_genome * len(seed_list),
        "eval_games": args.eval_games,
        "total_wall_time_seconds": round(total_time, 1),
        "results": [
            {
                "opponent": r["opponent"],
                "fitness": r["fitness"],
                "eval_winrate": r.get("eval_winrate", 0),
                "wall_time_seconds": r["wall_time"],
            }
            for r in results
            if r["exists"] and r["returncode"] == 0
        ]
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Reporte: {report_path}")


if __name__ == "__main__":
    main()
