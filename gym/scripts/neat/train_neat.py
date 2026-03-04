# gym/scripts/neat/train_neat.py
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import neat
import numpy as np

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.policy_factory import build_policy
from gym.policies.utils.features import extract_features
from gym.scripts.utils.naming import utc_stamp


class _GenerationLogger(neat.reporting.BaseReporter):
    """Reporter NEAT que escribe un JSONL por generación."""

    def __init__(self, log_path: Path):
        self._fh = open(log_path, "w", encoding="utf-8")
        self._t0 = time.perf_counter()
        self._gen = 0

    def post_evaluate(self, config, population, species_set, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        entry = {
            "generation": self._gen,
            "elapsed_seconds": round(time.perf_counter() - self._t0, 3),
            "best_fitness": round(float(best_genome.fitness), 6) if best_genome.fitness else None,
            "avg_fitness": round(float(np.mean(fitnesses)), 6) if fitnesses else None,
            "std_fitness": round(float(np.std(fitnesses)), 6) if fitnesses else None,
            "min_fitness": round(float(np.min(fitnesses)), 6) if fitnesses else None,
            "num_species": len(species_set.species),
            "num_genomes": len(population),
        }
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._gen += 1

    def close(self):
        self._fh.flush()
        self._fh.close()


def evaluate_genome(genome, config, env, opponent, seed, episodes):
    """Evaluate a single NEAT genome by playing episodes against an opponent."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0.0

    for ep in range(episodes):
        seed_game = seed + ep
        env.reset(seed=seed_game)
        opponent.reset(seed=seed_game)

        done = False
        turns = 0

        while not done:
            die = env.roll_die()
            obs = env._get_obs(dice_value=die)
            legal = env.legal_actions()

            player = int(obs["current_player"])

            if player == 0:
                features = extract_features(obs)
                output = net.activate(features.tolist())
                scored = [(output[a], a) for a in legal]
                scored.sort(reverse=True)
                action = scored[0][1]
            else:
                step = opponent.select_action(obs=obs, legal_actions=legal)
                action = int(step.action)

            res = env.step(action, dice_value=die)
            done = bool(res.done)
            turns += 1
            if turns > 500:
                break

        info = res.info
        s0 = info.get("final_score_p0", 0) or 0
        s1 = info.get("final_score_p1", 0) or 0
        winner = info.get("winner")

        if winner == 0:
            total_reward += 1.0
        elif winner is None:
            total_reward += 0.5
        total_reward += (s0 - s1) / 200.0

    return total_reward / episodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a NEAT genome for Knucklebones.")
    ap.add_argument("--opponent", type=str, default="baseline:first", help="Opponent spec")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--episodes-per-genome", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--config", type=str, default="gym/config/neat/neat_config.ini")
    ap.add_argument("--out", type=str, default="", help="Output name (optional)")
    args = ap.parse_args()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    env = KnucklebonesEnv()
    opponent = build_policy(args.opponent)

    def eval_genomes(genomes, neat_config):
        for genome_id, genome in genomes:
            genome.fitness = evaluate_genome(
                genome, neat_config, env, opponent,
                seed=args.seed + genome_id * 1000,
                episodes=args.episodes_per_genome,
            )

    # --- Preparar logging ---
    out_dir = Path("gym/data/models/neat")
    out_dir.mkdir(parents=True, exist_ok=True)

    opp_safe = args.opponent.replace(":", "_").replace("/", "_")
    base_name = args.out.strip() or f"neat__vs_{opp_safe}"
    log_path = out_dir / f"{base_name}.generation_log.jsonl"

    gen_logger = _GenerationLogger(log_path)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(gen_logger)

    # --- Entrenar ---
    t0 = time.perf_counter()

    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None

    winner = pop.run(eval_genomes, args.generations)

    wall_time = time.perf_counter() - t0
    gen_logger.close()

    # --- Guardar ---
    genome_path = out_dir / f"{base_name}.pkl"
    meta_path = out_dir / f"{base_name}.meta.json"

    with open(genome_path, "wb") as f:
        pickle.dump(winner, f)

    genome_size = os.path.getsize(genome_path)

    peak_memory_mb = None
    if proc is not None:
        try:
            peak_memory_mb = round(proc.memory_info().peak_wset / (1024 * 1024), 2)
        except AttributeError:
            peak_memory_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    meta = {
        "created_utc": utc_stamp(),
        "generations": int(args.generations),
        "episodes_per_genome": int(args.episodes_per_genome),
        "seed": int(args.seed),
        "opponent": args.opponent,
        "config_path": args.config,
        "genome_path": str(genome_path),
        "fitness": float(winner.fitness),
        "obs_features": 21,
        "feature_extractor": "gym.policies.utils.features.extract_features",
        # --- Métricas nuevas ---
        "wall_time_seconds": round(wall_time, 3),
        "genome_size_bytes": genome_size,
        "peak_memory_mb": peak_memory_mb,
        "generation_log": str(log_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Best genome saved: {genome_path} ({genome_size:,} bytes)")
    print(f"[OK] Generation log: {log_path}")
    print(f"[OK] Meta saved: {meta_path}")
    print(f"Fitness: {winner.fitness:.4f} | Time: {wall_time:.1f}s | Memory: {peak_memory_mb} MB")


if __name__ == "__main__":
    main()

