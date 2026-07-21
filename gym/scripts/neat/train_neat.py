# gym/scripts/neat/train_neat.py
"""
Entrena un genoma NEAT contra un oponente heurístico.

Soporta multi-semilla real: cada genoma se evalúa contra N seeds
distintas en cada generación, promediando el fitness. Esto previene
sobreajuste a una secuencia determinista particular.

Uso básico (una seed, seed rotation):
    python -m gym.scripts.neat.train_neat `
        --opponent "heuristic:greedy" `
        --generations 300 --episodes-per-genome 100

Multi-semilla real (recomendado):
    python -m gym.scripts.neat.train_neat `
        --opponent "heuristic:greedy" `
        --generations 300 --episodes-per-genome 30 `
        --seeds "123,456,789"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import time
from pathlib import Path
from typing import List

import neat
import numpy as np

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.policy_factory import build_policy
from gym.policies.utils.features import extract_features
from gym.scripts.utils.naming import utc_stamp


# ─── Worker globals (inicializados una vez por proceso) ───
_worker_env = None
_worker_opponent = None
_worker_opponent_spec = None


def _init_worker(opponent_spec: str):
    """Initializer para cada proceso worker del pool."""
    global _worker_env, _worker_opponent, _worker_opponent_spec
    _worker_env = KnucklebonesEnv()
    _worker_opponent = build_policy(opponent_spec)
    _worker_opponent_spec = opponent_spec


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


def _eval_genome_worker(args):
    """Función top-level pickleable para multiprocessing.Pool.map.
    Recibe (genome, config, seeds_list, episodes_per_seed) y usa globals del worker.
    seeds_list: lista de base_seeds. El fitness se promedia sobre todas."""
    genome, config, seeds_list, episodes_per_seed = args
    fitnesses = []
    for base_seed in seeds_list:
        f = _evaluate_genome(genome, config, _worker_env, _worker_opponent,
                             base_seed, episodes_per_seed)
        fitnesses.append(f)
    return float(np.mean(fitnesses))


def _evaluate_genome(genome, config, env, opponent, seed, episodes):
    """Evaluate a single NEAT genome by playing episodes against an opponent."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0.0

    for ep in range(episodes):
        seed_game = seed + ep
        env.reset(seed=seed_game)
        opponent.reset(seed=seed_game)

        done = False
        turns = 0
        destructions = 0

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

                # Contar destrucciones antes del step
                op_col = obs["op_cols"][action]
                destructions += sum(1 for v in op_col if v == die)
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

        # --- Fitness multi-componente ---
        if winner == 0:
            total_reward += 1.0
        elif winner is None:
            total_reward += 0.4

        score_diff = max(-1.0, min(1.0, (s0 - s1) / 50.0))
        total_reward += score_diff * 0.3

        total_reward += min(destructions * 0.02, 0.2)

    return total_reward / episodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a NEAT genome for Knucklebones.")
    ap.add_argument("--opponent", type=str, default="baseline:first", help="Opponent spec")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--episodes-per-genome", type=int, default=20,
                    help="Episodes PER SEED. Total episodes = episodes * num_seeds")
    ap.add_argument("--seed", type=int, default=123, help="Single seed (legacy, use --seeds)")
    ap.add_argument("--seeds", type=str, default="",
                    help="Comma-separated seeds for multi-seed fitness (e.g. '123,456,789')")
    ap.add_argument("--config", type=str, default="gym/config/neat/neat_config.ini")
    ap.add_argument("--out", type=str, default="", help="Output name (optional)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = auto, 1 = sequential)")
    args = ap.parse_args()

    # --- Parsear seeds ---
    if args.seeds.strip():
        seed_list: List[int] = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seed_list = [args.seed]

    multi_seed = len(seed_list) > 1
    print(f"[INFO] Seeds: {seed_list} ({'multi-seed' if multi_seed else 'single-seed + rotation'})")
    print(f"[INFO] Episodes per seed per genome: {args.episodes_per_genome}")
    print(f"[INFO] Total episodes per genome: {args.episodes_per_genome * len(seed_list)}")

    # --- Workers ---
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() - 2)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )

    # --- Eval function: parallel o sequential ---
    # Multi-seed: cada genoma se evalúa contra TODAS las seeds, promediando fitness.
    # Single-seed: seed rotation por generación (legacy).
    _gen_counter = [0]  # mutable para closure

    def _build_seeds_for_genome(genome_id: int) -> List[int]:
        """Construye la lista de seeds para un genoma en esta generación."""
        if multi_seed:
            # Multi-seed real: cada genoma juega contra TODAS las seeds,
            # con rotación por generación para variabilidad.
            gen_offset = _gen_counter[0] * 10007
            return [s + gen_offset + genome_id * 1000 for s in seed_list]
        else:
            # Single-seed con rotación
            gen_seed = seed_list[0] + _gen_counter[0] * 10007
            return [gen_seed + genome_id * 1000]

    if num_workers > 1:
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(args.opponent,),
        )
        print(f"[INFO] Parallel mode: {num_workers} workers")

        def eval_genomes(genomes, neat_config):
            tasks = []
            for genome_id, genome in genomes:
                seeds_for_genome = _build_seeds_for_genome(genome_id)
                tasks.append((genome, neat_config,
                              seeds_for_genome,
                              args.episodes_per_genome))
            results = pool.map(_eval_genome_worker, tasks)
            for (genome_id, genome), fitness in zip(genomes, results):
                genome.fitness = fitness
            _gen_counter[0] += 1
    else:
        # Modo secuencial (sin overhead de multiprocessing)
        env = KnucklebonesEnv()
        opponent = build_policy(args.opponent)
        print("[INFO] Sequential mode: 1 worker")

        def eval_genomes(genomes, neat_config):
            for genome_id, genome in genomes:
                seeds_for_genome = _build_seeds_for_genome(genome_id)
                fitnesses = []
                for base_seed in seeds_for_genome:
                    f = _evaluate_genome(
                        genome, neat_config, env, opponent,
                        seed=base_seed,
                        episodes=args.episodes_per_genome,
                    )
                    fitnesses.append(f)
                genome.fitness = float(np.mean(fitnesses))
            _gen_counter[0] += 1

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

    try:
        winner = pop.run(eval_genomes, args.generations)
    finally:
        if num_workers > 1:
            pool.close()
            pool.join()

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
        "episodes_per_seed": int(args.episodes_per_genome),
        "num_seeds": len(seed_list),
        "total_episodes_per_genome": int(args.episodes_per_genome) * len(seed_list),
        "seeds": seed_list,
        "multi_seed": multi_seed,
        "seed_rotation": True,
        "seed_rotation_prime": 10007,
        "opponent": args.opponent,
        "config_path": args.config,
        "genome_path": str(genome_path),
        "fitness": float(winner.fitness),
        "obs_features": 21,
        "feature_extractor": "gym.policies.utils.features.extract_features",
        # --- Métricas ---
        "wall_time_seconds": round(wall_time, 3),
        "genome_size_bytes": genome_size,
        "peak_memory_mb": peak_memory_mb,
        "generation_log": str(log_path),
        "num_workers": num_workers,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Best genome saved: {genome_path} ({genome_size:,} bytes)")
    print(f"[OK] Generation log: {log_path}")
    print(f"[OK] Meta saved: {meta_path}")
    print(f"Fitness: {winner.fitness:.4f} | Time: {wall_time:.1f}s | Workers: {num_workers} | Memory: {peak_memory_mb} MB")


if __name__ == "__main__":
    main()

