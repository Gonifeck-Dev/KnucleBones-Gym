# gym/scripts/rl/train_sb3.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from gym.env.knucklebones_sb3_env import KnucklebonesSB3Env
from gym.scripts.utils.naming import utc_stamp


class _TrainingLogger:
    """SB3 callback que escribe métricas por rollout a un JSONL."""

    def __init__(self, log_path: Path):
        self._fh = open(log_path, "w", encoding="utf-8")
        self._t0 = time.perf_counter()

    @staticmethod
    def _safe(v):
        """Convierte numpy scalars a Python nativos para json.dumps."""
        if v is None:
            return None
        try:
            return round(float(v), 8)
        except (TypeError, ValueError):
            return None

    def on_rollout_end(self, model) -> None:
        logger = model.logger
        # SB3 almacena métricas internas en name_to_value
        kv = {}
        if hasattr(logger, "name_to_value"):
            kv = dict(logger.name_to_value)

        g = self._safe
        entry = {
            "timestep": int(model.num_timesteps),
            "elapsed_seconds": round(time.perf_counter() - self._t0, 3),
            "ep_reward_mean": g(kv.get("rollout/ep_rew_mean")),
            "ep_len_mean": g(kv.get("rollout/ep_len_mean")),
            "policy_loss": g(kv.get("train/policy_gradient_loss")),
            "value_loss": g(kv.get("train/value_loss")),
            "entropy_loss": g(kv.get("train/entropy_loss")),
            "approx_kl": g(kv.get("train/approx_kl")),
            "clip_fraction": g(kv.get("train/clip_fraction")),
            "explained_variance": g(kv.get("train/explained_variance")),
            "learning_rate": g(kv.get("train/learning_rate")),
        }
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


def _make_callback(training_logger: _TrainingLogger):
    """Crea un callback SB3 estándar que llama al logger."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Cb(BaseCallback):
        def _on_rollout_end(self) -> None:
            training_logger.on_rollout_end(self.model)

        def _on_step(self) -> bool:
            return True

    return _Cb()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RL agent with Stable-Baselines3 on Knucklebones.")
    ap.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN"])
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--opponent", type=str, default="baseline:first")
    ap.add_argument("--reward-mode", type=str, default="diff_delta", choices=["diff_delta", "outcome"])
    ap.add_argument("--out", type=str, default="", help="Optional output model name (without extension)")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="Parallel environments (SubprocVecEnv). 1=sequential, >1=parallel.")
    args = ap.parse_args()

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except Exception as e:
        raise RuntimeError("Missing stable-baselines3. Install: pip install stable-baselines3 gymnasium") from e

    n_envs = max(1, args.n_envs)

    def make_env_fn(env_seed):
        """Factory que devuelve una función sin args (requerido por SubprocVecEnv)."""
        def _init():
            return KnucklebonesSB3Env(
                opponent_spec=args.opponent,
                seed=env_seed,
                reward_mode=args.reward_mode,
            )
        return _init

    if n_envs > 1:
        env_fns = [make_env_fn(args.seed + i * 10000) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        print(f"[INFO] Parallel envs: {n_envs} (SubprocVecEnv)")
    else:
        vec_env = DummyVecEnv([make_env_fn(args.seed)])
        print("[INFO] Sequential env: 1 (DummyVecEnv)")

    # Forzar CPU: MlpPolicy no se beneficia de GPU (warning oficial de SB3)
    device = "cpu"

    algo = args.algo.upper()
    if algo == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, device=device)
    else:
        from stable_baselines3 import DQN
        model = DQN("MlpPolicy", vec_env, verbose=1, seed=args.seed, device=device)

    # --- Preparar logging ---
    out_dir = Path("gym/data/models/rl")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.out.strip() or f"{algo}__vs_{args.opponent.replace(':','_')}"
    log_path = out_dir / f"{base_name}.training_log.jsonl"

    training_logger = _TrainingLogger(log_path)
    callback = _make_callback(training_logger)

    # --- Entrenar ---
    t0 = time.perf_counter()

    try:
        import psutil
        proc = psutil.Process()
        mem_before = proc.memory_info().rss
    except ImportError:
        proc = None
        mem_before = 0

    model.learn(total_timesteps=int(args.timesteps), callback=callback)

    wall_time = time.perf_counter() - t0
    training_logger.close()

    # --- Guardar modelo ---
    model_path = out_dir / f"{base_name}.zip"
    meta_path = out_dir / f"{base_name}.meta.json"

    model.save(str(model_path))
    model_size = os.path.getsize(model_path)

    peak_memory_mb = None
    if proc is not None:
        try:
            mem_info = proc.memory_info()
            rss = getattr(mem_info, 'rss', 0)
            peak_memory_mb = round(rss / (1024 * 1024), 2)
        except Exception:
            pass

    meta = {
        "created_utc": utc_stamp(),
        "algo": algo,
        "timesteps": int(args.timesteps),
        "seed": int(args.seed),
        "opponent": args.opponent,
        "reward_mode": args.reward_mode,
        "model_path": str(model_path),
        "obs_features": 21,
        "feature_extractor": "gym.policies.utils.features.extract_features",
        # --- Métricas ---
        "wall_time_seconds": round(wall_time, 3),
        "model_size_bytes": model_size,
        "peak_memory_mb": peak_memory_mb,
        "training_log": str(log_path),
        "n_envs": n_envs,
        "device": device,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] RL model saved: {model_path} ({model_size:,} bytes)")
    print(f"[OK] Training log: {log_path}")
    print(f"[OK] Meta saved: {meta_path}")
    print(f"Training time: {wall_time:.1f}s | Memory: {peak_memory_mb} MB")


if __name__ == "__main__":
    main()