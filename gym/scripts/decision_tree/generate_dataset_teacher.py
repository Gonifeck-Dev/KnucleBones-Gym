# gym/scripts/decision_tree/generate_dataset_teacher.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.policy_factory import build_policy
from gym.policies.utils.features import extract_features
from gym.scripts.utils.naming import utc_stamp, safe_name


@dataclass
class DatasetMeta:
    dataset_name: str
    created_utc: str
    episodes: int
    seed: int
    teacher: str
    opponent: str
    n_samples: int
    n_features: int
    format: str
    feature_extractor: str
    turn_protocol: str
    notes: str
    # --- Métricas nuevas ---
    wall_time_seconds: float = 0.0
    episodes_per_second: float = 0.0
    dataset_size_bytes: int = 0
    action_distribution: Dict[str, int] = field(default_factory=dict)
    action_balance_ratio: float = 0.0


def play_episode(
    env: KnucklebonesEnv,
    teacher_pol,
    opponent_pol,
    seed_game: int,
) -> Tuple[List[np.ndarray], List[int]]:
    env.reset(seed=seed_game)
    teacher_pol.reset(seed=seed_game)
    opponent_pol.reset(seed=seed_game)

    X: List[np.ndarray] = []
    y: List[int] = []

    done = False
    guard = 0

    while not done:
        guard += 1
        if guard > 600:
            raise RuntimeError("Guard: too many turns (>600). Something is wrong.")

        die = env.roll_die()
        obs = env._get_obs(dice_value=die)
        legal = env.legal_actions()

        player = int(obs["current_player"])
        if player == 0:
            # teacher turn
            action = int(teacher_pol.select_action(obs=obs, legal_actions=legal).action)
            # record supervised sample (state -> action)
            feats = extract_features(obs).astype(np.float32)
            X.append(feats)
            y.append(action)
        else:
            action = int(opponent_pol.select_action(obs=obs, legal_actions=legal).action)

        res = env.step(action, dice_value=die)
        done = bool(res.done)

    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate behavior cloning dataset from any teacher spec vs any opponent spec.")
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--teacher", type=str, required=True, help="Policy spec for teacher (p0).")
    ap.add_argument("--opponent", type=str, required=True, help="Policy spec for opponent (p1).")
    ap.add_argument("--out-dir", type=str, default="gym/data/datasets/raw", help="Base output dir.")
    ap.add_argument("--notes", type=str, default="Behavior cloning dataset.")
    args = ap.parse_args()

    teacher = build_policy(args.teacher)
    opponent = build_policy(args.opponent)

    env = KnucklebonesEnv()

    name = f"bc__teacher_{safe_name(args.teacher)}__vs_{safe_name(args.opponent)}__eps{args.episodes}__seed{args.seed}"
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    X_all: List[np.ndarray] = []
    y_all: List[int] = []

    t0 = time.perf_counter()

    for ep in range(int(args.episodes)):
        seed_game = int(args.seed) + ep
        X, y = play_episode(env, teacher, opponent, seed_game)
        X_all.extend(X)
        y_all.extend(y)

        if (ep + 1) % 1000 == 0:
            print(f"[{ep+1}/{args.episodes}] samples={len(y_all)}")

    wall_time = time.perf_counter() - t0

    X_np = np.stack(X_all, axis=0).astype(np.float32)
    y_np = np.array(y_all, dtype=np.int64)

    npz_path = out_dir / "samples.npz"
    np.savez_compressed(npz_path, X=X_np, y=y_np)
    dataset_size = os.path.getsize(npz_path)

    # preview
    preview_n = min(2000, X_np.shape[0])
    preview = np.concatenate([X_np[:preview_n], y_np[:preview_n, None].astype(np.float32)], axis=1)
    np.savetxt(out_dir / "preview.csv", preview, delimiter=",", fmt="%.6g")

    # Action distribution
    unique, counts = np.unique(y_np, return_counts=True)
    action_dist = {str(int(a)): int(c) for a, c in zip(unique, counts)}
    balance_ratio = float(counts.min() / counts.max()) if len(counts) > 0 else 0.0

    eps_per_sec = args.episodes / wall_time if wall_time > 0 else 0.0

    meta = DatasetMeta(
        dataset_name=name,
        created_utc=utc_stamp(),
        episodes=int(args.episodes),
        seed=int(args.seed),
        teacher=args.teacher,
        opponent=args.opponent,
        n_samples=int(X_np.shape[0]),
        n_features=int(X_np.shape[1]),
        format="npz (X,y)",
        feature_extractor="gym.policies.utils.features.extract_features",
        turn_protocol="die=env.roll_die(); obs=env._get_obs(die); step(action, dice_value=die)",
        notes=str(args.notes),
        wall_time_seconds=round(wall_time, 3),
        episodes_per_second=round(eps_per_sec, 2),
        dataset_size_bytes=dataset_size,
        action_distribution=action_dist,
        action_balance_ratio=round(balance_ratio, 4),
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    print(f"[OK] Dataset saved to: {out_dir}")
    print(f"Time: {wall_time:.1f}s ({eps_per_sec:.0f} eps/s) | Size: {dataset_size:,} bytes")
    print(f"Actions: {action_dist} | Balance: {balance_ratio:.4f}")


if __name__ == "__main__":
    main()