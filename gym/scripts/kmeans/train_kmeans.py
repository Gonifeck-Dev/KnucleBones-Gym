# gym/scripts/kmeans/train_kmeans.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from gym.env.knucklebones_env import KnucklebonesEnv
from gym.policies.utils.base_policy import BasePolicy, PolicyStep
from gym.policies.utils.policy_factory import build_policy
from gym.policies.decision_tree import BaselinePolicy
from gym.policies.kmeans.profile_features import OpponentProfileWindowV2
from gym.scripts.utils.naming import utc_stamp, safe_name


def majority_cluster_to_style(cluster_labels: np.ndarray, style_ids: np.ndarray, style_names: List[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        votes = style_ids[idx]
        winner = int(np.bincount(votes).argmax())
        mapping[int(c)] = style_names[winner]
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Train KMeans v2 using a realistic opponent pool.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--episodes-per-style", type=int, default=5000)
    ap.add_argument("--opponents", nargs="+", required=True, help="Pool specs (baseline/dt/rl/neat)")
    ap.add_argument("--k", type=int, default=0, help="n_clusters (0 => len(opponents))")
    ap.add_argument("--out", type=str, default="", help="Optional output .joblib name")
    args = ap.parse_args()

    style_specs = args.opponents
    style_names = [safe_name(s.replace(":", "__")) for s in style_specs]
    n_styles = len(style_specs)
    k = int(args.k) if int(args.k) > 0 else n_styles

    env = KnucklebonesEnv()

    # p0 “sparring” para diversificar estados
    # (usar random evita sesgo de estado fijo)
    sparring = BaselinePolicy(name="sparring_random", mode="random", seed=args.seed)

    Z_list: List[np.ndarray] = []
    y_style: List[int] = []

    t0 = time.perf_counter()

    for style_id, opp_spec in enumerate(style_specs):
        opp = build_policy(opp_spec)
        profile = OpponentProfileWindowV2(window=int(args.window))

        for ep in range(int(args.episodes_per_style)):
            seed_game = int(args.seed) + style_id * 1_000_000 + ep
            env.reset(seed=seed_game)
            opp.reset(seed=seed_game)
            sparring.reset(seed=seed_game)
            profile.reset()

            done = False
            turns = 0

            while not done:
                die = env.roll_die()
                obs = env._get_obs(dice_value=die)
                legal = env.legal_actions()
                player = int(obs["current_player"])

                if player == 0:
                    step0 = sparring.select_action(obs=obs, legal_actions=legal)
                    action = int(step0.action)
                else:
                    step1 = opp.select_action(obs=obs, legal_actions=legal)
                    action = int(step1.action)

                res = env.step(action, dice_value=die)
                done = bool(res.done)

                # actualizar perfil solo cuando juega el oponente (p1)
                if player == 1:
                    profile.update(action=action, legal_actions=legal)
                    if profile.ready():
                        Z_list.append(profile.vector())
                        y_style.append(style_id)

                turns += 1
                if turns > 500:
                    break

    Z = np.stack(Z_list, axis=0).astype(np.float32)
    y = np.array(y_style, dtype=np.int64)

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    kmeans = KMeans(n_clusters=k, random_state=int(args.seed), n_init="auto")
    kmeans.fit(Zs)

    labels = kmeans.predict(Zs)
    cluster_to_style = majority_cluster_to_style(labels, y, style_names)

    # Diagnóstico: purity por cluster
    purity = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        votes = y[idx]
        counts = np.bincount(votes, minlength=n_styles).astype(int).tolist()
        purity[int(c)] = {"counts_per_style": counts, "total": int(len(idx))}

    out_dir = Path("gym/data/models/kmeans")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = args.out.strip() or f"kmeans__v2__k{k}__win{args.window}__seed{args.seed}.joblib"
    out_path = out_dir / out_name

    artifact = {
        "profile_version": 2,
        "window": int(args.window),
        "profile_dim": int(Z.shape[1]),
        "seed": int(args.seed),
        "k": int(k),
        "style_specs": style_specs,
        "style_names": style_names,
        "n_samples": int(Z.shape[0]),
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_to_style": cluster_to_style,
        "cluster_purity": purity,
    }
    joblib.dump(artifact, out_path)

    wall_time = time.perf_counter() - t0
    artifact_size = os.path.getsize(out_path)
    sil_score = float(silhouette_score(Zs, labels)) if k > 1 else 0.0

    # Meta JSON separado (sin objetos binarios, solo metadata legible)
    meta = {
        "created_utc": utc_stamp(),
        "profile_version": 2,
        "k": int(k),
        "window": int(args.window),
        "profile_dim": int(Z.shape[1]),
        "seed": int(args.seed),
        "style_specs": style_specs,
        "style_names": style_names,
        "n_samples": int(Z.shape[0]),
        "cluster_to_style": cluster_to_style,
        "cluster_purity": purity,
        "artifact_path": str(out_path),
        # --- Métricas nuevas ---
        "wall_time_seconds": round(wall_time, 3),
        "inertia": round(float(kmeans.inertia_), 4),
        "silhouette_score": round(sil_score, 6),
        "artifact_size_bytes": artifact_size,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] KMeans v2 artifact saved: {out_path}")
    print(f"[OK] Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
