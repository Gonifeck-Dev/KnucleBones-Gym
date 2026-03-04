# gym/scripts/utils/analysis_toolkit.py
"""
Toolkit de análisis que lee directamente de los artefactos generados
por los scripts del pipeline. No inventa formatos propios.

Fuentes de datos:
  - gym/data/models/rl/*.training_log.jsonl      (train_sb3.py)
  - gym/data/models/neat/*.generation_log.jsonl   (train_neat.py)
  - gym/data/models/rl/*.meta.json                (train_sb3.py)
  - gym/data/models/neat/*.meta.json              (train_neat.py)
  - gym/data/models/sklearn/*.meta.json           (train_tree.py)
  - gym/data/models/kmeans/*.meta.json            (train_kmeans.py)
  - gym/data/results/runs/*/summary.json          (evaluate_any.py)
  - gym/data/results/runs/*/config.json           (evaluate_any.py)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # backend sin GUI (seguro para scripts)
import matplotlib.pyplot as plt


# ── helpers ─────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> pd.DataFrame:
    """Carga un archivo JSONL como DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _glob_single(pattern: str, base: Path) -> Optional[Path]:
    hits = sorted(base.glob(pattern))
    return hits[0] if hits else None


# ── Cargadores ──────────────────────────────────────────────────────

def load_ppo_training_log(opponent: str) -> pd.DataFrame:
    """Carga el training_log.jsonl de un PPO especialista."""
    tag = opponent.replace(":", "_")
    path = _glob_single(f"PPO__vs_{tag}*.training_log.jsonl",
                        Path("gym/data/models/rl"))
    if path is None:
        return pd.DataFrame()
    return _load_jsonl(path)


def load_neat_generation_log(opponent: str) -> pd.DataFrame:
    """Carga el generation_log.jsonl de un NEAT especialista."""
    tag = opponent.replace(":", "_")
    path = _glob_single(f"neat__vs_{tag}*.generation_log.jsonl",
                        Path("gym/data/models/neat"))
    if path is None:
        return pd.DataFrame()
    return _load_jsonl(path)


def load_model_meta(family: str, opponent: str) -> Dict[str, Any]:
    """Carga el .meta.json de un modelo por familia y oponente."""
    tag = opponent.replace(":", "_")
    # Para DT, el nombre puede ser dt__ppo_vs_denial o dt__ppo_vs_heuristic_denial
    short_tag = tag.replace("heuristic_", "").replace("baseline_", "")
    dirs = {
        "ppo": ("gym/data/models/rl", f"PPO__vs_{tag}*.meta.json"),
        "neat": ("gym/data/models/neat", f"neat__vs_{tag}*.meta.json"),
        "dt": ("gym/data/models/sklearn", f"dt__*{short_tag}*.meta.json"),
        "kmeans": ("gym/data/models/kmeans", "kmeans__*.meta.json"),
    }
    base_dir, pattern = dirs.get(family, (".", "*.meta.json"))
    path = _glob_single(pattern, Path(base_dir))
    if path is None:
        return {}
    return _load_json(path)


def load_all_model_metas() -> pd.DataFrame:
    """Carga todos los .meta.json de modelos en un DataFrame."""
    rows = []
    for folder in ["rl", "neat", "sklearn", "kmeans"]:
        base = Path(f"gym/data/models/{folder}")
        if not base.exists():
            continue
        for p in sorted(base.glob("*.meta.json")):
            meta = _load_json(p)
            meta["_source_file"] = str(p)
            meta["_folder"] = folder
            rows.append(meta)
    return pd.DataFrame(rows)


def load_run_summaries() -> pd.DataFrame:
    """Carga todos los summary.json de runs/ en un DataFrame."""
    runs_dir = Path("gym/data/results/runs")
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for run_dir in sorted(runs_dir.iterdir()):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config.json"
        if not summary_path.exists():
            continue
        s = _load_json(summary_path)
        if config_path.exists():
            c = _load_json(config_path)
            s["algo_tag"] = c.get("algo_tag", "")
            s["p0_name"] = c.get("p0_name", "")
            s["p1_name"] = c.get("p1_name", "")
            s["seed"] = c.get("seed", 0)
        s["_run_dir"] = str(run_dir)
        rows.append(s)
    return pd.DataFrame(rows)


# ── Gráficos ────────────────────────────────────────────────────────

PLOTS_DIR = Path("gym/data/results/plots")


def _ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_ppo_training_curve(opponent: str = "heuristic:denial"):
    """Gráfico de convergencia PPO: ep_reward_mean vs timestep."""
    df = load_ppo_training_log(opponent)
    if df.empty:
        print(f"[SKIP] No hay training log PPO para {opponent}")
        return
    _ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestep"], df["ep_reward_mean"], linewidth=1.5, label="ep_reward_mean")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward (mean)")
    ax.set_title(f"PPO Training Curve vs {opponent}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / f"ppo_training_{opponent.replace(':', '_')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_neat_evolution_curve(opponent: str = "heuristic:denial"):
    """Gráfico de evolución NEAT: best/avg fitness vs generation."""
    df = load_neat_generation_log(opponent)
    if df.empty:
        print(f"[SKIP] No hay generation log NEAT para {opponent}")
        return
    _ensure_plots_dir()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["generation"], df["best_fitness"], linewidth=2, label="Best Fitness")
    ax.plot(df["generation"], df["avg_fitness"], linewidth=1.5, alpha=0.7, label="Avg Fitness")
    if "std_fitness" in df.columns:
        avg = df["avg_fitness"].values
        std = df["std_fitness"].values
        ax.fill_between(df["generation"], avg - std, avg + std, alpha=0.15, label="+/- 1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"NEAT Evolution vs {opponent}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / f"neat_evolution_{opponent.replace(':', '_')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_training_time_comparison():
    """Barras: tiempo de entrenamiento (s) por modelo."""
    df = load_all_model_metas()
    if df.empty or "wall_time_seconds" not in df.columns:
        print("[SKIP] No hay datos de wall_time")
        return
    _ensure_plots_dir()

    df = df[df["wall_time_seconds"].notna()].copy()
    labels = df["_source_file"].apply(lambda x: Path(x).stem.replace(".meta", ""))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(labels, df["wall_time_seconds"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time per Model")
    ax.grid(True, alpha=0.3, axis="x")

    out = PLOTS_DIR / "training_time_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_model_size_comparison():
    """Barras: tamano del modelo en KB."""
    df = load_all_model_metas()
    if df.empty:
        print("[SKIP] No hay datos de modelos")
        return
    _ensure_plots_dir()

    rows = []
    for _, row in df.iterrows():
        sz = 0
        for c in ["model_size_bytes", "genome_size_bytes", "artifact_size_bytes"]:
            if c in row and pd.notna(row[c]):
                sz = row[c]
                break
        rows.append({
            "model": Path(row["_source_file"]).stem.replace(".meta", ""),
            "size_kb": sz / 1024,
        })
    sdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(sdf["model"], sdf["size_kb"], color="coral", alpha=0.8)
    ax.set_xlabel("Model Size (KB)")
    ax.set_title("Model Size Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    out = PLOTS_DIR / "model_size_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_evaluation_winrates():
    """Barras: winrate_p0_decided de cada run de evaluacion."""
    df = load_run_summaries()
    if df.empty or "winrate_p0_decided" not in df.columns:
        print("[SKIP] No hay runs de evaluacion")
        return
    _ensure_plots_dir()

    labels = []
    for _, row in df.iterrows():
        p0 = row.get("p0_spec", row.get("p0_name", "?"))
        p1 = row.get("p1_spec", row.get("p1_name", "?"))
        labels.append(f"{p0}\nvs {p1}")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    wr = df["winrate_p0_decided"].values

    colors = ["green" if w > 0.55 else "red" if w < 0.45 else "gray" for w in wr]
    ax.bar(x, wr, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Winrate (decided)")
    ax.set_title("Evaluation Winrates (p0 decided)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    if "winrate_p0_decided_se" in df.columns:
        se = df["winrate_p0_decided_se"].fillna(0).values
        ax.errorbar(x, wr, yerr=se, fmt="none", ecolor="black", capsize=3)

    out = PLOTS_DIR / "evaluation_winrates.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_score_diff_boxplot():
    """Box plot de percentiles de score_diff por run."""
    df = load_run_summaries()
    if df.empty or "score_diff_percentiles" not in df.columns:
        print("[SKIP] No hay percentiles de score_diff")
        return
    _ensure_plots_dir()

    box_data = []
    labels = []
    for _, row in df.iterrows():
        pct = row.get("score_diff_percentiles")
        if not isinstance(pct, dict):
            continue
        box_data.append([pct["p5"], pct["p25"], pct["p50"], pct["p75"], pct["p95"]])
        p0 = row.get("p0_spec", row.get("p0_name", "?"))
        p1 = row.get("p1_spec", row.get("p1_name", "?"))
        labels.append(f"{p0} vs {p1}")

    if not box_data:
        print("[SKIP] No hay percentiles validos")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(box_data, tick_labels=labels, vert=True, patch_artist=True,
                    whis=[0, 100], showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Score Diff (p0 - p1)")
    ax.set_title("Score Differential Distribution (p5/p25/p50/p75/p95)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    out = PLOTS_DIR / "score_diff_boxplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_dt_feature_importances(opponent: str = "heuristic:denial"):
    """Barras: importancia de cada feature del DT destilado."""
    meta = load_model_meta("dt", opponent)
    if not meta or "feature_importances" not in meta:
        print(f"[SKIP] No hay feature_importances para DT vs {opponent}")
        return
    _ensure_plots_dir()

    importances = np.array(meta["feature_importances"])
    feature_names = (
        ["dice"]
        + [f"my_{c}_{r}" for c in range(3) for r in range(3)]
        + [f"op_{c}_{r}" for c in range(3) for r in range(3)]
        + ["score_my", "score_op"]
    )

    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(importances)), importances[idx], color="teal", alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_title(f"DT Feature Importances (vs {opponent})")
    ax.grid(True, alpha=0.3, axis="y")

    out = PLOTS_DIR / f"dt_feature_importances_{opponent.replace(':', '_')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ── Tablas ──────────────────────────────────────────────────────────

TABLES_DIR = Path("gym/data/results/tables")


def export_models_table():
    """Tabla Markdown con todos los modelos y sus metricas."""
    df = load_all_model_metas()
    if df.empty:
        print("[SKIP] No hay modelos")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, m in df.iterrows():
        sz = 0
        for c in ["model_size_bytes", "genome_size_bytes", "artifact_size_bytes"]:
            if c in m and pd.notna(m[c]):
                sz = int(m[c])
                break

        acc = "-"
        if isinstance(m.get("metrics"), dict):
            acc = f"{m['metrics']['accuracy']:.4f}"

        rows.append({
            "Family": m.get("_folder", "?"),
            "Artifact": Path(m["_source_file"]).stem.replace(".meta", ""),
            "Opponent": str(m.get("opponent", m.get("style_specs", "?"))),
            "Wall Time (s)": f"{m['wall_time_seconds']:.1f}" if pd.notna(m.get("wall_time_seconds")) else "-",
            "Size (KB)": f"{sz/1024:.1f}" if sz > 0 else "-",
            "Accuracy": acc,
            "Fitness": f"{m['fitness']:.4f}" if pd.notna(m.get("fitness")) else "-",
        })

    out_df = pd.DataFrame(rows)
    md = out_df.to_markdown(index=False)
    out = TABLES_DIR / "models_summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"[OK] {out}")
    return md


def export_evaluations_table():
    """Tabla Markdown con todos los runs de evaluacion."""
    df = load_run_summaries()
    if df.empty:
        print("[SKIP] No hay evaluaciones")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "p0": r.get("p0_spec", r.get("p0_name", "?")),
            "p1": r.get("p1_spec", r.get("p1_name", "?")),
            "Games": int(r.get("games", 0)),
            "Winrate": f"{r['winrate_p0_decided']:.4f}" if pd.notna(r.get("winrate_p0_decided")) else "-",
            "+/- SE": f"{r['winrate_p0_decided_se']:.4f}" if pd.notna(r.get("winrate_p0_decided_se")) else "-",
            "Avg Diff": f"{r['avg_score_diff_p0_minus_p1']:.2f}" if pd.notna(r.get("avg_score_diff_p0_minus_p1")) else "-",
            "Std Diff": f"{r['std_score_diff']:.2f}" if pd.notna(r.get("std_score_diff")) else "-",
            "Wall (s)": f"{r['wall_time_seconds']:.1f}" if pd.notna(r.get("wall_time_seconds")) else "-",
        })

    out_df = pd.DataFrame(rows)
    md = out_df.to_markdown(index=False)
    out = TABLES_DIR / "evaluations_summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"[OK] {out}")
    return md


# ── main: genera ───────────────────────────────────────────────

def main():
    """Genera todos los graficos y tablas disponibles."""
    opponents = ["heuristic:denial", "heuristic:spread", "heuristic:greedy"]

    print("=== Training Curves ===")
    for opp in opponents:
        plot_ppo_training_curve(opp)
        plot_neat_evolution_curve(opp)

    print("\n=== Model Comparisons ===")
    plot_training_time_comparison()
    plot_model_size_comparison()

    print("\n=== Evaluation Results ===")
    plot_evaluation_winrates()
    plot_score_diff_boxplot()

    print("\n=== DT Explainability ===")
    for opp in opponents:
        plot_dt_feature_importances(opp)

    print("\n=== Tables ===")
    export_models_table()
    export_evaluations_table()

    print(f"\nPlots  -> {PLOTS_DIR}/")
    print(f"Tables -> {TABLES_DIR}/")


if __name__ == "__main__":
    main()

