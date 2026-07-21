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

# Importar estilo unificado de tesis
from gym.scripts.utils.thesis_style import apply_thesis_style, get_paradigm_color, COLORS

# Aplicar estilo al importar el módulo
apply_thesis_style()


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


# Modelos finales del proyecto (los que se reportan en la tesis)
FINAL_MODEL_STEMS = {
    "PPO__vs_heuristic_denial",
    "PPO__vs_heuristic_spread",
    "PPO__vs_heuristic_greedy",
    "neat__vs_heuristic_denial__multiseed_real",
    "neat__vs_heuristic_spread__multiseed_real",
    "neat__vs_heuristic_greedy__multiseed_real",
    "dt__ppo_vs_denial",
    "dt__ppo_vs_spread",
    "dt__ppo_vs_greedy",
    "kmeans_profiler",
}

# Nombres cortos para barras de gráficos
_DISPLAY_NAMES = {
    "PPO__vs_heuristic_denial":                    "PPO vs denial",
    "PPO__vs_heuristic_spread":                    "PPO vs spread",
    "PPO__vs_heuristic_greedy":                    "PPO vs greedy",
    "neat__vs_heuristic_denial__multiseed_real":    "NEAT vs denial",
    "neat__vs_heuristic_spread__multiseed_real":    "NEAT vs spread",
    "neat__vs_heuristic_greedy__multiseed_real":    "NEAT vs greedy",
    "dt__ppo_vs_denial":                           "DT vs denial",
    "dt__ppo_vs_spread":                           "DT vs spread",
    "dt__ppo_vs_greedy":                           "DT vs greedy",
    "kmeans_profiler":                             "KMeans",
}


def _model_display_name(stem: str) -> str:
    """Convierte un stem de archivo .meta.json a nombre corto para gráfico."""
    clean = stem.replace(".meta", "")
    if clean in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[clean]
    return _shorten_spec(clean)


def _filter_final_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra un DataFrame de model metas para dejar solo los modelos finales."""
    def _is_final(source_file):
        stem = Path(source_file).stem.replace(".meta", "")
        return stem in FINAL_MODEL_STEMS
    return df[df["_source_file"].apply(_is_final)].copy()


def _shorten_spec(spec: str) -> str:
    """Acorta nombres de specs para gráficos legibles.
    'system:system_neat' → 'Sys NEAT', 'heuristic:greedy' → 'greedy', etc."""
    s = str(spec)

    # Limpiar rutas completas - extraer solo el nombre relevante
    if "GYM/CONFIG/SYSTEMS" in s.upper() or "gym/config/systems" in s.lower():
        # Extraer tipo de sistema (DT, PPO, NEAT)
        s_upper = s.upper()
        if "SYSTEM_DT" in s_upper:
            return "Sys DT"
        if "SYSTEM_PPO" in s_upper:
            return "Sys PPO"
        if "SYSTEM_NEAT" in s_upper:
            return "Sys NEAT"
        return "System"

    # Limpiar rutas de modelos NEAT
    if "GYM/DATA/MODELS/NEAT" in s.upper() or "neat_config.ini" in s.lower():
        # Extraer oponente del path
        s_lower = s.lower()
        if "greedy" in s_lower:
            return "NEAT vs greedy"
        if "denial" in s_lower:
            return "NEAT vs denial"
        if "spread" in s_lower:
            return "NEAT vs spread"
        return "NEAT"

    if s.startswith("system:system_"):
        return "Sys " + s.replace("system:system_", "").upper()
    if s.startswith("system:"):
        return "Sys " + s.replace("system:", "").upper()
    if s.startswith("heuristic:"):
        return s.replace("heuristic:", "")
    if s.startswith("baseline:"):
        return s.replace("baseline:", "")
    # sklearn_tree__vs_heuristic_greedy → DT vs greedy
    if "sklearn_tree__vs_" in s or "SKLEARN_TREE" in s.upper():
        opp = s.split("__vs_")[-1].replace("heuristic_", "") if "__vs_" in s else s
        return f"DT vs {opp}"
    # dt__ppo_vs_denial → DT vs denial
    if s.startswith("dt__ppo_vs_"):
        return f"DT vs {s.replace('dt__ppo_vs_', '')}"
    # neat__vs_heuristic_greedy__multiseed_real → NEAT vs greedy
    if "__vs_" in s:
        parts = s.split("__vs_")
        fam = parts[0].upper()
        opp = parts[1].replace("heuristic_", "")
        # Limpiar sufijos técnicos
        for suffix in ["__multiseed_real", "__extended", "__old_100gen",
                       "__old_multiseed_select", "__seed123", "__seed456", "__seed789"]:
            opp = opp.replace(suffix, "")
        return f"{fam} vs {opp}"
    if "kmeans" in s.lower():
        return "KMeans"
    return s


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
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colores consistentes
    color_best = COLORS.get("NEAT", "#FFB6C1")
    color_avg = "#4169E1"  # Azul royal para contraste

    ax.plot(df["generation"], df["best_fitness"], linewidth=2.5,
            label="Mejor Fitness", color=color_best, marker='o', markevery=30, markersize=5)
    ax.plot(df["generation"], df["avg_fitness"], linewidth=2.0, alpha=0.8,
            label="Fitness Promedio", color=color_avg, linestyle='--')
    if "std_fitness" in df.columns:
        avg = df["avg_fitness"].values
        std = df["std_fitness"].values
        ax.fill_between(df["generation"], avg - std, avg + std,
                        alpha=0.2, color=color_avg, label="±1 desv. est.")

    # Formatear nombre de oponente
    opp_name = opponent.split(":")[-1].capitalize() if ":" in opponent else opponent

    ax.set_xlabel("Generación", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fitness", fontsize=12, fontweight='bold')
    ax.set_title(f"Evolución de NEAT vs {opp_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    out = PLOTS_DIR / f"neat_evolution_{opponent.replace(':', '_')}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_training_time_comparison():
    """Barras: tiempo de entrenamiento (s) por modelo (solo modelos finales)."""
    df = load_all_model_metas()
    if df.empty or "wall_time_seconds" not in df.columns:
        print("[SKIP] No hay datos de wall_time")
        return
    _ensure_plots_dir()
    apply_thesis_style()

    df = df[df["wall_time_seconds"].notna()].copy()
    df = _filter_final_models(df)
    if df.empty:
        print("[SKIP] No hay modelos finales con wall_time")
        return

    labels = df["_source_file"].apply(lambda x: _model_display_name(Path(x).stem))

    # Asignar colores por paradigma
    def get_bar_color(source):
        s = str(source).lower()
        if "ppo" in s and "dt" not in s:
            return COLORS["PPO"]
        elif "neat" in s:
            return COLORS["NEAT"]
        elif "dt" in s or "sklearn" in s:
            return COLORS["DT"]
        elif "kmeans" in s:
            return COLORS["KMeans"]
        return "#808080"

    colors = df["_source_file"].apply(get_bar_color)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, df["wall_time_seconds"], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Tiempo de entrenamiento (segundos)", fontsize=12, fontweight='bold')
    ax.set_xscale("log")
    ax.set_title("Tiempo de Entrenamiento por Modelo", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x", linestyle='-', linewidth=0.8)
    ax.tick_params(axis='y', labelsize=10)

    # Añadir valores en las barras
    for bar, val in zip(bars, df["wall_time_seconds"]):
        if val >= 60:
            label = f"{val/60:.1f} min"
        else:
            label = f"{val:.1f} s"
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    out = PLOTS_DIR / "training_time_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_model_size_comparison():
    """Barras: tamano del modelo en KB (solo modelos finales)."""
    df = load_all_model_metas()
    if df.empty:
        print("[SKIP] No hay datos de modelos")
        return
    _ensure_plots_dir()
    apply_thesis_style()

    df = _filter_final_models(df)
    rows = []
    for _, row in df.iterrows():
        sz = 0
        for c in ["model_size_bytes", "genome_size_bytes", "artifact_size_bytes"]:
            if c in row and pd.notna(row[c]):
                sz = row[c]
                break
        raw_name = Path(row["_source_file"]).stem.replace(".meta", "")
        source = str(row["_source_file"]).lower()

        # Determinar paradigma para color
        if "ppo" in source and "dt" not in source:
            color = COLORS["PPO"]
        elif "neat" in source:
            color = COLORS["NEAT"]
        elif "dt" in source or "sklearn" in source:
            color = COLORS["DT"]
        else:
            color = "#808080"

        rows.append({
            "model": _model_display_name(raw_name),
            "size_kb": sz / 1024,
            "color": color,
        })
    sdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sdf["model"], sdf["size_kb"], color=sdf["color"], alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Tamaño del modelo (KB)", fontsize=12, fontweight='bold')
    ax.set_xscale("log")
    ax.set_title("Comparativa de Tamaño de Modelo", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis="x", linestyle='-', linewidth=0.8)
    ax.tick_params(axis='y', labelsize=10)

    # Añadir valores en las barras
    for bar, val in zip(bars, sdf["size_kb"]):
        label = f"{val:.1f} KB"
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

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
        p0 = _shorten_spec(row.get("p0_spec", row.get("p0_name", "?")))
        p1 = _shorten_spec(row.get("p1_spec", row.get("p1_name", "?")))
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
    """Genera 4 box plots de score_diff: uno por paradigma (DT, PPO, NEAT) y uno consolidado."""
    df = load_run_summaries()
    if df.empty or "score_diff_percentiles" not in df.columns:
        print("[SKIP] No hay percentiles de score_diff")
        return
    _ensure_plots_dir()

    # Recolectar datos por paradigma
    paradigm_data = {"DT": [], "PPO": [], "NEAT": []}

    # Ordenar por fecha (más reciente primero)
    df_sorted = df.sort_values("_source_file", ascending=False) if "_source_file" in df.columns else df

    seen_combos = {"DT": set(), "PPO": set(), "NEAT": set()}

    for _, row in df_sorted.iterrows():
        p0 = str(row.get("p0_spec", ""))
        p1 = str(row.get("p1_spec", ""))
        pct = row.get("score_diff_percentiles")

        if not isinstance(pct, dict):
            continue

        p0_lower = p0.lower()
        p1_short = _shorten_spec(p1)

        # Determinar paradigma
        paradigm = None
        if "system_dt" in p0_lower or "dt__ppo" in p0_lower or "sklearn_tree" in p0_lower:
            paradigm = "DT"
            label = f"Sys DT vs {p1_short}" if "system" in p0_lower else f"DT vs {p1_short}"
        elif "system_ppo" in p0_lower or ("ppo" in p0_lower and "dt" not in p0_lower):
            paradigm = "PPO"
            label = f"Sys PPO vs {p1_short}" if "system" in p0_lower else f"PPO vs {p1_short}"
        elif "system_neat" in p0_lower or "neat" in p0_lower:
            paradigm = "NEAT"
            label = f"Sys NEAT vs {p1_short}" if "system" in p0_lower else f"NEAT vs {p1_short}"

        if paradigm and label not in seen_combos[paradigm]:
            seen_combos[paradigm].add(label)
            box = [pct["p5"], pct["p25"], pct["p50"], pct["p75"], pct["p95"]]
            paradigm_data[paradigm].append((label, box))

    # Generar gráfico por cada paradigma
    colors = {"DT": COLORS["DT"], "PPO": COLORS["PPO"], "NEAT": COLORS["NEAT"]}

    for paradigm, data in paradigm_data.items():
        if not data:
            print(f"[SKIP] No hay datos para {paradigm}")
            continue
        apply_thesis_style()

        # Ordenar por label
        data_sorted = sorted(data, key=lambda x: x[0])
        labels = [d[0] for d in data_sorted]
        box_data = [d[1] for d in data_sorted]

        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(box_data, tick_labels=labels, vert=True, patch_artist=True,
                        whis=[0, 100], showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[paradigm])
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        for median in bp["medians"]:
            median.set_color('darkred')
            median.set_linewidth(2)
        ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_ylabel("Score Diff (agente − oponente)", fontsize=12, fontweight='bold')
        ax.set_title(f"Diferencial de Puntaje — {paradigm}", fontsize=14, fontweight='bold')
        ax.tick_params(axis="x", rotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="y", linestyle='-', linewidth=0.8)

        out = PLOTS_DIR / f"score_diff_{paradigm.lower()}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")

    # Gráfico consolidado (solo sistemas, uno por combinación sistema-oponente)
    consolidated = []
    seen_consolidated = set()

    for paradigm, data in paradigm_data.items():
        for label, box in data:
            # Solo incluir sistemas (no IAs individuales)
            if "Sys" in label and label not in seen_consolidated:
                seen_consolidated.add(label)
                consolidated.append((paradigm, label, box))

    if consolidated:
        # Ordenar por paradigma y luego por label
        consolidated_sorted = sorted(consolidated, key=lambda x: (x[0], x[1]))
        full_labels = [c[1] for c in consolidated_sorted]
        box_data = [c[2] for c in consolidated_sorted]
        paradigms = [c[0] for c in consolidated_sorted]

        # Crear variables cortas: A1, A2, A3 para DT; B1, B2, B3 para NEAT; C1, C2, C3 para PPO
        var_prefixes = {"DT": "A", "NEAT": "B", "PPO": "C"}
        var_counters = {"DT": 0, "NEAT": 0, "PPO": 0}
        short_labels = []
        var_mapping = []  # Lista de (variable, descripción completa)

        for i, (paradigm, full_label, _) in enumerate(consolidated_sorted):
            var_counters[paradigm] += 1
            var = f"{var_prefixes[paradigm]}{var_counters[paradigm]}"
            short_labels.append(var)
            var_mapping.append((var, full_label))

        apply_thesis_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(box_data, tick_labels=short_labels, vert=True, patch_artist=True,
                        whis=[0, 100], showfliers=False)

        # Colorear por paradigma con bordes
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[paradigms[i]])
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        for median in bp["medians"]:
            median.set_color('darkred')
            median.set_linewidth(2)

        ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_ylabel("Score Diff (agente − oponente)", fontsize=12, fontweight='bold')
        ax.set_title("Diferencial de Puntaje — Sistemas Adaptativos", fontsize=14, fontweight='bold')
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, alpha=0.3, axis="y", linestyle='-', linewidth=0.8)

        # Leyenda de colores
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors["DT"], edgecolor='black', label="A: Sistema DT"),
            Patch(facecolor=colors["NEAT"], edgecolor='black', label="B: Sistema NEAT"),
            Patch(facecolor=colors["PPO"], edgecolor='black', label="C: Sistema PPO"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

        out = PLOTS_DIR / "score_diff_boxplot.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")

        # Guardar mapeo de variables para referencia en la tesis
        mapping_file = PLOTS_DIR / "score_diff_variable_mapping.txt"
        with open(mapping_file, "w", encoding="utf-8") as f:
            f.write("Mapeo de variables para Figura de Diferencial de Puntaje:\n")
            f.write("="*60 + "\n\n")
            for var, desc in var_mapping:
                f.write(f"{var}: {desc}\n")
        print(f"[OK] {mapping_file}")


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


def plot_cross_eval_heatmap():
    """Heatmap de evaluación cruzada: 9 especialistas × 3 oponentes."""
    cross_path = Path("gym/data/results/reports/cross_evaluation.json")
    if not cross_path.exists():
        print("[SKIP] No hay cross_evaluation.json")
        return
    _ensure_plots_dir()

    data = _load_json(cross_path)
    results = data.get("results", {})
    opponents = ["denial", "spread", "greedy"]

    # Organizar por familia
    families = {"NEAT": [], "PPO": [], "DT": []}
    for spec_name in sorted(results.keys()):
        if spec_name.startswith("neat"):
            families["NEAT"].append(spec_name)
        elif spec_name.startswith("ppo"):
            families["PPO"].append(spec_name)
        elif spec_name.startswith("dt"):
            families["DT"].append(spec_name)

    ordered_specs = families["NEAT"] + families["PPO"] + families["DT"]
    n_specs = len(ordered_specs)

    matrix = np.zeros((n_specs, len(opponents)))
    labels_y = []
    for i, spec_name in enumerate(ordered_specs):
        short = spec_name.replace("_vs_", " → ")
        labels_y.append(short)
        for j, opp in enumerate(opponents):
            r = results.get(spec_name, {}).get(opp, {})
            matrix[i, j] = r.get("winrate", 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.45, vmax=0.70, aspect="auto")

    ax.set_xticks(range(len(opponents)))
    ax.set_xticklabels([f"vs {o}" for o in opponents], fontsize=10)
    ax.set_yticks(range(n_specs))
    ax.set_yticklabels(labels_y, fontsize=9)

    # Anotaciones
    for i in range(n_specs):
        for j in range(len(opponents)):
            val = matrix[i, j]
            # Marcar el oponente de entrenamiento
            spec_name = ordered_specs[i]
            trained_against = spec_name.split("_vs_")[1] if "_vs_" in spec_name else ""
            marker = " ★" if trained_against == opponents[j] else ""
            color = "white" if val < 0.52 or val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    # Líneas separadoras entre familias
    for sep in [len(families["NEAT"]), len(families["NEAT"]) + len(families["PPO"])]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, label="Winrate (decided)", shrink=0.8)
    ax.set_title("Evaluación Cruzada — Generalización de Especialistas\n★ = oponente de entrenamiento",
                 fontsize=11)

    out = PLOTS_DIR / "cross_eval_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_ablation_bars():
    """Barras agrupadas del ablation study: 4 condiciones × 3 familias."""
    stats_path = Path("gym/data/results/reports/statistical_analysis.json")
    if not stats_path.exists():
        print("[SKIP] No hay statistical_analysis.json — ejecutar statistical_analysis.py primero")
        return
    _ensure_plots_dir()

    stats = _load_json(stats_path)
    ablation = stats.get("ablation_study", {})
    if not ablation:
        print("[SKIP] No hay datos de ablation en statistical_analysis.json")
        return

    families = ["DT", "PPO", "NEAT"]
    cond_names = ["Peor\ngeneralista", "Mejor\ngeneralista", "Sistema\nadaptativo", "Oráculo"]
    cond_keys = ["worst_generalist", "best_generalist", "system_adaptive", "oracle"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cond_names))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, fam in enumerate(families):
        fkey = fam.lower()
        if fkey not in ablation:
            continue
        vals = [ablation[fkey].get(ck, {}).get("avg_wr", 0) for ck in cond_keys]
        offset = (i - len(families) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=fam, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=10)
    ax.set_ylabel("Winrate Promedio (decided)")
    ax.set_title("Ablation Study — ¿Aporta la Adaptación KMeans?")
    ax.legend(title="Familia")
    ax.set_ylim(0.50, 0.65)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    out = PLOTS_DIR / "ablation_study.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_paradigm_comparison():
    """Tabla visual de comparación de paradigmas (WR, inferencia, tamaño, entrenamiento)."""
    _ensure_plots_dir()

    paradigms = ["PPO", "NEAT", "DT"]
    metrics = {
        "Winrate": [0.606, 0.607, 0.609],
        "Velocidad inferencia\n(juegos/s, 5k)": [1/0.00826, 1/0.00628, 1/0.00526],  # inversa de wall/game
        "Tamaño modelo (KB)": [161.5, 2.0, 140.0],
        "Tiempo entren. (min)": [12.0, 10.0, 0.08],  # DT ~5s = 0.08 min
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for ax, (metric, values) in zip(axes, metrics.items()):
        bars = ax.bar(paradigms, values, color=colors, alpha=0.85)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, values):
            label = f"{val:.1f}" if val > 1 else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Comparación de Paradigmas de IA", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "paradigm_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_system_winrates_grouped():
    """Barras agrupadas: 3 sistemas × 3 oponentes con IC 95%."""
    import math
    _ensure_plots_dir()

    # Datos directos de las evaluaciones
    systems = {
        "System DT": {"denial": (0.587, 4903), "spread": (0.658, 4893), "greedy": (0.583, 4915)},
        "System PPO": {"denial": (0.584, 4891), "spread": (0.652, 4901), "greedy": (0.581, 4908)},
        "System NEAT": {"denial": (0.576, 4897), "spread": (0.667, 4893), "greedy": (0.578, 4898)},
    }
    opponents = ["denial", "spread", "greedy"]
    sys_names = list(systems.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(opponents))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, sys_name in enumerate(sys_names):
        wrs = [systems[sys_name][opp][0] for opp in opponents]
        ns = [systems[sys_name][opp][1] for opp in opponents]
        ses = [1.96 * math.sqrt(wr * (1 - wr) / n) for wr, n in zip(wrs, ns)]

        offset = (i - len(sys_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, wrs, width, yerr=ses, capsize=4,
                      label=sys_name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, wrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"vs {o}" for o in opponents], fontsize=11)
    ax.set_ylabel("Winrate (decided)")
    ax.set_title("Winrate de Sistemas Adaptativos vs Heurísticos\n(5,000 partidas c/u, barras = IC 95%)")
    ax.legend(loc="upper left")
    ax.set_ylim(0.50, 0.72)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    out = PLOTS_DIR / "system_winrates_grouped.png"
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

    print("\n=== Consolidated Analysis ===")
    plot_cross_eval_heatmap()
    plot_ablation_bars()
    plot_paradigm_comparison()
    plot_system_winrates_grouped()

    print("\n=== Tables ===")
    export_models_table()
    export_evaluations_table()

    print(f"\nPlots  -> {PLOTS_DIR}/")
    print(f"Tables -> {TABLES_DIR}/")


if __name__ == "__main__":
    main()

