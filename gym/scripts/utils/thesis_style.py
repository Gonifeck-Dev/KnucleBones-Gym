# gym/scripts/utils/thesis_style.py
"""
Estilo visual unificado para todos los gráficos de la tesis.
Garantiza consistencia tipográfica, colores y tamaños para impresión.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# Paleta de colores consistente para paradigmas
COLORS = {
    "DT": "#90EE90",      # Verde claro
    "PPO": "#87CEEB",     # Azul cielo
    "NEAT": "#FFB6C1",    # Rosa claro
    "KMeans": "#DDA0DD",  # Púrpura claro
    "greedy": "#FFA07A",  # Salmón
    "denial": "#ADD8E6",  # Azul claro
    "spread": "#98FB98",  # Verde pálido
    "session": "#32CD32", # Verde lima
    "required": "#DC143C", # Carmesí
}

# Configuración de estilo para tesis
THESIS_STYLE = {
    # Fuentes - tamaño legible para impresión
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,

    # Líneas y bordes
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.linewidth": 0.8,

    # Grilla
    "axes.grid": True,
    "grid.alpha": 0.3,

    # Leyenda
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",

    # Figuras
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,

    # Backend
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def apply_thesis_style():
    """Aplica el estilo unificado de la tesis a matplotlib."""
    # Usar estilo base compatible con múltiples versiones
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Usar defaults de matplotlib
    mpl.rcParams.update(THESIS_STYLE)


def get_paradigm_color(paradigm: str) -> str:
    """Retorna el color consistente para un paradigma."""
    key = paradigm.upper() if paradigm.upper() in ["DT", "PPO", "NEAT"] else paradigm.lower()
    return COLORS.get(key, "#808080")


def create_figure(width: float = 10, height: float = 5):
    """Crea una figura con el estilo de tesis aplicado."""
    apply_thesis_style()
    return plt.subplots(figsize=(width, height))


