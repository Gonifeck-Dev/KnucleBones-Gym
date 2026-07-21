# gym/scripts/utils/run_manager.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


def _utc_stamp() -> str:
    # timezone-aware UTC (evita deprecaciones)
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s).strip("_")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    summary_path: Path
    games_path: Path
    turns_path: Path


class RunManager:
    """
    Regla C:
    - Crea carpeta por corrida en gym/data/results/runs/
    - Escribe:
        - config.json
        - turns.jsonl (1 registro por turno)
        - games.jsonl (1 registro por partida)
        - summary.json
    """

    def __init__(self, base_dir: str | Path = "gym/data/results/runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.paths: Optional[RunPaths] = None
        self._turns_fh: Optional[TextIO] = None
        self._games_fh: Optional[TextIO] = None

    def start_run(
        self,
        algo_tag: str,
        p0_name: str,
        p1_name: str,
        games: int,
        seed: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RunPaths:
        stamp = _utc_stamp()
        run_name = (
            f"{stamp}__{_safe_name(algo_tag)}__"
            f"{_safe_name(p0_name)}_vs_{_safe_name(p1_name)}__games{int(games)}__seed{int(seed)}"
        )
        run_dir = self.base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        games_path = run_dir / "games.jsonl"
        turns_path = run_dir / "turns.jsonl"

        config = {
            "run_name": run_name,
            "algo_tag": algo_tag,
            "p0_name": p0_name,
            "p1_name": p1_name,
            "games": int(games),
            "seed": int(seed),
            "created_utc": stamp,
            "extra": extra or {},
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        self._turns_fh = turns_path.open("a", encoding="utf-8")
        self._games_fh = games_path.open("a", encoding="utf-8")

        self.paths = RunPaths(
            run_dir=run_dir,
            config_path=config_path,
            summary_path=summary_path,
            games_path=games_path,
            turns_path=turns_path,
        )
        return self.paths

    def log_turn(self, record: Dict[str, Any]) -> None:
        if self._turns_fh is None:
            raise RuntimeError("RunManager not started. Call start_run() first.")
        self._turns_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_game(self, record: Dict[str, Any]) -> None:
        if self._games_fh is None:
            raise RuntimeError("RunManager not started. Call start_run() first.")
        self._games_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, summary: Dict[str, Any]) -> None:
        if self.paths is None:
            raise RuntimeError("RunManager not started. Call start_run() first.")

        if self._turns_fh is not None:
            self._turns_fh.flush()
            self._turns_fh.close()
            self._turns_fh = None

        if self._games_fh is not None:
            self._games_fh.flush()
            self._games_fh.close()
            self._games_fh = None

        self.paths.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )