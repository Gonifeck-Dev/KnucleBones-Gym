# gym/scripts/utils/artifacts_doctor.py
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass(frozen=True)
class AliasRule:
    alias: Path
    folder: Path
    ext: str
    min_bytes: int = 1024
    prefer_contains: Optional[str] = None  # substring preference


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return -1


def is_valid_binary(p: Path, min_bytes: int) -> bool:
    return p.exists() and p.is_file() and file_size(p) >= min_bytes


def pick_best_candidate(rule: AliasRule) -> Optional[Path]:
    candidates = [p for p in rule.folder.glob(f"*{rule.ext}") if p.is_file()]
    # exclude the alias itself
    candidates = [p for p in candidates if p.resolve() != rule.alias.resolve()]
    # only non-trivial size
    candidates = [p for p in candidates if file_size(p) >= rule.min_bytes]

    if not candidates:
        return None

    # Prefer name containing substring (optional)
    if rule.prefer_contains:
        preferred = [p for p in candidates if rule.prefer_contains in p.name]
        if preferred:
            candidates = preferred

    # Pick largest file (most likely the real artifact)
    candidates.sort(key=lambda p: file_size(p), reverse=True)
    return candidates[0]


def make_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    # Try hardlink first (best, avoids corruption)
    try:
        os.link(src, dst)
        return
    except Exception:
        # fallback to copy
        shutil.copy2(src, dst)


def doctor(rules: List[AliasRule], fix: bool) -> Tuple[int, int]:
    ok = 0
    bad = 0

    for rule in rules:
        alias_ok = is_valid_binary(rule.alias, rule.min_bytes)

        if alias_ok:
            ok += 1
            print(f"[OK]  {rule.alias} ({file_size(rule.alias)} bytes)")
            continue

        bad += 1
        print(f"[BAD] {rule.alias} (size={file_size(rule.alias)} bytes)")

        best = pick_best_candidate(rule)
        if best is None:
            print(f"      -> No candidate found in {rule.folder} for ext={rule.ext}. You must retrain or restore the artifact.")
            continue

        print(f"      -> Best candidate: {best} ({file_size(best)} bytes)")
        if fix:
            make_hardlink_or_copy(best, rule.alias)
            post = file_size(rule.alias)
            if post >= rule.min_bytes:
                print(f"      -> FIXED: {rule.alias} ({post} bytes)")
            else:
                print(f"      -> FAILED: alias still too small ({post} bytes)")

    return ok, bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate and fix canonical model aliases.")
    ap.add_argument("--fix", action="store_true", help="Attempt to fix aliases by hardlink/copy from best candidate.")
    args = ap.parse_args()

    base = Path("gym/data/models")

    rules = [
        # DT specialists (destilled from PPO)
        AliasRule(
            alias=base / "sklearn" / "dt__ppo_vs_denial.joblib",
            folder=base / "sklearn",
            ext=".joblib",
            min_bytes=2048,
            prefer_contains="denial",
        ),
        AliasRule(
            alias=base / "sklearn" / "dt__ppo_vs_spread.joblib",
            folder=base / "sklearn",
            ext=".joblib",
            min_bytes=2048,
            prefer_contains="spread",
        ),
        AliasRule(
            alias=base / "sklearn" / "dt__ppo_vs_greedy.joblib",
            folder=base / "sklearn",
            ext=".joblib",
            min_bytes=2048,
            prefer_contains="greedy",
        ),
        # PPO specialists
        AliasRule(
            alias=base / "rl" / "PPO__vs_heuristic_denial.zip",
            folder=base / "rl",
            ext=".zip",
            min_bytes=2048,
            prefer_contains="denial",
        ),
        AliasRule(
            alias=base / "rl" / "PPO__vs_heuristic_spread.zip",
            folder=base / "rl",
            ext=".zip",
            min_bytes=2048,
            prefer_contains="spread",
        ),
        AliasRule(
            alias=base / "rl" / "PPO__vs_heuristic_greedy.zip",
            folder=base / "rl",
            ext=".zip",
            min_bytes=2048,
            prefer_contains="greedy",
        ),
        # KMeans v2
        AliasRule(
            alias=base / "kmeans" / "kmeans__v2__k4__win6__seed123.joblib",
            folder=base / "kmeans",
            ext=".joblib",
            min_bytes=2048,
            prefer_contains="kmeans",
        ),
    ]

    ok, bad = doctor(rules, fix=bool(args.fix))
    print(f"\nSummary: ok={ok}, bad={bad}")
    if bad > 0 and not args.fix:
        print("Run again with --fix to attempt repair.")
    if bad > 0 and args.fix:
        print("If some aliases could not be fixed, retrain or restore the missing artifacts.")


if __name__ == "__main__":
    main()