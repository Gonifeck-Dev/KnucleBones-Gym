"""Evaluate all 3 systems vs all 3 opponents (9 matchups, 5000 games each)."""
import json, subprocess, os, time

systems = {
    "system_neat": "system:gym/config/systems/system_neat.json",
    "system_ppo":  "system:gym/config/systems/system_ppo.json",
    "system_dt":   "system:gym/config/systems/system_dt.json",
}
opponents = ["heuristic:denial", "heuristic:spread", "heuristic:greedy"]
GAMES = 5000
SEED = 123

results = {}
t0 = time.perf_counter()

for sname, sspec in systems.items():
    results[sname] = {}
    for opp in opponents:
        opp_short = opp.split(":")[1]
        print(f"\n=== {sname} vs {opp_short} ({GAMES} games) ===")
        cmd = [
            "python", "-m", "gym.scripts.utils.evaluate_any",
            "--p0", sspec, "--p1", opp,
            "--games", str(GAMES), "--seed", str(SEED)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd="F:/Programacion/entrenamiento")
        # Parse output
        for line in (r.stdout + r.stderr).split("\n"):
            if "winrate_p0_decided" in line:
                # Extract from dict-like output
                import ast
                try:
                    d = ast.literal_eval(line.strip())
                    wr = d["winrate_p0_decided"]
                    se = d.get("winrate_p0_decided_se", 0)
                    diff = d.get("avg_score_diff_p0_minus_p1", 0)
                    wall = d.get("wall_time_seconds", 0)
                    results[sname][opp_short] = {
                        "wr": round(wr, 4), "se": round(se, 4),
                        "diff": round(diff, 2), "wall": round(wall, 1)
                    }
                    print(f"  WR={wr:.4f} ± {se:.4f}  diff={diff:.1f}  time={wall:.1f}s")
                except:
                    pass

elapsed = time.perf_counter() - t0
print(f"\n{'='*60}")
print(f"TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*60}")

# Print summary table
print("\n| Sistema | vs denial | vs spread | vs greedy | Promedio |")
print("|---------|-----------|-----------|-----------|----------|")
for sname in systems:
    vals = results.get(sname, {})
    d = vals.get("denial", {}).get("wr", 0)
    s = vals.get("spread", {}).get("wr", 0)
    g = vals.get("greedy", {}).get("wr", 0)
    avg = (d + s + g) / 3 if (d and s and g) else 0
    print(f"| {sname:12s} | {d:.4f} | {s:.4f} | {g:.4f} | {avg:.4f} |")

# Save JSON
with open("gym/data/results/reports/system_evaluation_final.json", "w") as f:
    json.dump({"results": results, "games_per_matchup": GAMES, "seed": SEED,
               "total_time_seconds": round(elapsed, 1)}, f, indent=2)
print("\n[OK] Results saved to: gym/data/results/reports/system_evaluation_final.json")

