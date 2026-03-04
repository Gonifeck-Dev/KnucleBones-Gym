# gym/scripts/utils/test_env_sanity.py
"""
Test mínimo de consistencia interna para KnucklebonesEnv.

Objetivo (P0):
- Verificar invariantes estructurales del estado.
- Verificar legal_actions() y rechazo de acciones ilegales.
- Verificar transición determinista T(s,a)=s' usando dice_value fijo.
- Verificar regla de destrucción.
- Verificar criterio de término y consistencia de winner/score.

Uso:
    python -m gym.scripts.utils.test_env_sanity

Notas:
- Este test NO depende de pytest/unittest: es ejecutable y falla con AssertionError.
- Está pensado para correr rápido (< 1s) y ser "sanity check" antes de entrenar/evaluar.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from gym.env.knucklebones_env import KnucklebonesEnv


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _is_board_shape_ok(board: List[List[int]]) -> bool:
    if not isinstance(board, list) or len(board) != 3:
        return False
    for col in board:
        if not isinstance(col, list) or len(col) != 3:
            return False
        for v in col:
            if not isinstance(v, int) or v < 0 or v > 6:
                return False
    return True


def _count_value_in_col(board: List[List[int]], col_idx: int, value: int) -> int:
    return sum(1 for v in board[col_idx] if v == value)


def _board_full(board: List[List[int]]) -> bool:
    return all(v != 0 for col in board for v in col)


def _print_ok(title: str) -> None:
    print(f"[OK] {title}")


def test_invariants_after_reset() -> None:
    env = KnucklebonesEnv()
    obs = env.reset(seed=123)

    required_obs = ["game_id", "episode_id", "turn", "seed", "current_player", "dice_value", "my_cols", "op_cols"]
    for k in required_obs:
        _assert(k in obs, f"obs missing key: {k}")

    _assert(obs["turn"] == 0, "turn should start at 0")
    _assert(obs["current_player"] in (0, 1), "current_player should be 0 or 1")
    _assert(obs["seed"] == 123, "seed mismatch in obs")

    _assert(_is_board_shape_ok(obs["my_cols"]), "my_cols shape/values invalid")
    _assert(_is_board_shape_ok(obs["op_cols"]), "op_cols shape/values invalid")

    _assert(sum(sum(col) for col in obs["my_cols"]) == 0, "my board should be empty at reset")
    _assert(sum(sum(col) for col in obs["op_cols"]) == 0, "op board should be empty at reset")

    _print_ok("invariants after reset")


def test_legal_actions_and_illegal_rejection() -> None:
    env = KnucklebonesEnv()
    env.reset(seed=1)

    legal = env.legal_actions()
    _assert(legal == [0, 1, 2], f"expected all cols legal at start, got {legal}")

    # Llenar col0 de p0: p0 col0, p1 col1, p0 col0, p1 col1, p0 col0
    env.step(0, dice_value=1)  # p0
    env.step(1, dice_value=1)  # p1
    env.step(0, dice_value=2)  # p0
    env.step(1, dice_value=2)  # p1
    env.step(0, dice_value=3)  # p0 -> col0 de p0 ahora llena

    legal_p1 = env.legal_actions()
    _assert(set(legal_p1).issubset({0, 1, 2}), "legal actions must be subset of {0,1,2}")
    _assert(len(legal_p1) >= 1, "p1 should have at least one legal move")

    # p1 hace una jugada legal cualquiera para avanzar
    env.step(legal_p1[0], dice_value=4)

    # Turno vuelve a p0; para p0 col0 llena => no debe ser legal
    legal_p0 = env.legal_actions()
    _assert(0 not in legal_p0, f"expected col0 illegal for p0 (full), got {legal_p0}")

    # Debe rechazar acción ilegal
    try:
        env.step(0, dice_value=5)  # p0 intenta col0 lleno
        _assert(False, "expected ValueError for illegal action but step succeeded")
    except ValueError:
        pass

    _print_ok("legal_actions and illegal action rejection")


def test_transition_determinism_with_fixed_die() -> None:
    env = KnucklebonesEnv()
    env.reset(seed=999)

    s0 = env.get_state()
    r1 = env.step(2, dice_value=6)
    s1 = env.get_state()

    env.set_state(s0)
    r2 = env.step(2, dice_value=6)
    s1b = env.get_state()

    _assert(json.dumps(s1, sort_keys=True) == json.dumps(s1b, sort_keys=True),
            "determinism failed: s1 != s1b for same (s,a,die)")

    _assert(r1.info["die"] == r2.info["die"] == 6, "die mismatch in info")
    _assert(r1.info["action"] == r2.info["action"] == 2, "action mismatch in info")

    _print_ok("transition determinism with fixed dice_value")


def test_destruction_rule() -> None:
    env = KnucklebonesEnv()
    env.reset(seed=7)

    env.step(0, dice_value=3)  # p0
    env.step(0, dice_value=3)  # p1

    before = env._get_obs(dice_value=None)  # obs para p0
    op_before = before["op_cols"]
    _assert(_count_value_in_col(op_before, 0, 3) == 1, "setup failed: expected opponent col0 to have one '3'")

    env.step(0, dice_value=3)  # p0 destruye 3 en col0 de p1

    after = env._get_obs(dice_value=None)  # obs para p1
    my_after = after["my_cols"]  # ahora my_after corresponde a p1
    _assert(_count_value_in_col(my_after, 0, 3) == 0, "destruction failed: expected all '3' removed from p1 col0")

    _print_ok("destruction rule")


def _choose_legal_action(env: KnucklebonesEnv) -> int:
    legal = env.legal_actions()
    _assert(len(legal) > 0, "no legal actions available but game not done?")
    return legal[0]  # determinista para el test


def test_done_and_winner_consistency() -> None:
    env = KnucklebonesEnv()
    env.reset(seed=42)

    # Strategy del test:
    # - p0 intenta llenar su tablero de forma sistemática.
    # - p1 juega siempre una acción legal (la primera legal) para no romper el test.
    # - Usamos dice_value fijo para hacer el test reproducible.
    #
    # Nota: No buscamos "ganar", solo forzar término (board full) y validar consistencia.

    # p0 intentará llenar por columnas 0,1,2; repetimos pattern.
    p0_cols_cycle = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    p0_dice_cycle = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    done = False
    last_info: Optional[Dict[str, Any]] = None

    # Límite de seguridad para evitar loops
    for i in range(200):
        if env._compute_done():
            done = True
            break

        if env.current_player == 0:
            # p0: intenta la columna planificada; si no es legal, elige otra legal.
            idx = min(i // 2, len(p0_cols_cycle) - 1)  # avance lento
            preferred_col = p0_cols_cycle[idx]
            die = p0_dice_cycle[idx]
            legal = env.legal_actions()
            action = preferred_col if preferred_col in legal else legal[0]
            res = env.step(action, dice_value=die)
        else:
            # p1: juega algo legal (determinista) con dado fijo.
            action = _choose_legal_action(env)
            res = env.step(action, dice_value=1)

        if res.done:
            done = True
            last_info = res.info
            break

    _assert(done, "expected episode to finish but it did not within step limit")

    # Verificar que algún tablero esté lleno en el estado actual (usamos obs).
    obs = env._get_obs(dice_value=None)

    # obs es desde el jugador actual; reconstruimos ambos tableros desde state interno
    state = env.get_state()
    b0 = state["boards"][0]
    b1 = state["boards"][1]
    _assert(_board_full(b0) or _board_full(b1), "done but no board is full in env state")

    # Consistencia winner vs scores finales (si están disponibles en info)
    if last_info and last_info.get("final_score_p0") is not None:
        s0 = last_info["final_score_p0"]
        s1 = last_info["final_score_p1"]
        w = last_info["winner"]
        if s0 > s1:
            _assert(w == 0, "winner should be 0 when final_score_p0 > final_score_p1")
        elif s1 > s0:
            _assert(w == 1, "winner should be 1 when final_score_p1 > final_score_p0")
        else:
            _assert(w is None, "winner should be None on tie")

    _print_ok("done condition and winner consistency")


def main() -> None:
    print("Running KnucklebonesEnv sanity tests...\n")
    test_invariants_after_reset()
    test_legal_actions_and_illegal_rejection()
    test_transition_determinism_with_fixed_die()
    test_destruction_rule()
    test_done_and_winner_consistency()
    print("\nAll sanity tests PASSED [OK]")


if __name__ == "__main__":
    main()