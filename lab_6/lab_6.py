from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

N_AGENTS = 10
SEED = 2026
EPS = 1e-6
MAX_ITER = 10_000

INITIAL_LOW = 1
INITIAL_HIGH = 20

PLAYER_1_LOW = 0
PLAYER_1_HIGH = 100
PLAYER_2_LOW = -100
PLAYER_2_HIGH = 0


@dataclass(frozen=True)
class SimulationResult:
    x0: np.ndarray
    final_x: np.ndarray
    iterations: int
    last_delta: float
    convergence_rows: list[tuple[int, float, float, float, float]]


@dataclass(frozen=True)
class InfluenceScenario:
    player_1_agents: list[int]
    player_2_agents: list[int]
    neutral_agents: list[int]
    u: int
    v: int
    x0: np.ndarray


def fmt_vector(values: Iterable[float], digits: int = 3) -> str:
    return "(" + "; ".join(f"{x:.{digits}f}" for x in values) + ")"


def fmt_int_vector(values: Iterable[int]) -> str:
    return "(" + "; ".join(str(int(x)) for x in values) + ")"


def print_matrix(name: str, matrix: np.ndarray, digits: int = 3) -> None:
    print(f"{name} =")
    for row in matrix:
        print("  " + " ".join(f"{x:8.{digits}f}" for x in row))
    print()


def to_agent_numbers(indices: Iterable[int]) -> list[int]:
    return [i + 1 for i in indices]

def generate_trust_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.random((n, n)) + 1e-3
    return raw / raw.sum(axis=1, keepdims=True)


def generate_initial_opinions(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(INITIAL_LOW, INITIAL_HIGH + 1, size=n).astype(float)


def choose_influence_scenario(
        n: int,
        rng: np.random.Generator,
        base_x0: np.ndarray,
) -> InfluenceScenario:
    max_count = max(1, n // 3)
    count_1 = int(rng.integers(1, max_count + 1))
    count_2 = int(rng.integers(1, max_count + 1))

    permutation = rng.permutation(n)
    player_1_agents = sorted(map(int, permutation[:count_1]))
    player_2_agents = sorted(map(int, permutation[count_1:count_1 + count_2]))
    used = set(player_1_agents) | set(player_2_agents)
    neutral_agents = [i for i in range(n) if i not in used]

    u = int(rng.integers(PLAYER_1_LOW, PLAYER_1_HIGH + 1))
    v = int(rng.integers(PLAYER_2_LOW, PLAYER_2_HIGH + 1))

    x0 = base_x0.astype(float).copy()

    x0[player_1_agents] = u
    x0[player_2_agents] = v

    return InfluenceScenario(
        player_1_agents=player_1_agents,
        player_2_agents=player_2_agents,
        neutral_agents=neutral_agents,
        u=u,
        v=v,
        x0=x0,
    )


def is_row_stochastic(A: np.ndarray, tol: float = 1e-10) -> bool:
    non_negative = np.all(A >= -tol)
    row_sums_are_one = np.allclose(A.sum(axis=1), 1.0, atol=tol)
    return bool(non_negative and row_sums_are_one)


def simulate_opinions(A: np.ndarray, x0: np.ndarray, eps: float, max_iter: int) -> SimulationResult:
    x_prev = x0.astype(float).copy()
    convergence_rows: list[tuple[int, float, float, float, float]] = []

    for t in range(1, max_iter + 1):
        x_next = A @ x_prev
        delta = float(np.max(np.abs(x_next - x_prev)))
        min_x = float(np.min(x_next))
        max_x = float(np.max(x_next))
        spread = max_x - min_x
        convergence_rows.append((t, delta, min_x, max_x, spread))

        if delta < eps:
            return SimulationResult(
                x0=x0,
                final_x=x_next,
                iterations=t,
                last_delta=delta,
                convergence_rows=convergence_rows,
            )
        x_prev = x_next

    raise RuntimeError(f"Мнения не сошлись за {max_iter} итераций. Последняя delta = {delta}")


def stationary_row(A: np.ndarray, eps: float, max_iter: int) -> tuple[np.ndarray, int, float]:
    n = A.shape[0]
    r_prev = np.full(n, 1.0 / n)

    for t in range(1, max_iter + 1):
        r_next = r_prev @ A
        delta = float(np.max(np.abs(r_next - r_prev)))
        if delta < eps:
            return r_next, t, delta
        r_prev = r_next

    raise RuntimeError("Предельная строка матрицы доверия не была найдена за max_iter итераций.")


def limiting_matrix_power(A: np.ndarray, eps: float, max_iter: int) -> tuple[np.ndarray, int, float]:
    n = A.shape[0]
    A_prev = np.eye(n)

    for k in range(1, max_iter + 1):
        A_next = A_prev @ A
        delta = float(np.max(np.abs(A_next - A_prev)))

        if k > 1 and delta < eps:
            return A_next, k, delta

        A_prev = A_next

    raise RuntimeError(
        f"Матрица A^k не сошлась за {max_iter} итераций. Последняя delta = {delta}"
    )


def interpret_winner(final_value: float, player_1_target: float, player_2_target: float) -> str:
    distance_to_player_1 = abs(final_value - player_1_target)
    distance_to_player_2 = abs(final_value - player_2_target)

    if distance_to_player_1 < distance_to_player_2:
        return (
            "выиграл первый игрок\n"
            f"расстояние до цели первого игрока = {distance_to_player_1:.6f}\n"
            f"расстояние до цели второго игрока = {distance_to_player_2:.6f}\n"
        )

    if distance_to_player_2 < distance_to_player_1:
        return (
            "выиграл второй игрок\n"
            f"расстояние до цели первого игрока = {distance_to_player_1:.6f}\n"
            f"расстояние до цели второго игрока = {distance_to_player_2:.6f}\n"
        )

    return (
        "ничья: итоговое мнение равноудалено от целей игроков "
        f"(расстояние = {distance_to_player_1:.6f})"
    )


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Лабораторная работа № 6")
    print("Информационное противоборство в социальных сетях")
    print("Закон изменения мнений: x(t) = A x(t-1)")
    print(f"Количество агентов: {N_AGENTS}")
    print(f"Seed случайной генерации: {SEED}")
    print(f"Точность остановки eps: {EPS:g}")
    print("=" * 80)

    A = generate_trust_matrix(N_AGENTS, rng)
    print("1. Сгенерированная матрица доверия")
    print_matrix("A", A, digits=3)
    print("Проверка стохастичности матрицы доверия:")
    print(f"  все элементы неотрицательны: {np.all(A >= 0)}")
    print(f"  суммы строк: {fmt_vector(A.sum(axis=1), digits=6)}")
    print(f"  матрица стохастическая по строкам: {is_row_stochastic(A)}")
    print("=" * 80)

    x0_plain = generate_initial_opinions(N_AGENTS, rng)
    plain_result = simulate_opinions(A, x0_plain, EPS, MAX_ITER)
    r, r_iters, r_delta = stationary_row(A, EPS, MAX_ITER)
    A_limit, A_limit_iters, A_limit_delta = limiting_matrix_power(A, EPS, MAX_ITER)
    theoretical_plain = float(r @ x0_plain)

    print("2. Моделирование без информационного управления")
    print(f"Начальные мнения x(0): {fmt_int_vector(x0_plain.astype(int))}")
    print_convergence_table(plain_result.convergence_rows)
    print(f"Результирующие мнения x({plain_result.iterations}): {fmt_vector(plain_result.final_x)}")
    print(f"Число итераций до сходимости: {plain_result.iterations}")
    print(f"Последнее max|x(t)-x(t-1)|: {plain_result.last_delta:.8f}")
    print(f"Предельная строка r матрицы A^k: {fmt_vector(r, digits=6)}")
    print(f"Полная матрица A^k при k = {A_limit_iters}:")
    print_matrix(f"A^{A_limit_iters}", A_limit, digits=6)
    print(f"Последнее max|A^k - A^(k-1)|: {A_limit_delta:.8f}")
    print(f"Проверка по формуле X = r * x(0): {theoretical_plain:.6f}")
    print("=" * 80)

    scenario = choose_influence_scenario(N_AGENTS, rng, x0_plain)
    influence_result = simulate_opinions(A, scenario.x0, EPS, MAX_ITER)
    theoretical_influence = float(r @ scenario.x0)
    final_value = float(np.mean(influence_result.final_x))

    print("3. Моделирование с информационным управлением")
    print(f"Агенты влияния первого игрока: {to_agent_numbers(scenario.player_1_agents)}")
    print(f"Агенты влияния второго игрока: {to_agent_numbers(scenario.player_2_agents)}")
    print(f"Нейтральные агенты: {to_agent_numbers(scenario.neutral_agents)}")
    print(f"Управление первого игрока u: {scenario.u}")
    print(f"Управление второго игрока v: {scenario.v}")
    print("Начальные мнения нейтральных агентов:")
    for i in scenario.neutral_agents:
        print(f"  агент {i + 1}: {int(scenario.x0[i])}")
    print(f"Начальный вектор с учетом управления x(0): {fmt_int_vector(scenario.x0.astype(int))}")
    print_convergence_table(influence_result.convergence_rows)
    print(f"Результирующие мнения x({influence_result.iterations}): {fmt_vector(influence_result.final_x)}")
    print(f"Число итераций до сходимости: {influence_result.iterations}")
    print(f"Последнее max|x(t)-x(t-1)|: {influence_result.last_delta:.8f}")
    print(f"Проверка по формуле X = r * x(0): {theoretical_influence:.6f}")
    print(f"Итоговое мнение: {final_value:.6f};")
    print(f"{interpret_winner(final_value, scenario.u, scenario.v)}.")
    print("=" * 80)


def print_convergence_table(rows: list[tuple[int, float, float, float, float]]) -> None:
    print("Таблица сходимости:")
    print("  t | max|x(t)-x(t-1)| | min x(t) | max x(t) | разброс")
    print("----+-------------------+----------+----------+---------")
    for t, delta, min_x, max_x, spread in rows:
        print(f"{t:3d} | {delta:17.8f} | {min_x:8.3f} | {max_x:8.3f} | {spread:7.3f}")
    print()


if __name__ == "__main__":
    main()
