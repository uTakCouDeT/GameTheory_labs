"""
Лабораторная работа № 6.
Информационное противоборство.

По методичке:
1) для 10 агентов генерируется стохастическая матрица доверия A;
2) находится результирующая матрица доверия A^inf;
3) выбираются непересекающиеся агенты влияния двух игроков;
4) по параметрам варианта строятся функции выигрыша и целевые функции;
5) аналитически находится равновесие Нэша по управлениям u и v;
6) вычисляется итоговое мнение X(u, v), идеальные мнения игроков и расстояния до них.

По умолчанию решается вариант 13:
    g_f = 3, g_s = 2, a = 1, b = 3, c = 3, d = 2.

Чтобы решить другой вариант, измените константу VARIANT или значения в словаре VARIANTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# -----------------------------------------------------------------------------
# Настройки, которые обычно меняют при решении другого варианта
# -----------------------------------------------------------------------------
VARIANT = 13
N_AGENTS = 10
SEED = 2026
EPS = 1e-9
MAX_ITER = 10_000

# Случайные начальные мнения нейтральных агентов генерируются из этого отрезка.
# В методичке функции H_f(X) и H_s(X) записаны для числовой шкалы мнений,
# поэтому по умолчанию используется удобный нормированный отрезок [0; 1].
INITIAL_OPINION_LOW = 0.0
INITIAL_OPINION_HIGH = 1.0

# Количество агентов влияния выбирается случайно.
# Если нужно задать их вручную, заполните MANUAL_F_AGENTS и MANUAL_S_AGENTS
# номерами агентов от 1 до 10, например: [4, 5, 6].
MAX_INFLUENCE_AGENTS_PER_PLAYER = 3
ALLOW_ZERO_INFLUENCE_AGENTS = False
MANUAL_F_AGENTS: list[int] | None = None
MANUAL_S_AGENTS: list[int] | None = None


@dataclass(frozen=True)
class VariantParameters:
    gf: float
    gs: float
    a: float
    b: float
    c: float
    d: float


# Таблица 8.1 из методички: g_f, g_s, a, b, c, d.
VARIANTS: dict[int, VariantParameters] = {
    1: VariantParameters(1, 2, 1, 5, 1, 2),
    2: VariantParameters(2, 3, 2, 4, 2, 1),
    3: VariantParameters(3, 4, 3, 3, 2, 3),
    4: VariantParameters(4, 5, 4, 2, 4, 6),
    5: VariantParameters(5, 6, 1, 1, 1, 5),
    6: VariantParameters(1, 1, 2, 5, 5, 6),
    7: VariantParameters(2, 2, 3, 4, 3, 1),
    8: VariantParameters(3, 3, 4, 3, 4, 4),
    9: VariantParameters(4, 4, 1, 2, 2, 2),
    10: VariantParameters(5, 5, 2, 1, 4, 2),
    11: VariantParameters(1, 6, 3, 5, 6, 6),
    12: VariantParameters(2, 1, 4, 4, 1, 3),
    13: VariantParameters(3, 2, 1, 3, 3, 2),
    14: VariantParameters(4, 3, 2, 2, 3, 3),
    15: VariantParameters(5, 4, 3, 1, 1, 4),
    16: VariantParameters(1, 5, 4, 5, 2, 5),
    17: VariantParameters(2, 6, 1, 4, 6, 5),
    18: VariantParameters(3, 1, 2, 3, 4, 5),
    19: VariantParameters(4, 2, 3, 2, 2, 6),
    20: VariantParameters(5, 3, 4, 1, 4, 1),
}


@dataclass(frozen=True)
class SimulationResult:
    final_x: np.ndarray
    iterations: int
    last_delta: float
    rows: list[tuple[int, float, float, float, float]]


@dataclass(frozen=True)
class InfluenceSets:
    f_agents: list[int]  # индексы Python: 0..n-1
    s_agents: list[int]
    neutral_agents: list[int]


@dataclass(frozen=True)
class NashResult:
    u: float
    v: float
    matrix: np.ndarray
    rhs: np.ndarray
    x_final: float
    phi_f: float
    phi_s: float
    h_f: float
    h_s: float
    ideal_x_f: float
    ideal_x_s: float
    ideal_h_f: float
    ideal_h_s: float
    distance_f: float
    distance_s: float


# -----------------------------------------------------------------------------
# Форматирование вывода
# -----------------------------------------------------------------------------
def fmt_float(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def fmt_vector(values: Iterable[float], digits: int = 6) -> str:
    return "(" + "; ".join(fmt_float(float(x), digits) for x in values) + ")"


def print_matrix(name: str, matrix: np.ndarray, digits: int = 6) -> None:
    print(f"{name} =")
    for row in matrix:
        print("  " + " ".join(f"{float(x):10.{digits}f}" for x in row))
    print()


def to_agent_numbers(indices: Iterable[int]) -> list[int]:
    return [i + 1 for i in indices]


# -----------------------------------------------------------------------------
# Модель доверия и динамика мнений
# -----------------------------------------------------------------------------
def generate_trust_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Генерирует положительную стохастическую по строкам матрицу доверия."""
    raw = rng.random((n, n)) + 1e-3
    return raw / raw.sum(axis=1, keepdims=True)


def is_row_stochastic(A: np.ndarray, tol: float = 1e-10) -> bool:
    non_negative = bool(np.all(A >= -tol))
    row_sums_are_one = bool(np.allclose(A.sum(axis=1), 1.0, atol=tol))
    return non_negative and row_sums_are_one


def generate_initial_opinions(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(INITIAL_OPINION_LOW, INITIAL_OPINION_HIGH, size=n)


def simulate_opinions(A: np.ndarray, x0: np.ndarray, eps: float, max_iter: int) -> SimulationResult:
    x_prev = x0.astype(float).copy()
    rows: list[tuple[int, float, float, float, float]] = []

    for t in range(1, max_iter + 1):
        x_next = A @ x_prev
        delta = float(np.max(np.abs(x_next - x_prev)))
        min_x = float(np.min(x_next))
        max_x = float(np.max(x_next))
        spread = max_x - min_x
        rows.append((t, delta, min_x, max_x, spread))

        if delta < eps:
            return SimulationResult(
                final_x=x_next,
                iterations=t,
                last_delta=delta,
                rows=rows,
            )

        x_prev = x_next

    raise RuntimeError(f"Мнения не сошлись за {max_iter} итераций. Последняя delta = {delta}")


def limiting_matrix_power(A: np.ndarray, eps: float, max_iter: int) -> tuple[np.ndarray, int, float]:
    """Итерационно вычисляет приближение A^inf = lim A^t."""
    n = A.shape[0]
    previous = np.eye(n)

    for k in range(1, max_iter + 1):
        current = previous @ A
        delta = float(np.max(np.abs(current - previous)))

        if k > 1 and delta < eps:
            return current, k, delta

        previous = current

    raise RuntimeError(f"A^k не сошлась за {max_iter} итераций. Последняя delta = {delta}")


# -----------------------------------------------------------------------------
# Выбор агентов влияния
# -----------------------------------------------------------------------------
def choose_influence_sets(n: int, rng: np.random.Generator) -> InfluenceSets:
    if MANUAL_F_AGENTS is not None or MANUAL_S_AGENTS is not None:
        if MANUAL_F_AGENTS is None or MANUAL_S_AGENTS is None:
            raise ValueError("Если задаете агентов вручную, заполните оба списка: MANUAL_F_AGENTS и MANUAL_S_AGENTS.")

        f_agents = sorted(i - 1 for i in MANUAL_F_AGENTS)
        s_agents = sorted(i - 1 for i in MANUAL_S_AGENTS)

        if any(i < 0 or i >= n for i in f_agents + s_agents):
            raise ValueError("Номера агентов должны быть в диапазоне от 1 до N_AGENTS.")
        if set(f_agents) & set(s_agents):
            raise ValueError("Множества агентов первого и второго игроков не должны пересекаться.")

        used = set(f_agents) | set(s_agents)
        neutral_agents = [i for i in range(n) if i not in used]
        return InfluenceSets(f_agents=f_agents, s_agents=s_agents, neutral_agents=neutral_agents)

    low_count = 0 if ALLOW_ZERO_INFLUENCE_AGENTS else 1
    count_f = int(rng.integers(low_count, MAX_INFLUENCE_AGENTS_PER_PLAYER + 1))
    count_s = int(rng.integers(low_count, MAX_INFLUENCE_AGENTS_PER_PLAYER + 1))

    if count_f + count_s > n:
        raise ValueError("Суммарное число агентов влияния больше количества агентов в сети.")

    permutation = rng.permutation(n)
    f_agents = sorted(int(i) for i in permutation[:count_f])
    s_agents = sorted(int(i) for i in permutation[count_f:count_f + count_s])
    used = set(f_agents) | set(s_agents)
    neutral_agents = [i for i in range(n) if i not in used]

    return InfluenceSets(f_agents=f_agents, s_agents=s_agents, neutral_agents=neutral_agents)


# -----------------------------------------------------------------------------
# Аналитическое решение игры с непротивоположными интересами
# -----------------------------------------------------------------------------
def payoff_h_f(x: float, p: VariantParameters) -> float:
    return p.a * x - p.b * x * x


def payoff_h_s(x: float, p: VariantParameters) -> float:
    return p.c * x - p.d * x * x


def objective_phi_f(u: float, v: float, rf: float, rs: float, x0_const: float, p: VariantParameters) -> float:
    x = rf * u + rs * v + x0_const
    return payoff_h_f(x, p) - p.gf * u * u / 2.0


def objective_phi_s(u: float, v: float, rf: float, rs: float, x0_const: float, p: VariantParameters) -> float:
    x = rf * u + rs * v + x0_const
    return payoff_h_s(x, p) - p.gs * v * v / 2.0


def solve_nash_equilibrium(rf: float, rs: float, x0_const: float, p: VariantParameters) -> NashResult:
    """
    Решает систему первых производных:
        d Phi_f / d u = 0,
        d Phi_s / d v = 0.

    X(u,v) = rf*u + rs*v + X0.

    Phi_f = aX - bX^2 - gf*u^2/2,
    Phi_s = cX - dX^2 - gs*v^2/2.
    """
    system_matrix = np.array(
        [
            [p.gf + 2.0 * p.b * rf * rf, 2.0 * p.b * rf * rs],
            [2.0 * p.d * rf * rs, p.gs + 2.0 * p.d * rs * rs],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            p.a * rf - 2.0 * p.b * rf * x0_const,
            p.c * rs - 2.0 * p.d * rs * x0_const,
        ],
        dtype=float,
    )

    try:
        u, v = np.linalg.solve(system_matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Система условий равновесия Нэша вырождена; измените параметры или агентов влияния.") from exc

    x_final = float(rf * u + rs * v + x0_const)
    h_f = payoff_h_f(x_final, p)
    h_s = payoff_h_s(x_final, p)
    phi_f = objective_phi_f(u, v, rf, rs, x0_const, p)
    phi_s = objective_phi_s(u, v, rf, rs, x0_const, p)

    ideal_x_f = p.a / (2.0 * p.b)
    ideal_x_s = p.c / (2.0 * p.d)
    ideal_h_f = payoff_h_f(ideal_x_f, p)
    ideal_h_s = payoff_h_s(ideal_x_s, p)
    distance_f = abs(x_final - ideal_x_f)
    distance_s = abs(x_final - ideal_x_s)

    return NashResult(
        u=float(u),
        v=float(v),
        matrix=system_matrix,
        rhs=rhs,
        x_final=x_final,
        phi_f=phi_f,
        phi_s=phi_s,
        h_f=h_f,
        h_s=h_s,
        ideal_x_f=ideal_x_f,
        ideal_x_s=ideal_x_s,
        ideal_h_f=ideal_h_f,
        ideal_h_s=ideal_h_s,
        distance_f=distance_f,
        distance_s=distance_s,
    )


def make_controlled_initial_vector(
        base_x0: np.ndarray,
        influence: InfluenceSets,
        u: float,
        v: float,
) -> np.ndarray:
    x0 = base_x0.astype(float).copy()
    x0[influence.f_agents] = u
    x0[influence.s_agents] = v
    return x0


# -----------------------------------------------------------------------------
# Печать промежуточных результатов
# -----------------------------------------------------------------------------
def print_convergence_table(rows: list[tuple[int, float, float, float, float]]) -> None:
    print("Таблица сходимости:")
    print("  t | max|x(t)-x(t-1)| | min x(t) | max x(t) | разброс")
    print("----+-------------------+----------+----------+---------")
    for t, delta, min_x, max_x, spread in rows:
        print(f"{t:3d} | {delta:17.10f} | {min_x:8.6f} | {max_x:8.6f} | {spread:7.6f}")
    print()


def print_variant_parameters(variant: int, p: VariantParameters) -> None:
    print(f"Вариант: {variant}")
    print("Параметры варианта из таблицы 8.1:")
    print(f"  g_f = {fmt_float(p.gf, 6)}")
    print(f"  g_s = {fmt_float(p.gs, 6)}")
    print(f"  a   = {fmt_float(p.a, 6)}")
    print(f"  b   = {fmt_float(p.b, 6)}")
    print(f"  c   = {fmt_float(p.c, 6)}")
    print(f"  d   = {fmt_float(p.d, 6)}")
    print()


def print_game_formulas(p: VariantParameters, rf: float, rs: float, x0_const: float) -> None:
    print("Функции выигрыша игроков:")
    print(f"  H_f(X) = {fmt_float(p.a)} * X - {fmt_float(p.b)} * X^2")
    print(f"  H_s(X) = {fmt_float(p.c)} * X - {fmt_float(p.d)} * X^2")
    print()

    print("Итоговое мнение как функция управлений:")
    print(f"  X(u, v) = r_f * u + r_s * v + X0")
    print(f"          = {fmt_float(rf)} * u + {fmt_float(rs)} * v + {fmt_float(x0_const)}")
    print()

    print("Целевые функции с учетом затрат:")
    print(f"  Phi_f(u, v) = H_f(X(u,v)) - g_f * u^2 / 2")
    print(f"              = H_f(X(u,v)) - {fmt_float(p.gf)} * u^2 / 2")
    print(f"  Phi_s(u, v) = H_s(X(u,v)) - g_s * v^2 / 2")
    print(f"              = H_s(X(u,v)) - {fmt_float(p.gs)} * v^2 / 2")
    print()

    print("Условия равновесия Нэша:")
    print("  d Phi_f / d u = a*r_f - 2*b*r_f*(r_f*u + r_s*v + X0) - g_f*u = 0")
    print("  d Phi_s / d v = c*r_s - 2*d*r_s*(r_f*u + r_s*v + X0) - g_s*v = 0")
    print()


# -----------------------------------------------------------------------------
# Основной сценарий
# -----------------------------------------------------------------------------
def main() -> None:
    if VARIANT not in VARIANTS:
        raise ValueError(f"Неизвестный вариант {VARIANT}. Доступны варианты: {sorted(VARIANTS)}")

    params = VARIANTS[VARIANT]
    rng = np.random.default_rng(SEED)

    print("Лабораторная работа № 6")
    print("Информационное противоборство")
    print("=" * 90)
    print_variant_parameters(VARIANT, params)
    print(f"Количество агентов: {N_AGENTS}")
    print(f"Seed случайной генерации: {SEED}")
    print(f"Точность остановки eps: {EPS:g}")
    print("=" * 90)

    # 1. Матрица доверия
    A = generate_trust_matrix(N_AGENTS, rng)
    print("1. Сгенерированная стохастическая матрица доверия")
    print_matrix("A", A, digits=6)
    print("Проверка стохастичности матрицы A:")
    print(f"  все элементы неотрицательны: {bool(np.all(A >= 0))}")
    print(f"  суммы строк: {fmt_vector(A.sum(axis=1), digits=6)}")
    print(f"  матрица стохастическая по строкам: {is_row_stochastic(A)}")
    print("=" * 90)

    # 2. Результирующая матрица доверия
    A_inf, a_inf_iters, a_inf_delta = limiting_matrix_power(A, EPS, MAX_ITER)
    r = A_inf[0]
    print("2. Результирующая матрица доверия A^inf")
    print(f"Матрица A^k сошлась при k = {a_inf_iters}")
    print(f"Последнее max|A^k - A^(k-1)| = {a_inf_delta:.12f}")
    print_matrix(f"A^{a_inf_iters}", A_inf, digits=6)
    print(f"Предельная строка r: {fmt_vector(r, digits=6)}")
    print(f"Сумма элементов r: {float(np.sum(r)):.12f}")
    print("=" * 90)

    # 3. Мнения без управления
    base_x0 = generate_initial_opinions(N_AGENTS, rng)
    plain_result = simulate_opinions(A, base_x0, EPS, MAX_ITER)
    theoretical_plain = float(r @ base_x0)

    print("3. Моделирование мнений без информационного управления")
    print(f"Начальные мнения x(0): {fmt_vector(base_x0, digits=6)}")
    print_convergence_table(plain_result.rows)
    print(f"Результирующие мнения x({plain_result.iterations}): {fmt_vector(plain_result.final_x, digits=6)}")
    print(f"Итоговое мнение по итерациям: {float(np.mean(plain_result.final_x)):.12f}")
    print(f"Проверка по формуле X = r * x(0): {theoretical_plain:.12f}")
    print(f"Последнее max|x(t)-x(t-1)|: {plain_result.last_delta:.12f}")
    print("=" * 90)

    # 4. Агенты влияния и нейтральный вклад
    influence = choose_influence_sets(N_AGENTS, rng)
    rf = float(np.sum(r[influence.f_agents])) if influence.f_agents else 0.0
    rs = float(np.sum(r[influence.s_agents])) if influence.s_agents else 0.0
    x0_const = float(np.sum(r[influence.neutral_agents] * base_x0[influence.neutral_agents]))

    print("4. Выбор агентов влияния")
    print(f"Агенты влияния первого игрока F: {to_agent_numbers(influence.f_agents)}")
    print(f"Агенты влияния второго игрока S: {to_agent_numbers(influence.s_agents)}")
    print(f"Нейтральные агенты: {to_agent_numbers(influence.neutral_agents)}")
    print()
    print("Начальные мнения нейтральных агентов:")
    for i in influence.neutral_agents:
        print(f"  агент {i + 1}: {base_x0[i]:.6f}; вес r_{i + 1} = {r[i]:.6f}; вклад = {r[i] * base_x0[i]:.6f}")
    print()
    print(f"r_f = сумма весов агентов F = {rf:.12f}")
    print(f"r_s = сумма весов агентов S = {rs:.12f}")
    print(f"X0 = вклад нейтральных агентов = {x0_const:.12f}")
    print(f"r_f + r_s + r_neutral = {rf + rs + float(np.sum(r[influence.neutral_agents])):.12f}")
    print("=" * 90)

    # 5. Аналитическое решение игры
    print("5. Аналитическое решение игры с непротивоположными интересами")
    print_game_formulas(params, rf, rs, x0_const)

    nash = solve_nash_equilibrium(rf, rs, x0_const, params)
    print("Линейная система для нахождения равновесия Нэша:")
    print_matrix("M", nash.matrix, digits=6)
    print(f"Правая часть q: {fmt_vector(nash.rhs, digits=6)}")
    print("Решение системы M * (u, v)^T = q:")
    print(f"  u* = {nash.u:.12f}")
    print(f"  v* = {nash.v:.12f}")
    print()
    print(f"Итоговое мнение X(u*, v*) = {nash.x_final:.12f}")
    print(f"Доход первого игрока H_f(X*) = {nash.h_f:.12f}")
    print(f"Доход второго игрока H_s(X*) = {nash.h_s:.12f}")
    print(f"Целевая функция первого игрока Phi_f(u*,v*) = {nash.phi_f:.12f}")
    print(f"Целевая функция второго игрока Phi_s(u*,v*) = {nash.phi_s:.12f}")
    print("=" * 90)

    # 6. Проверка через моделирование с найденными управлениями
    controlled_x0 = make_controlled_initial_vector(base_x0, influence, nash.u, nash.v)
    controlled_result = simulate_opinions(A, controlled_x0, EPS, MAX_ITER)
    theoretical_controlled = float(r @ controlled_x0)

    print("6. Проверка найденных управлений через итерационное моделирование")
    print(f"Начальный вектор с учетом управлений x(0): {fmt_vector(controlled_x0, digits=6)}")
    print_convergence_table(controlled_result.rows)
    print(f"Результирующие мнения x({controlled_result.iterations}): {fmt_vector(controlled_result.final_x, digits=6)}")
    print(f"Итоговое мнение по итерациям: {float(np.mean(controlled_result.final_x)):.12f}")
    print(f"Проверка по формуле X = r * x(0): {theoretical_controlled:.12f}")
    print(f"Аналитическое X(u*, v*): {nash.x_final:.12f}")
    print(f"Последнее max|x(t)-x(t-1)|: {controlled_result.last_delta:.12f}")
    print("=" * 90)

    # 7. Точка утопии и расстояния
    print("7. Идеальные мнения игроков, точка утопии и расстояния")
    print(f"Идеальное мнение первого игрока X_max_f = a / (2b) = {nash.ideal_x_f:.12f}")
    print(f"Идеальное мнение второго игрока X_max_s = c / (2d) = {nash.ideal_x_s:.12f}")
    print(f"Максимальный доход первого игрока H_f(X_max_f) = {nash.ideal_h_f:.12f}")
    print(f"Максимальный доход второго игрока H_s(X_max_s) = {nash.ideal_h_s:.12f}")
    print(f"Расстояние до цели первого игрока |X* - X_max_f| = {nash.distance_f:.12f}")
    print(f"Расстояние до цели второго игрока |X* - X_max_s| = {nash.distance_s:.12f}")
    print()

    if nash.distance_f < nash.distance_s:
        winner = "первый игрок"
    elif nash.distance_s < nash.distance_f:
        winner = "второй игрок"
    else:
        winner = "ничья"

    print(f"Итог: {winner}")
    print("=" * 90)


if __name__ == "__main__":
    main()
