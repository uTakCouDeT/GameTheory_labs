import math
from fractions import Fraction
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# ЛР3. Непрерывная выпукло-вогнутая антагонистическая игра
#
# Что делает программа:
# 1) Решает задачу аналитически для ядра
#       H(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y,
#    x, y in [0, 1]
# 2) Решает задачу численно методом аппроксимации на сетке:
#       x_i = i / N, y_j = j / N, i,j = 0..N
#    то есть получаем матрицу размера (N+1) x (N+1).
# 3) Для каждой сеточной аппроксимации:
#       - ищет седловую точку в чистых стратегиях;
#       - если седла нет, применяет метод Брауна-Робинсона.
# 4) Останавливается по внешнему критерию:
#       |h^(N) - h^(N-1)| <= epsilon_outer
#    где h^(N) — найденная цена игры на текущей сетке.
#
# В отчёт обычно удобно брать:
# - текст условия;
# - аналитическое решение;
# - первые 10 итераций по N;
# - итерацию, на которой выполнен критерий остановки;
# - итоговые численные значения.
# ============================================================


# ------------------------------------------------------------
# НАСТРОЙКИ ВАРИАНТА
# ------------------------------------------------------------
# Подставьте сюда коэффициенты своего варианта.
A_COEF = -3.0
B_COEF = 1.5
C_COEF = 18.0 / 5.0
D_COEF = -18.0 / 50.0
E_COEF = -72.0 / 25.0

# Внешний критерий остановки по ЛР3:
# останавливаем рост N, когда соседние оценки цены игры близки.
EPSILON_OUTER = 1e-3

# Внутренний критерий для метода Брауна-Робинсона
# на фиксированной матрице.
EPSILON_BR = 1e-3

# Начинаем с N = 2, потому что это даёт сетку 3x3:
# 0, 0.5, 1.0
N_START = 2
N_MAX = 200

# Ограничение на число итераций Брауна-Робинсона
BR_MAX_ITER = 100000

# Сколько первых итераций по N печатать отдельно для отчёта
OUTER_ITERS_FOR_REPORT = 10

# Печатать ли матрицы для первых итераций по N
PRINT_MATRICES_FOR_FIRST_ITERATIONS = 3


# ------------------------------------------------------------
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ------------------------------------------------------------

def fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def fr(x: float, max_den: int = 10000) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(Fraction(float(x)).limit_denominator(max_den))


def vector_to_str(v: np.ndarray, digits: int = 6) -> str:
    return "[" + ", ".join(fmt(float(x), digits) for x in v) + "]"


def vector_to_fraction_str(v: np.ndarray) -> str:
    return "[" + ", ".join(fr(float(x)) for x in v) + "]"


def matrix_to_pretty_str(M: np.ndarray, digits: int = 6) -> str:
    rows = []
    for row in M:
        rows.append("[" + ", ".join(f"{float(x): .{digits}f}" for x in row) + "]")
    return "\n".join(rows)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ------------------------------------------------------------
# ЯДРО ИГРЫ
# ------------------------------------------------------------

def H(x: float, y: float, a: float, b: float, c: float, d: float, e: float) -> float:
    return a * x * x + b * y * y + c * x * y + d * x + e * y


def Hx(x: float, y: float, a: float, c: float, d: float) -> float:
    return 2.0 * a * x + c * y + d


def Hy(x: float, y: float, b: float, c: float, e: float) -> float:
    return 2.0 * b * y + c * x + e


def x_best_response(y: float, a: float, c: float, d: float) -> float:
    # Игрок A максимизирует H по x на [0,1], H_xx = 2a < 0
    if abs(a) < 1e-14:
        # Вне выпукло-вогнутого случая, но на всякий случай
        candidates = [0.0, 1.0]
        values = [H(x, y, a, 0.0, c, d, 0.0) for x in candidates]
        return candidates[int(np.argmax(values))]
    x0 = -(c * y + d) / (2.0 * a)
    return clamp01(x0)


def y_best_response(x: float, b: float, c: float, e: float) -> float:
    # Игрок B минимизирует H по y на [0,1], H_yy = 2b > 0
    if abs(b) < 1e-14:
        candidates = [0.0, 1.0]
        values = [H(x, y, 0.0, b, c, 0.0, e) for y in candidates]
        return candidates[int(np.argmin(values))]
    y0 = -(c * x + e) / (2.0 * b)
    return clamp01(y0)


# ------------------------------------------------------------
# АНАЛИТИЧЕСКОЕ РЕШЕНИЕ НА [0,1]x[0,1]
# ------------------------------------------------------------

@dataclass
class AnalyticSolution:
    x_star: float
    y_star: float
    h_star: float
    kind: str


def is_saddle_point_on_unit_square(
    x_star: float,
    y_star: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    tol: float = 1e-9,
) -> bool:
    # Проверяем через точные лучшие ответы на [0,1]
    x_br = x_best_response(y_star, a, c, d)
    y_br = y_best_response(x_star, b, c, e)

    h_star = H(x_star, y_star, a, b, c, d, e)
    h_xbr = H(x_br, y_star, a, b, c, d, e)
    h_ybr = H(x_star, y_br, a, b, c, d, e)

    return abs(h_xbr - h_star) <= tol and abs(h_ybr - h_star) <= tol


def continuous_analytic_solution(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    tol: float = 1e-9,
) -> Optional[AnalyticSolution]:
    # Кандидаты: внутренняя точка, граничные точки по BR, углы.
    candidates: List[Tuple[float, float, str]] = []

    # 1) Внутренняя стационарная точка
    det = 4.0 * a * b - c * c
    if abs(det) > 1e-14:
        # Решаем систему:
        # 2ax + cy + d = 0
        # cx + 2by + e = 0
        x0 = (c * e - 2.0 * b * d) / det
        y0 = (c * d - 2.0 * a * e) / det
        if -tol <= x0 <= 1.0 + tol and -tol <= y0 <= 1.0 + tol:
            candidates.append((clamp01(x0), clamp01(y0), "внутренняя стационарная точка"))

    # 2) x = 0 и x = 1
    for x_fixed in [0.0, 1.0]:
        y0 = y_best_response(x_fixed, b, c, e)
        candidates.append((x_fixed, y0, f"граница x={int(x_fixed)}"))

    # 3) y = 0 и y = 1
    for y_fixed in [0.0, 1.0]:
        x0 = x_best_response(y_fixed, a, c, d)
        candidates.append((x0, y_fixed, f"граница y={int(y_fixed)}"))

    # 4) углы
    for x_corner in [0.0, 1.0]:
        for y_corner in [0.0, 1.0]:
            candidates.append((x_corner, y_corner, "угловая точка"))

    unique_candidates: List[Tuple[float, float, str]] = []
    seen: List[Tuple[float, float]] = []
    for x0, y0, kind in candidates:
        key = (round(x0, 12), round(y0, 12))
        if key not in seen:
            seen.append(key)
            unique_candidates.append((x0, y0, kind))

    for x0, y0, kind in unique_candidates:
        if is_saddle_point_on_unit_square(x0, y0, a, b, c, d, e, tol=tol):
            return AnalyticSolution(
                x_star=x0,
                y_star=y0,
                h_star=H(x0, y0, a, b, c, d, e),
                kind=kind,
            )

    return None


# ------------------------------------------------------------
# МАТРИЧНАЯ ИГРА НА СЕТКЕ
# ------------------------------------------------------------

def build_grid_matrix(
    N: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, N + 1)
    C = np.zeros((N + 1, N + 1), dtype=float)
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            C[i, j] = H(float(x), float(y), a, b, c, d, e)
    return grid, C


def pure_strategy_analysis(C: np.ndarray, tol: float = 1e-12) -> Tuple[float, float, List[Tuple[int, int]]]:
    row_mins = C.min(axis=1)
    col_maxs = C.max(axis=0)

    v_lower = float(row_mins.max())
    v_upper = float(col_maxs.min())

    saddle_points: List[Tuple[int, int]] = []
    if abs(v_lower - v_upper) <= tol:
        good_rows = np.where(np.isclose(row_mins, v_lower, atol=tol))[0]
        good_cols = np.where(np.isclose(col_maxs, v_upper, atol=tol))[0]
        for i in good_rows:
            for j in good_cols:
                if abs(C[i, j] - v_lower) <= tol:
                    saddle_points.append((int(i), int(j)))

    return v_lower, v_upper, saddle_points


# ------------------------------------------------------------
# МЕТОД БРАУНА-РОБИНСОНА ДЛЯ ФИКСИРОВАННОЙ МАТРИЦЫ
# ------------------------------------------------------------

@dataclass
class BrownRobinsonResult:
    iterations: int
    x_est: np.ndarray
    y_est: np.ndarray
    row_counts: np.ndarray
    col_counts: np.ndarray
    upper_best: float
    lower_best: float
    value_mid: float
    gap: float
    history_first_10: List[Dict[str, float]]


def brown_robinson(
    C: np.ndarray,
    epsilon: float = 1e-3,
    max_iter: int = 100000,
    start_row: int = 0,
    start_col: int = 0,
) -> BrownRobinsonResult:
    m, n = C.shape

    row_counts = np.zeros(m, dtype=int)
    col_counts = np.zeros(n, dtype=int)

    row_cum = np.zeros(m, dtype=float)
    col_cum = np.zeros(n, dtype=float)

    i = start_row
    j = start_col

    best_upper = math.inf
    best_lower = -math.inf

    history_first_10: List[Dict[str, float]] = []

    for k in range(1, max_iter + 1):
        row_counts[i] += 1
        col_counts[j] += 1

        row_cum += C[:, j]
        col_cum += C[i, :]

        upper_curr = float(row_cum.max() / k)
        lower_curr = float(col_cum.min() / k)

        best_upper = min(best_upper, upper_curr)
        best_lower = max(best_lower, lower_curr)
        gap = best_upper - best_lower

        if k <= 10:
            history_first_10.append({
                "k": k,
                "row_choice": i,
                "col_choice": j,
                "upper_curr": upper_curr,
                "lower_curr": lower_curr,
                "upper_best": best_upper,
                "lower_best": best_lower,
                "gap": gap,
            })

        if gap <= epsilon:
            x_est = row_counts / k
            y_est = col_counts / k
            value_mid = 0.5 * (best_upper + best_lower)
            return BrownRobinsonResult(
                iterations=k,
                x_est=x_est,
                y_est=y_est,
                row_counts=row_counts.copy(),
                col_counts=col_counts.copy(),
                upper_best=best_upper,
                lower_best=best_lower,
                value_mid=value_mid,
                gap=gap,
                history_first_10=history_first_10,
            )

        i = int(np.argmax(row_cum))
        j = int(np.argmin(col_cum))

    x_est = row_counts / max_iter
    y_est = col_counts / max_iter
    value_mid = 0.5 * (best_upper + best_lower)
    return BrownRobinsonResult(
        iterations=max_iter,
        x_est=x_est,
        y_est=y_est,
        row_counts=row_counts.copy(),
        col_counts=col_counts.copy(),
        upper_best=best_upper,
        lower_best=best_lower,
        value_mid=value_mid,
        gap=best_upper - best_lower,
        history_first_10=history_first_10,
    )


# ------------------------------------------------------------
# РЕШЕНИЕ ОДНОЙ СЕТОЧНОЙ АППРОКСИМАЦИИ
# ------------------------------------------------------------

@dataclass
class GridIterationResult:
    N: int
    grid: np.ndarray
    matrix: np.ndarray
    method: str
    value: float
    x_repr: float
    y_repr: float
    x_strategy: np.ndarray
    y_strategy: np.ndarray
    delta_from_prev: Optional[float]
    saddle_points: List[Tuple[int, int]]
    br_result: Optional[BrownRobinsonResult]


def solve_grid_game(
    N: int,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    epsilon_br: float,
    br_max_iter: int,
) -> GridIterationResult:
    grid, C = build_grid_matrix(N, a, b, c, d, e)
    v_lower, v_upper, saddles = pure_strategy_analysis(C)

    if saddles:
        i, j = saddles[0]
        x_strategy = np.zeros(N + 1, dtype=float)
        y_strategy = np.zeros(N + 1, dtype=float)
        x_strategy[i] = 1.0
        y_strategy[j] = 1.0
        return GridIterationResult(
            N=N,
            grid=grid,
            matrix=C,
            method="седловая точка",
            value=C[i, j],
            x_repr=float(grid[i]),
            y_repr=float(grid[j]),
            x_strategy=x_strategy,
            y_strategy=y_strategy,
            delta_from_prev=None,
            saddle_points=saddles,
            br_result=None,
        )

    br = brown_robinson(C, epsilon=epsilon_br, max_iter=br_max_iter, start_row=0, start_col=0)
    x_repr = float(np.dot(br.x_est, grid))
    y_repr = float(np.dot(br.y_est, grid))

    return GridIterationResult(
        N=N,
        grid=grid,
        matrix=C,
        method="Браун-Робинсон",
        value=br.value_mid,
        x_repr=x_repr,
        y_repr=y_repr,
        x_strategy=br.x_est,
        y_strategy=br.y_est,
        delta_from_prev=None,
        saddle_points=[],
        br_result=br,
    )


# ------------------------------------------------------------
# ОСНОВНОЙ ВНЕШНИЙ ЦИКЛ ПО N
# ------------------------------------------------------------

def solve_continuous_game_numerically(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    n_start: int,
    n_max: int,
    epsilon_outer: float,
    epsilon_br: float,
    br_max_iter: int,
) -> Tuple[List[GridIterationResult], Optional[GridIterationResult]]:
    results: List[GridIterationResult] = []
    prev_value: Optional[float] = None
    final_result: Optional[GridIterationResult] = None

    for N in range(n_start, n_max + 1):
        result = solve_grid_game(N, a, b, c, d, e, epsilon_br, br_max_iter)

        if prev_value is None:
            result.delta_from_prev = None
        else:
            result.delta_from_prev = abs(result.value - prev_value)

        results.append(result)

        if prev_value is not None and result.delta_from_prev is not None and result.delta_from_prev <= epsilon_outer:
            final_result = result
            break

        prev_value = result.value

    if final_result is None and results:
        final_result = results[-1]

    return results, final_result


# ------------------------------------------------------------
# ПЕЧАТЬ
# ------------------------------------------------------------

def print_recognized_task() -> None:
    print("РАСПОЗНАННОЕ УСЛОВИЕ ЛР3")
    print("-" * 80)
    print("Нужно найти оптимальные стратегии непрерывной выпукло-вогнутой")
    print("антагонистической игры на единичном квадрате аналитическим и")
    print("численным методами.")
    print()
    print("Ядро игры:")
    print("    H(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y,   x,y in [0,1].")
    print()
    print("Аналитическая часть:")
    print("- проверить условия выпукло-вогнутости: H_xx = 2a < 0, H_yy = 2b > 0;")
    print("- найти оптимальные стратегии из условий первого порядка и")
    print("  ограничений x,y in [0,1];")
    print("- вычислить цену игры h = H(x*, y*).")
    print()
    print("Численная часть:")
    print("- заменить непрерывную игру матричной на сетке x_i=i/N, y_j=j/N, i,j=0..N;")
    print("- при N=2 получается матрица 3x3 с точками 0, 0.5, 1.0;")
    print("- на каждой сетке сначала искать седловую точку;")
    print("- если седловой точки нет, решать матричную игру методом")
    print("  Брауна-Робинсона (на основе кода из предыдущей лабораторной);")
    print("- критерий остановки для ЛР3: сравнивать соседние оценки h^(N) и")
    print("  останавливать процесс, когда |h^(N)-h^(N-1)| <= epsilon.")
    print("- в отчёт удобно вывести первые 10 итераций по N и номер итерации,")
    print("  на которой получен итоговый ответ.")
    print("-" * 80)
    print()


def print_analytic_part(a: float, b: float, c: float, d: float, e: float) -> Optional[AnalyticSolution]:
    print("АНАЛИТИЧЕСКОЕ РЕШЕНИЕ")
    print("-" * 80)
    print(f"H(x, y) = {fmt(a)}*x^2 + {fmt(b)}*y^2 + {fmt(c)}*x*y + {fmt(d)}*x + {fmt(e)}*y")
    print(f"H_xx = 2a = {fmt(2*a)}")
    print(f"H_yy = 2b = {fmt(2*b)}")

    if 2 * a < 0 and 2 * b > 0:
        print("Условия выпукло-вогнутости выполнены: H_xx < 0, H_yy > 0.")
    else:
        print("ВНИМАНИЕ: условия выпукло-вогнутости НЕ выполнены.")
        print("Код всё равно попытается найти седловую точку на [0,1]x[0,1].")

    print()
    print("Производные:")
    print(f"H_x = 2*a*x + c*y + d = {fmt(2*a)}*x + {fmt(c)}*y + {fmt(d)}")
    print(f"H_y = 2*b*y + c*x + e = {fmt(2*b)}*y + {fmt(c)}*x + {fmt(e)}")
    print()

    sol = continuous_analytic_solution(a, b, c, d, e)
    if sol is None:
        print("Аналитическое решение на [0,1]x[0,1] не найдено.")
        print()
        return None

    print(f"Тип найденного решения: {sol.kind}")
    print(f"x* = {fmt(sol.x_star)} = {fr(sol.x_star)}")
    print(f"y* = {fmt(sol.y_star)} = {fr(sol.y_star)}")
    print(f"h  = H(x*, y*) = {fmt(sol.h_star)} = {fr(sol.h_star)}")
    print()
    return sol


def print_outer_iterations(results: List[GridIterationResult], final_result: GridIterationResult) -> None:
    print("ЧИСЛЕННОЕ РЕШЕНИЕ")
    print("-" * 80)
    print("Первые итерации по N (для отчёта):")
    print(
        f"{'iter':>4} | {'N':>4} | {'размер':>8} | {'метод':>18} | {'h^(N)':>12} | {'x~':>10} | {'y~':>10} | {'|Δh|':>12}"
    )
    print("-" * 100)

    shown = min(OUTER_ITERS_FOR_REPORT, len(results))
    for idx in range(shown):
        r = results[idx]
        delta_text = "-" if r.delta_from_prev is None else fmt(r.delta_from_prev)
        print(
            f"{idx + 1:>4} | {r.N:>4} | {r.N + 1:>8} | {r.method:>18} | {fmt(r.value):>12} | {fmt(r.x_repr):>10} | {fmt(r.y_repr):>10} | {delta_text:>12}"
        )

    print("-" * 100)
    print()

    stop_iter = len(results)
    print(f"Критерий остановки выполнен на итерации по N: {stop_iter}")
    print(f"Соответствующее значение N = {final_result.N}")
    print(f"Размер матрицы на последней итерации: {(final_result.N + 1)} x {(final_result.N + 1)}")
    print()


def print_iteration_details(r: GridIterationResult, iteration_index: int) -> None:
    print(f"ДЕТАЛИ ИТЕРАЦИИ #{iteration_index} (N={r.N}, размер {r.N + 1}x{r.N + 1})")
    print("-" * 80)

    if r.N - N_START < PRINT_MATRICES_FOR_FIRST_ITERATIONS:
        print("Матрица игры H^(N):")
        print(matrix_to_pretty_str(r.matrix, digits=6))
        print()

    if r.method == "седловая точка":
        print("На сетке найдена седловая точка в чистых стратегиях.")
        for i, j in r.saddle_points:
            print(
                f"Седловая точка: i={i}, j={j}, x={fmt(r.grid[i])}, y={fmt(r.grid[j])}, H={fmt(r.matrix[i, j])}"
            )
    else:
        print("Седловой точки нет, применён метод Брауна-Робинсона.")
        assert r.br_result is not None
        br = r.br_result
        print(f"Число итераций Брауна-Робинсона: {br.iterations}")
        print(f"Лучшая верхняя оценка: {fmt(br.upper_best)}")
        print(f"Лучшая нижняя оценка: {fmt(br.lower_best)}")
        print(f"Погрешность BR:       {fmt(br.gap)}")
        print(f"Средняя оценка цены:  {fmt(br.value_mid)}")
        print()
        print("Первые 10 итераций Брауна-Робинсона:")
        print(
            f"{'k':>4} | {'i':>4} | {'j':>4} | {'v_max/k':>12} | {'v_min/k':>12} | {'min v_max':>12} | {'max v_min':>12} | {'E':>12}"
        )
        print("-" * 96)
        for rec in br.history_first_10:
            print(
                f"{rec['k']:>4} | {rec['row_choice']:>4} | {rec['col_choice']:>4} | "
                f"{fmt(rec['upper_curr']):>12} | {fmt(rec['lower_curr']):>12} | "
                f"{fmt(rec['upper_best']):>12} | {fmt(rec['lower_best']):>12} | {fmt(rec['gap']):>12}"
            )
        print()

    print(f"Представитель x~ = {fmt(r.x_repr)} = {fr(r.x_repr)}")
    print(f"Представитель y~ = {fmt(r.y_repr)} = {fr(r.y_repr)}")
    print(f"h^(N) = {fmt(r.value)} = {fr(r.value)}")
    if r.delta_from_prev is not None:
        print(f"|h^(N) - h^(N-1)| = {fmt(r.delta_from_prev)}")
    print()


def print_final_summary(final_result: GridIterationResult, analytic: Optional[AnalyticSolution]) -> None:
    print("ИТОГ")
    print("-" * 80)
    print(f"Численный метод на последней итерации: {final_result.method}")
    print(f"N = {final_result.N}")
    print(f"Размер матрицы: {final_result.N + 1} x {final_result.N + 1}")
    print(f"Численное решение: x~ = {fmt(final_result.x_repr)}, y~ = {fmt(final_result.y_repr)}")
    print(f"Численная цена игры: h^(N) = {fmt(final_result.value)}")
    print()

    print("Смешанная стратегия игрока A на последней сетке:")
    print(vector_to_str(final_result.x_strategy))
    print("Смешанная стратегия игрока B на последней сетке:")
    print(vector_to_str(final_result.y_strategy))
    print()

    if analytic is not None:
        print("Сравнение с аналитическим решением:")
        print(f"x*  = {fmt(analytic.x_star)}")
        print(f"y*  = {fmt(analytic.y_star)}")
        print(f"h   = {fmt(analytic.h_star)}")
        print(f"|x~-x*| = {fmt(abs(final_result.x_repr - analytic.x_star))}")
        print(f"|y~-y*| = {fmt(abs(final_result.y_repr - analytic.y_star))}")
        print(f"|h^(N)-h| = {fmt(abs(final_result.value - analytic.h_star))}")
    print("-" * 80)
    print()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    a = A_COEF
    b = B_COEF
    c = C_COEF
    d = D_COEF
    e = E_COEF

    print_recognized_task()
    analytic = print_analytic_part(a, b, c, d, e)

    results, final_result = solve_continuous_game_numerically(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        n_start=N_START,
        n_max=N_MAX,
        epsilon_outer=EPSILON_OUTER,
        epsilon_br=EPSILON_BR,
        br_max_iter=BR_MAX_ITER,
    )

    if final_result is None:
        print("Численное решение не получено.")
        return

    print_outer_iterations(results, final_result)

    # Подробно печатаем первые несколько итераций и последнюю
    detail_count = min(OUTER_ITERS_FOR_REPORT, len(results))
    for idx in range(detail_count):
        print_iteration_details(results[idx], idx + 1)

    if len(results) > detail_count:
        print("ПОСЛЕДНЯЯ ИТЕРАЦИЯ")
        print("=" * 80)
        print_iteration_details(final_result, len(results))

    print_final_summary(final_result, analytic)


if __name__ == "__main__":
    main()
