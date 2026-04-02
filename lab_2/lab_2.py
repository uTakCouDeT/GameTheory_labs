from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from lab_1 import fmt, fr, pure_strategy_analysis, brown_robinson

OUTER_EPSILON = 1e-5
BR_EPSILON = 1e-1
MAX_N = 2000
BR_MAX_ITER = 10000
PRINT_FIRST_ITERATIONS = 10
PRINT_DETAILS_ITER = 10


def q(value: str) -> float:
    return float(Fraction(value))


@dataclass
class ContinuousSolution:
    x: float
    y: float
    h: float
    regime: str
    hx: float
    hy: float


@dataclass
class IterationResult:
    iteration_no: int
    N: int
    matrix: np.ndarray
    method: str
    x: float
    y: float
    h: float
    delta_prev: float | None
    v_lower: float
    v_upper: float
    saddle_points: list[tuple[int, int]]
    br_iterations: int | None = None
    br_gap: float | None = None
    br_upper: float | None = None
    br_lower: float | None = None


def kernel(x: float, y: float, a: float, b: float, c: float, d: float, e: float) -> float:
    return float(a * x * x + b * y * y + c * x * y + d * x + e * y)


def hx(x: float, y: float, a: float, c: float, d: float) -> float:
    return float(2 * a * x + c * y + d)


def hy(x: float, y: float, b: float, c: float, e: float) -> float:
    return float(c * x + 2 * b * y + e)


def build_matrix(a: float, b: float, c: float, d: float, e: float, N: int):
    grid = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    C = a * X * X + b * Y * Y + c * X * Y + d * X + e * Y
    return grid, C.astype(float)


def solve_continuous_game_exact(a: float, b: float, c: float, d: float, e: float,
                                tol: float = 1e-10) -> ContinuousSolution:
    candidates: list[ContinuousSolution] = []

    for x_case in ("0", "1", "i"):
        for y_case in ("0", "1", "i"):
            A = []
            rhs = []

            if x_case == "0":
                A.append([1.0, 0.0])
                rhs.append(0.0)
            elif x_case == "1":
                A.append([1.0, 0.0])
                rhs.append(1.0)
            else:
                A.append([2 * a, c])
                rhs.append(-d)

            if y_case == "0":
                A.append([0.0, 1.0])
                rhs.append(0.0)
            elif y_case == "1":
                A.append([0.0, 1.0])
                rhs.append(1.0)
            else:
                A.append([c, 2 * b])
                rhs.append(-e)

            A = np.array(A, dtype=float)
            rhs = np.array(rhs, dtype=float)

            try:
                x, y = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue

            if not (-tol <= x <= 1 + tol and -tol <= y <= 1 + tol):
                continue

            x = min(max(float(x), 0.0), 1.0)
            y = min(max(float(y), 0.0), 1.0)

            dx = hx(x, y, a, c, d)
            dy = hy(x, y, b, c, e)

            ok = True

            if x_case == "0":
                ok = ok and dx <= tol
            elif x_case == "1":
                ok = ok and dx >= -tol
            else:
                ok = ok and (tol < x < 1 - tol) and abs(dx) <= 1e-7

            if y_case == "0":
                ok = ok and dy >= -tol
            elif y_case == "1":
                ok = ok and dy <= tol
            else:
                ok = ok and (tol < y < 1 - tol) and abs(dy) <= 1e-7

            if ok:
                candidates.append(
                    ContinuousSolution(
                        x=x,
                        y=y,
                        h=kernel(x, y, a, b, c, d, e),
                        regime=f"x={x_case}, y={y_case}",
                        hx=dx,
                        hy=dy,
                    )
                )

    if not candidates:
        raise RuntimeError("Не удалось найти аналитическое решение непрерывной игры.")

    candidates.sort(key=lambda s: (round(s.h, 12), s.x, s.y), reverse=True)
    return candidates[0]


def solve_matrix_game(
        C: np.ndarray,
        grid: np.ndarray,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        br_epsilon: float,
        br_max_iter: int,
) -> IterationResult:
    v_lower, v_upper, saddle_points = pure_strategy_analysis(C)

    if saddle_points:
        i, j = saddle_points[0]
        return IterationResult(
            iteration_no=0,
            N=len(grid) - 1,
            matrix=C.copy(),
            method="sedlo",
            x=float(grid[i]),
            y=float(grid[j]),
            h=float(C[i, j]),
            delta_prev=None,
            v_lower=v_lower,
            v_upper=v_upper,
            saddle_points=saddle_points,
        )

    history = brown_robinson(C, epsilon=br_epsilon, max_iter=br_max_iter, start_row=0, start_col=0)
    last = history[-1]

    x_mix = np.asarray(last["x_est"], dtype=float)
    y_mix = np.asarray(last["y_est"], dtype=float)

    x_value = float(np.dot(x_mix, grid))
    y_value = float(np.dot(y_mix, grid))
    h_value = kernel(x_value, y_value, a, b, c, d, e)

    return IterationResult(
        iteration_no=0,
        N=len(grid) - 1,
        matrix=C.copy(),
        method="brown_robinson",
        x=x_value,
        y=y_value,
        h=h_value,
        delta_prev=None,
        v_lower=v_lower,
        v_upper=v_upper,
        saddle_points=[],
        br_iterations=int(last["k"]),
        br_gap=float(last["gap"]),
        br_upper=float(last["upper_best"]),
        br_lower=float(last["lower_best"]),
    )


def solve_iteratively(
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        outer_epsilon: float,
        br_epsilon: float,
        max_N: int,
        br_max_iter: int,
):
    results: list[IterationResult] = []
    prev_h: float | None = None

    for N in range(2, max_N + 1):
        grid, C = build_matrix(a, b, c, d, e, N)
        res = solve_matrix_game(C, grid, a, b, c, d, e, br_epsilon, br_max_iter)
        res.iteration_no = N - 1
        res.delta_prev = None if prev_h is None else abs(res.h - prev_h)
        results.append(res)

        if prev_h is not None and res.delta_prev <= outer_epsilon:
            return results, True

        prev_h = res.h

    return results, False


def matrix_to_string(C: np.ndarray, digits: int = 6) -> str:
    rows = []
    for row in C:
        rows.append(" ".join(f"{float(value):>{digits + 6}.{digits}f}" for value in row))
    return "\n".join(rows)


def print_configuration(coeffs: dict[str, float]):
    print(f"\nВариант: 13")
    print("-" * 80)
    print("Рассматривается непрерывная антагонистическая игра на единичном квадрате")
    print("0 <= x <= 1, 0 <= y <= 1 с функцией выигрыша:")
    print("H(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y")
    print(
        f"a = {fr(coeffs['a'])}, b = {fr(coeffs['b'])}, c = {fr(coeffs['c'])}, d = {fr(coeffs['d'])}, e = {fr(coeffs['e'])}")
    print(f"\nКритерий останова по внешней итерации: {OUTER_EPSILON}")
    print(f"Точность Брауна–Робинсон: {BR_EPSILON}")
    print(f"Максимальное N: {MAX_N}")
    print(f"Максимум итераций Брауна–Робинсон: {BR_MAX_ITER}")
    print("-" * 80)
    print()


def print_analytic_solution(sol: ContinuousSolution, a: float, b: float, c: float, d: float, e: float):
    print("Аналитическое решение непрерывной игры")
    print("-" * 80)
    print(f"H(x, y) = {fr(a)}*x^2 + {fr(b)}*y^2 + {fr(c)}*x*y + {fr(d)}*x + {fr(e)}*y")
    print(f"x* = {fmt(sol.x)} = {fr(sol.x)}")
    print(f"y* = {fmt(sol.y)} = {fr(sol.y)}")
    print(f"H(x*, y*) = {fmt(sol.h)} = {fr(sol.h)}")
    print(f"Активный режим: {sol.regime}")
    print(f"Hx(x*, y*) = {fmt(sol.hx)}")
    print(f"Hy(x*, y*) = {fmt(sol.hy)}")
    print("-" * 80)
    print()


def print_first_iterations_summary(results: list[IterationResult], limit: int = PRINT_FIRST_ITERATIONS):
    if PRINT_FIRST_ITERATIONS == 0:
        return 0

    print(f"Первые {min(limit, len(results))} итераций внешнего уточнения")
    print("-" * 120)
    print(f"{'Ит.':>4} | {'N':>4} | {'Размер':>8} | {'Метод':>16} | {'x':>10} | {'y':>10} | {'H':>12} | {'Δ_prev':>12}")
    print("-" * 120)

    for res in results[:limit]:
        size = f"{res.N + 1}x{res.N + 1}"
        method = "седло" if res.method == "sedlo" else "Брауна-Робинсон"
        delta_str = "---" if res.delta_prev is None else fmt(res.delta_prev)
        print(
            f"{res.iteration_no:>4} | {res.N:>4} | {size:>8} | {method:>16} | "
            f"{fmt(res.x):>10} | {fmt(res.y):>10} | {fmt(res.h):>12} | {delta_str:>12}"
        )

    print("-" * 120)
    print()


def print_iteration_details(results: list[IterationResult], limit: int = PRINT_DETAILS_ITER):
    if PRINT_DETAILS_ITER == 0:
        return 0

    print(f"Подробности по первым {min(limit, len(results))} итерациям")
    print("-" * 80)

    for res in results[:limit]:
        print(f"Итерация {res.iteration_no}")
        print(f"Разбиение: N = {res.N}, матрица {res.N + 1}x{res.N + 1}")
        print("Матрица игры:")
        print(matrix_to_string(res.matrix))
        print(f"Метод решения: {'седловая точка' if res.method == 'sedlo' else 'Браун–Робинсон'}")
        print(f"x = {fmt(res.x)} = {fr(res.x)}")
        print(f"y = {fmt(res.y)} = {fr(res.y)}")
        print(f"H = {fmt(res.h)} = {fr(res.h)}")
        print(f"Нижняя цена игры max(min) = {fmt(res.v_lower)}")
        print(f"Верхняя цена игры min(max) = {fmt(res.v_upper)}")

        if res.delta_prev is None:
            print("Δ с предыдущим приближением: ---")
        else:
            print(f"Δ с предыдущим приближением: {fmt(res.delta_prev)}")

        if res.method == "sedlo":
            saddles_str = ", ".join(f"({i + 1}, {j + 1})" for i, j in res.saddle_points)
            print(f"Седловые точки: {saddles_str}")
        else:
            print(f"Итераций Брауна–Робинсон: {res.br_iterations}")
            print(f"Верхняя оценка цены игры: {fmt(res.br_upper)}")
            print(f"Нижняя оценка цены игры: {fmt(res.br_lower)}")
            print(f"Погрешность Брауна–Робинсон: {fmt(res.br_gap)}")

        print("-" * 80)

    print()


def print_final_summary(results: list[IterationResult], analytic: ContinuousSolution, outer_epsilon: float,
                        converged: bool):
    last = results[-1]

    print("Итоговый результат")
    print("-" * 80)
    print(f"Критерий останова: |H_k - H_(k-1)| <= {outer_epsilon}")
    print(f"Номер итерации, на которой получен итог: {last.iteration_no}")
    print(f"Разбиение: N = {last.N}, матрица {last.N + 1}x{last.N + 1}")
    print(f"Критерий останова {'выполнен' if converged else 'не выполнен, достигнут лимит по N'}")
    print(f"Метод на последнем шаге: {'седловая точка' if last.method == 'sedlo' else 'Браун–Робинсон'}")

    if last.delta_prev is not None:
        print(f"Последнее изменение |ΔH| = {fmt(last.delta_prev)}")

    if last.method == "brown_robinson":
        print(f"Итераций Брауна–Робинсон на последнем шаге: {last.br_iterations}")
        print(f"Верхняя оценка цены игры: {fmt(last.br_upper)}")
        print(f"Нижняя оценка цены игры: {fmt(last.br_lower)}")
        print(f"Погрешность Брауна–Робинсон: {fmt(last.br_gap)}")

    print()
    print(f"Аналитическое решение: x* = {fmt(analytic.x)}, y* = {fmt(analytic.y)}, H* = {fmt(analytic.h)}")
    print(f"Итерационное решение:  x  = {fmt(last.x)}, y  = {fmt(last.y)}, H  = {fmt(last.h)}")
    print(f"|x - x*| = {fmt(abs(last.x - analytic.x))}")
    print(f"|y - y*| = {fmt(abs(last.y - analytic.y))}")
    print(f"|H - H*| = {fmt(abs(last.h - analytic.h))}")
    print("-" * 80)


VARIANTS = {
    1: {"a": q("-5"), "b": q("5/12"), "c": q("10/3"), "d": q("-2/3"), "e": q("-4/3")},
    2: {"a": q("-10"), "b": q("15/4"), "c": q("10"), "d": q("-4"), "e": q("-8")},
    3: {"a": q("-4"), "b": q("4"), "c": q("8"), "d": q("-12/5"), "e": q("-28/5")},
    4: {"a": q("-15"), "b": q("20/3"), "c": q("40"), "d": q("-12"), "e": q("-24")},
    5: {"a": q("-3"), "b": q("12/5"), "c": q("6"), "d": q("-3/5"), "e": q("-24/5")},
    6: {"a": q("-5"), "b": q("5/2"), "c": q("15"), "d": q("-3"), "e": q("-12")},
    7: {"a": q("-3"), "b": q("3/2"), "c": q("5/2"), "d": q("-4"), "e": q("-11/5")},
    8: {"a": q("-5"), "b": q("9/2"), "c": q("15"), "d": q("-9/2"), "e": q("-9")},
    9: {"a": q("-6"), "b": q("32/5"), "c": q("16"), "d": q("-16/5"), "e": q("-64/5")},
    10: {"a": q("-3"), "b": q("9"), "c": q("18"), "d": q("-9/5"), "e": q("-81/5")},
    11: {"a": q("-5"), "b": q("5/6"), "c": q("10/3"), "d": q("-2/3"), "e": q("-2")},
    12: {"a": q("-10"), "b": q("40/3"), "c": q("40"), "d": q("-16"), "e": q("-32")},
    13: {"a": q("-4"), "b": q("2"), "c": q("8"), "d": q("-4/5"), "e": q("-32/5")},
    14: {"a": q("-6"), "b": q("16/5"), "c": q("16"), "d": q("-16/5"), "e": q("-48/5")},
    15: {"a": q("-15"), "b": q("9/2"), "c": q("24"), "d": q("-36/5"), "e": q("-84/5")},
    16: {"a": q("-5"), "b": q("5/4"), "c": q("10/3"), "d": q("-2/3"), "e": q("-8/3")},
    17: {"a": q("-4"), "b": q("10/3"), "c": q("16/3"), "d": q("-16/30"), "e": q("-112/30")},
    18: {"a": q("-10"), "b": q("15"), "c": q("60"), "d": q("-12"), "e": q("-48")},
    19: {"a": q("-15"), "b": q("15"), "c": q("75"), "d": q("-45/2"), "e": q("-105/2")},
    20: {"a": q("-5"), "b": q("10/3"), "c": q("10"), "d": q("-2"), "e": q("-8")},
}


def main():
    coeffs = VARIANTS[13]
    a, b, c, d, e = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"], coeffs["e"]

    print_configuration(coeffs)

    analytic = solve_continuous_game_exact(a, b, c, d, e)
    print_analytic_solution(analytic, a, b, c, d, e)

    results, converged = solve_iteratively(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        outer_epsilon=OUTER_EPSILON,
        br_epsilon=BR_EPSILON,
        max_N=MAX_N,
        br_max_iter=BR_MAX_ITER,
    )

    print_first_iterations_summary(results)
    print_iteration_details(results)
    print_final_summary(results, analytic, OUTER_EPSILON, converged)


if __name__ == "__main__":
    main()
