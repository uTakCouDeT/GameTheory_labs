from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_VARIANT = 13
DEFAULT_OUTER_EPSILON = 1e-5      # критерий останова по |H_k - H_{k-1}|
DEFAULT_BR_EPSILON = 1e-1         # точность метода Брауна–Робинсон для дискретной матрицы
DEFAULT_MAX_N = 200               # максимальное число разбиений [0, 1]
DEFAULT_BR_MAX_ITER = 10000
PRINT_FIRST_ITERATIONS = 10
SHOW_MATRICES_FOR_FIRST_ITERATIONS = True


def q(value: str) -> float:
    return float(Fraction(value))


VARIANTS = {
    1:  {"a": q("-5"),  "b": q("5/12"), "c": q("10/3"), "d": q("-2/3"),   "e": q("-4/3")},
    2:  {"a": q("-10"), "b": q("15/4"), "c": q("10"),   "d": q("-4"),     "e": q("-8")},
    3:  {"a": q("-4"),  "b": q("4"),    "c": q("8"),    "d": q("-12/5"),  "e": q("-28/5")},
    4:  {"a": q("-15"), "b": q("20/3"), "c": q("40"),   "d": q("-12"),    "e": q("-24")},
    5:  {"a": q("-3"),  "b": q("12/5"), "c": q("6"),    "d": q("-3/5"),   "e": q("-24/5")},
    6:  {"a": q("-5"),  "b": q("5/2"),  "c": q("15"),   "d": q("-3"),     "e": q("-12")},
    7:  {"a": q("-3"),  "b": q("3/2"),  "c": q("5/2"),  "d": q("-4"),     "e": q("-11/5")},
    8:  {"a": q("-5"),  "b": q("9/2"),  "c": q("15"),   "d": q("-9/2"),   "e": q("-9")},
    9:  {"a": q("-6"),  "b": q("32/5"), "c": q("16"),   "d": q("-16/5"),  "e": q("-64/5")},
    10: {"a": q("-3"),  "b": q("9"),    "c": q("18"),   "d": q("-9/5"),   "e": q("-81/5")},
    11: {"a": q("-5"),  "b": q("5/6"),  "c": q("10/3"), "d": q("-2/3"),   "e": q("-2")},
    12: {"a": q("-10"), "b": q("40/3"), "c": q("40"),   "d": q("-16"),    "e": q("-32")},
    13: {"a": q("-4"),  "b": q("2"),    "c": q("8"),    "d": q("-4/5"),   "e": q("-32/5")},
    14: {"a": q("-6"),  "b": q("16/5"), "c": q("16"),   "d": q("-16/5"),  "e": q("-48/5")},
    15: {"a": q("-15"), "b": q("9/2"),  "c": q("24"),   "d": q("-36/5"),  "e": q("-84/5")},
    16: {"a": q("-5"),  "b": q("5/4"),  "c": q("10/3"), "d": q("-2/3"),   "e": q("-8/3")},
    17: {"a": q("-4"),  "b": q("10/3"), "c": q("16/3"), "d": q("-16/30"), "e": q("-112/30")},
    18: {"a": q("-10"), "b": q("15"),   "c": q("60"),   "d": q("-12"),    "e": q("-48")},
    19: {"a": q("-15"), "b": q("15"),   "c": q("75"),   "d": q("-45/2"),  "e": q("-105/2")},
    20: {"a": q("-5"),  "b": q("10/3"), "c": q("10"),   "d": q("-2"),     "e": q("-8")},
}


def _fmt_default(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def _fr_default(x: float, max_den: int = 10000) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(Fraction(float(x)).limit_denominator(max_den))


def _vector_to_str_default(v: np.ndarray, digits: int = 6) -> str:
    return "[" + ", ".join(_fmt_default(float(x), digits) for x in v) + "]"


def _vector_to_fraction_str_default(v: np.ndarray) -> str:
    return "[" + ", ".join(_fr_default(float(x)) for x in v) + "]"


def _pure_strategy_analysis_default(C: np.ndarray):
    row_mins = C.min(axis=1)
    col_maxs = C.max(axis=0)

    v_lower = float(row_mins.max())
    v_upper = float(col_maxs.min())

    saddle_points = []
    if np.isclose(v_lower, v_upper):
        good_rows = np.where(np.isclose(row_mins, v_lower))[0]
        good_cols = np.where(np.isclose(col_maxs, v_upper))[0]
        for i in good_rows:
            for j in good_cols:
                if np.isclose(C[i, j], v_lower):
                    saddle_points.append((int(i), int(j)))

    return v_lower, v_upper, saddle_points


def _brown_robinson_default(
    C: np.ndarray,
    epsilon: float = 0.1,
    max_iter: int = 10000,
    start_row: int = 0,
    start_col: int = 0,
):
    m, n = C.shape

    row_counts = np.zeros(m, dtype=int)
    col_counts = np.zeros(n, dtype=int)

    row_cum = np.zeros(m, dtype=float)
    col_cum = np.zeros(n, dtype=float)

    i = start_row
    j = start_col

    best_upper = math.inf
    best_lower = -math.inf
    history = []

    for k in range(1, max_iter + 1):
        row_counts[i] += 1
        col_counts[j] += 1

        row_cum += C[:, j]
        col_cum += C[i, :]

        upper_curr = float(row_cum.max() / k)
        lower_curr = float(col_cum.min() / k)

        best_upper = min(best_upper, upper_curr)
        best_lower = max(best_lower, lower_curr)
        gap = float(best_upper - best_lower)

        x_est = row_counts / k
        y_est = col_counts / k

        history.append({
            "k": k,
            "row_choice": i + 1,
            "col_choice": j + 1,
            "row_cum": row_cum.copy(),
            "col_cum": col_cum.copy(),
            "upper_curr": upper_curr,
            "lower_curr": lower_curr,
            "upper_best": float(best_upper),
            "lower_best": float(best_lower),
            "gap": gap,
            "x_est": x_est.copy(),
            "y_est": y_est.copy(),
            "row_counts": row_counts.copy(),
            "col_counts": col_counts.copy(),
        })

        if gap <= epsilon:
            break

        i = int(np.argmax(row_cum))
        j = int(np.argmin(col_cum))

    return history


def _load_lab1_functions():
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parents[1] / "lab_1" / "lab_1.py",
        current_file.parent.parent / "lab_1" / "lab_1.py",
        Path.cwd() / "lab_1" / "lab_1.py",
        Path.cwd().parent / "lab_1" / "lab_1.py",
        Path("/mnt/data/lab_1.py"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location("lab1_shared", path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return {
            "fmt": getattr(module, "fmt", _fmt_default),
            "fr": getattr(module, "fr", _fr_default),
            "vector_to_str": getattr(module, "vector_to_str", _vector_to_str_default),
            "vector_to_fraction_str": getattr(module, "vector_to_fraction_str", _vector_to_fraction_str_default),
            "pure_strategy_analysis": getattr(module, "pure_strategy_analysis", _pure_strategy_analysis_default),
            "brown_robinson": getattr(module, "brown_robinson", _brown_robinson_default),
            "source_path": path,
        }

    return {
        "fmt": _fmt_default,
        "fr": _fr_default,
        "vector_to_str": _vector_to_str_default,
        "vector_to_fraction_str": _vector_to_fraction_str_default,
        "pure_strategy_analysis": _pure_strategy_analysis_default,
        "brown_robinson": _brown_robinson_default,
        "source_path": None,
    }


LAB1 = _load_lab1_functions()
fmt: Callable[[float, int], str] = LAB1["fmt"]
fr: Callable[[float], str] = LAB1["fr"]
vector_to_str = LAB1["vector_to_str"]
vector_to_fraction_str = LAB1["vector_to_fraction_str"]
pure_strategy_analysis = LAB1["pure_strategy_analysis"]
brown_robinson = LAB1["brown_robinson"]


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
    grid: np.ndarray
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
    x_mix: np.ndarray | None = None
    y_mix: np.ndarray | None = None


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


def solve_continuous_game_exact(a: float, b: float, c: float, d: float, e: float, tol: float = 1e-10) -> ContinuousSolution:
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
        x = float(grid[i])
        y = float(grid[j])
        h = float(C[i, j])
        return IterationResult(
            iteration_no=0,
            N=len(grid) - 1,
            grid=grid,
            matrix=C,
            method="sedlo",
            x=x,
            y=y,
            h=h,
            delta_prev=None,
            v_lower=v_lower,
            v_upper=v_upper,
            saddle_points=saddle_points,
        )

    history = brown_robinson(
        C,
        epsilon=br_epsilon,
        max_iter=br_max_iter,
        start_row=0,
        start_col=0,
    )
    last = history[-1]

    x_mix = np.asarray(last["x_est"], dtype=float)
    y_mix = np.asarray(last["y_est"], dtype=float)

    x = float(np.dot(x_mix, grid))
    y = float(np.dot(y_mix, grid))
    h = kernel(x, y, a, b, c, d, e)

    return IterationResult(
        iteration_no=0,
        N=len(grid) - 1,
        grid=grid,
        matrix=C,
        method="brown_robinson",
        x=x,
        y=y,
        h=h,
        delta_prev=None,
        v_lower=v_lower,
        v_upper=v_upper,
        saddle_points=[],
        br_iterations=int(last["k"]),
        br_gap=float(last["gap"]),
        br_upper=float(last["upper_best"]),
        br_lower=float(last["lower_best"]),
        x_mix=x_mix,
        y_mix=y_mix,
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


def print_condition_text():
    print("Рассматривается непрерывная антагонистическая игра на единичном квадрате")
    print("0 <= x <= 1, 0 <= y <= 1 с функцией выигрыша:")
    print("H(x, y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y")
    print()


def print_analytic_solution(sol: ContinuousSolution, a: float, b: float, c: float, d: float, e: float):
    print("Аналитическое решение непрерывной игры")
    print("-" * 80)
    print(f"H(x, y) = {fr(a)}*x^2 + {fr(b)}*y^2 + {fr(c)}*x*y + {fr(d)}*x + {fr(e)}*y")
    print(f"Оптимальная стратегия игрока A: x* = {fmt(sol.x)} = {fr(sol.x)}")
    print(f"Оптимальная стратегия игрока B: y* = {fmt(sol.y)} = {fr(sol.y)}")
    print(f"Цена игры: H(x*, y*) = {fmt(sol.h)} = {fr(sol.h)}")
    print(f"Активный режим решения: {sol.regime}")
    print(f"Hx(x*, y*) = {fmt(sol.hx)}")
    print(f"Hy(x*, y*) = {fmt(sol.hy)}")
    print("-" * 80)
    print()


def print_iteration_details(result: IterationResult):
    size = result.N + 1
    print(f"Итерация {result.iteration_no}: разбиение N = {result.N}, матрица {size}x{size}")

    if SHOW_MATRICES_FOR_FIRST_ITERATIONS:
        print("Матрица H^(N):")
        print(matrix_to_string(result.matrix))
        print()

    if result.method == "sedlo":
        saddles_str = ", ".join(
            f"({i + 1}, {j + 1})" for i, j in result.saddle_points
        )
        print("Решение дискретной игры: седловая точка")
        print(f"Координаты седловых позиций матрицы: {saddles_str}")
    else:
        print("Решение дискретной игры: метод Брауна–Робинсон")
        print(f"Итераций Брауна–Робинсон: {result.br_iterations}")
        print(f"Лучшая верхняя оценка: {fmt(result.br_upper)}")
        print(f"Лучшая нижняя оценка: {fmt(result.br_lower)}")
        print(f"Погрешность Брауна–Робинсон: {fmt(result.br_gap)}")
        print(f"x~ по сетке: {vector_to_str(result.x_mix)}")
        print(f"y~ по сетке: {vector_to_str(result.y_mix)}")

    print(f"x = {fmt(result.x)} = {fr(result.x)}")
    print(f"y = {fmt(result.y)} = {fr(result.y)}")
    print(f"H = {fmt(result.h)} = {fr(result.h)}")

    if result.delta_prev is None:
        print("Δ с предыдущим приближением: ---")
    else:
        print(f"Δ с предыдущим приближением: {fmt(result.delta_prev)}")
    print("-" * 80)
    print()


def print_first_iterations_summary(results: list[IterationResult], limit: int = PRINT_FIRST_ITERATIONS):
    print("Первые итерации уточнения")
    print("-" * 120)
    header = (
        f"{'Ит.':>4} | {'N':>4} | {'Размер':>8} | {'Метод':>16} | "
        f"{'x':>10} | {'y':>10} | {'H':>12} | {'Δ_prev':>12}"
    )
    print(header)
    print("-" * 120)

    for res in results[:limit]:
        size = f"{res.N + 1}x{res.N + 1}"
        method = "седло" if res.method == "sedlo" else "Браун-Робинсон"
        delta_str = "---" if res.delta_prev is None else fmt(res.delta_prev)
        print(
            f"{res.iteration_no:>4} | {res.N:>4} | {size:>8} | {method:>16} | "
            f"{fmt(res.x):>10} | {fmt(res.y):>10} | {fmt(res.h):>12} | {delta_str:>12}"
        )

    print("-" * 120)
    print()


def print_final_summary(results: list[IterationResult], outer_epsilon: float, converged: bool):
    last = results[-1]

    print("Итоговый результат итерационного уточнения")
    print("-" * 80)
    print(f"Критерий останова: |H_k - H_(k-1)| <= {outer_epsilon}")
    print(f"Итерация, на которой получен текущий результат: {last.iteration_no}")
    print(f"Разбиение: N = {last.N}, матрица размера {last.N + 1}x{last.N + 1}")

    if converged:
        print("Критерий останова выполнен.")
    else:
        print("Критерий останова не выполнен: достигнут лимит по N.")

    print(f"Метод решения последней дискретной игры: {'седловая точка' if last.method == 'sedlo' else 'Браун-Робинсон'}")
    print(f"x = {fmt(last.x)} = {fr(last.x)}")
    print(f"y = {fmt(last.y)} = {fr(last.y)}")
    print(f"H = {fmt(last.h)} = {fr(last.h)}")

    if last.delta_prev is not None:
        print(f"Последнее изменение |ΔH| = {fmt(last.delta_prev)}")

    if last.method != "sedlo":
        print(f"Итераций Брауна–Робинсон на последнем шаге: {last.br_iterations}")
        print(f"Погрешность Брауна–Робинсон на последнем шаге: {fmt(last.br_gap)}")

    print("-" * 80)
    print()


def print_configuration(args, coeffs: dict[str, float]):
    print("Используемые параметры")
    print("-" * 80)
    print(f"Вариант: {args.variant}")
    print(f"a = {fr(coeffs['a'])}, b = {fr(coeffs['b'])}, c = {fr(coeffs['c'])}, d = {fr(coeffs['d'])}, e = {fr(coeffs['e'])}")
    print(f"Критерий останова по внешней итерации: {args.outer_epsilon}")
    print(f"Точность Брауна–Робинсон: {args.br_epsilon}")
    print(f"Максимальное N: {args.max_n}")
    print(f"Максимум итераций Брауна–Робинсон: {args.br_max_iter}")

    if LAB1["source_path"] is not None:
        print(f"Функции из lab_1.py загружены из: {LAB1['source_path']}")
    else:
        print("Файл lab_1.py не найден, использованы встроенные совместимые функции.")
    print("-" * 80)
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ЛР №3. Решение непрерывной антагонистической игры на единичном квадрате."
    )
    parser.add_argument("--variant", type=int, default=DEFAULT_VARIANT, choices=sorted(VARIANTS.keys()))
    parser.add_argument("--outer-epsilon", type=float, default=DEFAULT_OUTER_EPSILON)
    parser.add_argument("--br-epsilon", type=float, default=DEFAULT_BR_EPSILON)
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    parser.add_argument("--br-max-iter", type=int, default=DEFAULT_BR_MAX_ITER)
    parser.add_argument(
        "--no-matrices",
        action="store_true",
        help="Не печатать матрицы на первых итерациях.",
    )
    return parser.parse_args()


def main():
    global SHOW_MATRICES_FOR_FIRST_ITERATIONS

    args = parse_args()
    if args.no_matrices:
        SHOW_MATRICES_FOR_FIRST_ITERATIONS = False

    coeffs = VARIANTS[args.variant]
    a, b, c, d, e = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"], coeffs["e"]

    print_condition_text()
    print_configuration(args, coeffs)

    analytic = solve_continuous_game_exact(a, b, c, d, e)
    print_analytic_solution(analytic, a, b, c, d, e)

    results, converged = solve_iteratively(
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        outer_epsilon=args.outer_epsilon,
        br_epsilon=args.br_epsilon,
        max_N=args.max_n,
        br_max_iter=args.br_max_iter,
    )

    print_first_iterations_summary(results)

    for res in results[:PRINT_FIRST_ITERATIONS]:
        print_iteration_details(res)

    if len(results) > PRINT_FIRST_ITERATIONS:
        print("...")
        print(f"Подробный вывод после {PRINT_FIRST_ITERATIONS}-й итерации опущен.")
        print()

    print_final_summary(results, args.outer_epsilon, converged)

    last = results[-1]
    print("Сравнение аналитического и итерационного решения")
    print("-" * 80)
    print(f"Аналитическое:  x* = {fmt(analytic.x)}, y* = {fmt(analytic.y)}, H* = {fmt(analytic.h)}")
    print(f"Итерационное:   x  = {fmt(last.x)}, y  = {fmt(last.y)}, H  = {fmt(last.h)}")
    print(f"|x - x*| = {fmt(abs(last.x - analytic.x))}")
    print(f"|y - y*| = {fmt(abs(last.y - analytic.y))}")
    print(f"|H - H*| = {fmt(abs(last.h - analytic.h))}")
    print("-" * 80)


if __name__ == "__main__":
    main()
