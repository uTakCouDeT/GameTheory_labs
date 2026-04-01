import math
from fractions import Fraction
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt


def fr(x, max_den=10000):
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(Fraction(float(x)).limit_denominator(max_den))


def fmt(x, digits=6):
    return f"{x:.{digits}f}"


def vector_to_str(v, digits=6):
    return "[" + ", ".join(fmt(x, digits) for x in v) + "]"


def vector_to_fraction_str(v):
    return "[" + ", ".join(fr(x) for x in v) + "]"


def array_int_or_float_to_str(arr):
    out = []
    for x in arr:
        if abs(x - round(x)) < 1e-12:
            out.append(str(int(round(x))))
        else:
            out.append(fmt(x))
    return "[" + ", ".join(out) + "]"


def pure_strategy_analysis(C):
    row_mins = C.min(axis=1)
    col_maxs = C.max(axis=0)

    v_lower = row_mins.max()
    v_upper = col_maxs.min()

    saddle_points = []
    if np.isclose(v_lower, v_upper):
        good_rows = np.where(np.isclose(row_mins, v_lower))[0]
        good_cols = np.where(np.isclose(col_maxs, v_upper))[0]
        for i in good_rows:
            for j in good_cols:
                if np.isclose(C[i, j], v_lower):
                    saddle_points.append((i, j))

    return v_lower, v_upper, saddle_points


def analytical_solution(matrix):
    n = matrix.shape[0]
    ones = np.ones(n)

    det = np.linalg.det(matrix)
    if abs(det) < 1e-12:
        return None

    inv_matrix = np.linalg.inv(matrix)
    denom = float((ones @ inv_matrix @ ones))

    x = (ones @ inv_matrix).flatten() / denom
    y = (inv_matrix @ ones).flatten() / denom
    v = 1.0 / denom

    return inv_matrix, x, y, v


def analytic_solution_by_supports(C, tol=1e-9):
    m, n = C.shape
    solutions = []

    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                sub = C[np.ix_(I, J)].astype(float)

                if abs(np.linalg.det(sub)) < tol:
                    continue

                inv_sub = np.linalg.inv(sub)
                ones = np.ones(k)

                denom = float(ones @ inv_sub @ ones)
                if abs(denom) < tol:
                    continue

                x_support = (ones @ inv_sub) / denom
                y_support = (inv_sub @ ones) / denom
                v = 1.0 / denom

                if np.any(x_support < -tol) or np.any(y_support < -tol):
                    continue

                x = np.zeros(m)
                y = np.zeros(n)
                x[list(I)] = x_support
                y[list(J)] = y_support

                x[x < tol] = 0.0
                y[y < tol] = 0.0

                sx = x.sum()
                sy = y.sum()
                if sx <= tol or sy <= tol:
                    continue

                x /= sx
                y /= sy

                row_payoffs = C @ y
                col_payoffs = x @ C

                if np.any(row_payoffs > v + tol):
                    continue
                if np.any(col_payoffs < v - tol):
                    continue
                if np.any(np.abs(row_payoffs[list(I)] - v) > tol):
                    continue
                if np.any(np.abs(col_payoffs[list(J)] - v) > tol):
                    continue

                solutions.append({
                    "k": k,
                    "rows_support": tuple(i + 1 for i in I),
                    "cols_support": tuple(j + 1 for j in J),
                    "x": x,
                    "y": y,
                    "v": v,
                    "row_payoffs": row_payoffs,
                    "col_payoffs": col_payoffs,
                })

    if not solutions:
        return None

    solutions.sort(key=lambda s: (-s["k"], s["rows_support"], s["cols_support"]))
    return solutions[0]


def brown_robinson(C, epsilon=0.1, max_iter=10000, start_row=0, start_col=0):
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

        upper_curr = row_cum.max() / k
        lower_curr = col_cum.min() / k

        best_upper = min(best_upper, upper_curr)
        best_lower = max(best_lower, lower_curr)
        gap = best_upper - best_lower

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
            "upper_best": best_upper,
            "lower_best": best_lower,
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


def print_rec(rec):
    row_cum_str = array_int_or_float_to_str(rec["row_cum"])
    col_cum_str = array_int_or_float_to_str(rec["col_cum"])
    print(
        f"{rec['k']:>4} | "
        f" x{rec['row_choice']:<2} | "
        f" y{rec['col_choice']:<2} | "
        f"{row_cum_str:<30} | "
        f"{col_cum_str:<30} | "
        f"{fmt(rec['upper_curr']):>10} | "
        f"{fmt(rec['lower_curr']):>10} | "
        f"{fmt(rec['upper_best']):>10} | "
        f"{fmt(rec['lower_best']):>10} | "
        f"{fmt(rec['gap']):>10}"
    )


def print_brown_robinson_table(history, cut=False):
    if not history:
        return

    print("Таблица метода Брауна-Робинсон")
    print("-" * 150)
    header = (
        f"{'k':>4} | {'x_i':>4} | {'y_i':>4} | "
        f"{'накопл. выигрыши A по строкам':<30} | "
        f"{'накопл. выигрыши A по столбцам':<30} | "
        f"{'v_max/k':>10} | {'v_min/k':>10} | {'min v_max':>10} | {'max v_min':>10} | {'E':>10}"
    )
    print(header)
    print("-" * 150)

    history_start = history[:7] if cut else history

    for rec in history_start:
        print_rec(rec)

    if cut:
        print(" " * 70 + "." * 10 + " " * 70)
        for rec in history[-7:]:
            print_rec(rec)

    print("-" * 150)
    print()


def plot_brown_robinson_graphs(history, analytic=None, save=False, prefix="lab1"):
    if not history:
        print("История метода Брауна-Робинсон пуста, графики построить нельзя.")
        return

    ks = np.array([rec["k"] for rec in history], dtype=int)

    upper_curr = np.array([rec["upper_curr"] for rec in history], dtype=float)
    lower_curr = np.array([rec["lower_curr"] for rec in history], dtype=float)
    upper_best = np.array([rec["upper_best"] for rec in history], dtype=float)
    lower_best = np.array([rec["lower_best"] for rec in history], dtype=float)
    gaps = np.array([rec["gap"] for rec in history], dtype=float)

    x_est_all = np.array([rec["x_est"] for rec in history], dtype=float)  # shape = (K, m)
    y_est_all = np.array([rec["y_est"] for rec in history], dtype=float)  # shape = (K, n)

    m = x_est_all.shape[1]
    n = y_est_all.shape[1]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, upper_best, label="Лучшая верхняя оценка")
    plt.plot(ks, lower_best, label="Лучшая нижняя оценка")
    plt.plot(ks, upper_curr, linestyle="--", alpha=0.7, label="Текущая верхняя оценка")
    plt.plot(ks, lower_curr, linestyle="--", alpha=0.7, label="Текущая нижняя оценка")

    if analytic is not None:
        plt.axhline(analytic["v"], linestyle=":", linewidth=2,
                    label=f"Аналитическая цена игры v = {analytic['v']:.6f}")

    plt.xlabel("Номер итерации k")
    plt.ylabel("Оценка цены игры")
    plt.title("Сходимость оценок цены игры в методе Брауна-Робинсон")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"data/{prefix}_value_bounds.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    for i in range(m):
        plt.plot(ks, x_est_all[:, i], label=f"x{i + 1}")

    if analytic is not None:
        for i in range(m):
            plt.axhline(
                analytic["x"][i],
                linestyle=":",
                linewidth=1,
                alpha=0.9,
                label=f"x{i + 1}* = {analytic['x'][i]:.6f}" if ks[0] == 1 else None
            )

        plt.clf()
        plt.figure(figsize=(10, 6))
        for i in range(m):
            plt.plot(ks, x_est_all[:, i], label=f"x{i + 1} (числ.)")
            plt.axhline(analytic["x"][i], linestyle=":", linewidth=1, alpha=0.9, label=f"x{i + 1}* (аналит.)")

    plt.xlabel("Номер итерации k")
    plt.ylabel("Частота использования стратегии")
    plt.title("Сходимость смешанной стратегии игрока A")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"data/{prefix}_player_A_frequencies.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    for j in range(n):
        plt.plot(ks, y_est_all[:, j], label=f"y{j + 1}")

    if analytic is not None:
        for j in range(n):
            plt.axhline(
                analytic["y"][j],
                linestyle=":",
                linewidth=1,
                alpha=0.9,
                label=f"y{j + 1}* = {analytic['y'][j]:.6f}" if ks[0] == 1 else None
            )

        plt.clf()
        plt.figure(figsize=(10, 6))
        for j in range(n):
            plt.plot(ks, y_est_all[:, j], label=f"y{j + 1} (числ.)")
            plt.axhline(analytic["y"][j], linestyle=":", linewidth=1, alpha=0.9, label=f"y{j + 1}* (аналит.)")

    plt.xlabel("Номер итерации k")
    plt.ylabel("Частота использования стратегии")
    plt.title("Сходимость смешанной стратегии игрока B")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"data/{prefix}_player_B_frequencies.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(ks, gaps, label="E(k) = min(v_max/k) - max(v_min/k)")
    plt.xlabel("Номер итерации k")
    plt.ylabel("Погрешность")
    plt.title("Изменение погрешности метода Брауна-Робинсон")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"data/{prefix}_gap.png", dpi=300, bbox_inches="tight")

    plt.show()


def main():
    C = np.array([
        [9, 10, 13],
        [1, 18, 11],
        [17, 4, 0],
    ], dtype=float)

    print()
    print("Матрица игры A:")
    for row in C:
        print(' '.join(f'{int(x):>{4}}' for x in row))
    print()

    v_lower, v_upper, saddles = pure_strategy_analysis(C)

    print("Анализ игры в чистых стратегиях")
    print(f"Нижняя цена игры max(min): {fmt(v_lower)}")
    print(f"Верхняя цена игры min(max): {fmt(v_upper)}")

    if saddles:
        print("Седловые точки есть:")
        for i, j in saddles:
            print(f"  A{i + 1}, B{j + 1}, значение = {fmt(C[i, j])}")
    else:
        print("Седловой точки нет => решение ищем в смешанных стратегиях.")
    print()

    # Аналитическое решение
    print("Аналитическое решение")
    analytic = analytic_solution_by_supports(C)

    if analytic is None:
        print("Аналитическое решение этим способом не найдено.")
        print()
    else:
        print(f"Найдена опора игрока A: {analytic['rows_support']}")
        print(f"Найдена опора игрока B: {analytic['cols_support']}")
        print(f"Цена игры v = {fmt(analytic['v'])} = {fr(analytic['v'])}")
        print(f"Оптимальная стратегия A: x* = {vector_to_str(analytic['x'])}")
        print(f"Оптимальная стратегия A (дроби): {vector_to_fraction_str(analytic['x'])}")
        print(f"Оптимальная стратегия B: y* = {vector_to_str(analytic['y'])}")
        print(f"Оптимальная стратегия B (дроби): {vector_to_fraction_str(analytic['y'])}")
        print()
        print("Проверка:")
        print(f"Выигрыши строк против y*: {vector_to_str(analytic['row_payoffs'])}")
        print(f"Выигрыши столбцов против x*: {vector_to_str(analytic['col_payoffs'])}")
        print("Опорные должны быть равны v")
        print()

    # Браун-Робинсон
    print("Метод Брауна-Робинсон")
    print(f"Точность ε = 0.1")
    print(f"Начальная пара стратегий: A1, B1")
    print()

    history = brown_robinson(C, epsilon=0.1, max_iter=10000, start_row=0, start_col=0, )
    print_brown_robinson_table(history, cut=True)

    last = history[-1]
    k = last["k"]
    x_br = last["x_est"]
    y_br = last["y_est"]
    v_top = last["upper_best"]
    v_bottom = last["lower_best"]
    v_mid = (v_top + v_bottom) / 2

    print("Итоги метода Брауна-Робинсон")
    print(f"Число итераций: {k}")
    print(f"Лучшая верхняя оценка цены игры: {fmt(v_top)} = {fr(v_top)}")
    print(f"Лучшая нижняя оценка цены игры:  {fmt(v_bottom)} = {fr(v_bottom)}")
    print(f"Погрешность E = {fmt(last['gap'])}")
    print(f"Приближённая цена игры (середина интервала): {fmt(v_mid)} = {fr(v_mid)}")
    print()

    print("Число использований стратегий:")
    print(f"A: {last['row_counts']}")
    print(f"B: {last['col_counts']}")
    print()

    print("Частоты использования стратегий (приближение смешанных стратегий):")
    print(f"A: x~ = {vector_to_str(x_br)}")
    print(f"A: x~ (дроби) = {vector_to_fraction_str(x_br)}")
    print(f"B: y~ = {vector_to_str(y_br)}")
    print(f"B: y~ (дроби) = {vector_to_fraction_str(y_br)}")
    print()

    # Сравнение
    print("Сравнение методов")
    if analytic is not None:
        print(f"Аналитическая цена игры:          {fmt(analytic['v'])}")
        print(f"Браун-Робинсон, средняя оценка:   {fmt(v_mid)}")
        print(f"Разница:                          {fmt(abs(analytic['v'] - v_mid))}")
        print()
        print("Сравнение стратегий:")
        print(f"Аналитическая x*: {vector_to_str(analytic['x'])}")
        print(f"Браун-Робинсон x~: {vector_to_str(x_br)}")
        print(f"Аналитическая y*: {vector_to_str(analytic['y'])}")
        print(f"Браун-Робинсон y~: {vector_to_str(y_br)}")
    else:
        print("Сравнение по стратегии и цене с аналитическим решением не выполняется,")
        print("так как аналитическое решение не найдено.")
    print()
    plot_brown_robinson_graphs(history, analytic=analytic, save=True)


if __name__ == "__main__":
    main()
