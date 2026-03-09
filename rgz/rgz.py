import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Исходные данные
# =========================================================
A = np.array([
    [1, 1, 0, 7, 6],
    [4, 3, 4, -1, -2],
    [4, 5, 4, 0, 0],
    [4, 3, 3, 1, -1],
    [4, 4, 4, 1, 1]
], dtype=float)


# =========================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ ДЛЯ КРАСИВОГО ВЫВОДА
# =========================================================
def fmt(x, digits=6):
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def print_matrix(M, title="Матрица", digits=6, row_labels=None, col_labels=None):
    M = np.array(M, dtype=float)
    rows, cols = M.shape

    text = [[fmt(M[i, j], digits) for j in range(cols)] for i in range(rows)]

    if row_labels is None:
        row_labels = [f"{i + 1}" for i in range(rows)]
    if col_labels is None:
        col_labels = [f"{j + 1}" for j in range(cols)]

    cell_w = max(
        max(len(x) for row in text for x in row),
        max(len(str(x)) for x in row_labels),
        max(len(str(x)) for x in col_labels),
        3
    )

    print(f"\n{title}:")
    print(" " * (cell_w + 3) + " ".join(f"{str(c):>{cell_w}}" for c in col_labels))
    print(" " * (cell_w + 2) + "-" * ((cell_w + 1) * cols))
    for i in range(rows):
        print(f"{str(row_labels[i]):>{cell_w}} | " + " ".join(f"{text[i][j]:>{cell_w}}" for j in range(cols)))


def print_vector(v, title="Вектор", digits=6, as_row=True):
    vv = [fmt(float(x), digits) for x in np.array(v).reshape(-1)]
    print(f"\n{title}:")
    if as_row:
        print("[ " + ", ".join(vv) + " ]")
    else:
        for x in vv:
            print(f"[ {x} ]")


def print_separator():
    print("\n" + "=" * 72)


# =========================================================
# 1. НОРМАЛИЗАЦИЯ
# =========================================================
def normalize_matrix(M):
    M = np.array(M, dtype=float)
    mmin = M.min()
    shift = -mmin if mmin < 0 else 0.0
    return M + shift, shift


# =========================================================
# 2. СОКРАЩЕНИЕ МАТРИЦЫ ПОГЛОЩЕНИЕМ ДОМИНИРУЕМЫХ СТРАТЕГИЙ
# =========================================================
def row_dominates(M, r1, r2, tol=1e-12):
    # Для игрока 1: строка r1 доминирует r2, если r1 >= r2 поэлементно
    return np.all(M[r1, :] >= M[r2, :] - tol) and np.any(M[r1, :] > M[r2, :] + tol)


def col_dominates(M, c1, c2, tol=1e-12):
    # Для игрока 2: столбец c1 доминирует c2, если c1 <= c2 поэлементно
    return np.all(M[:, c1] <= M[:, c2] + tol) and np.any(M[:, c1] < M[:, c2] - tol)


def eliminate_dominated_verbose(M):
    M = np.array(M, dtype=float)
    active_rows = list(range(M.shape[0]))
    active_cols = list(range(M.shape[1]))

    step = 1
    changed = True

    while changed:
        changed = False
        current = M[np.ix_(active_rows, active_cols)]
        print_matrix(
            current,
            title=f"Текущая матрица перед шагом доминирования {step}",
            row_labels=[f"R{r + 1}" for r in active_rows],
            col_labels=[f"C{c + 1}" for c in active_cols]
        )

        # Проверка строк
        removed_any = False
        i = 0
        while i < len(active_rows):
            r2 = active_rows[i]
            found = False
            for r1 in active_rows:
                if r1 == r2:
                    continue
                current_full = M[np.ix_(active_rows, active_cols)]
                rr1 = active_rows.index(r1)
                rr2 = active_rows.index(r2)
                if row_dominates(current_full, rr1, rr2):
                    print(f"\nШаг {step}. Строка R{r1 + 1} доминирует строку R{r2 + 1}. Удаляем R{r2 + 1}.")
                    active_rows.pop(i)
                    changed = True
                    removed_any = True
                    found = True
                    break
            if not found:
                i += 1

        if removed_any:
            step += 1
            continue

        # Проверка столбцов
        j = 0
        while j < len(active_cols):
            c2 = active_cols[j]
            found = False
            for c1 in active_cols:
                if c1 == c2:
                    continue
                current_full = M[np.ix_(active_rows, active_cols)]
                cc1 = active_cols.index(c1)
                cc2 = active_cols.index(c2)
                if col_dominates(current_full, cc1, cc2):
                    print(f"\nШаг {step}. Столбец C{c1 + 1} доминирует столбец C{c2 + 1}. Удаляем C{c2 + 1}.")
                    active_cols.pop(j)
                    changed = True
                    removed_any = True
                    found = True
                    break
            if not found:
                j += 1

        if removed_any:
            step += 1

    B = M[np.ix_(active_rows, active_cols)]
    print_matrix(
        B,
        title="Итоговая матрица после удаления доминируемых стратегий",
        row_labels=[f"R{r + 1}" for r in active_rows],
        col_labels=[f"C{c + 1}" for c in active_cols]
    )
    print(f"\nОставшиеся строки: {[r + 1 for r in active_rows]}")
    print(f"Оставшиеся столбцы: {[c + 1 for c in active_cols]}")
    return active_rows, active_cols, B


# =========================================================
# 3. СОКРАЩЕНИЕ МАТРИЦЫ УДАЛЕНИЕМ NBR-СТРАТЕГИЙ
# =========================================================
def eliminate_nbr_verbose(M):
    M = np.array(M, dtype=float)
    active_rows = list(range(M.shape[0]))
    active_cols = list(range(M.shape[1]))

    iteration = 1
    changed = True

    while changed:
        changed = False
        current = M[np.ix_(active_rows, active_cols)]
        print_matrix(
            current,
            title=f"Текущая матрица перед итерацией NBR {iteration}",
            row_labels=[f"R{r + 1}" for r in active_rows],
            col_labels=[f"C{c + 1}" for c in active_cols]
        )

        # Лучшие ответы игрока 1 на каждый столбец: максимум по столбцу
        best_rows = set()
        print("\nЛучшие ответы игрока 1 (по строкам) на каждый столбец:")
        for cj, c in enumerate(active_cols):
            col_vals = current[:, cj]
            mx = np.max(col_vals)
            rows_here = [active_rows[i] for i, val in enumerate(col_vals) if abs(val - mx) < 1e-12]
            print(f"  Для столбца C{c + 1}: максимум = {fmt(mx)}, лучшие строки = {[f'R{x + 1}' for x in rows_here]}")
            best_rows.update(rows_here)

        new_rows = [r for r in active_rows if r in best_rows]
        removed_rows = [r for r in active_rows if r not in best_rows]
        if removed_rows:
            print(f"\nУдаляем NBR-строки: {[f'R{r + 1}' for r in removed_rows]}")
            active_rows = new_rows
            changed = True

        current = M[np.ix_(active_rows, active_cols)]

        # Лучшие ответы игрока 2 на каждую строку: минимум по строке
        best_cols = set()
        print("\nЛучшие ответы игрока 2 (по столбцам) на каждую строку:")
        for ri, r in enumerate(active_rows):
            row_vals = current[ri, :]
            mn = np.min(row_vals)
            cols_here = [active_cols[j] for j, val in enumerate(row_vals) if abs(val - mn) < 1e-12]
            print(f"  Для строки R{r + 1}: минимум = {fmt(mn)}, лучшие столбцы = {[f'C{x + 1}' for x in cols_here]}")
            best_cols.update(cols_here)

        new_cols = [c for c in active_cols if c in best_cols]
        removed_cols = [c for c in active_cols if c not in best_cols]
        if removed_cols:
            print(f"\nУдаляем NBR-столбцы: {[f'C{c + 1}' for c in removed_cols]}")
            active_cols = new_cols
            changed = True

        iteration += 1

    result = M[np.ix_(active_rows, active_cols)]
    print_matrix(
        result,
        title="Итоговая матрица после удаления NBR-стратегий",
        row_labels=[f"R{r + 1}" for r in active_rows],
        col_labels=[f"C{c + 1}" for c in active_cols]
    )
    print(f"\nОставшиеся строки: {[r + 1 for r in active_rows]}")
    print(f"Оставшиеся столбцы: {[c + 1 for c in active_cols]}")
    return active_rows, active_cols, result


def reduce_dom_then_nbr_verbose(M):
    print_separator()
    print("КОМБИНИРОВАННОЕ СОКРАЩЕНИЕ: доминирование + NBR")
    rows1, cols1, M1 = eliminate_dominated_verbose(M)
    rows2_local, cols2_local, M2 = eliminate_nbr_verbose(M1)

    final_rows = [rows1[i] for i in rows2_local]
    final_cols = [cols1[j] for j in cols2_local]

    print_matrix(
        M2,
        title="Итог после комбинированного сокращения",
        row_labels=[f"R{r + 1}" for r in final_rows],
        col_labels=[f"C{c + 1}" for c in final_cols]
    )
    print(f"\nОкончательно оставшиеся строки: {[r + 1 for r in final_rows]}")
    print(f"Окончательно оставшиеся столбцы: {[c + 1 for c in final_cols]}")
    return final_rows, final_cols, M2


# =========================================================
# 4. ГРАФОАНАЛИТИЧЕСКИЙ МЕТОД ДЛЯ 2x2
# =========================================================
def solve_2x2_graphoanalytic_verbose(B):
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]

    print_separator()
    print("ГРАФОАНАЛИТИЧЕСКИЙ МЕТОД")

    print_matrix(B, title="Матрица B", row_labels=["R1", "R2"], col_labels=["C1", "C2"])

    # Игрок 2: ищем q
    print("\n1) Строим функции выигрыша по строкам:")
    print(f"u1(q) = {fmt(a)}q + {fmt(b)}(1-q)")
    print(f"u2(q) = {fmt(c)}q + {fmt(d)}(1-q)")

    q = (d - b) / (a - b - c + d)
    v = a * q + b * (1 - q)

    print(f"\nИз равенства u1(q)=u2(q): q* = {fmt(q)}")
    print(f"Цена игры: v' = {fmt(v)}")

    # Игрок 1: ищем p
    print("\n2) Строим функции выигрыша по столбцам:")
    print(f"v1(p) = {fmt(a)}p + {fmt(c)}(1-p)")
    print(f"v2(p) = {fmt(b)}p + {fmt(d)}(1-p)")

    p = (d - c) / (a - c - b + d)

    print(f"\nИз равенства v1(p)=v2(p): p* = {fmt(p)}")

    p_vec = np.array([p, 1 - p])
    q_vec = np.array([q, 1 - q])

    print(f"\nСтратегия игрока 1: ({fmt(p_vec[0])}, {fmt(p_vec[1])})")
    print(f"Стратегия игрока 2: ({fmt(q_vec[0])}, {fmt(q_vec[1])})")
    print(f"Цена игры: {fmt(v)}")

    return p_vec, q_vec, v


# =========================================================
# 5. АНАЛИТИЧЕСКИЙ (МАТРИЧНЫЙ) МЕТОД
# =========================================================
def inverse_2x2_verbose(C):
    a, b = C[0, 0], C[0, 1]
    c, d = C[1, 0], C[1, 1]

    det = a * d - b * c

    print("\nМатрица C:")
    print_matrix(C, title="C", row_labels=["R1", "R2"], col_labels=["C1", "C2"])
    print(f"\nОпределитель det(C) = {fmt(a)}*{fmt(d)} - {fmt(b)}*{fmt(c)} = {fmt(det)}")

    inv = (1 / det) * np.array([[d, -b], [-c, a]], dtype=float)

    print("\nОбратная матрица:")
    print("C^(-1) = (1 / det(C)) * [[d, -b], [-c, a]]")
    print_matrix(inv, title="C^(-1)", row_labels=["R1", "R2"], col_labels=["C1", "C2"])

    return inv


def solve_matrix_method_verbose(C):
    print_separator()
    print("АНАЛИТИЧЕСКИЙ (МАТРИЧНЫЙ) МЕТОД")

    invC = inverse_2x2_verbose(C)

    u_row = np.array([[1.0, 1.0]])
    u_col = np.array([[1.0], [1.0]])

    print_vector(u_row, title="Вектор u", as_row=True)

    uC_inv = u_row @ invC
    C_inv_uT = invC @ u_col
    scalar = (uC_inv @ u_col).item()

    print_vector(uC_inv, title="u * C^(-1)", as_row=True)
    print_vector(C_inv_uT, title="C^(-1) * u^T", as_row=False)
    print(f"\nu * C^(-1) * u^T = {fmt(scalar)}")

    p = (uC_inv / scalar).reshape(-1)
    q = (C_inv_uT / scalar).reshape(-1)
    v = 1 / scalar

    print("\nСмешанная стратегия игрока 1:")
    print(f"p* = (uC^(-1)) / (uC^(-1)u^T) = ({fmt(p[0])}, {fmt(p[1])})")

    print("\nСмешанная стратегия игрока 2:")
    print(f"q* = (C^(-1)u^T) / (uC^(-1)u^T) = ({fmt(q[0])}, {fmt(q[1])})")

    print(f"\nЦена нормализованной игры:")
    print(f"v' = 1 / (uC^(-1)u^T) = {fmt(v)}")

    return p, q, v


# =========================================================
# 6. ГРАФИЧЕСКОЕ РЕШЕНИЕ КАК ЗАДАЧИ ЛП
# =========================================================
def solve_lp_graphical_verbose(B):
    print_separator()
    print("ГРАФИЧЕСКОЕ РЕШЕНИЕ КАК ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")

    Bt = B.T
    print_matrix(B, title="Матрица B", row_labels=["R1", "R2"], col_labels=["C1", "C2"])
    print_matrix(Bt, title="Матрица B^T", row_labels=["C1", "C2"], col_labels=["R1", "R2"])

    print("\nРешаем двойственную задачу:")
    print("max (u1 + u2)")
    print("при ограничениях:")
    print(f"{fmt(Bt[0, 0])}u1 + {fmt(Bt[0, 1])}u2 <= 1")
    print(f"{fmt(Bt[1, 0])}u1 + {fmt(Bt[1, 1])}u2 <= 1")
    print("u1 >= 0, u2 >= 0")

    Aeq = np.array([
        [Bt[0, 0], Bt[0, 1]],
        [Bt[1, 0], Bt[1, 1]]
    ], dtype=float)
    beq = np.array([1.0, 1.0], dtype=float)

    u_opt = np.linalg.solve(Aeq, beq)
    u1, u2 = u_opt
    w = u1 + u2
    p = u_opt / w
    v = 1 / w

    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]
    q1 = (d - b) / (a - b - c + d)
    q = np.array([q1, 1 - q1])

    print("\nТочка оптимума dual:")
    print(f"u* = ({fmt(u1)}, {fmt(u2)})")
    print(f"w = {fmt(w)}")

    print("\nСтратегия игрока 1:")
    print(f"p* = ({fmt(p[0])}, {fmt(p[1])})")

    print("\nСтратегия игрока 2:")
    print(f"q* = ({fmt(q[0])}, {fmt(q[1])})")

    print(f"\nЦена игры: v' = {fmt(v)}")

    return p, q, v, u_opt, w


# =========================================================
# 7. СИМПЛЕКС-МЕТОД
# =========================================================
def simplex_max_leq_verbose(A, b, c, tol=1e-12, max_iter=1000):
    """
    Решает:
        max c^T x
        s.t. A x <= b, x >= 0
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    c = np.array(c, dtype=float).reshape(-1)

    m, n = A.shape

    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[m, :n] = -c

    basis = list(range(n, n + m))

    def print_tableau(T, basis, it):
        print(f"\nСимплекс-таблица, итерация {it}:")
        for row in T:
            print("  ".join(f"{fmt(x, 6):>10}" for x in row))
        print("Базис:", basis)

    def pivot(row, col):
        piv = T[row, col]
        T[row, :] /= piv
        for r in range(T.shape[0]):
            if r != row:
                T[r, :] -= T[r, col] * T[row, :]
        basis[row] = col

    print("\nНачальная симплекс-таблица:")
    print_tableau(T, basis, 0)

    for it in range(1, max_iter + 1):
        last = T[m, :-1]
        entering = int(np.argmin(last))
        if last[entering] >= -tol:
            print("\nОптимум достигнут: в последней строке нет отрицательных коэффициентов.")
            break

        ratios = []
        for i in range(m):
            if T[i, entering] > tol:
                ratios.append((T[i, -1] / T[i, entering], i))

        if not ratios:
            raise RuntimeError("Задача неограничена.")

        _, leaving = min(ratios, key=lambda x: x[0])

        print(f"\nИтерация {it}:")
        print(f"Входящая переменная: x{entering + 1}")
        print(f"Выходящая переменная: x{basis[leaving] + 1}")
        print(f"Разрешающий элемент: {fmt(T[leaving, entering])}")

        pivot(leaving, entering)
        print_tableau(T, basis, it)

    x = np.zeros(n + m)
    for i in range(m):
        x[basis[i]] = T[i, -1]

    x_opt = x[:n]
    z_opt = T[m, -1]

    print_vector(x_opt, title="Оптимальный план x*", as_row=True)
    print(f"\nОптимум целевой функции: {fmt(z_opt)}")

    return x_opt, z_opt


def solve_lp_simplex_verbose(B):
    print_separator()
    print("СИМПЛЕКС-МЕТОД ДЛЯ ДВОЙСТВЕННОЙ ЗАДАЧИ ЛП")

    Bt = B.T
    A_dual = Bt
    b_dual = np.ones(B.shape[1])
    c_dual = np.ones(B.shape[0])

    print("\nРешаем задачу:")
    print("max u1 + u2")
    print("при ограничениях:")
    for i in range(A_dual.shape[0]):
        print(f"{fmt(A_dual[i, 0])}u1 + {fmt(A_dual[i, 1])}u2 <= {fmt(b_dual[i])}")
    print("u1 >= 0, u2 >= 0")

    u_opt, w_opt = simplex_max_leq_verbose(A_dual, b_dual, c_dual)

    p = u_opt / w_opt
    v = 1 / w_opt

    # Восстанавливаем q через безразличие строк
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]
    q1 = (d - b) / (a - b - c + d)
    q = np.array([q1, 1 - q1])

    print("\nНормировка решения dual:")
    print(f"p* = ({fmt(p[0])}, {fmt(p[1])})")

    print("\nСтратегия игрока 2:")
    print(f"q* = ({fmt(q[0])}, {fmt(q[1])})")

    print(f"\nЦена игры: v' = {fmt(v)}")

    return p, q, v, u_opt, w_opt


# =========================================================
# Графики
# =========================================================

def plot_graphoanalytic_player2(B):
    """
    Графоаналитический метод для игрока 2:
    строятся u1(q), u2(q) и нижняя огибающая min(u1, u2).
    """
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]

    q = np.linspace(0, 1, 500)
    u1 = a * q + b * (1 - q)
    u2 = c * q + d * (1 - q)
    lower = np.minimum(u1, u2)

    q_star = (d - b) / (a - b - c + d)
    v_star = a * q_star + b * (1 - q_star)

    plt.figure(figsize=(8, 5))
    plt.plot(q, u1, label="u1(q) — строка 1")
    plt.plot(q, u2, label="u2(q) — строка 2")
    plt.plot(q, lower, label="min(u1(q), u2(q))")
    plt.scatter([q_star], [v_star], zorder=5,
                label=f"Оптимум: q*={q_star:.4f}, v'={v_star:.4f}")

    plt.xlabel("q — вероятность выбора 1-го столбца игроком 2")
    plt.ylabel("Ожидаемый выигрыш игрока 1")
    plt.title("Графоаналитический метод: стратегия игрока 2")
    plt.grid(True)
    plt.legend()
    plt.savefig("data/graph_anal_2.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_graphoanalytic_player1(B):
    """
    Графоаналитический метод для игрока 1:
    строятся v1(p), v2(p) и верхняя огибающая max(v1, v2).
    """
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]

    p = np.linspace(0, 1, 500)
    v1 = a * p + c * (1 - p)
    v2 = b * p + d * (1 - p)
    upper = np.maximum(v1, v2)

    p_star = (d - c) / (a - c - b + d)
    v_star = a * p_star + c * (1 - p_star)

    plt.figure(figsize=(8, 5))
    plt.plot(p, v1, label="v1(p) — столбец 1")
    plt.plot(p, v2, label="v2(p) — столбец 2")
    plt.plot(p, upper, label="max(v1(p), v2(p))")
    plt.scatter([p_star], [v_star], zorder=5,
                label=f"Оптимум: p*={p_star:.4f}, v'={v_star:.4f}")

    plt.xlabel("p — вероятность выбора 1-й строки игроком 1")
    plt.ylabel("Ожидаемый выигрыш игрока 1")
    plt.title("Графоаналитический метод: стратегия игрока 1")
    plt.grid(True)
    plt.legend()
    plt.savefig("data/graph_anal_1.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_lp_graphical_player1(B):
    """
    Графическое решение ЛП для игрока 1.
    Решается dual-задача:
        max(u1 + u2)
        при B^T u <= 1, u >= 0
    По найденному u восстанавливается p.
    """
    Bt = B.T
    a1, a2 = Bt[0, 0], Bt[0, 1]
    b1, b2 = Bt[1, 0], Bt[1, 1]

    u1_vals = np.linspace(0, 0.55, 600)

    raw_line1 = (1 - a1 * u1_vals) / a2
    raw_line2 = (1 - b1 * u1_vals) / b2

    line1_plot = np.where(raw_line1 >= 0, raw_line1, np.nan)
    line2_plot = np.where(raw_line2 >= 0, raw_line2, np.nan)

    feasible_mask = (raw_line1 >= 0) & (raw_line2 >= 0)
    feasible_top = np.where(feasible_mask, np.minimum(raw_line1, raw_line2), 0)

    Aeq = np.array([[a1, a2], [b1, b2]], dtype=float)
    beq = np.array([1.0, 1.0], dtype=float)
    u_opt = np.linalg.solve(Aeq, beq)
    u1_star, u2_star = u_opt

    w = u1_star + u2_star
    p_star = u1_star / w
    v_star = 1 / w

    plt.figure(figsize=(8, 5))
    plt.plot(u1_vals, line1_plot, label=f"{fmt(a1)}y1 + {fmt(a2)}y2 = 1")
    plt.plot(u1_vals, line2_plot, label=f"{fmt(b1)}y1 + {fmt(b2)}y2 = 1")

    plt.fill_between(
        u1_vals, 0, feasible_top,
        where=feasible_mask,
        alpha=0.2,
        label="Допустимая область"
    )

    plt.scatter([u1_star], [u2_star], zorder=5,
                label=f"Оптимум: ({u1_star:.4f}, {u2_star:.4f})")

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(f"Графический метод ЛП: стратегия игрока 1, p1*={p_star:.4f}, v'={v_star:.4f}")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig("data/graph_lp_1.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_lp_graphical_player2(B):
    """
    Графическое решение ЛП для игрока 2.
    Решается прямая задача:
        min(z1 + z2)
        при B z >= 1, z >= 0
    По найденному z восстанавливается q.
    """
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]

    z1_vals = np.linspace(0, 0.8, 600)

    # Границы:
    # a*z1 + b*z2 = 1  => z2 = (1 - a*z1)/b
    # c*z1 + d*z2 = 1  => z2 = (1 - c*z1)/d
    raw_line1 = (1 - a * z1_vals) / b
    raw_line2 = (1 - c * z1_vals) / d

    line1_plot = np.where(raw_line1 >= 0, raw_line1, np.nan)
    line2_plot = np.where(raw_line2 >= 0, raw_line2, np.nan)

    # Для ограничений вида >= допустимая область лежит ВЫШЕ обеих прямых
    feasible_bottom = np.maximum(raw_line1, raw_line2)
    feasible_bottom = np.maximum(feasible_bottom, 0)

    # Ограничим заливку сверху для картинки
    y_max = min(1.2, np.nanmax(feasible_bottom[np.isfinite(feasible_bottom)]) + 0.2)

    # Точка пересечения
    Aeq = np.array([[a, b], [c, d]], dtype=float)
    beq = np.array([1.0, 1.0], dtype=float)
    z_opt = np.linalg.solve(Aeq, beq)
    z1_star, z2_star = z_opt

    s = z1_star + z2_star
    q_star = z1_star / s
    v_star = 1 / s

    plt.figure(figsize=(8, 5))
    plt.plot(z1_vals, line1_plot, label=f"{fmt(a)}z1 + {fmt(b)}z2 = 1")
    plt.plot(z1_vals, line2_plot, label=f"{fmt(c)}z1 + {fmt(d)}z2 = 1")

    plt.fill_between(
        z1_vals, feasible_bottom, y_max,
        where=np.isfinite(feasible_bottom),
        alpha=0.2,
        label="Допустимая область"
    )

    plt.scatter([z1_star], [z2_star], zorder=5,
                label=f"Оптимум: ({z1_star:.4f}, {z2_star:.4f})")

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(f"Графический метод ЛП: стратегия игрока 2, q1*={q_star:.4f}, v'={v_star:.4f}")
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=y_max)
    plt.grid(True)
    plt.legend()
    plt.savefig("data/graph_lp_2.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print_separator()
    print("РЕШЕНИЕ ПРЯМОУГОЛЬНОЙ МАТРИЧНОЙ ИГРЫ С НУЛЕВОЙ СУММОЙ")

    print_matrix(A, title="Исходная матрица A", row_labels=["R1", "R2", "R3", "R4", "R5"],
                 col_labels=["C1", "C2", "C3", "C4", "C5"])

    # 1. Нормализация
    print_separator()
    print("1. НОРМАЛИЗАЦИЯ МАТРИЦЫ")
    A_pos, shift = normalize_matrix(A)
    print(f"\nМинимальный элемент матрицы A: {fmt(A.min())}")
    print(f"Добавляем ко всем элементам константу: {fmt(shift)}")
    print_matrix(A_pos, title="Нормализованная матрица A+", row_labels=["R1", "R2", "R3", "R4", "R5"],
                 col_labels=["C1", "C2", "C3", "C4", "C5"])

    # 2a. Доминирование
    print_separator()
    print("2а. СОКРАЩЕНИЕ МАТРИЦЫ ПОГЛОЩЕНИЕМ ДОМИНИРУЕМЫХ СТРАТЕГИЙ")
    rows_dom, cols_dom, B_dom = eliminate_dominated_verbose(A_pos)

    # 2b. NBR
    print_separator()
    print("2б. СОКРАЩЕНИЕ МАТРИЦЫ УДАЛЕНИЕМ NBR-СТРАТЕГИЙ")
    rows_nbr, cols_nbr, M_nbr = eliminate_nbr_verbose(A_pos)

    # Комбинированное сокращение
    rows_final, cols_final, B = reduce_dom_then_nbr_verbose(A_pos)

    # 3a. Графоаналитика
    p_graph, q_graph, v_graph = solve_2x2_graphoanalytic_verbose(B)
    plot_graphoanalytic_player2(B)
    plot_graphoanalytic_player1(B)

    # 3b. Аналитический (матричный)
    p_mat, q_mat, v_mat = solve_matrix_method_verbose(B)

    # 3c. Графически как задача ЛП
    p_lp_graph, q_lp_graph, v_lp_graph, u_opt_graph, w_graph = solve_lp_graphical_verbose(B)
    plot_lp_graphical_player1(B)
    plot_lp_graphical_player2(B)

    # 3d. Симплекс-метод
    p_simplex, q_simplex, v_simplex, u_opt_simplex, w_simplex = solve_lp_simplex_verbose(B)

    print_separator()
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПО МЕТОДАМ")

    print(f"\nГрафоаналитический метод:")
    print(f"p* = ({fmt(p_graph[0])}, {fmt(p_graph[1])})")
    print(f"q* = ({fmt(q_graph[0])}, {fmt(q_graph[1])})")
    print(f"v' = {fmt(v_graph)}")

    print(f"\nМатричный метод:")
    print(f"p* = ({fmt(p_mat[0])}, {fmt(p_mat[1])})")
    print(f"q* = ({fmt(q_mat[0])}, {fmt(q_mat[1])})")
    print(f"v' = {fmt(v_mat)}")

    print(f"\nГрафический метод ЛП:")
    print(f"p* = ({fmt(p_lp_graph[0])}, {fmt(p_lp_graph[1])})")
    print(f"q* = ({fmt(q_lp_graph[0])}, {fmt(q_lp_graph[1])})")
    print(f"v' = {fmt(v_lp_graph)}")

    print(f"\nСимплекс-метод:")
    print(f"p* = ({fmt(p_simplex[0])}, {fmt(p_simplex[1])})")
    print(f"q* = ({fmt(q_simplex[0])}, {fmt(q_simplex[1])})")
    print(f"v' = {fmt(v_simplex)}")

    # 4. Цена исходной игры
    print_separator()
    print("4. ЦЕНА ИСХОДНОЙ ИГРЫ")
    v_original = v_mat - shift

    print(f"\nЦена нормализованной игры: v' = {fmt(v_mat)}")
    print(f"Сдвиг при нормализации: {fmt(shift)}")
    print(f"Цена исходной игры: v = v' - {fmt(shift)} = {fmt(v_original)}")

    # Итог
    print_separator()
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")

    print("\nСокращённая матрица 2x2:")
    print_matrix(B,
                 title="Итоговая матрица B",
                 row_labels=[f"R{r + 1}" for r in rows_final],
                 col_labels=[f"C{c + 1}" for c in cols_final])

    print(f"\nОптимальная смешанная стратегия игрока 1:")
    print(f"p* = ({fmt(p_mat[0])}, {fmt(p_mat[1])})")

    print(f"\nОптимальная смешанная стратегия игрока 2:")
    print(f"q* = ({fmt(q_mat[0])}, {fmt(q_mat[1])})")

    print(f"\nЦена нормализованной игры: v' = {fmt(v_mat)}")
    print(f"Цена исходной игры: v = {fmt(v_original)}")
