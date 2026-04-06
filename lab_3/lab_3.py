from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

Number = float
Matrix = List[List[Number]]
Position = Tuple[int, int]  # индексы с нуля

RANDOM_GAME_SEED = 567
RANDOM_GAME_SIZE = 10
RANDOM_GAME_LOW = -50
RANDOM_GAME_HIGH = 50


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE_BG = "\033[44;97m"  # Nash
    GREEN_BG = "\033[42;30m"  # Pareto
    MAGENTA_BG = "\033[45;97m"  # Nash ∩ Pareto
    CYAN = "\033[36m"
    YELLOW = "\033[33m"


def colorize(text: str, style: str) -> str:
    if not style:
        return text
    return f"{style}{text}{Ansi.RESET}"


CLASSIC_GAMES = {
    "Дилемма заключённого": {
        "rows": ["Говорить", "Молчать"],
        "cols": ["Говорить", "Молчать"],
        "pairs": [
            [(-5, -5), (0, -10)],
            [(-10, 0), (-1, -1)],
        ],
        "comment": "Устойчивой по Нэшу считается только ситуация (Г, Г).",
    },
    "Семейный спор": {
        "rows": ["Футбол", "Театр"],
        "cols": ["Футбол", "Театр"],
        "pairs": [
            [(4, 1), (0, 0)],
            [(0, 0), (1, 4)],
        ],
        "comment": "Классическая игра с двумя чистыми и одной смешанной равновесной ситуацией.",
    },
    "Перекрёсток (Феррари / КамАЗ)": {
        "rows": ["Стоп", "Ехать"],
        "cols": ["Стоп", "Ехать"],
        "pairs": [
            [(1.0, 1.0), (0.7, 2.0)],
            [(2.0, 0.7), (-2, -12)],
        ],
        "comment": "Несимметричный учебный вариант перекрёстка для проверки алгоритма.",
    },
}


def fmt(x: Number) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def pair_str(a: Number, b: Number) -> str:
    return f"({fmt(a)}, {fmt(b)})"


def pairs_to_matrices(pairs: Sequence[Sequence[Tuple[Number, Number]]]) -> Tuple[Matrix, Matrix]:
    a = [[cell[0] for cell in row] for row in pairs]
    b = [[cell[1] for cell in row] for row in pairs]
    return a, b


def size_of(matrix: Matrix) -> Tuple[int, int]:
    return len(matrix), len(matrix[0])


def print_separator(char: str = "=", width: int = 96) -> None:
    print(char * width)


def positions_to_text(
        positions: Sequence[Position],
        pairs: Sequence[Sequence[Tuple[Number, Number]]],
        row_labels: Optional[Sequence[str]] = None,
        col_labels: Optional[Sequence[str]] = None,
) -> str:
    if not positions:
        return "нет"
    parts = []
    for i, j in positions:
        r_name = row_labels[i] if row_labels else f"A{i + 1}"
        c_name = col_labels[j] if col_labels else f"B{j + 1}"
        parts.append(f"({r_name}, {c_name}) = {pair_str(*pairs[i][j])}")
    return "; ".join(parts)


def intersection(xs: Sequence[Position], ys: Sequence[Position]) -> List[Position]:
    return sorted(set(xs) & set(ys))


def find_nash_equilibria(a: Matrix, b: Matrix, strict: bool = False) -> List[Position]:
    m, n = size_of(a)
    result = []
    for i in range(m):
        for j in range(n):
            if strict:
                row_ok = all(a[i][j] > a[k][j] for k in range(m) if k != i)
                col_ok = all(b[i][j] > b[i][l] for l in range(n) if l != j)
            else:
                row_ok = all(a[i][j] >= a[k][j] for k in range(m) if k != i)
                col_ok = all(b[i][j] >= b[i][l] for l in range(n) if l != j)
            if row_ok and col_ok:
                result.append((i, j))
    return result


def find_pareto_strong(a: Matrix, b: Matrix) -> List[Position]:
    m, n = size_of(a)
    result = []
    for i in range(m):
        for j in range(n):
            dominated = False
            for p in range(m):
                for q in range(n):
                    better_or_equal_for_both = a[p][q] >= a[i][j] and b[p][q] >= b[i][j]
                    strictly_better_for_someone = a[p][q] > a[i][j] or b[p][q] > b[i][j]
                    if better_or_equal_for_both and strictly_better_for_someone:
                        dominated = True
                        break
                if dominated:
                    break
            if not dominated:
                result.append((i, j))
    return result


def find_pareto_weak(a: Matrix, b: Matrix) -> List[Position]:
    m, n = size_of(a)
    result = []
    for i in range(m):
        for j in range(n):
            strongly_dominated = False
            for p in range(m):
                for q in range(n):
                    if a[p][q] > a[i][j] and b[p][q] > b[i][j]:
                        strongly_dominated = True
                        break
                if strongly_dominated:
                    break
            if not strongly_dominated:
                result.append((i, j))
    return result


def cell_marker(pos: Position, nash: set[Position], pareto: set[Position]) -> str:
    in_nash = pos in nash
    in_pareto = pos in pareto
    if in_nash and in_pareto:
        return "NP"
    if in_nash:
        return "N"
    if in_pareto:
        return "P"
    return " "


def cell_style(pos: Position, nash: set[Position], pareto: set[Position]) -> str:
    in_nash = pos in nash
    in_pareto = pos in pareto
    if in_nash and in_pareto:
        return Ansi.MAGENTA_BG
    if in_nash:
        return Ansi.BLUE_BG
    if in_pareto:
        return Ansi.GREEN_BG
    return ""


def print_legend() -> None:
    print("Легенда:")
    print("  " + colorize("  N  ", Ansi.BLUE_BG) + " — строгий Нэш")
    print("  " + colorize("  P  ", Ansi.GREEN_BG) + " — сильный Парето")
    print("  " + colorize(" NP  ", Ansi.MAGENTA_BG) + " — пересечение")
    print()


def print_colored_matrix(
        pairs: Sequence[Sequence[Tuple[Number, Number]]],
        nash: Sequence[Position],
        pareto: Sequence[Position],
        row_labels: Optional[Sequence[str]] = None,
        col_labels: Optional[Sequence[str]] = None,
) -> None:
    m = len(pairs)
    n = len(pairs[0])
    if row_labels is None:
        row_labels = [f"A{i + 1}" for i in range(m)]
    if col_labels is None:
        col_labels = [f"B{j + 1}" for j in range(n)]

    nash_set = set(nash)
    pareto_set = set(pareto)

    widths = []
    for j in range(n):
        max_len = len(str(col_labels[j]))
        for i in range(m):
            marker = cell_marker((i, j), nash_set, pareto_set)
            cell_text = f"{pair_str(*pairs[i][j])} {marker}".rstrip()
            max_len = max(max_len, len(cell_text))
        widths.append(max_len + 2)

    first_col_w = max(len(max(row_labels, key=len)), 8)
    header = " " * (first_col_w + 3)
    for j in range(n):
        header += f"{str(col_labels[j]):^{widths[j]}}"
    print(header)

    for i in range(m):
        line = f"{row_labels[i]:>{first_col_w}} |"
        for j in range(n):
            marker = cell_marker((i, j), nash_set, pareto_set)
            cell_text = f"{pair_str(*pairs[i][j])} {marker}".rstrip()
            padded = f"{cell_text:^{widths[j]}}"
            style = cell_style((i, j), nash_set, pareto_set)
            line += colorize(padded, style) if style else padded
        print(line)
    print()


def has_strictly_dominant_row_strategy(a: Matrix) -> Optional[int]:
    m, n = size_of(a)
    for i in range(m):
        dominates_all = True
        for k in range(m):
            if k == i:
                continue
            for j in range(n):
                if not (a[i][j] > a[k][j]):
                    dominates_all = False
                    break
            if not dominates_all:
                break
        if dominates_all:
            return i
    return None


def has_strictly_dominant_col_strategy(b: Matrix) -> Optional[int]:
    m, n = size_of(b)
    for j in range(n):
        dominates_all = True
        for l in range(n):
            if l == j:
                continue
            for i in range(m):
                if not (b[i][j] > b[i][l]):
                    dominates_all = False
                    break
            if not dominates_all:
                break
        if dominates_all:
            return j
    return None


def inverse_2x2(mtx: Matrix) -> Tuple[Optional[Matrix], Number]:
    a, b = mtx[0]
    c, d = mtx[1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None, det
    inv = [[d / det, -b / det], [-c / det, a / det]]
    return inv, det


def mixed_equilibrium_2x2(a: Matrix, b: Matrix) -> Dict[str, object]:
    a11, a12 = a[0]
    a21, a22 = a[1]
    b11, b12 = b[0]
    b21, b22 = b[1]

    result: Dict[str, object] = {"exists": False, "reason": None}

    denom_q = a11 - a12 - a21 + a22
    denom_p = b11 - b12 - b21 + b22
    result["denom_q"] = denom_q
    result["denom_p"] = denom_p

    if abs(denom_q) < 1e-12 or abs(denom_p) < 1e-12:
        result["reason"] = (
            "Одна из формул безразличия вырождается: знаменатель равен нулю, "
            "поэтому полностью смешанную ситуацию построить нельзя."
        )
        return result

    q = (a22 - a12) / denom_q
    p = (b22 - b21) / denom_p
    x = [p, 1 - p]
    y = [q, 1 - q]

    result["p"] = p
    result["q"] = q
    result["x"] = x
    result["y"] = y

    if not (0 < p < 1 and 0 < q < 1):
        result["reason"] = (
            "Решение уравнений безразличия лежит на границе или вне [0, 1], "
            "поэтому полностью смешанной равновесной ситуации нет."
        )
        return result

    v1_from_row1 = q * a11 + (1 - q) * a12
    v1_from_row2 = q * a21 + (1 - q) * a22
    v2_from_col1 = p * b11 + (1 - p) * b21
    v2_from_col2 = p * b12 + (1 - p) * b22

    result["v1_from_row1"] = v1_from_row1
    result["v1_from_row2"] = v1_from_row2
    result["v2_from_col1"] = v2_from_col1
    result["v2_from_col2"] = v2_from_col2
    result["v1"] = (v1_from_row1 + v1_from_row2) / 2
    result["v2"] = (v2_from_col1 + v2_from_col2) / 2

    a_inv, det_a = inverse_2x2(a)
    b_inv, det_b = inverse_2x2(b)
    result["det_a"] = det_a
    result["det_b"] = det_b
    result["a_inv"] = a_inv
    result["b_inv"] = b_inv

    if a_inv is not None and b_inv is not None:
        sum_a_inv = sum(sum(row) for row in a_inv)
        sum_b_inv = sum(sum(row) for row in b_inv)
        result["sum_a_inv"] = sum_a_inv
        result["sum_b_inv"] = sum_b_inv

        if abs(sum_a_inv) > 1e-12 and abs(sum_b_inv) > 1e-12:
            v1_formula = 1 / sum_a_inv
            v2_formula = 1 / sum_b_inv
            y_formula = [
                v1_formula * (a_inv[0][0] + a_inv[0][1]),
                v1_formula * (a_inv[1][0] + a_inv[1][1]),
            ]
            x_formula = [
                v2_formula * (b_inv[0][0] + b_inv[1][0]),
                v2_formula * (b_inv[0][1] + b_inv[1][1]),
            ]
            result["v1_formula"] = v1_formula
            result["v2_formula"] = v2_formula
            result["x_formula"] = x_formula
            result["y_formula"] = y_formula

    result["exists"] = True
    return result


def print_mixed_analysis(
        a: Matrix,
        b: Matrix,
        pairs: Sequence[Sequence[Tuple[Number, Number]]],
        row_labels: Optional[Sequence[str]] = None,
        col_labels: Optional[Sequence[str]] = None,
) -> None:
    if size_of(a) != (2, 2):
        print("Смешанные стратегии: анализ выполняется только для игр 2x2.\n")
        return

    print("Смешанные стратегии")
    print("-" * 96)

    dom_row = has_strictly_dominant_row_strategy(a)
    dom_col = has_strictly_dominant_col_strategy(b)

    if dom_row is None:
        print("У игрока 1 строго доминирующей стратегии нет.")
    else:
        name = row_labels[dom_row] if row_labels else f"A{dom_row + 1}"
        print(f"У игрока 1 есть строго доминирующая стратегия: {name}.")

    if dom_col is None:
        print("У игрока 2 строго доминирующей стратегии нет.")
    else:
        name = col_labels[dom_col] if col_labels else f"B{dom_col + 1}"
        print(f"У игрока 2 есть строго доминирующая стратегия: {name}.")

    pure_nash = find_nash_equilibria(a, b, strict=False)
    print("Чистые равновесия по Нэшу:")
    print(positions_to_text(pure_nash, pairs, row_labels, col_labels))

    mix = mixed_equilibrium_2x2(a, b)
    if not mix["exists"]:
        print("Полностью смешанная равновесная ситуация: не существует.")
        print(f"Причина: {mix['reason']}")
        print()
        return

    print("Полностью смешанная равновесная ситуация существует.")
    print(f"Знаменатель для q: {fmt(mix['denom_q'])}")
    print(f"Знаменатель для p: {fmt(mix['denom_p'])}")
    print(
        f"x = [p, 1-p] = [{fmt(mix['x'][0])}, {fmt(mix['x'][1])}], "
        f"y = [q, 1-q] = [{fmt(mix['y'][0])}, {fmt(mix['y'][1])}]"
    )
    print(
        f"Проверка игрока 1: v1(row1) = {fmt(mix['v1_from_row1'])}, "
        f"v1(row2) = {fmt(mix['v1_from_row2'])}"
    )
    print(
        f"Проверка игрока 2: v2(col1) = {fmt(mix['v2_from_col1'])}, "
        f"v2(col2) = {fmt(mix['v2_from_col2'])}"
    )
    print(f"Равновесные выигрыши: v1 = {fmt(mix['v1'])}, v2 = {fmt(mix['v2'])}")

    if mix.get("a_inv") is not None and mix.get("b_inv") is not None:
        print("Проверка по формуле из методички:")
        print(f"det(A) = {fmt(mix['det_a'])}, det(B) = {fmt(mix['det_b'])}")
        print(
            f"A^(-1) = [[{fmt(mix['a_inv'][0][0])}, {fmt(mix['a_inv'][0][1])}], "
            f"[{fmt(mix['a_inv'][1][0])}, {fmt(mix['a_inv'][1][1])}]]"
        )
        print(
            f"B^(-1) = [[{fmt(mix['b_inv'][0][0])}, {fmt(mix['b_inv'][0][1])}], "
            f"[{fmt(mix['b_inv'][1][0])}, {fmt(mix['b_inv'][1][1])}]]"
        )
        if mix.get("v1_formula") is not None and mix.get("v2_formula") is not None:
            print(
                f"v1 = 1 / (u * A^(-1) * u) = {fmt(mix['v1_formula'])}, "
                f"v2 = 1 / (u * B^(-1) * u) = {fmt(mix['v2_formula'])}"
            )
            print(f"x = v2 * u * B^(-1) = [{fmt(mix['x_formula'][0])}, {fmt(mix['x_formula'][1])}]")
            print(f"y = v1 * A^(-1) * u = [{fmt(mix['y_formula'][0])}, {fmt(mix['y_formula'][1])}]")
    print()


def generate_random_bimatrix(
        rows: int,
        cols: int,
        low: int,
        high: int,
        seed: int,
) -> List[List[Tuple[Number, Number]]]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append((rng.randint(low, high), rng.randint(low, high)))
        pairs.append(row)
    return pairs


def analyze_game(
        name: str,
        pairs: Sequence[Sequence[Tuple[Number, Number]]],
        row_labels: Optional[Sequence[str]] = None,
        col_labels: Optional[Sequence[str]] = None,
        extra_comment: Optional[str] = None,
) -> None:
    a, b = pairs_to_matrices(pairs)

    nash_classic = find_nash_equilibria(a, b, strict=False)
    nash_strict = find_nash_equilibria(a, b, strict=True)
    pareto_strong = find_pareto_strong(a, b)
    pareto_weak = find_pareto_weak(a, b)
    inter = intersection(nash_strict, pareto_strong)

    print_separator("=")
    print(name)
    print_separator("=")
    if extra_comment:
        print(extra_comment)
        print()

    print_legend()
    print("Матрица игры (в клетке: (A_ij, B_ij) и метка результата)")
    print_colored_matrix(pairs, nash_strict, pareto_strong, row_labels, col_labels)

    print("Краткая сводка:")
    print(f"- Нэш (нестрогий, по лекции): {positions_to_text(nash_classic, pairs, row_labels, col_labels)}")
    print(f"- Нэш (строгий, по семинару): {positions_to_text(nash_strict, pairs, row_labels, col_labels)}")
    print(f"- Парето (сильный): {positions_to_text(pareto_strong, pairs, row_labels, col_labels)}")
    print(f"- Парето (слабый): {positions_to_text(pareto_weak, pairs, row_labels, col_labels)}")
    print(f"- Пересечение: {positions_to_text(inter, pairs, row_labels, col_labels)}")

    if inter:
        print(f"- Итоговый выбор: {positions_to_text(inter, pairs, row_labels, col_labels)}")
    else:
        print(f"- Итоговый выбор: {positions_to_text(nash_strict, pairs, row_labels, col_labels)}")
    print()

    print_mixed_analysis(a, b, pairs, row_labels, col_labels)


def main() -> None:
    print("ПРОВЕРКА АЛГОРИТМОВ НА ТРЁХ КЛАССИЧЕСКИХ ИГРАХ")
    for game_name, data in CLASSIC_GAMES.items():
        analyze_game(
            game_name,
            data["pairs"],
            row_labels=data["rows"],
            col_labels=data["cols"],
            extra_comment=data.get("comment"),
        )

    print("СЛУЧАЙНАЯ БИМАТРИЧНАЯ ИГРА 10x10")
    print_separator("-")
    random_pairs = generate_random_bimatrix(
        rows=RANDOM_GAME_SIZE,
        cols=RANDOM_GAME_SIZE,
        low=RANDOM_GAME_LOW,
        high=RANDOM_GAME_HIGH,
        seed=RANDOM_GAME_SEED,
    )
    analyze_game("Случайная игра 10x10", random_pairs)

    print("ВАРИАНТ ИЗ ТАБЛИЦЫ Л5.1")
    print_separator("-")
    analyze_game(
        f"Вариант 13",
        [[(4, 1), (6, 2)], [(11, 7), (0, 5)]],
        row_labels=["α1", "α2"],
        col_labels=["β1", "β2"],
        extra_comment="Для этой игры дополнительно ищем равновесные ситуации в смешанном расширении.",
    )


if __name__ == "__main__":
    main()
