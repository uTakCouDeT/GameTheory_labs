#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Лабораторная работа №5 (в вашем наборе файлов она переименована в lab_3.py).
Тема: равновесие по Нэшу, оптимальность по Парето и смешанные стратегии в биматричных играх.

Что делает скрипт:
1) проверяет алгоритмы на трёх классических играх;
2) генерирует случайную биматричную игру 10x10;
3) решает вариант из таблицы Л5.1 (по умолчанию вариант 13);
4) для игр 2x2 ищет равновесия в чистых и смешанных стратегиях;
5) выводит компактный цветной результат: матрицу игры и выделяет на ней:
   - N  : строгие равновесия по Нэшу (по замечанию с семинара),
   - P  : Парето-оптимальные ситуации,
   - NP : пересечение этих множеств.

Дополнительно:
- классическое определение Нэша (>=) тоже вычисляется и печатается в сводке;
- цвет можно отключить флагом --no-color;
- более подробный вывод можно включить флагом --verbose.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Sequence, Tuple

Number = float
Matrix = List[List[Number]]
Position = Tuple[int, int]  # индексы с нуля


# ============================================================================
# Настройки запуска
# ============================================================================
DEFAULT_VARIANT = 13
DEFAULT_RANDOM_SEED = 567
RANDOM_SIZE = 10
RANDOM_LOW = -50
RANDOM_HIGH = 50


# ============================================================================
# Цвета ANSI
# ============================================================================
class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[44;97m"     # Nash
    GREEN = "\033[42;30m"    # Pareto
    MAGENTA = "\033[45;97m"  # Intersection
    CYAN = "\033[36m"
    YELLOW = "\033[33m"


def colorize(text: str, style: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{style}{text}{Ansi.RESET}"


# ============================================================================
# Исходные данные
# ============================================================================
CLASSIC_GAMES = {
    "Дилемма заключённого": {
        "rows": ["М", "Г"],
        "cols": ["М", "Г"],
        # Матрица по вашим записям с семинара
        "pairs": [
            [(-0.5, -0.5), (-10, 0)],
            [(0, -10), (-5, -5)],
        ],
        "comment": "По замечанию с семинара устойчивой по Нэшу считается только (Г, Г).",
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
        # Несимметричный учебный вариант: Ferrari vs КамАЗ
        "pairs": [
            [(1.0, 1.0), (0.7, 2.0)],
            [(2.0, 0.9), (-12.0, -2.0)],
        ],
        "comment": "Несимметричный вариант перекрёстка для наглядной проверки алгоритма.",
    },
}


VARIANTS: Dict[int, List[List[Tuple[Number, Number]]]] = {
    1: [[(5, 0), (8, 4)], [(7, 6), (6, 3)]],
    2: [[(6, 7), (8, 4)], [(2, 1), (9, 3)]],
    3: [[(3, 1), (5, 0)], [(9, 6), (2, 3)]],
    4: [[(4, 7), (5, 2)], [(0, 2), (7, 3)]],
    5: [[(5, 8), (7, 4)], [(11, 7), (6, 9)]],
    6: [[(3, 8), (2, 4)], [(1, 3), (12, 5)]],
    7: [[(0, 1), (9, 2)], [(4, 6), (6, 3)]],
    8: [[(4, 7), (8, 3)], [(2, 1), (10, 6)]],
    9: [[(5, 1), (10, 4)], [(8, 6), (6, 9)]],
    10: [[(6, 8), (7, 4)], [(0, 1), (9, 3)]],
    11: [[(3, 0), (5, 4)], [(11, 6), (6, 7)]],
    12: [[(10, 7), (0, 4)], [(2, 1), (9, 3)]],
    13: [[(4, 1), (6, 2)], [(11, 7), (0, 5)]],
    14: [[(9, 8), (7, 4)], [(2, 1), (10, 3)]],
    15: [[(0, 10), (9, 1)], [(7, 8), (6, 11)]],
    16: [[(8, 7), (1, 2)], [(2, 0), (3, 4)]],
    17: [[(1, 5), (6, 4)], [(7, 9), (3, 8)]],
    18: [[(2, 7), (8, 4)], [(1, 1), (11, 3)]],
}


# ============================================================================
# Служебные функции
# ============================================================================
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


def intersection(xs: Sequence[Position], ys: Sequence[Position]) -> List[Position]:
    return sorted(set(xs) & set(ys))


def positions_brief(
    positions: Sequence[Position],
    pairs: Sequence[Sequence[Tuple[Number, Number]]],
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
) -> str:
    if not positions:
        return "нет"
    items = []
    for i, j in positions:
        r = row_labels[i] if row_labels else f"A{i+1}"
        c = col_labels[j] if col_labels else f"B{j+1}"
        items.append(f"({r}, {c}) = {pair_str(*pairs[i][j])}")
    return "; ".join(items)


# ============================================================================
# Нэш и Парето
# ============================================================================
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


# ============================================================================
# Цветная печать матрицы
# ============================================================================
def cell_marker(pos: Position, nash: set[Position], pareto: set[Position]) -> str:
    in_nash = pos in nash
    in_pareto = pos in pareto
    if in_nash and in_pareto:
        return "NP"
    if in_nash:
        return "N"
    if in_pareto:
        return "P"
    return "  "


def cell_style(pos: Position, nash: set[Position], pareto: set[Position]) -> str:
    in_nash = pos in nash
    in_pareto = pos in pareto
    if in_nash and in_pareto:
        return Ansi.MAGENTA
    if in_nash:
        return Ansi.BLUE
    if in_pareto:
        return Ansi.GREEN
    return ""


def print_legend(use_color: bool) -> None:
    print("Легенда:")
    print("  " + colorize("  N  ", Ansi.BLUE, use_color) + " — строгий Нэш")
    print("  " + colorize("  P  ", Ansi.GREEN, use_color) + " — Парето")
    print("  " + colorize(" NP  ", Ansi.MAGENTA, use_color) + " — пересечение")
    print()


def print_bimatrix_colored(
    title: str,
    pairs: Sequence[Sequence[Tuple[Number, Number]]],
    nash_positions: Sequence[Position],
    pareto_positions: Sequence[Position],
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    use_color: bool = True,
) -> None:
    m = len(pairs)
    n = len(pairs[0])
    if row_labels is None:
        row_labels = [f"A{i+1}" for i in range(m)]
    if col_labels is None:
        col_labels = [f"B{j+1}" for j in range(n)]

    nash_set = set(nash_positions)
    pareto_set = set(pareto_positions)

    print(title)
    print_legend(use_color)

    widths = []
    for j in range(n):
        max_len = len(str(col_labels[j]))
        for i in range(m):
            content = f"{pair_str(*pairs[i][j])} {cell_marker((i, j), nash_set, pareto_set)}"
            max_len = max(max_len, len(content))
        widths.append(max_len + 2)

    first_col_w = max(len(max(row_labels, key=len)), 8)
    header = " " * (first_col_w + 3)
    for j in range(n):
        header += f"{str(col_labels[j]):^{widths[j]}}"
    print(header)

    for i in range(m):
        line = f"{row_labels[i]:>{first_col_w}} |"
        for j in range(n):
            pos = (i, j)
            raw = f"{pair_str(*pairs[i][j])} {cell_marker(pos, nash_set, pareto_set)}"
            visible = f"{raw:^{widths[j]}}"
            styled = colorize(visible, cell_style(pos, nash_set, pareto_set), use_color)
            line += styled
        print(line)
    print()


# ============================================================================
# Смешанные стратегии для 2x2
# ============================================================================
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

    result: Dict[str, object] = {
        "exists": False,
        "reason": None,
    }

    denom_q = a11 - a12 - a21 + a22
    denom_p = b11 - b12 - b21 + b22
    result["denom_q"] = denom_q
    result["denom_p"] = denom_p

    if abs(denom_q) < 1e-12 or abs(denom_p) < 1e-12:
        result["reason"] = "Один из знаменателей в формулах безразличия равен нулю."
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
        result["reason"] = "Решение лежит вне интервала (0, 1), полностью смешанного равновесия нет."
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
    verbose: bool = False,
) -> None:
    if size_of(a) != (2, 2):
        return

    dom_row = has_strictly_dominant_row_strategy(a)
    dom_col = has_strictly_dominant_col_strategy(b)
    pure_nash = find_nash_equilibria(a, b, strict=False)
    mix = mixed_equilibrium_2x2(a, b)

    print("Смешанные стратегии:")
    if dom_row is None:
        print("  • У игрока 1 строго доминирующей стратегии нет.")
    else:
        name = row_labels[dom_row] if row_labels else f"A{dom_row + 1}"
        print(f"  • У игрока 1 есть строго доминирующая стратегия: {name}.")

    if dom_col is None:
        print("  • У игрока 2 строго доминирующей стратегии нет.")
    else:
        name = col_labels[dom_col] if col_labels else f"B{dom_col + 1}"
        print(f"  • У игрока 2 есть строго доминирующая стратегия: {name}.")

    print(f"  • Чистые равновесия по Нэшу: {positions_brief(pure_nash, pairs, row_labels, col_labels)}")

    if not mix["exists"]:
        print("  • Полностью смешанная ситуация равновесия: не существует.")
        print(f"    Причина: {mix['reason']}")
        print()
        return

    print("  • Полностью смешанная ситуация равновесия существует.")
    print(
        f"    x = [{fmt(mix['x'][0])}, {fmt(mix['x'][1])}], "
        f"y = [{fmt(mix['y'][0])}, {fmt(mix['y'][1])}]"
    )
    print(f"    v1 = {fmt(mix['v1'])}, v2 = {fmt(mix['v2'])}")

    if verbose:
        print(
            f"    Проверка безразличия игрока 1: {fmt(mix['v1_from_row1'])} = {fmt(mix['v1_from_row2'])}"
        )
        print(
            f"    Проверка безразличия игрока 2: {fmt(mix['v2_from_col1'])} = {fmt(mix['v2_from_col2'])}"
        )
        if mix.get("a_inv") is not None and mix.get("b_inv") is not None:
            print(f"    det(A) = {fmt(mix['det_a'])}, det(B) = {fmt(mix['det_b'])}")
            print(
                f"    A^(-1) = [[{fmt(mix['a_inv'][0][0])}, {fmt(mix['a_inv'][0][1])}], "
                f"[{fmt(mix['a_inv'][1][0])}, {fmt(mix['a_inv'][1][1])}]]"
            )
            print(
                f"    B^(-1) = [[{fmt(mix['b_inv'][0][0])}, {fmt(mix['b_inv'][0][1])}], "
                f"[{fmt(mix['b_inv'][1][0])}, {fmt(mix['b_inv'][1][1])}]]"
            )
            if mix.get("v1_formula") is not None and mix.get("v2_formula") is not None:
                print(
                    f"    По формуле методички: x = [{fmt(mix['x_formula'][0])}, {fmt(mix['x_formula'][1])}], "
                    f"y = [{fmt(mix['y_formula'][0])}, {fmt(mix['y_formula'][1])}]"
                )
    print()


# ============================================================================
# Генерация случайной биматричной игры
# ============================================================================
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


# ============================================================================
# Анализ одной игры
# ============================================================================
def analyze_game(
    name: str,
    pairs: Sequence[Sequence[Tuple[Number, Number]]],
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    extra_comment: Optional[str] = None,
    use_color: bool = True,
    verbose: bool = False,
) -> None:
    a, b = pairs_to_matrices(pairs)

    nash_classic = find_nash_equilibria(a, b, strict=False)
    nash_strict = find_nash_equilibria(a, b, strict=True)
    pareto_strong = find_pareto_strong(a, b)
    pareto_weak = find_pareto_weak(a, b)
    inter = intersection(nash_strict, pareto_strong)

    print_separator("=")
    print(colorize(name, Ansi.BOLD + Ansi.CYAN, use_color))
    print_separator("=")
    if extra_comment:
        print(extra_comment)
        print()

    print_bimatrix_colored(
        title="Матрица игры (цветом выделены искомые клетки):",
        pairs=pairs,
        nash_positions=nash_strict,
        pareto_positions=pareto_strong,
        row_labels=row_labels,
        col_labels=col_labels,
        use_color=use_color,
    )

    print("Краткий итог:")
    print(f"  • Строгий Нэш: {positions_brief(nash_strict, pairs, row_labels, col_labels)}")
    print(f"  • Парето: {positions_brief(pareto_strong, pairs, row_labels, col_labels)}")
    print(f"  • Пересечение: {positions_brief(inter, pairs, row_labels, col_labels)}")

    if inter:
        print(f"  • Выбор по правилу семинара: {positions_brief(inter, pairs, row_labels, col_labels)}")
    else:
        print(f"  • Выбор по правилу семинара: {positions_brief(nash_strict, pairs, row_labels, col_labels)}")

    if nash_classic != nash_strict:
        print(f"  • Классический Нэш (>=): {positions_brief(nash_classic, pairs, row_labels, col_labels)}")

    if verbose:
        print(f"  • Слабый Парето: {positions_brief(pareto_weak, pairs, row_labels, col_labels)}")
    print()

    print_mixed_analysis(a, b, pairs, row_labels, col_labels, verbose=verbose)


# ============================================================================
# Аргументы командной строки
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа: Нэш, Парето и смешанные стратегии в биматричных играх."
    )
    parser.add_argument(
        "variant",
        nargs="?",
        type=int,
        default=DEFAULT_VARIANT,
        help=f"номер варианта из таблицы Л5.1 (по умолчанию {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"seed для случайной игры 10x10 (по умолчанию {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="отключить цветной вывод",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="печатать дополнительные промежуточные вычисления",
    )
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        parser.error("вариант должен быть целым числом от 1 до 18")

    return args


# ============================================================================
# Главная программа
# ============================================================================
def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    print_separator("#")
    print(colorize("ЛАБОРАТОРНАЯ РАБОТА №5: НЭШ, ПАРЕТО, СМЕШАННЫЕ СТРАТЕГИИ", Ansi.BOLD, use_color))
    print_separator("#")
    print(f"Вариант: {args.variant}")
    print(f"Seed случайной игры 10x10: {args.seed}")
    print()

    print(colorize("1. Проверка на классических играх", Ansi.BOLD, use_color))
    print_separator("-")
    for game_name, data in CLASSIC_GAMES.items():
        analyze_game(
            game_name,
            data["pairs"],
            row_labels=data["rows"],
            col_labels=data["cols"],
            extra_comment=data.get("comment"),
            use_color=use_color,
            verbose=args.verbose,
        )

    print(colorize("2. Случайная биматричная игра 10x10", Ansi.BOLD, use_color))
    print_separator("-")
    random_pairs = generate_random_bimatrix(
        rows=RANDOM_SIZE,
        cols=RANDOM_SIZE,
        low=RANDOM_LOW,
        high=RANDOM_HIGH,
        seed=args.seed,
    )
    analyze_game(
        "Случайная игра 10x10",
        random_pairs,
        use_color=use_color,
        verbose=False,
    )

    print(colorize("3. Вариант из таблицы Л5.1", Ansi.BOLD, use_color))
    print_separator("-")
    analyze_game(
        f"Вариант {args.variant}",
        VARIANTS[args.variant],
        row_labels=["α1", "α2"],
        col_labels=["β1", "β2"],
        extra_comment="Для этой игры дополнительно ищутся равновесия в смешанном расширении.",
        use_color=use_color,
        verbose=True if args.verbose else False,
    )


if __name__ == "__main__":
    main()
