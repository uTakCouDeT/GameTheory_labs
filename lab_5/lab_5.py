from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations
from math import factorial
from typing import Dict, FrozenSet, Iterable, List, Tuple

Coalition = FrozenSet[int]
CharacteristicFunction = Dict[Coalition, Fraction]

PLAYERS: Tuple[int, ...] = (1, 2, 3, 4)

COALITION_ORDER: Tuple[Tuple[int, ...], ...] = (
    (),
    (1,), (2,), (3,), (4,),
    (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    (1, 2, 3, 4),
)

# Исходные данные вариантов из таблицы Л7.1.
VARIANT_ROWS: Dict[int, Tuple[int, ...]] = {
    1:  (0, 4, 1, 3, 1, 6, 8, 6, 5, 3, 5, 9, 8, 10, 7, 11),
    2:  (0, 4, 3, 2, 3, 8, 6, 7, 5, 7, 6, 10, 12, 10, 9, 13),
    3:  (0, 3, 4, 1, 2, 7, 5, 6, 5, 7, 3, 10, 10, 9, 9, 12),
    4:  (0, 1, 1, 3, 3, 4, 4, 6, 4, 4, 7, 9, 9, 8, 8, 11),
    5:  (0, 3, 2, 3, 2, 6, 6, 5, 6, 4, 7, 10, 10, 9, 10, 12),
    6:  (0, 4, 2, 2, 1, 7, 7, 5, 4, 4, 3, 11, 9, 10, 7, 13),
    7:  (0, 4, 3, 2, 2, 7, 7, 6, 6, 6, 4, 11, 10, 11, 10, 13),
    8:  (0, 2, 1, 1, 2, 4, 4, 4, 2, 4, 4, 7, 8, 8, 6, 10),
    9:  (0, 3, 3, 1, 2, 7, 6, 7, 4, 7, 3, 10, 10, 9, 7, 12),
    10: (0, 3, 1, 2, 4, 4, 5, 8, 3, 5, 6, 7, 10, 11, 8, 13),
    11: (0, 3, 4, 4, 4, 8, 7, 7, 9, 8, 9, 13, 13, 13, 14, 16),
    12: (0, 2, 3, 4, 1, 6, 7, 4, 7, 5, 6, 12, 9, 9, 10, 14),
    13: (0, 3, 3, 2, 4, 6, 5, 8, 6, 9, 8, 9, 12, 10, 12, 14),
    14: (0, 4, 1, 1, 1, 7, 7, 7, 3, 2, 2, 10, 10, 10, 6, 12),
    15: (0, 2, 3, 2, 4, 6, 5, 7, 6, 8, 8, 9, 11, 10, 12, 14),
    16: (0, 4, 4, 4, 2, 9, 9, 6, 10, 7, 7, 15, 12, 12, 14, 17),
    17: (0, 1, 2, 1, 1, 4, 2, 3, 4, 4, 2, 7, 6, 5, 6, 9),
    18: (0, 2, 3, 4, 4, 6, 6, 7, 7, 8, 8, 11, 11, 12, 12, 14),
}


def coalition(items: Iterable[int]) -> Coalition:
    return frozenset(items)


def build_characteristic_function(row: Tuple[int, ...]) -> CharacteristicFunction:
    if len(row) != len(COALITION_ORDER):
        raise ValueError("В строке варианта должно быть 16 значений характеристической функции.")
    return {coalition(c): Fraction(value) for c, value in zip(COALITION_ORDER, row)}


def all_coalitions(players: Tuple[int, ...] = PLAYERS) -> List[Coalition]:
    result: List[Coalition] = []
    for r in range(len(players) + 1):
        for comb in combinations(players, r):
            result.append(coalition(comb))
    return result


def coalition_to_str(s: Coalition) -> str:
    if not s:
        return "∅"
    return "{" + ", ".join(map(str, sorted(s))) + "}"


def fmt_fraction(x: Fraction, *, with_decimal: bool = True) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    if with_decimal:
        return f"{x.numerator}/{x.denominator} ≈ {float(x):.6f}"
    return f"{x.numerator}/{x.denominator}"


def print_characteristic_function(v: CharacteristicFunction) -> None:
    print("\nХарактеристическая функция:")
    for c in COALITION_ORDER:
        s = coalition(c)
        print(f"  v({coalition_to_str(s):>12}) = {fmt_fraction(v[s])}")


def check_superadditivity(v: CharacteristicFunction, players: Tuple[int, ...] = PLAYERS):
    coalitions = all_coalitions(players)
    checks = []
    violations = []
    for idx_s, s in enumerate(coalitions):
        if not s:
            continue
        for t in coalitions[idx_s + 1:]:
            if not t or s & t:
                continue
            left = v[s | t]
            right = v[s] + v[t]
            ok = left >= right
            item = (s, t, left, right, ok)
            checks.append(item)
            if not ok:
                violations.append(item)
    return checks, violations


def check_convexity(v: CharacteristicFunction, players: Tuple[int, ...] = PLAYERS):
    """Проверка v(S∪T) + v(S∩T) >= v(S) + v(T) для всех S и T."""
    coalitions = all_coalitions(players)
    checks = []
    violations = []
    for idx_s, s in enumerate(coalitions):
        for t in coalitions[idx_s:]:
            left = v[s | t] + v[s & t]
            right = v[s] + v[t]
            ok = left >= right
            item = (s, t, left, right, ok)
            checks.append(item)
            if not ok:
                violations.append(item)
    return checks, violations


def print_check_result(name: str, checks, violations, *, verbose: bool = False) -> None:
    print(f"\n{name}:")
    print(f"  Проверено неравенств: {len(checks)}")
    if not violations:
        print("  Нарушений нет.")
    else:
        print(f"  Найдены нарушения: {len(violations)}")
        for s, t, left, right, _ in violations:
            print(
                f"  S={coalition_to_str(s):>12}, T={coalition_to_str(t):>12}: "
                f"левая часть = {fmt_fraction(left)}, правая часть = {fmt_fraction(right)}"
            )
    if verbose:
        print("  Все проверенные неравенства:")
        for s, t, left, right, ok in checks:
            sign = ">=" if ok else "<"
            print(
                f"    S={coalition_to_str(s):>12}, T={coalition_to_str(t):>12}: "
                f"{fmt_fraction(left)} {sign} {fmt_fraction(right)}"
            )


def make_superadditive(v: CharacteristicFunction, players: Tuple[int, ...] = PLAYERS):
    """
    Минимальная монотонная корректировка вверх:
    если для непересекающихся S и T нарушено v(S∪T) >= v(S)+v(T),
    значение v(S∪T) повышается до v(S)+v(T). Процедура повторяется
    до исчезновения всех нарушений.
    """
    result = dict(v)
    coalitions = all_coalitions(players)
    changes = []
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for s in coalitions:
            if not s:
                continue
            for t in coalitions:
                if not t or s & t:
                    continue
                u = s | t
                required = result[s] + result[t]
                if result[u] < required:
                    old = result[u]
                    result[u] = required
                    changes.append((iteration, s, t, u, old, required))
                    changed = True
    return result, changes


def shapley_vector(v: CharacteristicFunction, players: Tuple[int, ...] = PLAYERS):
    """Вычисление вектора Шепли и подробных слагаемых формулы."""
    n = len(players)
    all_sets = all_coalitions(players)
    phi: Dict[int, Fraction] = {i: Fraction(0) for i in players}
    details = {i: [] for i in players}

    for i in players:
        for s in all_sets:
            if i in s:
                continue
            size = len(s)
            coeff = Fraction(factorial(size) * factorial(n - size - 1), factorial(n))
            marginal = v[s | {i}] - v[s]
            term = coeff * marginal
            phi[i] += term
            details[i].append((s, coeff, marginal, term))
    return phi, details


def print_shapley_details(phi, details) -> None:
    print("\nРасчет компонент вектора Шепли:")
    for i in PLAYERS:
        print(f"\n  Игрок {i}:")
        for s, coeff, marginal, term in details[i]:
            print(
                f"    S={coalition_to_str(s):>10}; "
                f"коэф.={fmt_fraction(coeff, with_decimal=False):>5}; "
                f"v(S∪{{{i}}})-v(S)={fmt_fraction(marginal):>12}; "
                f"слагаемое={fmt_fraction(term)}"
            )
        print(f"    x_{i}(v) = {fmt_fraction(phi[i])}")


def print_rationality_checks(v: CharacteristicFunction, phi: Dict[int, Fraction]) -> None:
    total_coalition = coalition(PLAYERS)
    total = sum(phi.values(), Fraction(0))
    print("\nПроверка условий рационализации:")
    print("  Групповая рационализация:")
    print(
        f"    Σ x_i(v) = {fmt_fraction(total)}; "
        f"v(I) = {fmt_fraction(v[total_coalition])}; "
        f"результат: {'выполнено' if total == v[total_coalition] else 'НЕ выполнено'}"
    )
    print("  Индивидуальная рационализация:")
    for i in PLAYERS:
        single = coalition([i])
        ok = phi[i] >= v[single]
        print(
            f"    x_{i}(v) = {fmt_fraction(phi[i])}; "
            f"v({{{i}}}) = {fmt_fraction(v[single])}; "
            f"результат: {'выполнено' if ok else 'НЕ выполнено'}"
        )


def solve_variant(variant: int, *, verbose_inequalities: bool = False, auto_fix: bool = True) -> None:
    if variant not in VARIANT_ROWS:
        raise ValueError(f"Неизвестный вариант {variant}. Доступны варианты: {sorted(VARIANT_ROWS)}")

    print("=" * 90)
    print(f"Лабораторная работа № 5. Вариант {variant}")
    print("=" * 90)

    original = build_characteristic_function(VARIANT_ROWS[variant])
    print_characteristic_function(original)

    super_checks, super_violations = check_superadditivity(original)
    print_check_result(
        "Проверка исходной игры на супераддитивность",
        super_checks,
        super_violations,
        verbose=verbose_inequalities,
    )

    convex_checks, convex_violations = check_convexity(original)
    print_check_result(
        "Проверка исходной игры на выпуклость",
        convex_checks,
        convex_violations,
        verbose=verbose_inequalities,
    )

    working = original
    if super_violations and auto_fix:
        print("\nИсходная игра не является супераддитивной. Выполняется корректировка ХФ.")
        working, changes = make_superadditive(original)
        print("  Измененные значения характеристической функции:")
        for iteration, s, t, u, old, new in changes:
            print(
                f"    итерация {iteration}: из S={coalition_to_str(s)}, T={coalition_to_str(t)} "
                f"получено требование v({coalition_to_str(u)}) >= {fmt_fraction(new)}; "
                f"старое значение {fmt_fraction(old)} заменено на {fmt_fraction(new)}"
            )
        print_characteristic_function(working)
        super_checks2, super_violations2 = check_superadditivity(working)
        print_check_result(
            "Повторная проверка скорректированной игры на супераддитивность",
            super_checks2,
            super_violations2,
            verbose=verbose_inequalities,
        )
        convex_checks2, convex_violations2 = check_convexity(working)
        print_check_result(
            "Проверка скорректированной игры на выпуклость",
            convex_checks2,
            convex_violations2,
            verbose=verbose_inequalities,
        )
    elif super_violations and not auto_fix:
        print("\nВнимание: игра не супераддитивна, но автоматическая корректировка отключена.")

    phi, details = shapley_vector(working)
    print_shapley_details(phi, details)

    print("\nИтоговый вектор Шепли:")
    print("  X(v) = (" + ", ".join(fmt_fraction(phi[i]) for i in PLAYERS) + ")")

    print_rationality_checks(working, phi)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа № 5: проверка кооперативной игры и расчет вектора Шепли."
    )
    parser.add_argument("--variant", "-v", type=int, default=13, help="номер варианта из таблицы Л7.1")
    parser.add_argument("--all", action="store_true", help="решить все варианты из таблицы Л7.1")
    parser.add_argument(
        "--verbose-inequalities",
        action="store_true",
        help="вывести не только нарушения, но и все проверенные неравенства",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="не исправлять характеристическую функцию, если игра не супераддитивна",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = sorted(VARIANT_ROWS) if args.all else [args.variant]
    for variant in variants:
        solve_variant(
            variant,
            verbose_inequalities=args.verbose_inequalities,
            auto_fix=not args.no_auto_fix,
        )


if __name__ == "__main__":
    main()
