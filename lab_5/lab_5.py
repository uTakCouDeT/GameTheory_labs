from __future__ import annotations

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

VARIANT_13_ROW: Tuple[int, ...] = (
    0, 3, 3, 2, 4,
    6, 5, 8, 6, 9, 8,
    9, 12, 10, 12,
    14,
)


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


def print_check_result(name: str, checks, violations) -> None:
    print(f"\n{name}:")
    print(f"  Проверено неравенств: {len(checks)}")

    if not violations:
        print("  Нарушений нет.")
        return

    print(f"  Найдены нарушения: {len(violations)}")
    for s, t, left, right, _ in violations:
        print(
            f"  S={coalition_to_str(s):>12}, T={coalition_to_str(t):>12}: "
            f"левая часть = {fmt_fraction(left)}, правая часть = {fmt_fraction(right)}"
        )


def make_superadditive(v: CharacteristicFunction, players: Tuple[int, ...] = PLAYERS):
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


def main() -> None:
    original = build_characteristic_function(VARIANT_13_ROW)
    print_characteristic_function(original)

    super_checks, super_violations = check_superadditivity(original)
    print_check_result(
        "Проверка исходной игры на супераддитивность",
        super_checks,
        super_violations,
    )

    convex_checks, convex_violations = check_convexity(original)
    print_check_result(
        "Проверка исходной игры на выпуклость",
        convex_checks,
        convex_violations,
    )

    working = original
    if super_violations:
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
        )

        convex_checks2, convex_violations2 = check_convexity(working)
        print_check_result(
            "Проверка скорректированной игры на выпуклость",
            convex_checks2,
            convex_violations2,
        )

    phi, details = shapley_vector(working)
    print_shapley_details(phi, details)

    print("\nИтоговый вектор Шепли:")
    print("  X(v) = (" + ", ".join(fmt_fraction(phi[i]) for i in PLAYERS) + ")")

    print_rationality_checks(working, phi)
    print()


if __name__ == "__main__":
    main()
