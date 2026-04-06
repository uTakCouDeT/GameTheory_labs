from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ============================================================
# ЛР №3. Теория закрытого аукциона первой цены
# Вариант 13
# ============================================================

DEFAULT_VARIANT = 13
DEFAULT_BUYERS = 17
DEFAULT_LOT_COUNT = 1
ROUND_DIGITS = 6

# Константный массив "красивых" внутренних оценок Xi.
# Числа специально взяты целыми и с двумя нулями на конце.
VALUATIONS: List[int] = [
    400,
    1300,
    2100,
    2800,
    3600,
    4700,
    5900,
    6800,
    7400,
    8300,
    9100,
    10200,
    11800,
    13100,
    14900,
    16700,
    18500,
]


@dataclass
class BidderResult:
    bidder_id: int
    valuation: int           # Xi
    equilibrium_bid: float   # bi*
    profit_if_wins: float    # Xi - bi*


def fmt(x: float, digits: int = ROUND_DIGITS) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def equilibrium_bid(valuation: int, n: int) -> float:
    """
    Симметричное равновесие Нэша для закрытого аукциона первой цены:
        b(x) = (n - 1) / n * x
    """
    return ((n - 1) / n) * valuation


def solve_first_price_auction(
    n: int,
    valuations: List[int],
) -> tuple[List[BidderResult], int, float, float]:
    """
    Возвращает:
    - результаты по всем игрокам,
    - индекс победителя (0-based),
    - цену аукциона,
    - прибыль победителя.
    """
    results: List[BidderResult] = []

    for i, x_i in enumerate(valuations, start=1):
        bid = equilibrium_bid(x_i, n)
        profit_if_wins = x_i - bid
        results.append(
            BidderResult(
                bidder_id=i,
                valuation=x_i,
                equilibrium_bid=bid,
                profit_if_wins=profit_if_wins,
            )
        )

    winner_index = max(range(len(results)), key=lambda idx: results[idx].equilibrium_bid)
    auction_price = results[winner_index].equilibrium_bid
    winner_profit = results[winner_index].profit_if_wins

    return results, winner_index, auction_price, winner_profit


def print_theoretical_part(n: int) -> None:
    coeff = (n - 1) / n
    print("=" * 100)
    print("ЛАБОРАТОРНАЯ РАБОТА №3 — ТЕОРИЯ ЗАКРЫТОГО АУКЦИОНА ПЕРВОЙ ЦЕНЫ")
    print("=" * 100)
    print(f"Вариант: {DEFAULT_VARIANT}")
    print(f"Число покупателей n: {n}")
    print(f"Количество товаров: {DEFAULT_LOT_COUNT}")
    print()
    print("АНАЛИТИЧЕСКОЕ РЕШЕНИЕ")
    print("-" * 100)
    print("Симметричное равновесие Нэша для закрытого аукциона первой цены:")
    print("    b(x) = (n - 1) / n * x")
    print()
    print(f"Для n = {n}:")
    print(f"    b(x) = ({n - 1} / {n}) * x = {fmt(coeff)} * x")
    print()
    print("Прибыль победителя:")
    print("    π = X - b(X)")
    print(f"    π = X - {fmt(coeff)} * X = (1/{n}) * X = {fmt(1 / n)} * X")
    print()


def print_input_data(valuations: List[int]) -> None:
    print("ИСХОДНЫЕ ДАННЫЕ")
    print("-" * 100)
    print("Внутренние оценки Xi заданы константным массивом:")
    for i, value in enumerate(valuations, start=1):
        print(f"    X{i:02d} = {value}")
    print()


def print_detailed_bids(results: List[BidderResult], n: int) -> None:
    print("ПОЭТАПНЫЙ РАСЧЁТ ОПТИМАЛЬНЫХ СТАВОК ПО НЭШУ")
    print("-" * 100)
    for row in results:
        print(
            f"Оптимальная по Нэшу ставка a{row.bidder_id - 1} = "
            f"(n - 1) / n * X{row.bidder_id - 1} = "
            f"{n - 1}/{n} * {row.valuation} = {fmt(row.equilibrium_bid)}"
        )
    print()


def print_results_table(results: List[BidderResult], winner_index: int) -> None:
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("-" * 100)
    print(
        f"{'Игрок':>5} | {'Xi':>8} | {'bi*':>12} | {'Прибыль Xi - bi*':>18}"
    )
    print("-" * 100)
    for idx, row in enumerate(results):
        mark = "  <-- победитель" if idx == winner_index else ""
        print(
            f"{row.bidder_id:>5} | "
            f"{row.valuation:>8} | "
            f"{fmt(row.equilibrium_bid):>12} | "
            f"{fmt(row.profit_if_wins):>18}{mark}"
        )
    print()


def print_final_conclusion(
    results: List[BidderResult],
    winner_index: int,
    auction_price: float,
    winner_profit: float,
) -> None:
    winner = results[winner_index]

    print("ИТОГ")
    print("-" * 100)
    print(f"Победитель аукциона: игрок a{winner.bidder_id - 1} (№{winner.bidder_id})")
    print(f"Максимальная внутренняя оценка: X_max = {winner.valuation}")
    print(f"Равновесная ставка победителя: b* = {fmt(auction_price)}")
    print(f"Цена аукциона: {fmt(auction_price)}")
    print(f"Прибыль победителя: π = {fmt(winner_profit)}")
    print()
    print("Проверка:")
    print(f"    b* = 16/17 * {winner.valuation} = {fmt((16 / 17) * winner.valuation)}")
    print(f"    π  = 1/17 * {winner.valuation} = {fmt((1 / 17) * winner.valuation)}")
    print()


def main() -> None:
    n = DEFAULT_BUYERS

    if len(VALUATIONS) != n:
        raise ValueError(
            f"Размер массива VALUATIONS должен быть равен {n}, "
            f"сейчас: {len(VALUATIONS)}"
        )

    results, winner_index, auction_price, winner_profit = solve_first_price_auction(
        n=n,
        valuations=VALUATIONS,
    )

    print_theoretical_part(n)
    print_input_data(VALUATIONS)
    print_detailed_bids(results, n)
    print_results_table(results, winner_index)
    print_final_conclusion(results, winner_index, auction_price, winner_profit)


if __name__ == "__main__":
    main()