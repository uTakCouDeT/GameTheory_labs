from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

tol = 1e-9


@dataclass
class StrategyResult:
    name: str
    row: List[float]
    bernoulli: float
    wald: float
    maximax: float
    hurwicz: float
    savage: float


def avg(values: List[float]) -> float:
    return sum(values) / len(values)


def matrix_dimensions(matrix: List[List[float]]) -> Tuple[int, int]:
    if not matrix:
        raise ValueError("Матрица пуста.")
    row_len = len(matrix[0])
    if row_len == 0:
        raise ValueError("Матрица содержит пустую строку.")
    for row in matrix:
        if len(row) != row_len:
            raise ValueError("Все строки матрицы должны быть одной длины.")
    return len(matrix), row_len


def regret_matrix(payoff_matrix: List[List[float]]) -> List[List[float]]:
    m, n = matrix_dimensions(payoff_matrix)
    col_max = [max(payoff_matrix[i][j] for i in range(m)) for j in range(n)]
    regrets = []
    for i in range(m):
        regrets.append([col_max[j] - payoff_matrix[i][j] for j in range(n)])
    return regrets


def compute_criteria(payoff_matrix: List[List[float]], hurwicz_alpha: float) -> Tuple[
    List[StrategyResult], List[List[float]]]:
    m, _ = matrix_dimensions(payoff_matrix)
    regrets = regret_matrix(payoff_matrix)
    results: List[StrategyResult] = []

    for i in range(m):
        row = payoff_matrix[i]
        row_regrets = regrets[i]

        bernoulli = avg(row)
        wald = min(row)
        maximax = max(row)
        hurwicz = hurwicz_alpha * min(row) + (1 - hurwicz_alpha) * max(row)
        savage = max(row_regrets)  # потом ранжируем по возрастанию

        results.append(
            StrategyResult(
                name=f"a{i + 1}",
                row=row,
                bernoulli=bernoulli,
                wald=wald,
                maximax=maximax,
                hurwicz=hurwicz,
                savage=savage,
            )
        )

    return results, regrets


def average_ranks(values: List[float], bigger_is_better: bool = True) -> List[float]:
    """
    Возвращает ранги 1..n.
    При равенствах ставится среднее место.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=bigger_is_better)

    ranks = [0.0] * len(values)
    pos = 1
    i = 0

    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and math.isclose(indexed[j + 1][1], indexed[i][1], abs_tol=tol):
            j += 1

        avg_rank = (pos + (pos + (j - i))) / 2
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank

        pos += (j - i + 1)
        i = j + 1

    return ranks


def print_matrix(matrix: List[List[float]], row_names: List[str], col_names: List[str], title: str) -> None:
    print(f"\n{title}")
    header = [" " * 8] + [f"{c:>10}" for c in col_names]
    print("".join(header))
    for name, row in zip(row_names, matrix):
        print(f"{name:>8}" + "".join(f"{x:>10.3f}" for x in row))


def print_strategy_table(results: List[StrategyResult]) -> None:
    print("\nЗначения критериев по стратегиям")
    print(
        f"{'Стратегия':>10}"
        f"{'Бернулли':>12}"
        f"{'Вальд':>12}"
        f"{'Максимакс':>12}"
        f"{'Гурвиц':>12}"
        f"{'Сэвидж':>12}"
    )
    for r in results:
        print(
            f"{r.name:>10}"
            f"{r.bernoulli:>12.3f}"
            f"{r.wald:>12.3f}"
            f"{r.maximax:>12.3f}"
            f"{r.hurwicz:>12.3f}"
            f"{r.savage:>12.3f}"
        )


def print_ranking(
        criterion_name: str,
        strategy_names: List[str],
        values: List[float],
        ranks: List[float],
        bigger_is_better: bool,
) -> None:
    print(f"\nРанжирование по критерию: {criterion_name}")
    rows = list(zip(strategy_names, values, ranks))
    rows.sort(key=lambda x: (x[2], -x[1] if bigger_is_better else x[1], x[0]))

    print(f"{'Стратегия':>10}{'Значение':>12}{'Ранг':>10}")
    for name, value, rank in rows:
        print(f"{name:>10}{value:>12.3f}{rank:>10.3f}")

def build_gamma_matrix_from_all_criteria(strategy_names, criterion_ranks):
    """
    Строит матрицу парных сравнений gamma по аналогии со слайдом 10 МКО.

    gamma[i][j] =
        1.0  если стратегия i предпочтительнее стратегии j
        0.5  если стратегии равноценны
        0.0  если стратегия i хуже стратегии j

    Сравнение двух стратегий выполняется по числу побед по критериям:
    - у кого меньше ранг по большему числу критериев, тот лучше;
    - если побед поровну, считаем стратегии равнозначными.
    """
    n = len(strategy_names)
    crit_names = list(criterion_ranks.keys())

    gamma = [[0.5 if i == j else 0.0 for j in range(n)] for i in range(n)]
    pair_details = [["" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            wins_i = 0
            wins_j = 0
            ties = 0
            details = []

            for crit in crit_names:
                ri = criterion_ranks[crit][i]
                rj = criterion_ranks[crit][j]

                if math.isclose(ri, rj, abs_tol=tol):
                    ties += 1
                    details.append(f"{crit}: ничья")
                elif ri < rj:
                    wins_i += 1
                    details.append(f"{crit}: {strategy_names[i]}")
                else:
                    wins_j += 1
                    details.append(f"{crit}: {strategy_names[j]}")

            if wins_i > wins_j:
                gamma[i][j] = 1.0
                gamma[j][i] = 0.0
                result_ij = f"{strategy_names[i]} лучше {strategy_names[j]}"
                result_ji = f"{strategy_names[j]} хуже {strategy_names[i]}"
            elif wins_j > wins_i:
                gamma[i][j] = 0.0
                gamma[j][i] = 1.0
                result_ij = f"{strategy_names[i]} хуже {strategy_names[j]}"
                result_ji = f"{strategy_names[j]} лучше {strategy_names[i]}"
            else:
                gamma[i][j] = 0.5
                gamma[j][i] = 0.5
                result_ij = f"{strategy_names[i]} эквивалентна {strategy_names[j]}"
                result_ji = f"{strategy_names[j]} эквивалентна {strategy_names[i]}"

            pair_details[i][j] = (
                f"{result_ij}; победы {wins_i}:{wins_j}, ничьи={ties}; "
                + "; ".join(details)
            )
            pair_details[j][i] = (
                f"{result_ji}; победы {wins_j}:{wins_i}, ничьи={ties}; "
                + "; ".join(details)
            )

    return gamma, pair_details


def compute_alpha_from_gamma(gamma):
    """
    alpha_i = sum_{j != i} gamma_ij
    дополнительно считаются нормированные веса
    """
    n = len(gamma)
    alpha = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += gamma[i][j]
        alpha.append(s)

    total = sum(alpha)
    if total > tol:
        alpha_norm = [a / total for a in alpha]
    else:
        alpha_norm = [0.0 for _ in alpha]

    return alpha, alpha_norm


def relation_symbol_from_gamma(value):
    if math.isclose(value, 1.0, abs_tol=tol):
        return ">"
    if math.isclose(value, 0.5, abs_tol=tol):
        return "~"
    return "<"


def try_build_transitive_chain_from_gamma(strategy_names, gamma):
    """
    Пытаемся упорядочить стратегии по:
    1) числу строгих побед,
    2) alpha,
    3) числу ничьих.
    Это не гарантирует существование строгой транзитивной цепочки,
    но даёт естественный порядок для вывода.
    """
    alpha, _ = compute_alpha_from_gamma(gamma)
    score = []

    for i, name in enumerate(strategy_names):
        wins = sum(1 for j in range(len(strategy_names)) if i != j and math.isclose(gamma[i][j], 1.0, abs_tol=tol))
        ties = sum(1 for j in range(len(strategy_names)) if i != j and math.isclose(gamma[i][j], 0.5, abs_tol=tol))
        score.append((name, wins, alpha[i], ties))

    score.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    return [x[0] for x in score]


# ============================================================
# ОСНОВНОЕ РЕШЕНИЕ
# ============================================================
def solve_lab4_rk1(payoff_matrix: List[List[float]], hurwicz_alpha: float = 0.5) -> None:
    m, n = matrix_dimensions(payoff_matrix)
    strategy_names = [f"a{i + 1}" for i in range(m)]
    state_names = [f"b{j + 1}" for j in range(n)]

    print_matrix(payoff_matrix, strategy_names, state_names, "Исходная матрица выигрышей")

    results, regrets = compute_criteria(payoff_matrix, hurwicz_alpha)
    print_matrix(regrets, strategy_names, state_names, "Матрица рисков Сэвиджа")
    print_strategy_table(results)

    # -------------------------------
    # 1. Ранжирование по 5 критериям
    # -------------------------------
    criterion_values: Dict[str, List[float]] = {
        "Бернулли": [r.bernoulli for r in results],
        "Вальд": [r.wald for r in results],
        "Максимакс": [r.maximax for r in results],
        "Гурвиц": [r.hurwicz for r in results],
        "Сэвидж": [r.savage for r in results],
    }

    criterion_directions: Dict[str, bool] = {
        "Бернулли": True,
        "Вальд": True,
        "Максимакс": True,
        "Гурвиц": True,
        "Сэвидж": False,  # меньше риск = лучше
    }

    criterion_ranks: Dict[str, List[float]] = {}
    total_scores = [0.0] * m

    for crit_name, values in criterion_values.items():
        ranks = average_ranks(values, bigger_is_better=criterion_directions[crit_name])
        criterion_ranks[crit_name] = ranks
        total_scores = [s + r for s, r in zip(total_scores, ranks)]

        print_ranking(
            criterion_name=crit_name,
            strategy_names=strategy_names,
            values=values,
            ranks=ranks,
            bigger_is_better=criterion_directions[crit_name],
        )

    print("\nИтог по суммарным баллам (меньше = лучше)")
    total_rows = list(zip(strategy_names, total_scores))
    total_rows.sort(key=lambda x: (x[1], x[0]))

    print(f"{'Стратегия':>10}{'Сумма баллов':>16}")
    for name, score in total_rows:
        print(f"{name:>10}{score:>16.3f}")

    print("\nИтоговое ранжирование по 5 критериям:")
    print(" > ".join(name for name, _ in total_rows))

    # -------------------------------
    # 2. Метод парных сравнений (по МКО)
    # -------------------------------
    gamma, pair_details = build_gamma_matrix_from_all_criteria(strategy_names, criterion_ranks)
    alpha, alpha_norm = compute_alpha_from_gamma(gamma)

    print("\n" + "=" * 72)
    print("Метод парных сравнений стратегий")
    print("=" * 72)

    print("\nМатрица gamma")
    print("Обозначения:")
    print("1   - стратегия строки предпочтительнее стратегии столбца")
    print("0.5 - стратегии равнозначны")
    print("0   - стратегия строки менее предпочтительна")

    header = [" " * 8] + [f"{name:>10}" for name in strategy_names]
    print("".join(header))
    for i, row in enumerate(gamma):
        print(f"{strategy_names[i]:>8}" + "".join(f"{x:>10.1f}" for x in row))

    print("\nПояснения по попарным сравнениям:")
    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            rel = relation_symbol_from_gamma(gamma[i][j])
            print(f"{strategy_names[i]} {rel} {strategy_names[j]}  |  {pair_details[i][j]}")

    print("\nРасчёт весов стратегий alpha_i")
    print("Формула: alpha_i = sum(gamma_ij), j != i")
    print(f"{'Стратегия':>10}{'alpha_i':>12}{'alpha_norm':>14}")

    alpha_rows = list(zip(strategy_names, alpha, alpha_norm))
    alpha_rows.sort(key=lambda x: (-x[1], -x[2], x[0]))
    for name, a, an in alpha_rows:
        print(f"{name:>10}{a:>12.3f}{an:>14.5f}")

    print("\nРанжирование стратегий по убыванию alpha_i:")
    print(" > ".join(name for name, _, _ in alpha_rows))

    # -------------------------------
    # 3. Попытка построения транзитивной цепочки
    # -------------------------------
    chain = try_build_transitive_chain_from_gamma(strategy_names, gamma)

    print("\n" + "=" * 72)
    print("Попытка построения транзитивной цепочки")
    print("=" * 72)

    print("Цепочка, полученная по числу попарных побед и alpha:")
    print(" > ".join(chain))

    print("\nВсе попарные отношения:")
    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            rel = relation_symbol_from_gamma(gamma[i][j])
            print(f"{strategy_names[i]} {rel} {strategy_names[j]}")

    # Проверка на наличие цикла
    has_cycle = False
    n = len(strategy_names)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    if (
                            math.isclose(gamma[i][j], 1.0, abs_tol=tol)
                            and math.isclose(gamma[j][k], 1.0, abs_tol=tol)
                            and math.isclose(gamma[k][i], 1.0, abs_tol=tol)
                    ):
                        has_cycle = True

    if has_cycle:
        print("\nОбнаружен цикл предпочтений: строгая транзитивная цепочка нарушается.")
    else:
        print("\nЦикл строгих предпочтений не обнаружен.")


if __name__ == "__main__":
    # matrix = [
    #     [5, 8, 7, 5, 4],
    #     [1, 10, 5, 5, 6],
    #     [2, 4, 3, 6, 2],
    #     [3, 5, 4, 12, 3],
    # ]

    # 13
    # matrix = [
    #     [10, 11, 16, 15, 2],
    #     [9, 7, 6, 17, 1],
    #     [3, 0, 19, 15, 4],
    #     [0, 15, 13, 10, 6],
    # ]

    matrix = [
        [1, 7, 16, 9, 17],
        [11, 5, 6, 9, 18],
        [12, 7, 13, 16, 15],
        [11, 7, 2, 13, 7],
    ]

    solve_lab4_rk1(matrix)
