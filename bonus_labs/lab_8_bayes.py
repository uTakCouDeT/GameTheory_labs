#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительное задание 2. Байесовские игры.

Реализован универсальный перебор чистых байесовских стратегий.
Стратегия игрока — функция от его типа к действию; равновесие проверяется по каждому типу.

Для подстановки своего варианта создайте новую функцию-конструктор по образцу build_sheriff_game().
Минимально нужно заменить players, types, actions, prior и payoff.

Что считает скрипт:
1. Задает статическую байесовскую игру в чистых стратегиях.
2. Стратегия игрока трактуется как функция от его типа к действию.
3. Полным перебором строятся все профили чистых байесовских стратегий.
4. Для каждого профиля проверяется условие равновесия Байеса — Нэша:
   ни одному типу ни одного игрока не выгодно односторонне менять свое действие.
5. Выводятся найденные равновесия, ex ante ожидаемые выигрыши и проверка по типам.

Запуск:
    python lab_8_bayes.py

Как менять задачу:
    - для дилеммы шерифа измените q в build_sheriff_game(q=...);
    - для своей игры создайте функцию-конструктор по образцу build_sheriff_game();
    - нужно задать players, types, actions, prior и payoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Hashable, List, Mapping, Tuple

Action = Hashable
TypeName = Hashable
Player = str
TypeProfile = Tuple[TypeName, ...]
ActionProfile = Tuple[Action, ...]
Payoff = Tuple[float, ...]


# -----------------------------------------------------------------------------
# Форматирование результата
# -----------------------------------------------------------------------------


def fmt_number(x: float, digits: int = 6) -> str:
    """Красиво печатает число: целые без .0, остальные с ограниченной точностью."""
    if abs(x - round(x)) < 10 ** (-(digits - 1)):
        return str(int(round(x)))
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# -----------------------------------------------------------------------------
# Универсальная модель байесовской игры
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianGame:
    """Статическая байесовская игра в чистых стратегиях."""

    players: Tuple[Player, ...]
    types: Tuple[Tuple[TypeName, ...], ...]
    actions: Tuple[Tuple[Action, ...], ...]
    prior: Mapping[TypeProfile, float]
    payoff: Mapping[Tuple[TypeProfile, ActionProfile], Payoff]

    def __post_init__(self) -> None:
        """Проверяет корректность входных данных."""
        if len(self.players) != len(self.types) or len(self.players) != len(self.actions):
            raise ValueError("players, types и actions должны иметь одинаковую длину")

        n = len(self.players)
        total = sum(self.prior.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Сумма вероятностей типов должна быть 1, сейчас {total}")

        all_type_profiles = set(product(*self.types))
        if set(self.prior.keys()) != all_type_profiles:
            missing = all_type_profiles - set(self.prior.keys())
            extra = set(self.prior.keys()) - all_type_profiles
            raise ValueError(f"prior должен быть задан для всех профилей типов; missing={missing}, extra={extra}")

        for type_profile in all_type_profiles:
            for action_profile in product(*self.actions):
                key = (type_profile, action_profile)
                if key not in self.payoff:
                    raise ValueError(f"Нет платежей для type_profile={type_profile}, action_profile={action_profile}")
                if len(self.payoff[key]) != n:
                    raise ValueError(f"Платеж должен содержать {n} чисел")

    def pure_strategies_for_player(self, player_index: int) -> List[Tuple[Action, ...]]:
        """
        Все чистые байесовские стратегии игрока.

        Если у игрока два типа и два действия, то стратегии имеют вид:
            (действие_для_типа_1, действие_для_типа_2).
        """
        return list(product(self.actions[player_index], repeat=len(self.types[player_index])))

    def all_strategy_profiles(self) -> List[Tuple[Tuple[Action, ...], ...]]:
        """Все профили чистых байесовских стратегий всех игроков."""
        strategies = [self.pure_strategies_for_player(i) for i in range(len(self.players))]
        return list(product(*strategies))

    def action_by_strategy(self, player_index: int, strategy: Tuple[Action, ...], type_i: TypeName) -> Action:
        """Возвращает действие, которое стратегия предписывает данному типу."""
        type_position = self.types[player_index].index(type_i)
        return strategy[type_position]

    def action_profile_from_strategies(
        self,
        strategy_profile: Tuple[Tuple[Action, ...], ...],
        type_profile: TypeProfile,
    ) -> ActionProfile:
        """Строит профиль фактических действий по профилю стратегий и профилю типов."""
        return tuple(
            self.action_by_strategy(i, strategy_profile[i], type_profile[i])
            for i in range(len(self.players))
        )

    def conditional_type_profiles(self, player_index: int, own_type: TypeName) -> List[Tuple[TypeProfile, float]]:
        """Возвращает условное распределение P(t | t_i=own_type)."""
        denom = sum(prob for tp, prob in self.prior.items() if tp[player_index] == own_type)
        if denom <= 0:
            raise ValueError(f"Тип {own_type!r} игрока {self.players[player_index]} имеет нулевую вероятность")
        return [(tp, prob / denom) for tp, prob in self.prior.items() if tp[player_index] == own_type]

    def expected_payoff_for_type_action(
        self,
        player_index: int,
        own_type: TypeName,
        action: Action,
        opponents_strategy_profile: Tuple[Tuple[Action, ...], ...],
    ) -> float:
        """Ожидаемый выигрыш типа own_type при выборе конкретного действия action."""
        n = len(self.players)
        total = 0.0

        for type_profile, cond_prob in self.conditional_type_profiles(player_index, own_type):
            action_profile: list[Action] = []
            opponent_pos = 0

            for j in range(n):
                if j == player_index:
                    action_profile.append(action)
                else:
                    strategy_j = opponents_strategy_profile[opponent_pos]
                    action_profile.append(self.action_by_strategy(j, strategy_j, type_profile[j]))
                    opponent_pos += 1

            total += cond_prob * self.payoff[(type_profile, tuple(action_profile))][player_index]

        return total

    def best_responses_for_type(
        self,
        player_index: int,
        own_type: TypeName,
        opponents_strategy_profile: Tuple[Tuple[Action, ...], ...],
        tol: float = 1e-9,
    ) -> Tuple[float, List[Action], Dict[Action, float]]:
        """Считает выигрыши всех действий и лучшие ответы для одного типа игрока."""
        utilities = {
            action: self.expected_payoff_for_type_action(player_index, own_type, action, opponents_strategy_profile)
            for action in self.actions[player_index]
        }
        best_value = max(utilities.values())
        best_actions = [action for action, value in utilities.items() if abs(value - best_value) <= tol]
        return best_value, best_actions, utilities

    def is_bayes_nash_equilibrium(
        self,
        strategy_profile: Tuple[Tuple[Action, ...], ...],
        tol: float = 1e-9,
    ) -> bool:
        """
        Проверяет равновесие Байеса — Нэша.

        Для каждого игрока и каждого его типа берется действие, предписанное стратегией.
        Затем проверяется, входит ли оно в множество лучших ответов на стратегии остальных.
        """
        n = len(self.players)
        for i in range(n):
            opponents = tuple(strategy_profile[j] for j in range(n) if j != i)
            for own_type in self.types[i]:
                chosen = self.action_by_strategy(i, strategy_profile[i], own_type)
                _, best_actions, _ = self.best_responses_for_type(i, own_type, opponents, tol=tol)
                if chosen not in best_actions:
                    return False
        return True

    def bayes_nash_equilibria(self, tol: float = 1e-9) -> List[Tuple[Tuple[Action, ...], ...]]:
        """Находит все чистые равновесия Байеса — Нэша полным перебором."""
        return [
            strategy_profile
            for strategy_profile in self.all_strategy_profiles()
            if self.is_bayes_nash_equilibrium(strategy_profile, tol=tol)
        ]

    def ex_ante_expected_payoff(self, strategy_profile: Tuple[Tuple[Action, ...], ...]) -> Tuple[float, ...]:
        """Ожидаемые выигрыши до раскрытия типов."""
        n = len(self.players)
        total = [0.0] * n

        for type_profile, prob in self.prior.items():
            action_profile = self.action_profile_from_strategies(strategy_profile, type_profile)
            payoff = self.payoff[(type_profile, action_profile)]
            for i in range(n):
                total[i] += prob * payoff[i]

        return tuple(total)

    def describe_strategy_profile(self, strategy_profile: Tuple[Tuple[Action, ...], ...]) -> str:
        """Текстовое описание профиля стратегий."""
        parts: list[str] = []
        for i, player in enumerate(self.players):
            mapping = ", ".join(f"{type_i}->{action}" for type_i, action in zip(self.types[i], strategy_profile[i]))
            parts.append(f"{player}: ({mapping})")
        return "; ".join(parts)

    def print_equilibrium_analysis(self, title: str) -> None:
        """Печатает исходные данные, найденные BNE и проверку по типам."""
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)

        equilibria = self.bayes_nash_equilibria()

        print(f"Игроки: {self.players}")
        print("Типы:")
        for player, types_i in zip(self.players, self.types):
            print(f"  {player}: {types_i}")
        print("Действия:")
        for player, actions_i in zip(self.players, self.actions):
            print(f"  {player}: {actions_i}")
        print("Априорное распределение типов:")
        for type_profile, prob in sorted(self.prior.items(), key=lambda item: str(item[0])):
            print(f"  P{type_profile} = {fmt_number(prob)}")

        print(f"\nНайдено чистых равновесий Байеса — Нэша: {len(equilibria)}")
        for idx, equilibrium in enumerate(equilibria, start=1):
            print(f"  {idx}) {self.describe_strategy_profile(equilibrium)}")
            payoffs = tuple(fmt_number(v) for v in self.ex_ante_expected_payoff(equilibrium))
            print(f"     Ex ante E[u] = {payoffs}")
            print("     Проверка по типам:")

            for i, player in enumerate(self.players):
                opponents = tuple(equilibrium[j] for j in range(len(self.players)) if j != i)
                for own_type in self.types[i]:
                    chosen = self.action_by_strategy(i, equilibrium[i], own_type)
                    _, best_actions, utilities = self.best_responses_for_type(i, own_type, opponents)
                    utilities_s = ", ".join(f"{action}: {fmt_number(value)}" for action, value in utilities.items())
                    print(
                        f"       {player}, тип {own_type}: выбрано {chosen}; "
                        f"полезности {{{utilities_s}}}; лучшие ответы {best_actions}"
                    )


# -----------------------------------------------------------------------------
# Примеры байесовских игр
# -----------------------------------------------------------------------------


def build_sheriff_game(q: float = 0.5) -> BayesianGame:
    """
    Байесовская игра «Дилемма шерифа».

    Игрок 1: Suspect. Типы: criminal с вероятностью q и civilian с вероятностью 1-q.
    Игрок 2: Sheriff. Один технический тип sheriff.
    Действия обоих игроков: Shoot / Not.
    """
    players = ("Suspect", "Sheriff")
    types = (("criminal", "civilian"), ("sheriff",))
    actions = (("Shoot", "Not"), ("Shoot", "Not"))
    prior = {
        ("criminal", "sheriff"): float(q),
        ("civilian", "sheriff"): float(1 - q),
    }

    # payoff[(type_profile, (suspect_action, sheriff_action))] = (u_suspect, u_sheriff)
    payoff: dict[tuple[TypeProfile, ActionProfile], Payoff] = {}

    criminal_matrix = {
        ("Shoot", "Shoot"): (0, 0),
        ("Shoot", "Not"): (2, -2),
        ("Not", "Shoot"): (-2, -1),
        ("Not", "Not"): (-1, 1),
    }
    civilian_matrix = {
        ("Shoot", "Shoot"): (-3, -1),
        ("Shoot", "Not"): (-1, -2),
        ("Not", "Shoot"): (-2, -1),
        ("Not", "Not"): (0, 0),
    }

    for action_profile, p in criminal_matrix.items():
        payoff[(("criminal", "sheriff"), action_profile)] = p
    for action_profile, p in civilian_matrix.items():
        payoff[(("civilian", "sheriff"), action_profile)] = p

    return BayesianGame(players, types, actions, prior, payoff)


def build_two_type_player2_example(p_type1: float = 0.75) -> BayesianGame:
    """
    Небольшая байесовская игра по примеру со слайдов 13-14.

    Игрок 1 имеет один тип и две стратегии: I, H/I.
    Игрок 2 имеет два типа t1, t2 и две стратегии: C, H/C.
    """
    players = ("Player1", "Player2")
    types = (("single",), ("t1", "t2"))
    actions = (("I", "H/I"), ("C", "H/C"))
    prior = {
        ("single", "t1"): float(p_type1),
        ("single", "t2"): float(1 - p_type1),
    }

    payoff: dict[tuple[TypeProfile, ActionProfile], Payoff] = {}

    matrix_t1 = {
        ("I", "C"): (2, 2),
        ("I", "H/C"): (0, 0),
        ("H/I", "C"): (-1, -1),
        ("H/I", "H/C"): (-1, 1),
    }
    matrix_t2 = {
        ("I", "C"): (-3, 2),
        ("I", "H/C"): (-1, -1),
        ("H/I", "C"): (-2, -1),
        ("H/I", "H/C"): (0, 0),
    }

    for action_profile, p in matrix_t1.items():
        payoff[(("single", "t1"), action_profile)] = p
    for action_profile, p in matrix_t2.items():
        payoff[(("single", "t2"), action_profile)] = p

    return BayesianGame(players, types, actions, prior, payoff)


# -----------------------------------------------------------------------------
# Сценарии расчета
# -----------------------------------------------------------------------------


def scan_sheriff_q_grid() -> None:
    """Показывает, как меняются чистые BNE дилеммы шерифа при разных q."""
    print("\n" + "-" * 92)
    print("Сканирование q для дилеммы шерифа")

    for q in [0.01, 0.10, 0.25, 1/3, 0.50, 0.75, 0.90, 0.99]:
        game = build_sheriff_game(q)
        equilibria = game.bayes_nash_equilibria()
        print(f"q={q:.2f}: {len(equilibria)} BNE")
        for equilibrium in equilibria:
            print(f"  {game.describe_strategy_profile(equilibrium)}")


def print_slide_threshold_example() -> None:
    """Печатает расчет порога из примера со слайдов 13-14."""
    print("\n" + "-" * 92)
    print("Пример со слайдов 13-14: порог по представлению p=P(t1)")
    print("Сравниваются два кандидата: (I, C) и (H/I, H/C).")
    print("U1(I, C) = 2p + (-3)(1-p) = 5p - 3")
    print("U1(H/I, H/C) = (-1)p + 0(1-p) = -p")
    print("I выгоднее, если 5p - 3 > -p, то есть p > 1/2.")

    for p in [0.25, 0.50, 0.75]:
        u_i = 5 * p - 3
        u_h = -p
        if abs(u_i - u_h) < 1e-9:
            conclusion = "игрок 1 безразличен; возможны оба кандидата"
        elif u_i > u_h:
            conclusion = "выбирается кандидат (I, C)"
        else:
            conclusion = "выбирается кандидат (H/I, H/C)"
        print(f"p={p:.2f}: U1(I,C)={fmt_number(u_i)}, U1(H/I,H/C)={fmt_number(u_h)} -> {conclusion}")


def solve_bayesian_task() -> None:
    """Основной сценарий расчета задания 2."""
    # sheriff = build_sheriff_game(q=1/3)
    sheriff = build_sheriff_game(q=0.5)
    sheriff.print_equilibrium_analysis("Пример: дилемма шерифа, q = 0.5")

    scan_sheriff_q_grid()
    print_slide_threshold_example()


if __name__ == "__main__":
    solve_bayesian_task()
