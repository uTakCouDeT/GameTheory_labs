#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительное задание 2. Байесовские игры.

Реализован универсальный перебор чистых байесовских стратегий.
Стратегия игрока — функция от его типа к действию; равновесие проверяется по каждому типу.

Для подстановки своего варианта создайте новую функцию-конструктор по образцу build_sheriff_game().
Минимально нужно заменить players, types, actions, prior и payoff.

Что считает скрипт:
1. Статическую байесовскую игру на основе вариантов 13 и 14 из методички.
2. Все чистые равновесия Байеса — Нэша в статической игре.
3. Динамическую байесовскую игру «выход на рынок» из лекции.
4. Чистые совершенные равновесия Байеса — Нэша для этой динамической игры.
5. Контрольные лекционные примеры: дилемма шерифа и пример со слайдов 13-14.

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

def build_methodical_variants_13_14_game(p_variant13: float = 0.5) -> BayesianGame:
    """
    Статическая байесовская игра на основе вариантов 13 и 14 из таблицы Л5.1 методички.

    Смысл модели:
    - игрок 1 выбирает строку a1 или a2 и не знает тип игрока 2;
    - игрок 2 знает свой тип: v13 или v14;
    - тип v13 означает, что реализуется биматричная игра варианта 13;
    - тип v14 означает, что реализуется биматричная игра варианта 14;
    - P(v13)=p_variant13, P(v14)=1-p_variant13.

    Вариант 13:
        [(4, 1)   (6, 2)]
        [(11, 7)  (0, 5)]

    Вариант 14:
        [(9, 8)   (7, 4)]
        [(2, 1)   (10, 3)]
    """
    if not 0.0 <= p_variant13 <= 1.0:
        raise ValueError("p_variant13 должен лежать в отрезке [0, 1]")

    players = ("Player1", "Player2")
    types = (("single",), ("v13", "v14"))
    actions = (("a1", "a2"), ("b1", "b2"))
    prior = {
        ("single", "v13"): float(p_variant13),
        ("single", "v14"): float(1 - p_variant13),
    }

    payoff: dict[tuple[TypeProfile, ActionProfile], Payoff] = {}

    variant13_matrix = {
        ("a1", "b1"): (4, 1),
        ("a1", "b2"): (6, 2),
        ("a2", "b1"): (11, 7),
        ("a2", "b2"): (0, 5),
    }
    variant14_matrix = {
        ("a1", "b1"): (9, 8),
        ("a1", "b2"): (7, 4),
        ("a2", "b1"): (2, 1),
        ("a2", "b2"): (10, 3),
    }

    for action_profile, pay in variant13_matrix.items():
        payoff[(("single", "v13"), action_profile)] = pay
    for action_profile, pay in variant14_matrix.items():
        payoff[(("single", "v14"), action_profile)] = pay

    return BayesianGame(players, types, actions, prior, payoff)


@dataclass(frozen=True)
class DynamicMarketEntryGame:
    """
    Динамическая байесовская игра «выход на рынок» из лекции.

    Порядок ходов:
    1. Природа выбирает тип incumbent: Normal с вероятностью p_normal или Tough с вероятностью 1-p_normal.
    2. Entrant не наблюдает тип и выбирает Out или In.
    3. Если выбран In, Incumbent, зная свой тип, выбирает Fight или Acquiesce.

    Выигрыши записаны в порядке (Incumbent, Entrant).
    """

    p_normal: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_normal <= 1.0:
            raise ValueError("p_normal должен лежать в отрезке [0, 1]")

    @property
    def p_tough(self) -> float:
        return 1.0 - self.p_normal

    def incumbent_payoff_after_entry(self, incumbent_type: str, response: str) -> float:
        """Выигрыш Incumbent после входа Entrant в зависимости от типа и ответа."""
        payoff = {
            ("Normal", "Fight"): -1,
            ("Normal", "Acquiesce"): 1,
            ("Tough", "Fight"): 1,
            ("Tough", "Acquiesce"): -1,
        }
        return payoff[(incumbent_type, response)]

    def terminal_payoff(self, incumbent_type: str, entrant_action: str, incumbent_response: str | None = None) -> Payoff:
        """Возвращает терминальный выигрыш (Incumbent, Entrant)."""
        if entrant_action == "Out":
            return (2, 0)
        if incumbent_response is None:
            raise ValueError("При entrant_action='In' нужно указать ответ Incumbent")
        payoff = {
            ("Normal", "Fight"): (-1, -1),
            ("Normal", "Acquiesce"): (1, 1),
            ("Tough", "Fight"): (1, -1),
            ("Tough", "Acquiesce"): (-1, 1),
        }
        return payoff[(incumbent_type, incumbent_response)]

    def incumbent_best_responses(self) -> dict[str, list[str]]:
        """Последовательные лучшие ответы Incumbent после входа Entrant."""
        result: dict[str, list[str]] = {}
        for incumbent_type in ("Normal", "Tough"):
            utilities = {
                response: self.incumbent_payoff_after_entry(incumbent_type, response)
                for response in ("Fight", "Acquiesce")
            }
            best = max(utilities.values())
            result[incumbent_type] = [r for r, u in utilities.items() if abs(u - best) <= 1e-9]
        return result

    def entrant_expected_payoff(self, entrant_action: str, incumbent_strategy: Mapping[str, str]) -> float:
        """Ожидаемый выигрыш Entrant при выборе Out/In."""
        if entrant_action == "Out":
            return 0.0
        normal_payoff = self.terminal_payoff("Normal", "In", incumbent_strategy["Normal"])[1]
        tough_payoff = self.terminal_payoff("Tough", "In", incumbent_strategy["Tough"])[1]
        return self.p_normal * normal_payoff + self.p_tough * tough_payoff

    def perfect_bayesian_equilibria(self) -> list[tuple[str, dict[str, str], tuple[float, float]]]:
        """
        Находит чистые совершенные равновесия Байеса — Нэша для данной игры.

        Возвращает список троек:
            (действие Entrant, стратегия Incumbent по типам, ex ante выигрыши).
        """
        br = self.incumbent_best_responses()
        incumbent_strategies: list[dict[str, str]] = []
        for normal_response in br["Normal"]:
            for tough_response in br["Tough"]:
                incumbent_strategies.append({"Normal": normal_response, "Tough": tough_response})

        equilibria: list[tuple[str, dict[str, str], tuple[float, float]]] = []
        for incumbent_strategy in incumbent_strategies:
            entrant_utilities = {
                action: self.entrant_expected_payoff(action, incumbent_strategy)
                for action in ("Out", "In")
            }
            best = max(entrant_utilities.values())
            best_actions = [a for a, u in entrant_utilities.items() if abs(u - best) <= 1e-9]

            for entrant_action in best_actions:
                equilibria.append((entrant_action, incumbent_strategy, self.ex_ante_payoff(entrant_action, incumbent_strategy)))

        return equilibria

    def ex_ante_payoff(self, entrant_action: str, incumbent_strategy: Mapping[str, str]) -> tuple[float, float]:
        """Ожидаемые выигрыши до реализации типа Incumbent."""
        if entrant_action == "Out":
            return (2.0, 0.0)

        normal = self.terminal_payoff("Normal", "In", incumbent_strategy["Normal"])
        tough = self.terminal_payoff("Tough", "In", incumbent_strategy["Tough"])
        return (
            self.p_normal * normal[0] + self.p_tough * tough[0],
            self.p_normal * normal[1] + self.p_tough * tough[1],
        )

    def print_analysis(self, title: str = "Динамическая байесовская игра: выход на рынок") -> None:
        """Печатает подробный расчет динамической байесовской игры."""
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        print(f"P(Normal) = {fmt_number(self.p_normal)}, P(Tough) = {fmt_number(self.p_tough)}")
        print("Порядок ходов: Nature -> Entrant -> Incumbent, если Entrant выбрал In")
        print("Тип Incumbent не наблюдается Entrant, но известен самому Incumbent.")

        print("\nТерминальные выигрыши (Incumbent, Entrant):")
        print("  Out: для любого типа -> (2, 0)")
        print("  Normal, In-Fight -> (-1, -1); Normal, In-Acquiesce -> (1, 1)")
        print("  Tough,  In-Fight -> (1, -1); Tough,  In-Acquiesce -> (-1, 1)")

        br = self.incumbent_best_responses()
        print("\nЛучшие ответы Incumbent после входа Entrant:")
        for incumbent_type, responses in br.items():
            utilities = {
                response: self.incumbent_payoff_after_entry(incumbent_type, response)
                for response in ("Fight", "Acquiesce")
            }
            utilities_s = ", ".join(f"{r}: {fmt_number(u)}" for r, u in utilities.items())
            print(f"  {incumbent_type}: полезности {{{utilities_s}}}; лучшие ответы {responses}")

        incumbent_strategy = {"Normal": br["Normal"][0], "Tough": br["Tough"][0]}
        u_out = self.entrant_expected_payoff("Out", incumbent_strategy)
        u_in = self.entrant_expected_payoff("In", incumbent_strategy)
        print("\nОжидаемый выигрыш Entrant при последовательной стратегии Incumbent:")
        print(f"  стратегия Incumbent: Normal->{incumbent_strategy['Normal']}, Tough->{incumbent_strategy['Tough']}")
        print(f"  U_Entrant(Out) = {fmt_number(u_out)}")
        print(f"  U_Entrant(In)  = {fmt_number(u_in)}")
        print("  Аналитически: U_Entrant(In) = p*1 + (1-p)*(-1) = 2p - 1")
        print("  Поэтому Entrant выбирает In при p >= 1/2, Out при p <= 1/2.")

        equilibria = self.perfect_bayesian_equilibria()
        print(f"\nНайдено чистых совершенных равновесий Байеса — Нэша: {len(equilibria)}")
        for idx, (entrant_action, inc_strategy, payoffs) in enumerate(equilibria, start=1):
            print(
                f"  {idx}) Entrant->{entrant_action}; "
                f"Incumbent: Normal->{inc_strategy['Normal']}, Tough->{inc_strategy['Tough']}; "
                f"Ex ante E[u] = ({fmt_number(payoffs[0])}, {fmt_number(payoffs[1])})"
            )


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


def scan_methodical_p_grid() -> None:
    """Показывает, как меняются BNE игры из методички при разных p=P(v13)."""
    print("\n" + "-" * 92)
    print("Сканирование p=P(v13) для статической игры из вариантов 13 и 14")

    for p in [0.01, 0.25, 0.50, 0.75, 0.99]:
        game = build_methodical_variants_13_14_game(p_variant13=p)
        equilibria = game.bayes_nash_equilibria()
        print(f"p={p:.2f}: {len(equilibria)} BNE")
        for equilibrium in equilibria:
            payoffs = tuple(fmt_number(v) for v in game.ex_ante_expected_payoff(equilibrium))
            print(f"  {game.describe_strategy_profile(equilibrium)}; Ex ante E[u]={payoffs}")


def scan_dynamic_market_entry_grid() -> None:
    """Показывает, как меняется решение динамической игры при разных p=P(Normal)."""
    print("\n" + "-" * 92)
    print("Сканирование p=P(Normal) для динамической игры 'выход на рынок'")

    for p in [0.10, 0.25, 0.50, 0.70, 0.90]:
        game = DynamicMarketEntryGame(p_normal=p)
        equilibria = game.perfect_bayesian_equilibria()
        print(f"p={p:.2f}: {len(equilibria)} PBE")
        for entrant_action, inc_strategy, payoffs in equilibria:
            print(
                f"  Entrant->{entrant_action}; "
                f"Incumbent: Normal->{inc_strategy['Normal']}, Tough->{inc_strategy['Tough']}; "
                f"Ex ante E[u]=({fmt_number(payoffs[0])}, {fmt_number(payoffs[1])})"
            )


def solve_bayesian_task() -> None:
    """Основной сценарий расчета задания 2."""
    methodical_game = build_methodical_variants_13_14_game(p_variant13=0.5)
    methodical_game.print_equilibrium_analysis(
        "Статическая байесовская игра на основе вариантов 13 и 14 из методички, p = 0.5"
    )

    scan_methodical_p_grid()

    dynamic_game = DynamicMarketEntryGame(p_normal=0.7)
    dynamic_game.print_analysis("Динамическая байесовская игра: выход на рынок, p = 0.7")

    scan_dynamic_market_entry_grid()

    # Контрольные примеры из лекции. Их удобно оставить для проверки,
    # но в отчете основным вариантом лучше считать игру из методички выше.
    sheriff = build_sheriff_game(q=0.5)
    sheriff.print_equilibrium_analysis("Контрольный пример: дилемма шерифа, q = 0.5")

    scan_sheriff_q_grid()
    print_slide_threshold_example()


if __name__ == "__main__":
    solve_bayesian_task()
