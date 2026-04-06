from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# ЛР №4 (по методичке это ЛР №6)
# Позиционные игры. Метод обратной индукции.
# Вариант 13:
#   глубина дерева = 5
#   игроков = 3
#   стратегии игроков = [2, 4, 2]
#   диапазон выигрышей = [-50, 20]
# ============================================================

VARIANT = 13
TREE_DEPTH = 5
NUM_PLAYERS = 3
STRATEGIES_PER_PLAYER = [2, 4, 2]
PAYOFF_MIN = -50
PAYOFF_MAX = 20

RANDOM_SEED = 13

OUTPUT_DIR = "data"
SAVE_STEP_PLOTS = True
SHOW_PLOTS = False

FIGSIZE = (22, 12)
NODE_FONT_SIZE = 8
EDGE_FONT_SIZE = 9
BASE_NODE_SIZE = 900
LEAF_NODE_SIZE = 1100
PATH_LINE_WIDTH = 3.5


# ------------------------------------------------------------
# Вспомогательные структуры
# ------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """Один возможный оптимальный результат в вершине:
    payoff  - вектор выигрышей
    path    - последовательность id вершин от текущей вершины до листа
    """
    payoff: Tuple[int, ...]
    path: Tuple[int, ...]


@dataclass
class Node:
    id: int
    depth: int
    player: Optional[int]                   # None для листа
    parent_id: Optional[int]
    action_from_parent: Optional[str]
    children_ids: List[int] = field(default_factory=list)
    terminal_payoff: Optional[Tuple[int, ...]] = None

    # Заполняется при обратной индукции
    resolved_candidates: List[Candidate] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.player is None

    def short_name(self) -> str:
        return f"v{self.id}"


# ------------------------------------------------------------
# Генерация дерева
# ------------------------------------------------------------

class PositionalGame:
    def __init__(
        self,
        depth: int,
        num_players: int,
        strategies_per_player: List[int],
        payoff_min: int,
        payoff_max: int,
        seed: int,
    ) -> None:
        self.depth = depth
        self.num_players = num_players
        self.strategies_per_player = strategies_per_player
        self.payoff_min = payoff_min
        self.payoff_max = payoff_max
        self.seed = seed

        self.nodes: Dict[int, Node] = {}
        self.root_id: Optional[int] = None
        self._next_id = 1

        import random
        self.rng = random.Random(seed)

    # ---------- общие функции ----------

    def player_for_depth(self, depth: int) -> Optional[int]:
        """На глубинах 0..depth-1 ходят игроки циклически 1,2,3,1,2,...
        На глубине == self.depth находятся листья.
        """
        if depth >= self.depth:
            return None
        return (depth % self.num_players) + 1

    def branching_for_player(self, player: int) -> int:
        return self.strategies_per_player[player - 1]

    def action_labels_for_player(self, player: int) -> List[str]:
        k = self.branching_for_player(player)
        # Для k<=26 используем a,b,c...; иначе s1,s2,...
        if k <= 26:
            return [chr(ord("a") + i) for i in range(k)]
        return [f"s{i + 1}" for i in range(k)]

    def random_payoff(self) -> Tuple[int, ...]:
        return tuple(
            self.rng.randint(self.payoff_min, self.payoff_max)
            for _ in range(self.num_players)
        )

    # ---------- построение дерева ----------

    def build_full_tree(self) -> None:
        self.nodes.clear()
        self._next_id = 1
        self.root_id = self._build_subtree(depth=0, parent_id=None, action_from_parent=None)

    def _build_subtree(
        self,
        depth: int,
        parent_id: Optional[int],
        action_from_parent: Optional[str],
    ) -> int:
        node_id = self._next_id
        self._next_id += 1

        player = self.player_for_depth(depth)

        node = Node(
            id=node_id,
            depth=depth,
            player=player,
            parent_id=parent_id,
            action_from_parent=action_from_parent,
        )
        self.nodes[node_id] = node

        # Лист
        if player is None:
            node.terminal_payoff = self.random_payoff()
            node.resolved_candidates = [
                Candidate(payoff=node.terminal_payoff, path=(node.id,))
            ]
            return node_id

        # Внутренняя вершина
        action_labels = self.action_labels_for_player(player)
        for action_label in action_labels:
            child_id = self._build_subtree(
                depth=depth + 1,
                parent_id=node_id,
                action_from_parent=action_label,
            )
            node.children_ids.append(child_id)

        return node_id

    # --------------------------------------------------------
    # Обратная индукция
    # --------------------------------------------------------

    def solve_by_backward_induction(self) -> None:
        """Выполняет обратную индукцию.
        На каждом внутреннем узле:
          - смотрим все результаты детей;
          - выбираем те, где максимум выигрыша у игрока, делающего ход;
          - при равенстве оставляем все.
        """
        # листья уже инициализированы
        for current_depth in range(self.depth - 1, -1, -1):
            for node in self.nodes_at_depth(current_depth):
                self._resolve_node(node)

    def _resolve_node(self, node: Node) -> None:
        assert not node.is_leaf
        assert node.player is not None

        mover_index = node.player - 1

        all_child_candidates: List[Candidate] = []
        for child_id in node.children_ids:
            child = self.nodes[child_id]
            for cand in child.resolved_candidates:
                # путь от текущего узла до листа
                extended = Candidate(
                    payoff=cand.payoff,
                    path=(node.id,) + cand.path,
                )
                all_child_candidates.append(extended)

        # Максимум по компоненте текущего игрока
        best_value = max(c.payoff[mover_index] for c in all_child_candidates)

        chosen = [c for c in all_child_candidates if c.payoff[mover_index] == best_value]

        # Убираем дубли одинаковых (payoff, path)
        unique = list({(c.payoff, c.path): c for c in chosen}.values())

        # Для красивого и стабильного вывода отсортируем
        unique.sort(key=lambda c: (c.payoff, c.path))

        node.resolved_candidates = unique

    # --------------------------------------------------------
    # Служебные функции
    # --------------------------------------------------------

    def nodes_at_depth(self, depth: int) -> List[Node]:
        result = [n for n in self.nodes.values() if n.depth == depth]
        result.sort(key=lambda x: x.id)
        return result

    def leaves(self) -> List[Node]:
        result = [n for n in self.nodes.values() if n.is_leaf]
        result.sort(key=lambda x: x.id)
        return result

    def internal_nodes(self) -> List[Node]:
        result = [n for n in self.nodes.values() if not n.is_leaf]
        result.sort(key=lambda x: x.id)
        return result

    def root(self) -> Node:
        assert self.root_id is not None
        return self.nodes[self.root_id]

    def path_to_actions(self, path: Tuple[int, ...]) -> List[Tuple[int, str, int]]:
        """Преобразует путь из id вершин в список шагов:
        (from_node_id, action_label, to_node_id)
        """
        result: List[Tuple[int, str, int]] = []
        for i in range(len(path) - 1):
            parent = self.nodes[path[i]]
            child = self.nodes[path[i + 1]]
            result.append((parent.id, child.action_from_parent or "?", child.id))
        return result

    def unique_payoffs_at_node(self, node: Node) -> List[Tuple[int, ...]]:
        payoffs = sorted(set(c.payoff for c in node.resolved_candidates))
        return payoffs

    def depth_player_description(self) -> List[str]:
        lines = []
        for d in range(self.depth):
            player = self.player_for_depth(d)
            assert player is not None
            lines.append(
                f"Глубина {d}: ходит игрок {player}, число альтернатив = {self.branching_for_player(player)}"
            )
        lines.append(f"Глубина {self.depth}: листья, выбор никто не делает")
        return lines

    # --------------------------------------------------------
    # Печать результатов
    # --------------------------------------------------------

    def print_summary(self) -> None:
        print("=" * 100)
        print("ЛР №4 (по методичке ЛР №6) — ПОЗИЦИОННЫЕ ИГРЫ")
        print("=" * 100)
        print(f"Вариант: {VARIANT}")
        print(f"Seed генератора: {self.seed}")
        print()
        print("Параметры варианта:")
        print(f"  Глубина дерева: {self.depth}")
        print(f"  Количество игроков: {self.num_players}")
        print(f"  Количество стратегий игроков: {self.strategies_per_player}")
        print(f"  Диапазон выигрышей: [{self.payoff_min}, {self.payoff_max}]")
        print()
        print("Кто ходит на каждом уровне:")
        for line in self.depth_player_description():
            print(" ", line)
        print()
        print(f"Всего вершин: {len(self.nodes)}")
        print(f"Внутренних вершин: {len(self.internal_nodes())}")
        print(f"Листьев: {len(self.leaves())}")
        print("=" * 100)

    def print_terminal_payoffs(self) -> None:
        print("\nТЕРМИНАЛЬНЫЕ ВЕРШИНЫ И ИХ ВЫИГРЫШИ")
        print("-" * 100)
        for node in self.leaves():
            print(f"{node.short_name():>4}  depth={node.depth}  payoff={node.terminal_payoff}")

    def print_step_results(self, current_depth: int) -> None:
        print()
        print("=" * 100)
        print(f"ИТЕРАЦИЯ ОБРАТНОЙ ИНДУКЦИИ: ОБРАБАТЫВАЕМ УРОВЕНЬ {current_depth}")
        player = self.player_for_depth(current_depth)
        print(f"На этом уровне ходит игрок {player}")
        print("=" * 100)

        for node in self.nodes_at_depth(current_depth):
            assert node.player is not None
            mover_index = node.player - 1

            print(f"\nВершина {node.short_name()} (глубина {node.depth}, ходит игрок {node.player})")
            print("Дочерние вершины и доступные результаты:")

            child_rows: List[Tuple[int, str, Tuple[int, ...], int]] = []
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                for cand in child.resolved_candidates:
                    child_rows.append(
                        (
                            child.id,
                            child.action_from_parent or "?",
                            cand.payoff,
                            cand.payoff[mover_index],
                        )
                    )

            child_rows.sort(key=lambda x: (x[0], x[2], x[1]))
            for child_id, action, payoff, value_for_mover in child_rows:
                print(
                    f"  -> {node.short_name()} --{action}--> v{child_id}: "
                    f"payoff={payoff}, компонент игрока {node.player} = {value_for_mover}"
                )

            best_value = max(x[3] for x in child_rows)
            print(f"Лучшее значение для игрока {node.player}: {best_value}")

            kept = node.resolved_candidates
            print("Оставляем в узле:")
            for cand in kept:
                print(f"  payoff={cand.payoff}, path={list(cand.path)}")

    def print_final_solutions(self) -> None:
        root = self.root()

        print()
        print("=" * 100)
        print("ИТОГОВОЕ РЕШЕНИЕ ИГРЫ")
        print("=" * 100)

        unique_root_payoffs = self.unique_payoffs_at_node(root)
        print("Множество значений в корне:")
        for payoff in unique_root_payoffs:
            print(f"  {payoff}")

        print()
        print(f"Количество оптимальных путей: {len(root.resolved_candidates)}")
        print()

        for idx, cand in enumerate(root.resolved_candidates, start=1):
            print(f"Оптимальный путь #{idx}")
            print(f"  Выигрыши в конце пути: {cand.payoff}")
            print(f"  Путь по вершинам: {list(cand.path)}")

            steps = self.path_to_actions(cand.path)
            for from_id, action, to_id in steps:
                print(f"    v{from_id} --{action}--> v{to_id}")
            print()

    # --------------------------------------------------------
    # Визуализация
    # --------------------------------------------------------

    def compute_positions(self) -> Dict[int, Tuple[float, float]]:
        """Располагаем дерево иерархически.
        x у листьев идёт слева направо.
        x внутренней вершины = среднее x её детей.
        y = -depth.
        """
        positions: Dict[int, Tuple[float, float]] = {}
        leaf_x_counter = 0

        def dfs(node_id: int) -> float:
            nonlocal leaf_x_counter
            node = self.nodes[node_id]

            if node.is_leaf:
                x = leaf_x_counter
                leaf_x_counter += 1
                positions[node_id] = (x, -node.depth)
                return x

            child_xs = [dfs(child_id) for child_id in node.children_ids]
            x = sum(child_xs) / len(child_xs)
            positions[node_id] = (x, -node.depth)
            return x

        dfs(self.root_id)
        return positions

    def node_label(self, node: Node, show_resolved: bool = True) -> str:
        lines = [node.short_name()]

        if node.is_leaf:
            lines.append("Лист")
            lines.append(str(node.terminal_payoff))
            return "\n".join(lines)

        lines.append(f"P{node.player}")

        if show_resolved and node.resolved_candidates:
            payoffs = self.unique_payoffs_at_node(node)
            if len(payoffs) == 1:
                lines.append(str(payoffs[0]))
            else:
                lines.append("{" + ", ".join(str(p) for p in payoffs) + "}")

        return "\n".join(lines)

    def resolved_upto_depth(self, processed_from_depth: int) -> Set[int]:
        """Какие узлы уже получили значения после обработки уровней
        depth-1, depth-2, ..., processed_from_depth.
        """
        result = set()
        for node in self.nodes.values():
            if node.is_leaf:
                result.add(node.id)
            elif node.depth >= processed_from_depth:
                if node.resolved_candidates:
                    result.add(node.id)
        return result

    def save_plot(
        self,
        filename: str,
        title: str,
        colored_paths: Optional[List[Tuple[Tuple[int, ...], str]]] = None,
        show_resolved: bool = True,
        processed_from_depth: Optional[int] = None,
    ) -> None:
        positions = self.compute_positions()
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.set_title(title, fontsize=16)
        ax.axis("off")

        # Сначала рисуем все рёбра
        for node in self.nodes.values():
            for child_id in node.children_ids:
                x1, y1 = positions[node.id]
                x2, y2 = positions[child_id]
                ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1.0, zorder=1)

                child = self.nodes[child_id]
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                ax.text(
                    mx,
                    my + 0.08,
                    child.action_from_parent or "",
                    fontsize=EDGE_FONT_SIZE,
                    ha="center",
                    va="center",
                    color="dimgray",
                )

        # Если нужно, выделяем пути разными цветами
        if colored_paths:
            for path, color in colored_paths:
                for i in range(len(path) - 1):
                    u = path[i]
                    v = path[i + 1]
                    x1, y1 = positions[u]
                    x2, y2 = positions[v]
                    ax.add_patch(
                        FancyArrowPatch(
                            (x1, y1),
                            (x2, y2),
                            arrowstyle="-",
                            mutation_scale=10,
                            linewidth=PATH_LINE_WIDTH,
                            color=color,
                            zorder=2,
                            alpha=0.9,
                        )
                    )

        # Какие вершины считать уже "оценёнными"
        resolved_nodes: Set[int] = set()
        if processed_from_depth is None:
            resolved_nodes = set(node.id for node in self.nodes.values() if node.resolved_candidates)
        else:
            resolved_nodes = self.resolved_upto_depth(processed_from_depth)

        # Затем рисуем вершины
        for node in self.nodes.values():
            x, y = positions[node.id]

            if node.is_leaf:
                facecolor = "#fef3c7"
                size = LEAF_NODE_SIZE
            else:
                if node.id in resolved_nodes:
                    facecolor = "#d1fae5"
                else:
                    facecolor = "#e5e7eb"
                size = BASE_NODE_SIZE

            ax.scatter(
                [x],
                [y],
                s=size,
                c=facecolor,
                edgecolors="black",
                zorder=3,
            )

            ax.text(
                x,
                y,
                self.node_label(node, show_resolved=show_resolved),
                fontsize=NODE_FONT_SIZE,
                ha="center",
                va="center",
                zorder=4,
            )

        # Подписи уровней слева
        max_depth = max(node.depth for node in self.nodes.values())
        min_x = min(x for x, _ in positions.values())
        for d in range(max_depth + 1):
            player = self.player_for_depth(d)
            if player is None:
                label = f"Глубина {d}: листья"
            else:
                label = f"Глубина {d}: игрок {player}"
            ax.text(
                min_x - 3.5,
                -d,
                label,
                fontsize=10,
                ha="left",
                va="center",
                color="black",
            )

        plt.tight_layout()
        full_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(full_path, dpi=180, bbox_inches="tight")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ------------------------------------------------------------
# Основной сценарий
# ------------------------------------------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def color_palette(n: int) -> List[str]:
    base = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "magenta",
        "cyan",
        "olive",
        "teal",
    ]
    if n <= len(base):
        return base[:n]

    # если путей больше, повторим палитру
    result = []
    for i in range(n):
        result.append(base[i % len(base)])
    return result


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    game = PositionalGame(
        depth=TREE_DEPTH,
        num_players=NUM_PLAYERS,
        strategies_per_player=STRATEGIES_PER_PLAYER,
        payoff_min=PAYOFF_MIN,
        payoff_max=PAYOFF_MAX,
        seed=RANDOM_SEED,
    )

    # 1. Построить дерево
    game.build_full_tree()

    # 2. Вывести общую информацию
    game.print_summary()
    game.print_terminal_payoffs()

    # 3. Исходное дерево
    if SAVE_STEP_PLOTS:
        game.save_plot(
            filename="step_00_initial_tree.png",
            title="Исходное дерево игры",
            colored_paths=None,
            show_resolved=False,
            processed_from_depth=None,
        )

    # 4. Обратная индукция по уровням снизу вверх
    for current_depth in range(game.depth - 1, -1, -1):
        for node in game.nodes_at_depth(current_depth):
            game._resolve_node(node)

        game.print_step_results(current_depth)

        if SAVE_STEP_PLOTS:
            game.save_plot(
                filename=f"step_{game.depth - current_depth:02d}_after_depth_{current_depth}.png",
                title=f"После обработки уровня {current_depth}",
                colored_paths=None,
                show_resolved=True,
                processed_from_depth=current_depth,
            )

    # 5. Итог
    game.print_final_solutions()

    # 6. Финальный рисунок с выделением всех оптимальных путей
    root = game.root()
    colors = color_palette(len(root.resolved_candidates))
    colored_paths = [
        (cand.path, colors[i])
        for i, cand in enumerate(root.resolved_candidates)
    ]

    if SAVE_STEP_PLOTS:
        game.save_plot(
            filename="step_final_optimal_paths.png",
            title="Финальное дерево. Все оптимальные пути",
            colored_paths=colored_paths,
            show_resolved=True,
            processed_from_depth=0,
        )

    print("=" * 100)
    print(f"Все изображения сохранены в папку: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()