from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import os
import random

from graphviz import Digraph


# ============================================================
# ЛР №4 (по методичке ЛР №6)
# Позиционные игры. Метод обратной индукции.
# Вариант 13
# ============================================================

VARIANT = 13
TREE_DEPTH = 5
NUM_PLAYERS = 3
STRATEGIES_PER_PLAYER = [2, 4, 2]
PAYOFF_MIN = -50
PAYOFF_MAX = 20
RANDOM_SEED = 56578

OUTPUT_DIR = "data"

GRAPH_FORMAT = "svg"
GRAPH_ENGINE = "dot"

SAVE_INITIAL_GRAPH = True
SAVE_STEP_GRAPHS = True
SAVE_FINAL_GRAPH = True

PRINT_TERMINAL_PAYOFFS = False
PRINT_STEP_SUMMARY = True
PRINT_DETAILED_STEP_INFO = False


# ============================================================
# Структуры
# ============================================================

@dataclass(frozen=True)
class Candidate:
    payoff: Tuple[int, ...]
    path: Tuple[int, ...]


@dataclass
class Node:
    id: int
    depth: int
    player: Optional[int]
    parent_id: Optional[int]
    action_from_parent: Optional[str]
    children_ids: List[int] = field(default_factory=list)
    terminal_payoff: Optional[Tuple[int, ...]] = None
    resolved_candidates: List[Candidate] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.player is None

    def name(self) -> str:
        return f"v{self.id}"


# ============================================================
# Основной класс
# ============================================================

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
        self.rng = random.Random(seed)

    # --------------------------------------------------------
    # Служебные функции
    # --------------------------------------------------------

    def player_for_depth(self, depth: int) -> Optional[int]:
        if depth >= self.depth:
            return None
        return (depth % self.num_players) + 1

    def branching_for_player(self, player: int) -> int:
        return self.strategies_per_player[player - 1]

    def action_labels_for_player(self, player: int) -> List[str]:
        return [chr(ord("a") + i) for i in range(self.branching_for_player(player))]

    def random_payoff(self) -> Tuple[int, ...]:
        return tuple(
            self.rng.randint(self.payoff_min, self.payoff_max)
            for _ in range(self.num_players)
        )

    def root(self) -> Node:
        assert self.root_id is not None
        return self.nodes[self.root_id]

    def nodes_at_depth(self, depth: int) -> List[Node]:
        arr = [n for n in self.nodes.values() if n.depth == depth]
        arr.sort(key=lambda x: x.id)
        return arr

    def leaves(self) -> List[Node]:
        arr = [n for n in self.nodes.values() if n.is_leaf]
        arr.sort(key=lambda x: x.id)
        return arr

    def internal_nodes(self) -> List[Node]:
        arr = [n for n in self.nodes.values() if not n.is_leaf]
        arr.sort(key=lambda x: x.id)
        return arr

    def unique_payoffs_at_node(self, node: Node) -> List[Tuple[int, ...]]:
        return sorted(set(c.payoff for c in node.resolved_candidates))

    def path_to_actions(self, path: Tuple[int, ...]) -> List[Tuple[int, str, int]]:
        result = []
        for i in range(len(path) - 1):
            parent = self.nodes[path[i]]
            child = self.nodes[path[i + 1]]
            result.append((parent.id, child.action_from_parent or "?", child.id))
        return result

    # --------------------------------------------------------
    # Построение дерева
    # --------------------------------------------------------

    def build_full_tree(self) -> None:
        self.nodes.clear()
        self._next_id = 1
        self.root_id = self._build_subtree(0, None, None)

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

        if player is None:
            node.terminal_payoff = self.random_payoff()
            node.resolved_candidates = [Candidate(node.terminal_payoff, (node.id,))]
            return node_id

        for action in self.action_labels_for_player(player):
            child_id = self._build_subtree(depth + 1, node_id, action)
            node.children_ids.append(child_id)

        return node_id

    # --------------------------------------------------------
    # Обратная индукция
    # --------------------------------------------------------

    def _resolve_node(self, node: Node) -> None:
        assert not node.is_leaf
        assert node.player is not None

        mover_idx = node.player - 1
        all_candidates: List[Candidate] = []

        for child_id in node.children_ids:
            child = self.nodes[child_id]
            for cand in child.resolved_candidates:
                all_candidates.append(
                    Candidate(
                        payoff=cand.payoff,
                        path=(node.id,) + cand.path,
                    )
                )

        best_value = max(c.payoff[mover_idx] for c in all_candidates)
        chosen = [c for c in all_candidates if c.payoff[mover_idx] == best_value]

        unique_map = {(c.payoff, c.path): c for c in chosen}
        unique = list(unique_map.values())
        unique.sort(key=lambda c: (c.payoff, c.path))

        node.resolved_candidates = unique

    # --------------------------------------------------------
    # Печать
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
        for d in range(self.depth):
            p = self.player_for_depth(d)
            print(f"  Глубина {d}: ходит игрок {p}, число альтернатив = {self.branching_for_player(p)}")
        print(f"  Глубина {self.depth}: листья, выбор никто не делает")
        print()
        print(f"Всего вершин: {len(self.nodes)}")
        print(f"Внутренних вершин: {len(self.internal_nodes())}")
        print(f"Листьев: {len(self.leaves())}")
        print("=" * 100)

    def print_terminal_payoffs(self) -> None:
        print("\nТЕРМИНАЛЬНЫЕ ВЕРШИНЫ И ИХ ВЫИГРЫШИ")
        print("-" * 100)
        for node in self.leaves():
            print(f"{node.name():>4}  depth={node.depth}  payoff={node.terminal_payoff}")

    def print_step_summary(self, current_depth: int) -> None:
        player = self.player_for_depth(current_depth)
        print()
        print("=" * 100)
        print(f"УРОВЕНЬ {current_depth} — ходит игрок {player}")
        print("=" * 100)
        for node in self.nodes_at_depth(current_depth):
            payoffs = self.unique_payoffs_at_node(node)
            if len(payoffs) == 1:
                print(f"{node.name():>4}: значение {payoffs[0]}")
            else:
                print(f"{node.name():>4}: несколько равнозначных исходов {payoffs}")

    def print_detailed_step_info(self, current_depth: int) -> None:
        player = self.player_for_depth(current_depth)
        print()
        print("=" * 100)
        print(f"ИТЕРАЦИЯ ОБРАТНОЙ ИНДУКЦИИ: ОБРАБАТЫВАЕМ УРОВЕНЬ {current_depth}")
        print(f"На этом уровне ходит игрок {player}")
        print("=" * 100)

        for node in self.nodes_at_depth(current_depth):
            assert node.player is not None
            mover_idx = node.player - 1

            print(f"\nВершина {node.name()} (глубина {node.depth}, ходит игрок {node.player})")
            print("Дочерние вершины и доступные результаты:")

            rows = []
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                for cand in child.resolved_candidates:
                    rows.append((child.id, child.action_from_parent or "?", cand.payoff, cand.payoff[mover_idx]))

            rows.sort(key=lambda x: (x[0], x[2], x[1]))
            for child_id, action, payoff, mover_value in rows:
                print(
                    f"  -> {node.name()} --{action}--> v{child_id}: "
                    f"payoff={payoff}, компонент игрока {node.player} = {mover_value}"
                )

            best_value = max(x[3] for x in rows)
            print(f"Лучшее значение для игрока {node.player}: {best_value}")
            print("Оставляем в узле:")
            for cand in node.resolved_candidates:
                print(f"  payoff={cand.payoff}, path={list(cand.path)}")

    def print_final_solutions(self) -> None:
        root = self.root()

        print()
        print("=" * 100)
        print("ИТОГОВОЕ РЕШЕНИЕ ИГРЫ")
        print("=" * 100)

        print("Значение игры в корне:")
        for payoff in self.unique_payoffs_at_node(root):
            print(f"  {payoff}")

        print()
        print(f"Количество оптимальных путей: {len(root.resolved_candidates)}")
        print()

        for idx, cand in enumerate(root.resolved_candidates, start=1):
            print(f"Оптимальный путь #{idx}")
            print(f"  Выигрыши: {cand.payoff}")
            print(f"  Путь по вершинам: {list(cand.path)}")
            for u, act, v in self.path_to_actions(cand.path):
                print(f"    v{u} --{act}--> v{v}")
            print()

    # --------------------------------------------------------
    # Graphviz helpers
    # --------------------------------------------------------

    def _make_graph(self, title: str) -> Digraph:
        g = Digraph(format=GRAPH_FORMAT, engine=GRAPH_ENGINE)
        g.attr(
            rankdir="LR",
            splines="polyline",
            overlap="false",
            nodesep="0.6",
            ranksep="0.9",
            pad="0.2",
            margin="0.05",
        )
        g.attr(label=title, labelloc="t", fontsize="18", fontname="Arial")
        g.attr("node", fontname="Arial", fontsize="10")
        g.attr("edge", fontname="Arial", fontsize="9")
        return g

    def _node_label_structure(self, node: Node) -> str:
        return f"{node.name()}\\nP{node.player}"

    def _node_label_value(self, node: Node) -> str:
        if node.is_leaf:
            return f"{node.name()}\\n{node.terminal_payoff}"

        vals = self.unique_payoffs_at_node(node)
        if len(vals) == 1:
            return f"{node.name()}\\nP{node.player}\\n{vals[0]}"
        if len(vals) > 1:
            return f"{node.name()}\\nP{node.player}\\n{len(vals)} исхода"
        return f"{node.name()}\\nP{node.player}"

    # --------------------------------------------------------
    # Граф 1: только структура внутренних уровней
    # --------------------------------------------------------

    def render_initial_graph(self, filename_no_ext: str) -> None:
        g = self._make_graph("Исходная структура дерева")

        internal_nodes = self.internal_nodes()

        for node in internal_nodes:
            g.node(
                node.name(),
                label=self._node_label_structure(node),
                shape="circle",
                style="filled",
                fillcolor="#F3F4F6",
                width="0.6",
            )

        for node in internal_nodes:
            for child_id in node.children_ids:
                child = self.nodes[child_id]
                if child.is_leaf:
                    continue
                g.edge(node.name(), child.name(), label=child.action_from_parent or "", color="#9CA3AF")

        g.render(os.path.join(OUTPUT_DIR, filename_no_ext), cleanup=True)

    # --------------------------------------------------------
    # Граф 2: локальный граф текущего уровня
    # --------------------------------------------------------

    def render_local_step_graph(self, current_depth: int, filename_no_ext: str) -> None:
        """
        Строим маленький граф только для текущего уровня.
        Для каждой вершины текущего уровня:
        - сама вершина;
        - её дети;
        - у ребёнка показываем уже вычисленное значение или терминальный выигрыш.
        """
        g = self._make_graph(f"Локальный граф для уровня {current_depth} (ходит игрок {self.player_for_depth(current_depth)})")

        current_nodes = self.nodes_at_depth(current_depth)

        # Фиктивный корень для выравнивания группы вершин
        g.node("LEVEL_ROOT", label=f"Уровень {current_depth}", shape="plaintext")

        for node in current_nodes:
            node_name = node.name()

            g.node(
                node_name,
                label=self._node_label_value(node),
                shape="circle",
                style="filled",
                fillcolor="#D1FAE5",
                penwidth="2",
                width="0.7",
            )
            g.edge("LEVEL_ROOT", node_name, color="#666666")

            for child_id in node.children_ids:
                child = self.nodes[child_id]
                child_name = child.name()

                if child.is_leaf:
                    g.node(
                        child_name,
                        label=self._node_label_value(child),
                        shape="box",
                        style="filled",
                        fillcolor="#FDE68A",
                    )
                else:
                    # ребёнок уже должен иметь вычисленное значение на более глубоком уровне
                    g.node(
                        child_name,
                        label=self._node_label_value(child),
                        shape="circle",
                        style="filled",
                        fillcolor="#EEF2FF",
                    )

                g.edge(node_name, child_name, label=child.action_from_parent or "")

        g.render(os.path.join(OUTPUT_DIR, filename_no_ext), cleanup=True)

    # --------------------------------------------------------
    # Граф 3: только финальные оптимальные пути
    # --------------------------------------------------------

    def render_final_graph(self, filename_no_ext: str) -> None:
        g = self._make_graph("Финальное дерево. Все оптимальные пути")

        root = self.root()
        final_paths = [cand.path for cand in root.resolved_candidates]

        colors = [
            "red", "blue", "green", "orange", "purple",
            "brown", "magenta", "cyan", "darkgreen", "darkgoldenrod"
        ]

        node_ids: Set[int] = set()
        edge_to_color: Dict[Tuple[int, int], str] = {}

        for i, path in enumerate(final_paths):
            color = colors[i % len(colors)]
            node_ids.update(path)
            for j in range(len(path) - 1):
                edge_to_color[(path[j], path[j + 1])] = color

        for nid in sorted(node_ids):
            node = self.nodes[nid]
            if node.is_leaf:
                g.node(
                    node.name(),
                    label=self._node_label_value(node),
                    shape="box",
                    style="filled",
                    fillcolor="#FDE68A",
                    penwidth="2",
                )
            else:
                g.node(
                    node.name(),
                    label=self._node_label_value(node),
                    shape="circle",
                    style="filled",
                    fillcolor="#D1FAE5",
                    penwidth="2" if nid == self.root_id else "1",
                    width="0.75",
                )

        for (u, v), color in edge_to_color.items():
            child = self.nodes[v]
            g.edge(
                self.nodes[u].name(),
                child.name(),
                label=child.action_from_parent or "",
                color=color,
                penwidth="3",
            )

        g.render(os.path.join(OUTPUT_DIR, filename_no_ext), cleanup=True)


# ============================================================
# main
# ============================================================

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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

    game.build_full_tree()
    game.print_summary()

    if PRINT_TERMINAL_PAYOFFS:
        game.print_terminal_payoffs()

    if SAVE_INITIAL_GRAPH:
        game.render_initial_graph("step_00_initial_tree")

    for current_depth in range(game.depth - 1, -1, -1):
        for node in game.nodes_at_depth(current_depth):
            game._resolve_node(node)

        if PRINT_STEP_SUMMARY:
            game.print_step_summary(current_depth)

        if PRINT_DETAILED_STEP_INFO:
            game.print_detailed_step_info(current_depth)

        if SAVE_STEP_GRAPHS:
            game.render_local_step_graph(
                current_depth=current_depth,
                filename_no_ext=f"step_{game.depth - current_depth:02d}_after_depth_{current_depth}",
            )

    game.print_final_solutions()

    if SAVE_FINAL_GRAPH:
        game.render_final_graph("step_final_optimal_paths")

    print("=" * 100)
    print(f"Все изображения сохранены в папку: {OUTPUT_DIR}")
    print(f"Формат графов: .{GRAPH_FORMAT}")
    print("=" * 100)


if __name__ == "__main__":
    main()