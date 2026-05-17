import numpy as np


# ==============================
# Таблица 8.1 из методички
# вариант: a, b, c, d, gf, gs
# ==============================

VARIANTS = {
    1:  (1, 2, 1, 5, 1, 2),
    2:  (2, 3, 2, 4, 2, 1),
    3:  (3, 4, 3, 3, 2, 3),
    4:  (4, 5, 4, 2, 4, 6),
    5:  (5, 6, 1, 1, 1, 5),
    6:  (1, 1, 2, 5, 5, 6),
    7:  (2, 2, 3, 4, 3, 1),
    8:  (3, 3, 4, 3, 4, 4),
    9:  (4, 4, 1, 2, 2, 2),
    10: (5, 5, 2, 1, 4, 2),
    11: (1, 6, 3, 5, 6, 6),
    12: (2, 1, 4, 4, 1, 3),
    13: (3, 2, 1, 3, 3, 2),
    14: (4, 3, 2, 2, 3, 3),
    15: (5, 4, 3, 1, 1, 4),
    16: (1, 5, 4, 5, 2, 5),
    17: (2, 6, 1, 4, 6, 5),
    18: (3, 1, 2, 3, 4, 5),
    19: (4, 2, 3, 2, 2, 6),
    20: (5, 3, 4, 1, 4, 1),
}


def format_matrix(matrix, decimals=3):
    """
    Красивый вывод матрицы в стиле методички:
    (
      0.125 0.096 ...
      ...
    )
    """
    rounded = np.round(matrix, decimals)
    lines = []
    for row in rounded:
        line = " ".join(f"{x:7.{decimals}f}" for x in row)
        lines.append(line)
    return "(\n" + "\n".join(lines) + "\n)"


def format_vector(vector, decimals=3):
    rounded = np.round(vector, decimals)
    return "(" + " ".join(f"{x:.{decimals}f}" for x in rounded) + ")"


def generate_stochastic_matrix(n, seed=1):
    """
    Генерация стохастической матрицы доверия.
    Все элементы положительные, сумма каждой строки равна 1.
    """
    rng = np.random.default_rng(seed)

    A = rng.random((n, n))
    A = A / A.sum(axis=1, keepdims=True)

    return A


def get_resulting_trust_matrix(A, eps=1e-12, max_iter=100000):
    """
    Получение результирующей матрицы доверия A^∞.

    Для положительной стохастической матрицы все строки A^∞ одинаковы.
    Эту строку обозначаем r.
    """
    n = A.shape[0]

    r = np.ones(n) / n

    for _ in range(max_iter):
        new_r = r @ A

        if np.max(np.abs(new_r - r)) < eps:
            r = new_r
            break

        r = new_r

    A_inf = np.tile(r, (n, 1))

    return A_inf, r


def choose_agents(n, seed=1, example_style=True):
    """
    Выбор агентов влияния.

    example_style=True:
        выбирает агентов как в примере методички:
        первый игрок: 4, 5, 6
        второй игрок: 1, 2, 3

    example_style=False:
        выбирает случайные непересекающиеся множества агентов.
    """
    if example_style:
        first_agents = np.array([3])
        second_agents = np.array([4, 5])
        return first_agents, second_agents

    rng = np.random.default_rng(seed)

    all_agents = np.arange(n)
    selected = rng.choice(all_agents, size=6, replace=False)

    first_agents = np.sort(selected[:3])
    second_agents = np.sort(selected[3:])

    return first_agents, second_agents


def solve_nash_equilibrium(a, b, c, d, gf, gs, rf, rs):
    """
    Решение игры с непротивоположными интересами.

    Итоговое мнение в стиле примера методички:
        X = rf*u + rs*v

    Функции выигрыша:
        Hf(X) = aX - bX^2
        Hs(X) = cX - dX^2

    Целевые функции:
        Фf(u, v) = aX - bX^2 - gf*u^2/2
        Фs(u, v) = cX - dX^2 - gs*v^2/2

    Равновесие Нэша находится из системы:
        dФf/du = 0
        dФs/dv = 0
    """

    system_matrix = np.array([
        [gf + 2 * b * rf ** 2, 2 * b * rf * rs],
        [2 * d * rf * rs, gs + 2 * d * rs ** 2]
    ])

    right_side = np.array([
        a * rf,
        c * rs
    ])

    u, v = np.linalg.solve(system_matrix, right_side)

    X = rf * u + rs * v

    Hf = a * X - b * X ** 2
    Hs = c * X - d * X ** 2

    Phi_f = Hf - gf * u ** 2 / 2
    Phi_s = Hs - gs * v ** 2 / 2

    Xmax_f = a / (2 * b)
    Xmax_s = c / (2 * d)

    delta_f = abs(Xmax_f - X)
    delta_s = abs(Xmax_s - X)

    if delta_f < delta_s:
        winner = "первый игрок"
    elif delta_s < delta_f:
        winner = "второй игрок"
    else:
        winner = "оба игрока находятся на одинаковом расстоянии от своих точек утопии"

    return {
        "system_matrix": system_matrix,
        "right_side": right_side,
        "u": u,
        "v": v,
        "X": X,
        "Hf": Hf,
        "Hs": Hs,
        "Phi_f": Phi_f,
        "Phi_s": Phi_s,
        "Xmax_f": Xmax_f,
        "Xmax_s": Xmax_s,
        "delta_f": delta_f,
        "delta_s": delta_s,
        "winner": winner
    }


def run_lab6(variant=1, seed=1, example_style=True):
    """
    Выполнение лабораторной работы №6.

    variant:
        номер варианта из таблицы 8.1

    seed:
        фиксирует случайную генерацию матрицы доверия

    example_style:
        True  — вывод и выбор агентов похожи на пример из методички
        False — агенты влияния выбираются случайно
    """

    if variant not in VARIANTS:
        raise ValueError("Такого варианта нет в таблице 8.1")

    n = 10
    a, b, c, d, gf, gs = VARIANTS[variant]

    print("Лабораторная работа № 6.")
    print("Информационное противоборство.")
    print()
    print(f"Вариант № {variant}")
    print()

    # 1. Генерация матрицы доверия
    print("Рассмотрим игру для 10 агентов.")
    print("Сначала сгенерируем матрицу доверия и убедимся, что она стохастическая:")
    print()

    A = generate_stochastic_matrix(n, seed=seed)

    print(format_matrix(A, decimals=3))
    print()

    print("Проверка стохастичности матрицы A:")
    print("Суммы строк матрицы A:")
    print(format_vector(A.sum(axis=1), decimals=3))
    print()

    # 2. Выбор агентов влияния
    first_agents, second_agents = choose_agents(
        n=n,
        seed=seed,
        example_style=example_style
    )

    print("Теперь выберем номера агентов первого и второго игрока.")
    print("Номера агентов не должны пересекаться.")
    print()

    print("Агенты первого игрока:", ", ".join(map(str, first_agents + 1)))
    print("Агенты второго игрока:", ", ".join(map(str, second_agents + 1)))
    print()

    # 3. Результирующая матрица доверия
    print("Рассчитаем результирующую матрицу доверия:")
    print()

    A_inf, r = get_resulting_trust_matrix(A)

    print(format_matrix(A_inf, decimals=3))
    print()

    print("Строка результирующей матрицы r:")
    print(format_vector(r, decimals=6))
    print()

    # 4. Расчет rf и rs
    rf = np.sum(r[first_agents])
    rs = np.sum(r[second_agents])

    print("Найдем rf и rs:")
    print(f"rf = {rf:.3f}")
    print(f"rs = {rs:.3f}")
    print()

    # 5. Параметры варианта
    print(f"Пусть a = {a}, b = {b}, c = {c}, d = {d}, gf = {gf}, gs = {gs}")
    print()

    print("Функции выигрыша имеют вид:")
    print(f"Hf(x) = {a}x - {b}x^2 -> max")
    print(f"Hs(x) = {c}x - {d}x^2 -> max")
    print()

    Xmax_f = a / (2 * b)
    Xmax_s = c / (2 * d)

    print("Оптимальные мнения:")
    print(f"Xmax_f = a / (2b) = {Xmax_f:.3f}")
    print(f"Xmax_s = c / (2d) = {Xmax_s:.3f}")
    print()

    print("Целевая функция первого игрока:")
    print(f"Фf(u, v) = {a}(rf*u + rs*v) - {b}(rf*u + rs*v)^2 - {gf}u^2/2")
    print()

    print("Целевая функция второго игрока:")
    print(f"Фs(u, v) = {c}(rf*u + rs*v) - {d}(rf*u + rs*v)^2 - {gs}v^2/2")
    print()

    print("Подставим значения параметров:")
    print(
        f"Фf(u, v) = {a}({rf:.3f}u + {rs:.3f}v) "
        f"- {b}({rf:.3f}u + {rs:.3f}v)^2 - {gf}u^2/2"
    )
    print(
        f"Фs(u, v) = {c}({rf:.3f}u + {rs:.3f}v) "
        f"- {d}({rf:.3f}u + {rs:.3f}v)^2 - {gs}v^2/2"
    )
    print()

    # 6. Решение равновесия Нэша
    solution = solve_nash_equilibrium(a, b, c, d, gf, gs, rf, rs)

    print("Равновесие Нэша найдем из системы:")
    print("dФf/du = 0")
    print("dФs/dv = 0")
    print()

    print("В матричном виде система имеет вид:")
    print(format_matrix(solution["system_matrix"], decimals=6))
    print("*")
    print("(u v)^T")
    print("=")
    print(format_vector(solution["right_side"], decimals=6))
    print()

    u = solution["u"]
    v = solution["v"]
    X = solution["X"]

    print("Решение системы:")
    print(f"u = {u:.3f}")
    print(f"v = {v:.3f}")
    print()

    print("Теперь найдем итоговое мнение X:")
    print("X = rf*u + rs*v")
    print(f"X = {rf:.3f} * {u:.3f} + {rs:.3f} * {v:.3f}")
    print(f"X = {X:.3f}")
    print()

    print("Значения функций выигрыша:")
    print(f"Hf(X) = {solution['Hf']:.3f}")
    print(f"Hs(X) = {solution['Hs']:.3f}")
    print()

    print("Значения целевых функций:")
    print(f"Фf(u, v) = {solution['Phi_f']:.3f}")
    print(f"Фs(u, v) = {solution['Phi_s']:.3f}")
    print()

    print("При Xmax_f и Xmax_s найдем расстояния:")
    print(f"Δxf = |Xmax_f - X| = |{Xmax_f:.3f} - {X:.3f}| = {solution['delta_f']:.3f}")
    print(f"Δxs = |Xmax_s - X| = |{Xmax_s:.3f} - {X:.3f}| = {solution['delta_s']:.3f}")
    print()

    print(f"Вывод: ближе к своей точке утопии находится {solution['winner']}.")


# ==============================
# Запуск для варианта 1
# ==============================

run_lab6(
    variant=13,
    seed=2026,
    example_style=True
)