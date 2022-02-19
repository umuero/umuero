import math
from functools import reduce
from collections import Counter
from collections import defaultdict

try:
    from queue import PriorityQueue
except:
    from Queue import PriorityQueue

import fractions


def solution1(x, y):
    x_c = Counter(x)
    y_c = Counter(y)
    if x_c - y_c:
        return (x_c - y_c).most_common()[0][0]
    if y_c - x_c:
        return (y_c - x_c).most_common()[0][0]
    return


def solution21(n, b):
    k = len(n)
    S = dict()
    ctr = 0
    while n not in S:
        S[n] = ctr
        ctr += 1
        x = "".join(sorted(n, reverse=True))
        y = x[::-1]
        z = str_base(int(x, b) - int(y, b), b)
        n = "0" * (k - len(z)) + z
    return ctr - S[n]


def str_base(number, base):
    (d, m) = divmod(number, base)
    if d > 0:
        return str_base(d, base) + str(m)
    return str(m)


def solution22(xs):
    p_max = None
    p_min_negative = None
    pos_count = 0
    zero_count = 0
    neg_count = 0
    for x in xs:
        if x != 0:
            if p_max is None:
                p_max = x
            else:
                p_max *= x
        else:
            zero_count += 1
        if x > 0:
            pos_count += 1
        if x < 0:
            neg_count += 1
            if p_min_negative is None or p_min_negative < x:
                p_min_negative = x
    if p_max is None:
        return "0"
    if p_max < 0:
        if pos_count > 0 or neg_count > 1:
            return str(p_max / p_min_negative)
        if zero_count > 0:
            return "0"
        return str(p_max)
    return str(p_max)


def solution31(n):
    binary = str_to_binary(n)
    ctr = 0
    while binary != ["1"]:
        if binary[-1] == "0":
            binary.pop()
            ctr += 1
        elif "".join(binary[-2:]) == "11" and len(binary) != 2:
            for i in range(len(binary) - 1, -1, -1):
                if binary[i] == "1":
                    binary[i] = "0"
                else:
                    binary[i] = "1"
                    break
            if i == 0:
                binary.insert(0, "1")
            ctr += 1
        else:
            binary[-1] = "0"
            ctr += 1
    return ctr


def str_to_binary(n):
    l = int(n)
    ret = ""
    while l:
        if l % 2 == 0:
            ret += "0"
        else:
            ret += "1"
        l = l // 2
    return [i for i in ret[::-1]]


def solution32(l):
    c = Counter()
    res = Counter()
    for i in l:
        items = list(c.items())
        for k, v in items:
            if i % k[-1] == 0:
                if len(k) == 2:
                    res[k + (i,)] += v
                else:
                    c[k + (i,)] += v
        c[(i,)] += 1
    return sum(res.values())


def solution33(m):
    import numpy as np

    terminal = []
    transient = []
    coef = []
    path = dict()
    for rctr, row in enumerate(m):
        if sum(row) == 0:
            terminal.append(rctr)
        else:
            transient.append(rctr)
            coef.append(sum(row))
            for cctr, col in enumerate(row):
                path[(rctr, cctr)] = col * 1.0 / coef[-1]

    M = len(transient)
    N = len(terminal)

    if 0 in terminal:
        return [1] + [0] * (N - 1) + [1]

    Q = [[0] * M for i in range(M)]
    R = [[0] * N for i in range(M)]

    for rctr, rval in enumerate(transient):
        for cctr, cval in enumerate(transient):
            Q[rctr][cctr] = path[(rval, cval)]
        for tctr, tval in enumerate(terminal):
            R[rctr][tctr] = path[(rval, tval)]
    Q = np.matrix(Q)
    R = np.matrix(R)
    FQ = np.linalg.inv(np.identity(M) - Q)
    ST1 = FQ[0] * R * np.prod(coef) / np.linalg.det(FQ)
    ST1 = ST1.round().astype(int).tolist()[0]
    gcd = np.gcd.reduce(ST1)
    res = [int(i / gcd) for i in ST1]
    res.append(sum(res))
    return res


"""
def matrix_inverse(arr):
    M = len(arr)
    ret = [[0] * M for i in range(M)]
    for x in range(M):
        for y in range(M):
            mult = 1 if (x + y) % 2 == 0 else -1
            ret[y][x] = mult * matrix_det(matrix_slice(arr, x, y))
    # don't divide by determinant
    return ret


def matrix_det(arr):
    d = len(arr)
    if d == 1:
        return arr[0][0]
    if d == 2:
        return arr[0][0] * arr[1][1] - arr[1][0] * arr[0][1]
    mult = 1
    sum = 0
    for i in range(d):
        print(arr, i, d)
        sum += mult * arr[0][i] * matrix_det(matrix_slice(arr, 0, i))
        mult *= -1


def matrix_slice(arr, r, c):
    return [[cell for col, cell in enumerate(row) if col != c] for rc, row in enumerate(arr) if rc != r]


def gcd(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    return x
"""

# from Queue import PriorityQueue


def solution41(times, times_limit):
    S = dict()  # state - time
    Q = PriorityQueue()  # -timeLeft, x, collected
    Q.put((-times_limit, 0, ()))
    minCost = min([min(t) for t in times])
    N = len(times[0])
    while not Q.empty():
        time_left, pos, bunnyIds = Q.get()
        time_left *= -1
        if 0 < pos < N - 1 and pos - 1 not in bunnyIds:
            bunnyIds = tuple(sorted(bunnyIds + (pos - 1,)))
        if (pos, bunnyIds) in S:
            if S[(pos, bunnyIds)] < time_left:
                # infinite positive loop
                return list(range(N - 2))
            continue
        S[(pos, bunnyIds)] = time_left
        if len(bunnyIds) == N - 1 and pos != N - 1:
            continue
        for pctr, cost in enumerate(times[pos]):
            if pctr == pos:
                continue
            if time_left - cost >= minCost:
                Q.put((cost - time_left, pctr, bunnyIds))
    ret = ()
    for k in S.keys():
        if k[0] == N - 1 and S[k] >= 0:
            if len(ret) < len(k[1]):
                ret = k[1]
            elif len(ret) == len(k[1]):
                if ret > k[1]:
                    ret = k[1]
    return list(ret)


def gcd(x, y):
    # testing in py3, solution @py2
    while y != 0:
        (x, y) = (y, x % y)
    if x < 0:
        x *= -1
    return x


def reflect2(dimensions, distance, pos, start):
    vectors = dict()  # gcd(posX, posY) ->  mult
    d2 = distance ** 2

    for dx in range(-(distance // dimensions[0]) - 1, distance // dimensions[0] + 2):
        if dx % 2 == 0:
            x = dimensions[0] * dx + pos[0]
        else:
            x = dimensions[0] * (dx + 1) - pos[0]
        for dy in range(-(distance // dimensions[1]) - 1, distance // dimensions[1] + 2):
            if dy % 2 == 0:
                y = dimensions[1] * dy + pos[1]
            else:
                y = dimensions[1] * (dy + 1) - pos[1]

            shot_x = x - start[0]
            shot_y = y - start[1]
            if shot_x ** 2 + shot_y ** 2 > d2:
                continue
            g = gcd(shot_x, shot_y)
            if g != 0:
                shot = (shot_x // g, shot_y // g)
                if shot in vectors:
                    vectors[shot] = min(g, vectors[shot])
                else:
                    vectors[shot] = g
    return vectors


def solution42(dimensions, your_position, trainer_position, distance):
    opp = reflect2(dimensions, distance, trainer_position, your_position)
    me = reflect2(dimensions, distance, your_position, your_position)

    hit_self = 0
    for shot in opp.keys():
        if shot in me and me[shot] < opp[shot]:
            hit_self += 1
    return len(opp) - hit_self


# print("==s1")
# print(solution1([13, 5, 6, 2, 5], [5, 2, 5, 13]))
# # 6
# print(solution1([14, 27, 1, 4, 2, 50, 3, 1], [2, 4, -4, 3, 1, 1, 14, 27, 50]))
# # -4

# print("==s2.1")
# print(solution21('1211', 10))
# # 1
# print(solution21('210022', 3))
# # 3
# print("==s2.2")
# print(solution22([2, 0, 2, 2, 0]))
# print(solution22([-2, -3, 4, -5]))
# print(solution22([2, -3, 1, 0, -5]))
# print(solution22([-3, 0]))  # -> 0
# print(solution22([-3]))  # -> -3
# print(solution22([-3, -2, -4, 0]))  # -> 12

# print("==s3.1")
# print(solution31('15'))
# # 5
# print(solution31('4'))
# # 2
# print("==s3.2")
# print(solution32([1, 1, 1]))
# # 1
# print(solution32([1, 2, 3, 4, 5, 6]))
# # 3
# print(solution32(range(1, 10)))
# print("==s3.3")
# print(solution331([
#     [0, 2, 1, 0, 0],
#     [0, 0, 0, 3, 4],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]]))
# #    [7, 6, 8, 21]

# print(solution331([
#     [0, 1, 0, 0, 0, 1],
#     [4, 0, 0, 3, 2, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0]]))
# #    [0, 3, 2, 9, 14]

# print(solution331([
#     [8, 1, 1, 0],
#     [4, 4, 1, 1],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
# ]))
# #    [7, 1, 8]

# print("===", solution41([
#     [0, 2, 2, 2, -1],
#     [9, 0, 2, 2, -1],
#     [9, 3, 0, 2, -1],
#     [9, 3, 2, 0, -1],
#     [9, 3, 2, 2, 0]], 0))
# #    [1, 2]

# print(solution41([
#     [0, 1, 1, 1, 1],
#     [1, 0, 1, 1, 1],
#     [1, 1, 0, 1, 1],
#     [1, 1, 1, 0, 1],
#     [1, 1, 1, 1, 0]], 900))
# #    [0, 1]

# solution42([3, 2], [1, 1], [2, 1], 4)
# #    7
# solution42([300, 275], [150, 150], [185, 100], 500)
# #    9
# print(solution42([10, 10], [4, 4], [3, 3], 100))
# # 291
# print(solution42([10, 10], [4, 4], [3, 3], 5000))
# # 739323


# def solution5v1(g):
#     S = dict()  #  x, y -> (down, me, right): count;  -> (FFF count, FFT count, FTF count, ... )
#     X = len(g)
#     if X == 0:
#         return 0
#     Y = len(g[0])
#     rules = {
#         True: {
#             (0, 0): [(0, 1), (1, 0)],
#             (0, 1): [(0, 0)],
#             (1, 0): [(0, 0)],
#             (1, 1): [],
#         },
#         False: {
#             (0, 0): [(0, 0), (1, 1)],  # 000 001
#             (0, 1): [(0, 1), (1, 0), (1, 1)],  # 011 010 011 011 - 2 tane 1
#             (1, 0): [(0, 1), (1, 0), (1, 1)],  # 101 100 101 101 - 2 tane 1
#             (1, 1): [(0, 0), (0, 1), (1, 0), (1, 1)],  # 110 111 110 111 - 110 111 2şer tane 0-1
#         },
#     }
#     for x in range(X - 1, -1, -1):
#         for y in range(Y - 1, -1, -1):
#             S[x, y] = defaultdict(int)
#             base = S.get((x, y + 1))
#             extra = S.get(
#                 (x + 1, y),
#                 {(0, 0, 0): 1, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1, (1, 0, 0): 0, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 0},
#             )
#             diag = S.get(
#                 (x + 1, y + 1),
#                 {(0, 0, 0): 1, (0, 0, 1): 0, (0, 1, 0): 1, (0, 1, 1): 0, (1, 0, 0): 0, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 0},
#             )
#             # print(x, y, base if y != Y - 1 else extra)
#             for rl, valids in rules[g[x][y]].items():
#                 for pr in valids:
#                     extraC = extra[(0, rl[1], pr[1])] + extra[(1, rl[1], pr[1])]
#                     diagC = diag[(0, pr[1], 0)] + diag[(0, pr[1], 1)] + diag[(1, pr[1], 0)] + diag[(1, pr[1], 1)]
#                     if y != Y - 1:
#                         baseC = base[(pr[1], pr[0], 0)] + base[(pr[1], pr[0], 1)]
#                         if baseC and extraC:
#                             if baseC == diagC or extraC == diagC:
#                                 S[x, y][rl + (pr[0],)] += (baseC * extraC) // diagC
#                                 # S[x, y][rl + (pr[0],)] += baseC + extraC - diagC
#                             else:
#                                 S[x, y][rl + (pr[0],)] += baseC + extraC
#                         print((x, y), rl, pr, baseC, extraC, diagC)
#                     else:
#                         # vertical & start
#                         S[x, y][rl + (pr[0],)] += extraC
#                         # print((x, y), rl, pr, S[x, y])
#             print(x, y, "=", sum(S[x, y].values()), S[x, y])

#     return sum(S[0, 0].values())


def solution5(g):
    return sol5_rec(g, 0, 0, [[True] * (len(g[0]) + 1) for i in range(len(g) + 1)], {}, [])


def sol5_rec(state, a=0, b=0, past=0, S=0, history=0):
    if b == len(state[0]) + 1:
        return 1

    res = 0
    index = ((a, b), tuple(history[-(len(state) + 2) :]))
    if index in S:
        return S[index]

    for cell in [True, False]:
        if (not a or not b) or ((past[a][b - 1] + past[a - 1][b] + past[a - 1][b - 1] + cell) == 1) == state[a - 1][b - 1]:
            history.append(cell)
            past[a][b] = cell
            res += sol5_rec(
                state,
                (a + 1) % (len(state) + 1),
                b + (a + 1) // (len(state) + 1),
                past,
                S,
                history,
            )
            history.pop()

    S[index] = res
    # print(index, res)
    return res


print("============", 20, solution5([[False, False], [True, True]]))
print("============", 10, solution5([[False, True]]))
print("============", 12, solution5([[True, False], [False, True]]))
print("============", 8, solution5([[True, False, True]]))
print("============", 16, solution5([[False, True, False], [True, False, True]]))

print("============", 4, solution5([[True, False, True], [False, True, False], [True, False, True]]))
print(
    "============",
    254,
    solution5(
        [
            [True, False, True, False, False, True, True, True],
            [True, False, True, False, False, False, True, False],
            [True, True, True, False, False, False, True, False],
            [True, False, True, False, False, False, True, False],
            [True, False, True, False, False, True, True, True],
        ]
    ),
)
print(
    "==========",
    11567,
    solution5(
        [
            [True, True, False, True, False, True, False, True, True, False],
            [True, True, False, False, False, False, True, True, True, False],
            [True, True, False, False, False, False, False, False, False, True],
            [False, True, False, False, False, False, True, True, False, False],
        ]
    ),
)
# #    11567
