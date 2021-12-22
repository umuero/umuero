# https://adventofcode.com/2020
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, product
from functools import reduce
from queue import PriorityQueue
import itertools
import re
import copy
import ast
import math
import argparse
import aocd

Nx = [1, 0, -1, 0]
Ny = [0, 1, 0, -1]
NDx = [1, 0, -1, 0, 1, 1, -1, -1]
NDy = [0, 1, 0, -1, 1, -1, 1, -1]
NMx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
NMy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]


def a1(f):
    arr = [int(i) for i in f.split("\n")]
    last = None
    ctr = 0
    for i in arr:
        if last is None:
            last = i
        if i > last:
            ctr += 1
        last = i
    print("p1:", ctr)

    last = None
    ctr = 0
    for i in range(len(arr) - 2):
        s = sum(arr[i : i + 3])
        if last is None:
            last = s
        if s > last:
            ctr += 1
        last = s
    print("p2:", ctr)


def a2(f):
    arr = [i.split() for i in f.split("\n")]
    hor = 0
    depth = 0
    aim = 0
    for cmd, ctr in arr:
        if cmd == "forward":
            hor += int(ctr)
            depth += aim * int(ctr)
        if cmd == "down":
            aim += int(ctr)
        if cmd == "up":
            aim -= int(ctr)
    print(hor * depth)


def a3(f):
    arr = [i.strip() for i in f.split("\n")]
    gamma = ""
    epsilon = ""
    nums = defaultdict(Counter)
    N = len(arr[0])
    for num in arr:
        for ctr, n in enumerate(num):
            nums[ctr][n] += 1

    for i in range(N):
        gamma += nums[i].most_common()[0][0]
        epsilon += nums[i].most_common()[-1][0]

    print(int(gamma, 2) * int(epsilon, 2))

    ox = ""
    co = ""
    ox_C = Counter()
    co_C = Counter()
    for i in range(N):
        for num in arr:
            if num.startswith(ox):
                ox_C[num[i]] += 1
            if num.startswith(co):
                co_C[num[i]] += 1

        ox += sorted(ox_C.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        ox_C = Counter()
        co += sorted(co_C.items(), key=lambda x: (x[1], x[0]), reverse=False)[0][0]
        co_C = Counter()

    print(int(ox, 2) * int(co, 2))


def a4(f):
    arr = [i.strip().replace("  ", " ").split(" ") for i in f.split("\n")]
    nums = [int(i) for i in arr[0][0].split(",")]
    Bc = []
    Br = []
    for ctr in range(1, len(arr), 6):
        Br.append([[int(i) for i in l] for l in arr[ctr + 1 : ctr + 6]])
        Bc.append(
            [[Br[-1][j][i] for j in range(len(Br[-1]))] for i in range(len(Br[-1][0]))]
        )

    lastScore = None
    for num in nums:
        for bc, br in zip(Bc, Br):
            if len(bc) == 0:
                continue
            finish = False
            for row in bc:
                if num in row:
                    row.pop(row.index(num))
                if len(row) == 0:
                    finish = True
            for row in br:
                if num in row:
                    row.pop(row.index(num))
                if len(row) == 0:
                    finish = True
            if finish:
                if lastScore is None:
                    print("a:", sum([sum(i) for i in bc]) * num)
                lastScore = sum([sum(i) for i in bc]) * num
                bc.clear()
    print("b:", lastScore)


def a5(f):
    arr = [
        [int(k) for n in i.strip().split(" -> ") for k in n.split(",")]
        for i in f.split("\n")
    ]
    m = defaultdict(int)

    for cord in arr:
        xMin = min(cord[0], cord[2])
        xMax = max(cord[0], cord[2])
        yMin = min(cord[1], cord[3])
        yMax = max(cord[1], cord[3])
        if xMin == xMax or yMin == yMax:
            for x in range(xMin, xMax + 1):
                for y in range(yMin, yMax + 1):
                    m[(x, y)] += 1
        if xMax - xMin == yMax - yMin:
            for x, y in zip(
                range(
                    cord[0],
                    cord[2] + (1 if cord[2] >= cord[0] else -1),
                    1 if cord[2] >= cord[0] else -1,
                ),
                range(
                    cord[1],
                    cord[3] + (1 if cord[3] >= cord[1] else -1),
                    1 if cord[3] >= cord[1] else -1,
                ),
            ):
                m[(x, y)] += 1
    print("b:", sum([1 for i in m.values() if i > 1]))


def a6(f):
    arr = [int(i) for i in f.strip().split(",")]
    c = Counter(arr)
    for d in range(256):
        cN = Counter()
        for k, v in c.items():
            if k == 0:
                cN[8] += v
                cN[6] += v
            else:
                cN[k - 1] += v
        c = cN
        if d == 79:
            print("a:", d, sum(cN.values()))
    print("b:", d, sum(cN.values()))


def a7(f):
    arr = [int(i) for i in f.strip().split(",")]
    minCtr = None
    for goal in range(min(arr), max(arr) + 1):
        # les, ortadan girip kuculen tarafa gitmeli
        ctr = 0
        for i in arr:
            ctr += int(abs(goal - i) * (abs(goal - i) + 1) / 2)
        if minCtr is None or minCtr > ctr:
            minCtr = ctr
    print("a:", minCtr)


def a8(f):
    arr = [
        [[i for i in k.strip().split(" ")] for k in l.split(" | ")]
        for l in f.split("\n")
    ]
    pA = 0
    pB = 0
    for guide, out in arr:
        m = a8_deduce(guide)
        outN = 0
        for n in out:
            if len(n) in [2, 3, 4, 7]:
                pA += 1
            outN *= 10
            outN += m["".join(sorted(n))]
        pB += outN
    print("a:", pA)
    print("b:", pB)


def a8_deduce(guide):
    m = {"abcdefg": 8}
    h = {}
    c = Counter()
    for g in guide:
        for char in g:
            c[char] += 1
        if len(g) == 2:  # 1
            m[1] = sorted(g)
            m["".join(m[1])] = 1
        if len(g) == 3:  # 7
            m[7] = sorted(g)
            m["".join(m[7])] = 7
        if len(g) == 4:  # 4
            m[4] = sorted(g)
            m["".join(m[4])] = 4
    # {'e': 4, 'b': 6, 'g': 7, 'd': 7, 'a': 8, 'c': 8, 'f': 9}
    h["e"] = [i[0] for i in c.items() if i[1] == 4][0]
    h["b"] = [i[0] for i in c.items() if i[1] == 6][0]
    h["f"] = [i[0] for i in c.items() if i[1] == 9][0]
    h["c"] = (set(m[1]) - set(h["f"])).pop()
    h["a"] = (set(m[7]) - set(m[1])).pop()
    h["d"] = [i[0] for i in c.items() if i[1] == 7 and i[0] in m[4]][0]
    h["g"] = [i[0] for i in c.items() if i[1] == 7 and i[0] != h["d"]][0]
    for g in guide:
        if len(g) == 6:  # 0 6 9
            if h["d"] not in g:
                m["".join(sorted(g))] = 0
            elif h["c"] not in g:
                m["".join(sorted(g))] = 6
            else:
                m["".join(sorted(g))] = 9
        if len(g) == 5:  # 2 3 5
            if h["c"] not in g:
                m["".join(sorted(g))] = 5
            elif h["e"] not in g:
                m["".join(sorted(g))] = 3
            else:
                m["".join(sorted(g))] = 2
    return m


def a9(f):
    arr = [[int(i) for i in l.strip()] for l in f.split("\n")]
    risks = []
    lows = []
    X = len(arr)
    Y = len(arr[0])
    for x in range(X):
        for y in range(Y):
            low = True
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= nx < X and 0 <= ny < Y and arr[nx][ny] <= arr[x][y]:
                    low = False
            if low:
                risks.append(arr[x][y])
                lows.append((x, y))
    print("a:", sum(risks) + len(risks))
    basins = []
    for lx, ly in lows:
        Q = [(lx, ly)]
        s = set()
        while len(Q) > 0:
            ax, ay = Q.pop()
            if (ax, ay) in s:
                continue
            s.add((ax, ay))
            for nx, ny in [(ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)]:
                if (
                    0 <= nx < X
                    and 0 <= ny < Y
                    and (nx, ny) not in s
                    and arr[nx][ny] != 9
                    and arr[nx][ny] >= arr[ax][ay]
                ):
                    Q.append((nx, ny))
        basins.append(len(s))
    maxB = sorted(basins)[-3:]
    print("b:", maxB[0] * maxB[1] * maxB[2])


def a10(f):
    arr = [i.strip() for i in f.split("\n")]
    Ps = {"(": ")", "{": "}", "[": "]", "<": ">"}
    P2 = {")": 1, "]": 2, "}": 3, ">": 4}
    auto = []
    E = Counter()
    for l in arr:
        p = deque()
        errors = Counter()
        for i in l:
            if i in Ps.keys():
                p.append(Ps[i])
            if i in Ps.values():
                if p[-1] == i:
                    p.pop()
                else:
                    errors[i] += 1
                    break
        if errors:
            E.update(errors)
        else:
            p2 = 0
            while len(p) > 0:
                p2 *= 5
                p2 += P2[p.pop()]
            auto.append(p2)
    print(E)
    print(3 * E[")"] + 57 * E["]"] + 1197 * E["}"] + 25137 * E[">"])
    print(sorted(auto)[int(len(auto) / 2)])


def a11(f):
    arr = [[int(i) for i in l.strip()] for l in f.split("\n")]
    X = len(arr)
    Y = len(arr[0])
    fl = 0
    for step in range(1, 1000):
        Q = deque()
        for x in range(X):
            for y in range(Y):
                arr[x][y] += 1
                if arr[x][y] > 9:
                    Q.append((x, y))

        S = set()
        while len(Q) > 0:
            x, y = Q.pop()
            if (x, y) in S:
                continue
            S.add((x, y))
            arr[x][y] = 0
            for dx, dy in zip(NDx, NDy):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < X and 0 <= ny < Y and (nx, ny) not in S:
                    arr[nx][ny] += 1
                    if arr[nx][ny] > 9:
                        Q.append((nx, ny))
        fl += len(S)
        if step == 100:
            print("a:", fl)
        if len(S) == X * Y:
            print("b:", step)
            return


def a12(f):
    m = defaultdict(list)
    for k, v in [[i for i in l.strip().split("-")] for l in f.split("\n")]:
        m[k].append(v)
        m[v].append(k)

    FIN = set()
    Q = deque([("start",)])
    while len(Q) > 0:
        path = Q.popleft()
        for p in m[path[-1]]:
            if p == "start":
                continue
            if p == "end":
                FIN.add(path + (p,))
                continue
            if p.lower() == p:
                if p not in path:
                    Q.append(path + (p,))
                elif path[0] == "start":
                    Q.append(("U",) + path + (p,))
            if p.lower() != p:
                Q.append(path + (p,))
    print("b:", len(FIN))


def a13(f):
    m = {}
    for line in f.split("\n"):
        if "," in line:
            x, y = line.strip().split(",", 1)
            m[(int(x), int(y))] = "#"
        if line.startswith("fold along"):
            fold = int(line.strip().split("=")[-1])
            m2 = dict()
            for item in m.keys():
                if "y=" in line:
                    if item[1] < fold:
                        m2[item] = "#"
                    else:
                        m2[(item[0], 2 * fold - item[1])] = "#"
                else:
                    if item[0] < fold:
                        m2[item] = "#"
                    else:
                        m2[(2 * fold - item[0], item[1])] = "#"
            m = m2
            print(len(m))
    X = 0
    Y = 0
    for item in m.keys():
        X = max(X, item[0] + 1)
        Y = max(Y, item[1] + 1)
    for y in range(Y):
        print("".join([m.get((x, y), " ") for x in range(100)]))


def a14(f):
    arr = [i.strip() for i in f.split("\n")]
    rules = dict()
    for line in arr:
        if " -> " in line:
            rules[line.split(" -> ")[0]] = line.split(" -> ")[-1].strip()
    E = Counter(["".join(i) for i in zip(arr[0], arr[0][1:])])
    for step in range(40):
        next = Counter()
        for pair in E:
            if pair in rules:
                next[pair[0] + rules[pair]] += E[pair]
                next[rules[pair] + pair[1]] += E[pair]
            else:
                next[pair] += E[pair]
        E = next
    C = Counter()
    for k, v in E.items():
        C[k[0]] += v
        C[k[1]] += v
    c = sorted(C.values())
    print(math.ceil(c[-1] / 2) - math.ceil(c[0] / 2))


def a15(f):
    # Part A, clean BFS
    arr = [[int(i) for i in l.strip()] for l in f.split("\n")]
    X = len(arr)
    Y = len(arr[0])
    S = set()
    Q = PriorityQueue()  # score, x, y
    Q.put((0, 0, 0))
    while not Q.empty():
        r, x, y = Q.get()
        if x == X - 1 and y == Y - 1:
            print("a:", r)
            break
        if (x, y) in S:
            continue
        S.add((x, y))
        for dx, dy in zip(Nx, Ny):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < X and 0 <= ny < Y and (nx, ny) not in S:
                Q.put((r + arr[nx][ny], nx, ny))

    S = set()
    Q = PriorityQueue()  # score, x, y
    Q.put((0, 0, 0))
    while not Q.empty():
        r, x, y = Q.get()
        if x == 5 * X - 1 and y == 5 * Y - 1:
            print("b:", r)
            break
        if (x, y) in S:
            continue
        S.add((x, y))
        for dx, dy in zip(Nx, Ny):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < 5 * X and 0 <= ny < 5 * Y and (nx, ny) not in S:
                val = arr[nx % X][ny % Y] + nx // X + ny // Y
                if val > 9:
                    val -= 9
                Q.put((r + val, nx, ny))


def a16(f):
    arr = [
        "".join(["{0:04b}".format(int(c, 16)) for c in l.strip()])
        for l in f.split("\n")
    ]
    for line in arr:
        # print("processing", line)
        ret = a16_process_packet(line, 0)
        print(ret[0])


def a16_process_packet(line, index):
    pVersion = int(line[index : index + 3], 2)
    pType = int(line[index + 3 : index + 6], 2)
    index += 6
    if pType == 4:
        lastLiteral = False
        literal = ""
        while not lastLiteral:
            lastLiteral = line[index] == "0"
            literal += line[index + 1 : index + 5]
            index += 5
        return int(literal, 2), index
    else:
        childVersions = []
        pLen = line[index]
        if pLen == "0":
            pTotalLen = int(line[index + 1 : index + 16], 2)
            index += 16
            stopIndex = index + pTotalLen
            # print("pLen 0", pTotalLen)
            while index < stopIndex:
                cVersion, index = a16_process_packet(line, index)
                # print(cVersion, index, "dondu 0")
                childVersions.append(cVersion)
        else:
            pCount = int(line[index + 1 : index + 12], 2)
            index += 12
            # print("pLen 1", pCount)
            for _ in range(pCount):
                cVersion, index = a16_process_packet(line, index)
                # print(cVersion, index, "dondu 1")
                childVersions.append(cVersion)
        if pType == 0:
            return sum(childVersions), index
        if pType == 1:
            return reduce((lambda x, y: x * y), childVersions), index
        if pType == 2:
            return min(childVersions), index
        if pType == 3:
            return max(childVersions), index
        if pType == 5:
            return 1 if childVersions[0] > childVersions[1] else 0, index
        if pType == 6:
            return 1 if childVersions[0] < childVersions[1] else 0, index
        if pType == 7:
            return 1 if childVersions[0] == childVersions[1] else 0, index


def a17(f):
    l = f.split("=")
    X = [int(i) for i in l[1].strip(" ,y").split("..")]
    Y = [int(i) for i in l[2].split("..")]
    print(">> ", X, Y)
    maxY = 0
    p2 = 0
    for x in range(int(math.sqrt(2 * X[0])), X[1] + 1):
        for y in range(Y[0] - 1, X[1]):
            vx = x
            vy = y
            success = False
            px = 0
            py = 0
            my = 0
            while px <= X[1] and py >= Y[0]:
                px += vx
                py += vy
                my = max(my, py)
                if vx == 0 and not X[0] <= px <= X[1]:
                    break
                if X[0] <= px <= X[1] and Y[0] <= py <= Y[1]:
                    success = True
                    break
                if vx > 0:
                    vx -= 1
                vy -= 1
            if success:
                # print("finished", x, y, px, py, X, Y)
                p2 += 1
                maxY = max(maxY, my)
    print(maxY)
    print(p2)


def a18(f):
    nums = [ast.literal_eval(l) for l in f.split("\n")]
    res = nums[0]
    for n in nums[1:]:
        # print(res, "<-", n)
        ne = copy.deepcopy([res, n])
        reduced = True
        while reduced:
            ne, reduced = a18_reduce(ne)
        res = ne
    print("a:", a18_magn(res))
    maxN = 0
    for c1 in range(len(nums)):
        for c2 in range(len(nums)):
            if c1 == c2:
                continue
            ne = copy.deepcopy([nums[c1], nums[c2]])
            reduced = True
            while reduced:
                ne, reduced = a18_reduce(ne)
            maxN = max(maxN, a18_magn(ne))
    print("b:", maxN)


def a18_magn(l):
    if type(l) == int:
        return l
    return 3 * a18_magn(l[0]) + 2 * a18_magn(l[1])


def a18_reduce(ne):
    reduced = False
    # depth > 4 - explode
    lastNo = None
    nextNo = None
    for c1, l1 in enumerate(ne):
        if type(l1) == int:
            lastNo = c1
            if nextNo is not None:
                ne[c1] += nextNo
                return ne, reduced
            continue
        for c2, l2 in enumerate(l1):
            if type(l2) == int:
                lastNo = (c1, c2)
                if nextNo is not None:
                    ne[c1][c2] += nextNo
                    return ne, reduced
                continue
            for c3, l3 in enumerate(l2):
                if type(l3) == int:
                    lastNo = (c1, c2, c3)
                    if nextNo is not None:
                        ne[c1][c2][c3] += nextNo
                        return ne, reduced
                    continue
                for c4, l4 in enumerate(l3):
                    if type(l4) == int:
                        lastNo = (c1, c2, c3, c4)
                        if nextNo is not None:
                            ne[c1][c2][c3][c4] += nextNo
                            return ne, reduced
                        continue
                    if nextNo is not None:
                        ne[c1][c2][c3][c4][0] += nextNo
                        return ne, reduced
                    # print(ne, "explode", l4)
                    if lastNo:
                        if len(lastNo) == 1:
                            ne[lastNo[0]] += l4[0]
                        if len(lastNo) == 2:
                            ne[lastNo[0]][lastNo[1]] += l4[0]
                        if len(lastNo) == 3:
                            ne[lastNo[0]][lastNo[1]][lastNo[2]] += l4[0]
                        if len(lastNo) == 4:
                            ne[lastNo[0]][lastNo[1]][lastNo[2]][lastNo[3]] += l4[0]
                    nextNo = l4[1]
                    ne[c1][c2][c3][c4] = 0
                    reduced = True
    if reduced is True:
        # nextNo aradim bulamadim
        return ne, True

    # c > 9 split
    for c1, l1 in enumerate(ne):
        if type(l1) == int:
            if l1 > 9:
                # print(ne, "split", l1)
                ne[c1] = [l1 // 2, math.ceil(l1 / 2)]
                return ne, True
            continue
        for c2, l2 in enumerate(l1):
            if type(l2) == int:
                if l2 > 9:
                    # print(ne, "split", l2)
                    ne[c1][c2] = [l2 // 2, math.ceil(l2 / 2)]
                    return ne, True
                continue
            for c3, l3 in enumerate(l2):
                if type(l3) == int:
                    if l3 > 9:
                        # print(ne, "split", l3)
                        ne[c1][c2][c3] = [l3 // 2, math.ceil(l3 / 2)]
                        return ne, True
                    continue
                for c4, l4 in enumerate(l3):
                    if l4 > 9:
                        # print(ne, "split", l4)
                        ne[c1][c2][c3][c4] = [l4 // 2, math.ceil(l4 / 2)]
                        return ne, True

    return ne, reduced


def a19(f):
    S = []  # 359 - 12292
    for l in f.split("\n"):
        if "---" in l:
            S.append([])
        if "," in l:
            S[-1].append(tuple(int(i) for i in l.strip().split(",")))
    M = defaultdict(dict)
    for c, s in enumerate(S):
        M[c] = a19_create_diffs(s)
        print(c, len(s), len(M[c]))

    merged = 1
    beacons = [(0, 0, 0)]
    while merged < len(S):
        for c in range(1, len(S)):
            match = set(M[c].keys()).intersection(M[0].keys())
            nodes0 = set(
                [node for key in match for item in M[0][key] for node in item[1]]
            )
            if len(nodes0) >= 12:
                pos = Counter()
                for key in list(match):
                    for orSUB in M[0][key]:
                        for toSUB in M[c][key]:
                            origKey = orSUB[0]
                            toKey = toSUB[0]
                            orient = [[0, 1], [1, 1], [2, 1]]  # pos, orient
                            for z1, z2 in zip(origKey, toKey):
                                if int(z1) % 3 != int(z2) % 3:
                                    orient[int(z1) % 3][0] = int(z2) % 3
                            for ch in origKey:
                                if ch not in toKey:
                                    orient[int(ch) % 3][1] = -1
                            for node0 in orSUB[1]:
                                for nodeC in toSUB[1]:
                                    pos[
                                        (
                                            node0[0]
                                            + orient[0][1] * nodeC[orient[0][0]],
                                            node0[1]
                                            + orient[1][1] * nodeC[orient[1][0]],
                                            node0[2]
                                            + orient[2][1] * nodeC[orient[2][0]],
                                            str(orient),
                                            1,
                                        )
                                    ] += 1
                print(c, len(nodes0), pos.most_common()[:4])
                sensor = pos.most_common()[0][0]
                if pos.most_common()[0][1] < 20:
                    for p in pos.most_common():
                        sensor = p[0]
                        orient = ast.literal_eval(sensor[3])
                        direction = sensor[4]
                        newNodes = set()
                        nodesC = set(
                            [
                                node
                                for key in match
                                for item in M[c][key]
                                for node in item[1]
                            ]
                        )
                        for nodeC in nodesC:
                            newNodes.add(
                                (
                                    sensor[0]
                                    - direction * orient[0][1] * nodeC[orient[0][0]],
                                    sensor[1]
                                    - direction * orient[1][1] * nodeC[orient[1][0]],
                                    sensor[2]
                                    - direction * orient[2][1] * nodeC[orient[2][0]],
                                )
                            )
                        print(
                            sensor,
                            len(nodes0),
                            len(newNodes),
                            len(nodes0.intersection(newNodes)),
                        )
                        if len(nodes0.intersection(newNodes)) == 12:
                            break
                beacons.append((sensor[0], sensor[1], sensor[2]))

                orient = ast.literal_eval(sensor[3])
                direction = sensor[4]
                for nodeC in S[c]:
                    S[0].append(
                        (
                            sensor[0] - direction * orient[0][1] * nodeC[orient[0][0]],
                            sensor[1] - direction * orient[1][1] * nodeC[orient[1][0]],
                            sensor[2] - direction * orient[2][1] * nodeC[orient[2][0]],
                        )
                    )
                S[0] = list(set(S[0]))
                M[0] = a19_create_diffs(S[0])
                M[c] = {}
                merged += 1
    print(len(S[0]))
    maxD = 0
    for x in beacons:
        for y in beacons:
            maxD = max(maxD, abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]))
    print("b:", maxD)


def a19_create_diffs(s):
    ret = defaultdict(list)
    for n in s:
        for m in s:
            if m == n:
                continue
            p = [[n[0] - m[0], "0"], [n[1] - m[1], "1"], [n[2] - m[2], "2"]]
            for cx in range(len(p)):
                if p[cx][0] < 0:
                    p[cx][1] = str(int(p[cx][1]) + 3)
                    p[cx][0] = abs(p[cx][0])
            sp = sorted(p)
            ret[" ".join([str(i[0]) for i in sp])].append(
                ("".join([i[1] for i in sp]), (n, m))
            )
            if sp[0][0] == sp[1][0]:
                ret[" ".join([str(i[0]) for i in sp])].append(
                    ("".join([i[1] for i in [sp[1], sp[0], sp[2]]]), (n, m))
                )
            if sp[1][0] == sp[2][0]:
                ret[" ".join([str(i[0]) for i in sp])].append(
                    ("".join([i[1] for i in [sp[0], sp[2], sp[1]]]), (n, m))
                )
    return ret


def a20(f):
    print(f)
    m = dict()
    ie = None
    XX = 0
    YY = 0
    for l in f.split("\n"):
        l = l.strip()
        if ie is None:
            ie = l
            continue
        if l:
            for c, ch in enumerate(l):
                if ch == "#":
                    m[(c, YY)] = 1
            XX = len(l)
            YY += 1

    for c in range(50):
        mNext = dict()
        X = [0 - 110, XX + 110]
        Y = [0 - 110, YY + 110]
        # for x, y in m.keys():
        #     if x < X[0]:
        #         X[0] = x
        #     if x > X[1]:
        #         X[1] = x
        #     if y < Y[0]:
        #         Y[0] = y
        #     if y > Y[1]:
        #         Y[1] = y
        for x in range(X[0], X[1] + 1):
            for y in range(Y[0], Y[1] + 1):
                if ie[a20_getIndex(m, x, y)] == "#":
                    mNext[(x, y)] = 1
        print(
            sum(
                [
                    1 if -55 <= i[0] <= XX + 55 and -55 <= i[1] <= YY + 55 else 0
                    for i in mNext.keys()
                ]
            )
        )
        print(c, X, Y, len(mNext))
        m = mNext


def a20_getIndex(m, x, y):
    ret = ""
    for dx, dy in zip(NMx, NMy):
        ret += "1" if (m.get((x + dx, y + dy)) == 1) else "0"
    return int(ret, 2)


def a21(f):
    P1 = 6
    P2 = 1
    p = [P1, P2]
    s = [0, 0]
    ctr = True
    dice = 0
    while True:
        dice += 3
        ctr = not ctr
        p[ctr] = (p[ctr] + 3 * (dice - 1)) % 10
        if p[ctr] == 0:
            p[ctr] = 10
        s[ctr] += p[ctr]
        if s[0] >= 1000 or s[1] >= 1000:
            print(s, dice)
            print(min(s) * dice)
            break
    p1 = 0
    p2 = 0
    C = Counter([(P1, P2, 0, 0, False)])
    mult = [(3, 1), (4, 3), (5, 6), (6, 7), (7, 6), (8, 3), (9, 1)]
    while len(C):
        CN = Counter()
        for item, val in C.items():
            for num, ctr in mult:
                qp = [item[0], item[1]]
                qs = [item[2], item[3]]
                qp[item[4]] = (qp[item[4]] + num) % 10
                if qp[item[4]] == 0:
                    qp[item[4]] = 10
                qs[item[4]] += qp[item[4]]
                if qs[0] >= 21:
                    p1 += val * ctr
                elif qs[1] >= 21:
                    p2 += val * ctr
                else:
                    CN[(qp[0], qp[1], qs[0], qs[1], not item[4])] += val * ctr
        C = CN
    print(p1, p2)


def a22(f):
    orders = []
    S = set()
    X = dict()
    Y = dict()
    Z = dict()
    for line in f.split("\n"):
        l = line.replace(" ", ",").replace("..", ",").replace("=", ",").split(",")
        if len(l) < 8:
            print(l)
            continue
        orders.append(
            (
                int(l[2]),
                int(l[3]) + 1,
                int(l[5]),
                int(l[6]) + 1,
                int(l[8]),
                int(l[9]) + 1,
                l[0] == "on",
            )
        )
        X[orders[-1][0]] = ""
        X[orders[-1][1]] = ""
        Y[orders[-1][2]] = ""
        Y[orders[-1][3]] = ""
        Z[orders[-1][4]] = ""
        Z[orders[-1][5]] = ""
    print(len(X.keys()), len(Y.keys()), len(Z.keys()))
    L = dict()
    last = ""
    for c, x in enumerate(sorted(X.keys())):
        X[x] = c
        if last:
            L["x" + str(last[0])] = x - last[1]
        last = (c, x)
    last = ""
    for c, y in enumerate(sorted(Y.keys())):
        Y[y] = c
        if last:
            L["y" + str(last[0])] = y - last[1]
        last = (c, y)
    last = ""
    for c, z in enumerate(sorted(Z.keys())):
        Z[z] = c
        if last:
            L["z" + str(last[0])] = z - last[1]
        last = (c, z)
    for co, o in enumerate(orders):
        if co % 10 == 0:
            print(co, len(orders))
        for x in range(X[o[0]], X[o[1]]):
            for y in range(Y[o[2]], Y[o[3]]):
                for z in range(Z[o[4]], Z[o[5]]):
                    if o[6]:
                        S.add((x, y, z))
                    else:
                        S.discard((x, y, z))
    print(len(S))
    s = 0
    for node in S:
        s += L["x" + str(node[0])] * L["y" + str(node[1])] * L["z" + str(node[2])]
    print("b:", s)


AoC = {
    "01": a1,
    "02": a2,
    "03": a3,
    "04": a4,
    "05": a5,
    "06": a6,
    "07": a7,
    "08": a8,
    "09": a9,
    "10": a10,
    "11": a11,
    "12": a12,
    "13": a13,
    "14": a14,
    "15": a15,
    "16": a16,
    "17": a17,
    "18": a18,
    "19": a19,
    "20": a20,
    "21": a21,
    "22": a22,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="advent day")
    parser.add_argument("-e", help="use example set -ee", action="count", default=0)
    args = parser.parse_args()

    if args.d is None:
        args.d = sorted(AoC.keys())[-1]
        print("DAY: ", args.d)
    f = aocd.get_data(year=2021, day=int(args.d))
    if args.e:
        f = (
            open("inp/inp%02de%s" % (int(args.d), str(args.e) if args.e > 1 else ""))
            .read()
            .strip()
        )
    AoC[args.d](f)
