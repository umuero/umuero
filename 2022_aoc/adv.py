# https://adventofcode.com/2022
from collections import defaultdict, deque, Counter
import functools
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

DD = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (1, 0),
    "D": (-1, 0),
}
Nx = [1, 0, -1, 0]
Ny = [0, 1, 0, -1]
NDx = [1, 0, -1, 0, 1, 1, -1, -1]
NDy = [0, 1, 0, -1, 1, -1, 1, -1]
NMx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
NMy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]


def a1(f):
    arr = [int(i) if i != "" else -1 for i in f.split("\n")]
    arr.append(-1)
    arr2 = []
    ctr = 0
    for i in arr:
        if i == -1:
            arr2.append(ctr)
            ctr = 0
            continue
        ctr += i
    print("p1:", max(arr2))
    print("p2:", sum(sorted(arr2)[-3:]))


def a2(f):
    arr = [i.split() for i in f.split("\n")]
    D = {"A": 1, "B": 2, "C": 3, "X": 1, "Y": 2, "Z": 3}
    score = 0
    for opp, own in arr:
        score += D[own]
        if D[own] == D[opp]:
            score += 3
        if (D[own] % 3) == ((D[opp] + 1) % 3):
            score += 6
    print("p1:", score)
    score = 0
    for opp, own in arr:
        if own == "X":
            score += D[opp] - 1 if D[opp] - 1 != 0 else 3
        if own == "Y":
            score += 3 + D[opp]
        if own == "Z":
            score += 6 + (D[opp] + 1 if D[opp] + 1 != 4 else 1)
    print("p2:", score)


def a3_score(c):
    s = ord(c)
    if s > 95:
        return s - 96
    return s - 38


def a3(f):
    arr = [i for i in f.split("\n")]
    score = 0
    for l in arr:
        split = len(l) // 2
        for c in set(Counter(l[:split])).intersection(Counter(l[split:])):
            score += a3_score(c)
    print("p1:", score)
    score = 0
    for l in range(0, len(arr), 3):
        for c in set(Counter(arr[l])).intersection(Counter(arr[l + 1])).intersection(Counter(arr[l + 2])):
            score += a3_score(c)
    print("p2:", score)


def a4(f):
    arr = [(int(m) for m in i.replace(",", "-").split("-")) for i in f.split("\n")]
    c1 = 0
    c2 = 0
    for l1, l2, l3, l4 in arr:
        if (l1 <= l3 and l4 <= l2) or (l3 <= l1 and l2 <= l4):
            c1 += 1
        if l2 >= l3 and l1 <= l4:
            c2 += 1
    print("p1:", c1)
    print("p1:", c2)


def a5(f):
    stacks = defaultdict(list)
    stacks2 = defaultdict(list)
    orders = []
    order = False
    for line in f.split("\n"):
        if line == "":
            order = True
            continue
        if order:
            parts = line.split(" ")
            orders.append((int(parts[3]), int(parts[5]), int(parts[1])))
            continue
        for index, chr in enumerate(line):
            if chr.isalpha():
                st = (index + 3) // 4
                stacks[st].append(chr)
                stacks2[st].append(chr)
    for ord in orders:
        s2 = []
        for count in range(ord[2]):
            st = stacks[ord[0]].pop(0)
            stacks[ord[1]].insert(0, st)
            s2.append(stacks2[ord[0]].pop(0))
        s2.reverse()
        for s in s2:
            stacks2[ord[1]].insert(0, s)

    print("".join([stacks[i + 1][0] for i in range(len(stacks))]))
    print("".join([stacks2[i + 1][0] for i in range(len(stacks2))]))


def a6(f):
    l = []
    for i, c in enumerate(f):
        l.append(c)
        if len(l) > 4:
            l.pop(0)
        if len(set(l)) == 4:
            print("a:", i + 1)
            break
    l = []
    for i, c in enumerate(f):
        l.append(c)
        if len(l) > 14:
            l.pop(0)
        if len(set(l)) == 14:
            print("b:", i + 1)
            break


def a7(f):
    arr = [i for i in f.strip().split("\n")]
    path = []
    sizes = {}
    dirs = set("/")
    for cmd in arr:
        # print(cmd[0], cmd[2:4])
        if cmd[0] == "$":
            if cmd[2:4] == "cd":
                if cmd[5:] == "/":
                    path = []
                elif cmd[5:] == "..":
                    path.pop()
                else:
                    path.append(cmd[5:])
            elif cmd[2:4] == "ls":
                pass
        else:
            size, name = cmd.split(" ", 1)
            if size == "dir":
                if path:
                    sizes["/" + "/".join(path) + "/" + name] = 0
                    dirs.add("/" + "/".join(path) + "/" + name)
                else:
                    sizes["/" + name] = 0
                    dirs.add("/" + name)
            else:
                if path:
                    sizes["/" + "/".join(path) + "/" + name] = int(size)
                else:
                    sizes["/" + name] = int(size)
    p1 = 0
    usage = 0
    for dir in dirs:
        s = sum([v for k, v in sizes.items() if k.startswith(dir)])
        if dir == "/":
            usage = s
        if s < 100000:
            p1 += s
    print("1:", p1)
    toDelete = usage - 40000000
    p2 = 70000000
    for dir in dirs:
        s = sum([v for k, v in sizes.items() if k.startswith(dir)])
        if toDelete < s:
            p2 = min(p2, s)
    print("2:", p2)


def a8b_score(arr, X, Y, xx, yy):
    S = 1
    H = arr[xx][yy]
    for dx, dy in zip(Nx, Ny):
        C, x, y = 0, xx + dx, yy + dy
        while 0 <= x < X and 0 <= y < Y:
            C += 1
            if arr[x][y] >= H:
                break
            x += dx
            y += dy
        S *= C
    return S


def a8(f):
    arr = [[int(i) for i in l] for l in f.strip().split("\n")]
    X = len(arr)
    Y = len(arr[0])
    vis = set()
    for x in range(X):
        cMax = -1
        for y in range(Y):
            if arr[x][y] > cMax:
                vis.add((x, y))
                cMax = arr[x][y]
        cMax = -1
        for y in range(Y - 1, -1, -1):
            if arr[x][y] > cMax:
                vis.add((x, y))
                cMax = arr[x][y]
    for y in range(Y):
        cMax = -1
        for x in range(X):
            if arr[x][y] > cMax:
                vis.add((x, y))
                cMax = arr[x][y]
        cMax = -1
        for x in range(X - 1, -1, -1):
            if arr[x][y] > cMax:
                vis.add((x, y))
                cMax = arr[x][y]
    print("a:", len(vis))

    bMax = -1
    for x in range(1, X - 1):
        for y in range(1, Y - 1):
            bMax = max(bMax, a8b_score(arr, X, Y, x, y))
    print("b:", bMax)


def a9_solve(moves, knots):
    X = [[0, 0] for _ in range(knots)]
    visited = set()
    for a, b in moves:
        for _ in range(int(b)):
            X[0][0] += {"R": 1, "L": -1}.get(a, 0)
            X[0][1] += {"D": 1, "U": -1}.get(a, 0)
            for j in range(1, knots):
                if (X[j - 1][0] - X[j][0]) ** 2 + (X[j - 1][1] - X[j][1]) ** 2 > 2:
                    X[j][0] += max(-1, min(1, X[j - 1][0] - X[j][0]))
                    X[j][1] += max(-1, min(1, X[j - 1][1] - X[j][1]))
            visited.add(tuple(X[-1]))
    return len(visited)


def a9(f):
    arr = [l.split(" ") for l in f.strip().split("\n")]
    print("a:", a9_solve(arr, 2))
    print("b:", a9_solve(arr, 10))


def a10(f):
    arr = [i.strip() for i in f.split("\n")]
    X = 1
    c = 1
    M = (0, 0)
    p1 = 0
    for ord in arr:
        if ord == "noop":
            pass
        elif ord.startswith("addx"):
            M = (c + 2, int(ord.split(" ")[1]))

        if abs(((c - 1) % 40) - X) < 2:
            print("#", end="")
        else:
            print(".", end="")
        if (c + 20) % 40 == 0:
            p1 += c * X
        if c % 40 == 0:
            print("")
        c += 1
        if ord.startswith("addx"):
            if abs(((c - 1) % 40) - X) < 2:
                print("#", end="")
            else:
                print(".", end="")
            if (c + 20) % 40 == 0:
                p1 += c * X
            if c % 40 == 0:
                print("")
            c += 1
            if c == M[0]:
                X += M[1]
    print("a: ", p1)


def a11(f):
    arr = [l for l in f.split("\n")]
    # M = {"0": {items: [], op: lambda x: x * 19, test: 23, t: "2", f: "3"}}
    M = []
    mCtr = 0
    divs = 1
    for line in arr:
        line = line.strip()
        if line.startswith("Monkey "):
            mCtr = int(line.split(" ")[1].strip(":"))
            M.append({"items": []})
        elif line.startswith("Starting items"):
            M[mCtr]["items"] = [int(i.strip()) for i in line.split(":")[1].split(",")]
        elif line.startswith("Operation"):
            rule = line.split(":")[1].strip()
            ruleInt = rule.split(" ")[-1]
            if ruleInt.isnumeric():
                ruleInt = int(ruleInt)
            if "*" in rule:
                if type(ruleInt) == int:
                    M[mCtr]["op"] = (lambda y: (lambda x: x * y))(ruleInt)
                else:
                    M[mCtr]["op"] = lambda x: x * x
            if "+" in rule:
                M[mCtr]["op"] = (lambda y: (lambda x: x + y))(ruleInt)
        elif line.startswith("Test:"):
            rule = line.split(":")[1].strip()
            if rule.startswith("divisible"):
                ruleInt = int(rule.split(" ")[-1])
                divs *= ruleInt
                M[mCtr]["test"] = (lambda y: (lambda x: x % y == 0))(ruleInt)
            else:
                print("non divisible rule")
        elif line.startswith("If true:"):
            rule = line.split(":")[1].strip()
            if rule.startswith("throw to monkey"):
                M[mCtr]["tr"] = int(rule.split(" ")[-1])
            else:
                print("non throw to monkey rule")
        elif line.startswith("If false:"):
            rule = line.split(":")[1].strip()
            if rule.startswith("throw to monkey"):
                M[mCtr]["fl"] = int(rule.split(" ")[-1])
            else:
                print("non throw to monkey rule")
    c = Counter()
    for r in range(10000):
        if r % 100 == 0:
            print(r)
        for mctr, m in enumerate(M):
            for it in m["items"]:
                c[mctr] += 1
                w = m["op"](it) % divs
                if m["test"](w):
                    M[m["tr"]]["items"].append(w)
                else:
                    M[m["fl"]]["items"].append(w)
                # print(it, w, m["test"](w), m["tr"], m["fl"])
            m["items"] = []
    print("b:", c.most_common()[0][1] * c.most_common()[1][1])


def a12(f):
    # Part A, clean BFS
    arr = [[i for i in l.strip()] for l in f.strip().split("\n")]
    X = len(arr)
    Y = len(arr[0])
    S = set()
    Q = PriorityQueue()  # score, x, y
    for x in range(X):
        for y in range(Y):
            if arr[x][y] == "S":
                Q.put((0, x, y))
                arr[x][y] = "a"
            if arr[x][y] == "E":
                Goal = (x, y)
                arr[x][y] = "z"
    while not Q.empty():
        r, x, y = Q.get()
        if x == Goal[0] and y == Goal[1]:
            print("a:", r)
            break
        if (x, y) in S:
            continue
        S.add((x, y))
        for dx, dy in zip(Nx, Ny):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < X and 0 <= ny < Y and (nx, ny) not in S and ord(arr[nx][ny]) <= ord(arr[x][y]) + 1:
                Q.put((r + 1, nx, ny))

    Q = PriorityQueue()  # score, x, y
    S = set()
    Q.put((0, Goal[0], Goal[1]))
    while not Q.empty():
        r, x, y = Q.get()
        if arr[x][y] == "a":
            print("b:", r)
            break
        if (x, y) in S:
            continue
        S.add((x, y))
        for dx, dy in zip(Nx, Ny):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < X and 0 <= ny < Y and (nx, ny) not in S and ord(arr[nx][ny]) >= ord(arr[x][y]) - 1:
                Q.put((r + 1, nx, ny))


def a13_comp(p1, p2):
    if type(p1) == int and type(p2) == int:
        return p2 - p1  # >0 true
    if type(p1) == list and type(p2) == list:
        for i in range(len(p1)):
            if i >= len(p2):
                return -1
            res = a13_comp(p1[i], p2[i])
            if res > 0:
                return 1
            elif res < 0:
                return -1
        if len(p1) < len(p2):
            return 1
        return 0
    if type(p1) == int:
        return a13_comp([p1], p2)
    else:
        return a13_comp(p1, [p2])


def a13(f):
    p1, p2 = None, None
    correct = 0
    ind = 1
    lines = [[[2]], [[6]]]
    for line in f.split("\n"):
        if len(line.strip()) == 0:
            p1, p2 = None, None
            ind += 1
            continue
        l = ast.literal_eval(line)
        if p1 is None:
            p1 = l
            lines.append(p1)
            continue
        p2 = l
        lines.append(p2)
        if a13_comp(p1, p2) > 0:
            correct += ind
    print("a:", correct)
    sl = sorted(lines, key=functools.cmp_to_key(a13_comp), reverse=True)
    print("b:", (sl.index([[2]]) + 1) * (sl.index([[6]]) + 1))


def a14_fall(M, maxY, sX, sY):
    # if sY > maxY: # partA
    #     return None
    if sY > maxY:  # partB
        return (sX, sY)
    if (sX, sY + 1) not in M:
        return a14_fall(M, maxY, sX, sY + 1)
    if (sX - 1, sY + 1) not in M:
        return a14_fall(M, maxY, sX - 1, sY + 1)
    if (sX + 1, sY + 1) not in M:
        return a14_fall(M, maxY, sX + 1, sY + 1)
    return (sX, sY)


def a14(f):
    arr = [i for i in f.strip().split("\n")]
    maxY = 0
    G = (500, 0)
    M = dict()
    sCtr = 0
    for line in arr:
        parts = line.split(" -> ")
        for p1, p2 in zip(parts, parts[1:]):
            p1x, p1y = (int(i) for i in p1.split(","))
            p2x, p2y = (int(i) for i in p2.split(","))
            maxY = max(maxY, p1y, p2y)
            for x in range(min(p1x, p2x), max(p1x, p2x) + 1):
                for y in range(min(p1y, p2y), max(p1y, p2y) + 1):
                    M[x, y] = "#"
    while maxY > 0:
        sand = a14_fall(M, maxY, 500, 0)
        if sand is None:
            break
        M[sand] = "o"
        sCtr += 1
        if sand == (500, 0):
            break
    print("a:", sCtr)


def manh_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def manh_distP(p):
    return abs(p[0] - p[2]) + abs(p[1] - p[3])


def a15(f):
    arr = [tuple([int(i.split(",")[0]) for i in l.replace(":", ",").split("=")[1:]]) for l in f.strip().split("\n")]
    Y = 2000000
    M = 4000000
    if len(arr) == 14:
        Y = 10
        M = 20
    intervals = []
    for sx, sy, bx, by in arr:
        dist = manh_dist(sx, sy, bx, by)
        areaDist = dist - abs(sy - Y)
        if areaDist >= 0:
            intervals.append((sx - areaDist, 0))  # start
            intervals.append((sx + areaDist, 1))  # end
        if by == Y:
            intervals.append((bx, 1, "B"))
    ctr = 0
    overlap = 0
    overlapStart = None
    lastB = -1
    for iv in sorted(intervals):
        if iv[1] == 0:
            if overlapStart is None:
                overlapStart = iv[0]
            overlap += 1
        if iv[1] == 1:
            if len(iv) == 3:
                if lastB != iv[0]:
                    ctr -= 1
                lastB = iv[0]
            else:
                overlap -= 1
                if overlap == 0:
                    ctr += iv[0] + 1 - overlapStart
                    overlapStart = None
    print("a:", ctr)
    posX = set()
    negX = set()
    for p in arr:
        posX.add(p[0] + p[1] + manh_distP(p) + 1)  # y = (X+Y+D) + x
        posX.add(p[0] + p[1] - manh_distP(p) - 1)
        negX.add(-p[0] + p[1] + manh_distP(p) + 1)
        negX.add(-p[0] + p[1] - manh_distP(p) - 1)

    for a in negX:
        for b in posX:
            p = ((b - a) // 2, (a + b) // 2)
            if not (0 < p[0] < M and 0 < p[1] < M):
                continue
            match = None
            for s in arr:
                if manh_dist(p[0], p[1], s[0], s[1]) <= manh_distP(s):
                    match = True
                    break
            if match is None:
                print("b:", p[0] * M + p[1])
                return


def a16(f):
    M = {l.split(" ")[1]: (int(l[23:26].replace(";", "")), l.replace(",", "").split(" ")[9:]) for l in f.strip().split("\n")}
    vals = len([i for i in M.keys() if M[i][0] > 0])
    S = set()
    Q = PriorityQueue()
    Q.put((0, 0, "AA", ()))
    while not Q.empty():
        m, rel, p, op = Q.get()
        if m == 30:
            print("a:", (m, rel))
            break
        flow = sum([M[l][0] for l in op])
        if vals == len(op):
            Q.put((m + 1, rel - flow, p, op))
            continue
        if (p, op) in S:
            continue
        S.add((p, op))
        print(m, flow, p, op)
        for n in M[p][1]:
            Q.put((m + 1, rel - flow, n, op))
        if p not in op and M[p][0] > 0:
            Q.put((m + 1, rel - flow, p, tuple(sorted(op + (p,)))))

    # S = set()
    # Q = PriorityQueue()
    # ctr = 0
    # mMax = (0, 0)
    # Q.put((0, 0, "AA", "AA", ()))
    # while not Q.empty():
    #     m, rel, p, e, op = Q.get()
    #     ctr += 1
    #     if ctr % 1000 == 0:
    #         print(m, rel, p, e, Q.qsize())
    #     if m == 26:
    #         print("b:", (m, rel))
    #         break
    #     flow = sum([M[l][0] for l in op])
    #     check = (p, e, op)
    #     if p > e:
    #         check = (e, p, op)
    #     if check in S:
    #         continue
    #     S.add(check)
    #     # if mMax[0] != m:
    #     #     mMax = (m, rel)
    #     # else:
    #     #     if Q.qsize() > 10000 and mMax[1] * 0.5 < rel:
    #     #         break
    #     if p not in op and M[p][0] > 0:
    #         if p != e and e not in op and M[e][0] > 0:
    #             nop = sorted(op+ (p,e,)))
    #             Q.put((m + 1, rel - flow, p, e, tuple(nop)))
    #         for n in M[e][1]:
    #             Q.put((m + 1, rel - flow, p, n, tuple(sorted(op + (p,)))))
    #     for n in M[p][1]:
    #         if p != e and e not in op and M[e][0] > 0:
    #             Q.put((m + 1, rel - flow, n, e, tuple(sorted(op + (e,)))))
    #         for nn in M[e][1]:
    #             Q.put((m + 1, rel - flow, n, nn, op))


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
            nodes0 = set([node for key in match for item in M[0][key] for node in item[1]])
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
                                            node0[0] + orient[0][1] * nodeC[orient[0][0]],
                                            node0[1] + orient[1][1] * nodeC[orient[1][0]],
                                            node0[2] + orient[2][1] * nodeC[orient[2][0]],
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
                        nodesC = set([node for key in match for item in M[c][key] for node in item[1]])
                        for nodeC in nodesC:
                            newNodes.add(
                                (
                                    sensor[0] - direction * orient[0][1] * nodeC[orient[0][0]],
                                    sensor[1] - direction * orient[1][1] * nodeC[orient[1][0]],
                                    sensor[2] - direction * orient[2][1] * nodeC[orient[2][0]],
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
            ret[" ".join([str(i[0]) for i in sp])].append(("".join([i[1] for i in sp]), (n, m)))
            if sp[0][0] == sp[1][0]:
                ret[" ".join([str(i[0]) for i in sp])].append(("".join([i[1] for i in [sp[1], sp[0], sp[2]]]), (n, m)))
            if sp[1][0] == sp[2][0]:
                ret[" ".join([str(i[0]) for i in sp])].append(("".join([i[1] for i in [sp[0], sp[2], sp[1]]]), (n, m)))
    return ret


def a20(f):
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
        print(sum([1 if -55 <= i[0] <= XX + 55 and -55 <= i[1] <= YY + 55 else 0 for i in mNext.keys()]))
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


def a23(f):
    S = set()
    Q = PriorityQueue()  # score, state
    # P = ((10, 1), (100, 1000), (10, 100), (1000, 1)) # ex
    # P1 = ((1000, 10), (100, 1), (1000, 1), (10, 100))
    P = ((1000, 1000, 1000, 10), (100, 100, 10, 1), (1000, 10, 1, 1), (10, 1, 100, 100))
    init = (0, 0, P[0], 0, P[1], 0, P[2], 0, P[3], 0, 0)
    Q.put((0, init, ()))
    while not Q.empty():
        r, s, last = Q.get()
        if s in S:
            continue
        # print(r, s, last)
        S.add(s)
        if sum([i for i in s if type(i) == int]) == 0 and r != 0:
            print("a:", r)
            break
        for avail in a23_avail(s):
            Q.put((r + avail[0], avail[1], s))


def a23_avail(s):
    ret = []
    for i, node in enumerate(s):
        if node == 0:
            continue
        path = 0
        item = 0
        L = list(s)
        if type(node) == tuple:
            # room -> hall
            for ctr in range(len(node)):
                if node[ctr] != 0:
                    path = ctr + 1
                    item = node[ctr]
                    L[i] = node[:ctr] + (0,) + node[ctr + 1 :]
                    break
            if item != 0:
                for c in range(1, 10):
                    if i + c <= 10 and (type(s[i + c]) != int or s[i + c] == 0):
                        if s[i + c] == 0:
                            L[i + c] = item
                            # print("adding+", item, c, ((path + c) * item, tuple(L)))
                            ret.append(((path + c) * item, tuple(L)))
                            L[i + c] = 0
                    else:
                        break
                for c in range(-1, -10, -1):
                    if i + c >= 0 and (type(s[i + c]) != int or s[i + c] == 0):
                        if s[i + c] == 0:
                            L[i + c] = item
                            # print("adding-", item, c, ((path - c) * item, tuple(L)))
                            ret.append(((path - c) * item, tuple(L)))
                            L[i + c] = 0
                    else:
                        break
        else:
            # room -> hall
            item = node
            goal = 0
            if item == 1:
                goal = 2
            if item == 10:
                goal = 4
            if item == 100:
                goal = 6
            if item == 1000:
                goal = 8
            if sum(s[goal]) <= item * 4 and sum(s[goal]) % item == 0:
                if i < goal:
                    for c in range(1, 10):
                        if i + c == goal:
                            L[i] = 0
                            home = L[i + c]
                            for ctr in range(len(home) - 1, -1, -1):
                                if home[ctr] == 0:
                                    path = ctr + 1
                                    L[i + c] = home[:ctr] + (item,) + home[ctr + 1 :]
                                    break
                            # print("home+", item, c, ((path + c) * item, tuple(L)))
                            ret.append(((path + c) * item, tuple(L)))
                            break
                        if i + c <= 10 and (type(s[i + c]) != int or s[i + c] == 0):
                            pass
                        else:
                            break
                else:
                    for c in range(-1, -10, -1):
                        if i + c == goal:
                            L[i] = 0
                            home = L[i + c]
                            for ctr in range(len(home) - 1, -1, -1):
                                if home[ctr] == 0:
                                    path = ctr + 1
                                    L[i + c] = home[:ctr] + (item,) + home[ctr + 1 :]
                                    break
                            # print("home-", item, c, ((path - c) * item, tuple(L)))
                            ret.append(((path - c) * item, tuple(L)))
                            break
                        if i + c >= 0 and (type(s[i + c]) != int or s[i + c] == 0):
                            pass
                        else:
                            break

    return ret


def a24(f):
    # INP = list("12996997829399")
    INP = list("11841231117189")
    R4 = [1, 1, 1, 1, 26, 1, 1, 26, 26, 26, 1, 26, 26, 26]
    R5 = [14, 15, 12, 11, -5, 14, 15, -13, -16, -8, 15, -8, 0, -4]
    R15 = [12, 7, 1, 2, 4, 15, 11, 5, 3, 9, 2, 3, 3, 11]

    w, x, z = (0, 0, 0)
    for i in range(len(INP)):
        w = int(INP[i])
        goal = (z % 26) + R5[i]
        z = z // R4[i]
        if goal != w:
            z = (26 * z) + w + R15[i]
        print(f"d:{R4[i]}\tg+:{R5[i]}\tm:{R15[i]}\tg:{goal}\tw:{w}\tx:{x}\tz:{z}")

    arr = [i.strip() for i in f.split("\n")]
    VAR = {"w": 0, "x": 0, "y": 0, "z": 0}
    for line in arr:
        order = line.split(" ")
        if order[0] == "inp":
            VAR[order[1]] = int(INP.pop(0))
            print("processed", 13 - len(INP), order, VAR)
        if order[0] == "add":
            VAR[order[1]] = VAR[order[1]] + VAR.get(order[2], int(order[2]) if not order[2].isalpha() else 0)
        if order[0] == "mul":
            VAR[order[1]] = VAR[order[1]] * VAR.get(order[2], int(order[2]) if not order[2].isalpha() else 0)
        if order[0] == "div":
            VAR[order[1]] = VAR[order[1]] // VAR.get(order[2], int(order[2]) if not order[2].isalpha() else 0)
        if order[0] == "mod":
            VAR[order[1]] = VAR[order[1]] % VAR.get(order[2], int(order[2]) if not order[2].isalpha() else 0)
        if order[0] == "eql":
            VAR[order[1]] = 1 if VAR[order[1]] == VAR.get(order[2], int(order[2]) if not order[2].isalpha() else 0) else 0
    print("valid", VAR["z"] == 0, VAR)


def a25(f):
    print(f)
    mE = set()
    mS = set()
    for Y, l in enumerate(f.split("\n")):
        for X, ch in enumerate(l.strip()):
            if ch == ">":
                mE.add((X, Y))
            if ch == "v":
                mS.add((X, Y))
    X = X + 1
    Y = Y + 1
    step = 0
    diff = 1
    while diff > 0:
        if step % 10 == 0:
            print(step, diff)
        step += 1
        diff = 0
        nE = set()
        nS = set()
        for x, y in mE:
            if ((x + 1) % X, y) not in mS and ((x + 1) % X, y) not in mE:
                nE.add(((x + 1) % X, y))
                diff += 1
            else:
                nE.add((x, y))
        for x, y in mS:
            if (x, (y + 1) % Y) not in mS and (x, (y + 1) % Y) not in nE:
                nS.add((x, (y + 1) % Y))
                diff += 1
            else:
                nS.add((x, y))
        mE = nE
        mS = nS
    print(step)


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
    "23": a23,
    "24": a24,
    "25": a25,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="advent day")
    parser.add_argument("-e", help="use example set -ee", action="count", default=0)
    args = parser.parse_args()

    if args.d is None:
        args.d = sorted(AoC.keys())[-1]
        print("DAY: ", args.d)
    f = aocd.get_data(year=2022, day=int(args.d))
    if args.e:
        f = open("inp/inp%02de%s" % (int(args.d), str(args.e) if args.e > 1 else "")).read()
    AoC[args.d](f)
