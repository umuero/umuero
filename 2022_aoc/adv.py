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
N3x = [1, 0, 0, -1, 0, 0]
N3y = [0, 1, 0, 0, -1, 0]
N3z = [0, 0, 1, 0, 0, -1]
reNUMS = re.compile("\d+")


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
        if p not in op and M[p][0] > 0:
            Q.put((m + 1, rel - flow, p, tuple(sorted(op + (p,)))))
        for n in M[p][1]:
            Q.put((m + 1, rel - flow, n, op))

    print(M)
    M2 = {k: {n: 1 for n in v[1]} for k, v in M.items() if v[0] > 0 or k == "AA"}
    for n in M2.keys():
        keys = M2[n][1].keys()
        for n2 in keys:
            if M2[n2][0] > 0:
                continue
            v2 = M2[n2][1][n2]
            del M2[n2][1][n2]
            for n3 in M[n2][1]:
                if n3 not in M2[n2][1]:
                    M2[n2][1][n3] = v2 + 1
    print(M2)
    # S = set()
    # P = dict()
    # Q = PriorityQueue()
    # ctr = 0
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
    #     check = (p, e)
    #     if p > e:
    #         check = (e, p)
    #     prev = P.get(check)
    #     if prev is not None and flow < prev[0] and len(op) < prev[1]:
    #         continue
    #     P[check] = (flow, len(op))
    #     if p not in op and M[p][0] > 0:
    #         if p != e and e not in op and M[e][0] > 0:
    #             nop = sorted(op + (p, e))
    #             Q.put((m + 1, rel - flow, p, e, tuple(nop)))
    #         for n in M[e][1]:
    #             Q.put((m + 1, rel - flow, p, n, tuple(sorted(op + (p,)))))
    #     for n in M[p][1]:
    #         if p != e and e not in op and M[e][0] > 0:
    #             Q.put((m + 1, rel - flow, n, e, tuple(sorted(op + (e,)))))
    #         for nn in M[e][1]:
    #             Q.put((m + 1, rel - flow, n, nn, op))


def a17_avail(M, x, y, item):
    if x < 0 or x > 6 or y < 0:
        return False
    for ix, iy in item:
        if x + ix > 6:
            return False
        if (x + ix, y + iy) in M:
            return False
    return True


def a17(f):
    commands = [i for i in f.strip()]
    cLen = len(commands)
    order = ["-", "+", "L", "I", "O"]
    shape = {
        "-": ((0, 0), (1, 0), (2, 0), (3, 0)),
        "+": ((0, 1), (1, 1), (1, 0), (1, 2), (2, 1)),
        "L": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        "I": ((0, 0), (0, 1), (0, 2), (0, 3)),
        "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    }
    M = set()
    maxY = -1
    itemC = 0
    cmdC = 0
    item = None
    loop = dict()
    ITER = 102022
    offset = 0
    while itemC <= ITER:
        if item is None:
            x, y = 2, maxY + 4
            item = shape[order[itemC % 5]]
            itemC += 1
            if itemC == 2022:
                print("a:", maxY + 1)
            loopKey = (order[itemC % 5], cmdC % cLen)
            if loopKey in loop and offset == 0:
                # print("loop", loopKey, maxY, loop[loopKey])
                if maxY - loop[loopKey][0] == loop[loopKey][2]:
                    print("looped", (maxY, itemC, maxY - loop[loopKey][2]), loop[loopKey])
                    # ITER = 1000000000000 - 1514285714288
                    # prev - loop[loopKey] -> (77, 51, 53)
                    # (maxY, itemC, maxY - loop[loopKey][2]) ->  (130, 86, 53)
                    loopHeight = loop[loopKey][2]
                    loopItemCount = itemC - loop[loopKey][1]
                    print(f"{loopHeight}h in {loopItemCount} items")
                    offset = (1000000000000 - itemC) // loopItemCount * loopHeight
                    ITER = itemC + 1 + ((1000000000000 - itemC) % loopItemCount)
                else:
                    loop[loopKey] = (maxY, itemC, maxY - loop[loopKey][2])
            else:
                loop[loopKey] = (maxY, itemC, maxY)
        cmd = commands[cmdC % cLen]
        cmdC += 1
        if cmd == "<" and a17_avail(M, x - 1, y, item):
            x -= 1
        if cmd == ">" and a17_avail(M, x + 1, y, item):
            x += 1
        if a17_avail(M, x, y - 1, item):
            y -= 1
        else:
            for ix, iy in item:
                M.add((x + ix, y + iy))
                maxY = max(maxY, y + iy)
            item = None
    print("b:", maxY + offset)


def a18(f):
    nums = [tuple([int(i) for i in l.split(",")]) for l in f.strip().split("\n")]
    M = set()
    minx, miny, minz = nums[0]
    maxx, maxy, maxz = nums[0]
    for n in nums:
        M.add((n[0], n[1], n[2]))
        minx = min(minx, n[0])
        maxx = max(maxx, n[0])
        miny = min(miny, n[1])
        maxy = max(maxy, n[1])
        minz = min(minz, n[2])
        maxz = max(maxz, n[2])
    ctr = 0
    Check = set()
    for n in nums:
        for dx, dy, dz in zip(N3x, N3y, N3z):
            if (n[0] + dx, n[1] + dy, n[2] + dz) not in M:
                ctr += 1
                Check.add((n[0] + dx, n[1] + dy, n[2] + dz))
    print("a:", ctr)
    Out = set()
    In = set()
    for ch in Check:
        S = set()
        Q = PriorityQueue()  # score, x, y
        Q.put((0, ch))
        isOut = False
        while not Q.empty():
            r, n = Q.get()
            if n in S:
                continue
            S.add(n)
            if minx >= n[0] + dx or n[0] + dx >= maxx or n in Out:
                isOut = True
                Out = Out.union(S)
                break
            if n in In:
                break
            for dx, dy, dz in zip(N3x, N3y, N3z):
                if (n[0] + dx, n[1] + dy, n[2] + dz) in M:
                    continue
                Q.put((r + 1, (n[0] + dx, n[1] + dy, n[2] + dz)))
        if isOut is False:
            In = In.union(S)
    ctr = 0
    for n in nums:
        for dx, dy, dz in zip(N3x, N3y, N3z):
            if (n[0] + dx, n[1] + dy, n[2] + dz) not in M and (n[0] + dx, n[1] + dy, n[2] + dz) not in In:
                ctr += 1
    print("b:", ctr)


def a19(f):
    # b = [bpNo, OreO, ClayO, ObsO, ObsC, GeoO, GeoObs]
    # ore, clay, obs, geo
    B = [tuple([int(i) for i in reNUMS.findall(l)]) for l in f.strip().split("\n")]
    bMax = {}
    # for blue in B:
    for blue in B[:3]:
        print(blue)
        Q = PriorityQueue()  # score, x, y
        Q.put((1, (1, 0, 0, 0), (0, 0, 0, 0), blue))
        S = dict()
        mPrint = 10
        while not Q.empty():
            minute, robots, inventory, bluep = Q.get()
            if minute == mPrint:
                print(minute, robots, inventory, bluep[0])
                mPrint += 1
            if minute == 32:
                # bMax[bluep[0]] = max(bMax.get(bluep[0], -1), (robots[3] - inventory[3]) * bluep[0])
                bMax[bluep[0]] = max(bMax.get(bluep[0], -1), robots[3] - inventory[3])
                continue
            if val := S.get((minute, robots, bluep[0])):
                if all(v <= i for v, i in zip(val, inventory)):
                    continue
            S[(minute, robots, bluep[0])] = inventory
            # print(minute, robots, inventory, bluep[0])
            Q.put((minute + 1, robots, tuple([i - r for r, i in zip(robots, inventory)]), bluep))
            if -inventory[0] >= bluep[5] and -inventory[2] >= bluep[6]:
                # Build Geo
                Q.put(
                    (
                        minute + 1,
                        (robots[0], robots[1], robots[2], robots[3] + 1),
                        (
                            inventory[0] - robots[0] + bluep[5],
                            inventory[1] - robots[1],
                            inventory[2] - robots[2] + bluep[6],
                            inventory[3] - robots[3],
                        ),
                        bluep,
                    )
                )
            if -inventory[0] >= bluep[3] and -inventory[1] >= bluep[4] and robots[2] <= bluep[6]:
                # Build Obs
                Q.put(
                    (
                        minute + 1,
                        (robots[0], robots[1], robots[2] + 1, robots[3]),
                        (
                            inventory[0] - robots[0] + bluep[3],
                            inventory[1] - robots[1] + bluep[4],
                            inventory[2] - robots[2],
                            inventory[3] - robots[3],
                        ),
                        bluep,
                    )
                )
            if -inventory[0] >= bluep[2] and robots[1] <= bluep[4]:
                # Build Clay
                Q.put(
                    (
                        minute + 1,
                        (robots[0], robots[1] + 1, robots[2], robots[3]),
                        (
                            inventory[0] - robots[0] + bluep[2],
                            inventory[1] - robots[1],
                            inventory[2] - robots[2],
                            inventory[3] - robots[3],
                        ),
                        bluep,
                    )
                )
            if -inventory[0] >= bluep[1] and robots[0] <= max(bluep[1], bluep[2], bluep[3], bluep[5]):
                # Build Ore
                Q.put(
                    (
                        minute + 1,
                        (robots[0] + 1, robots[1], robots[2], robots[3]),
                        (
                            inventory[0] - robots[0] + bluep[1],
                            inventory[1] - robots[1],
                            inventory[2] - robots[2],
                            inventory[3] - robots[3],
                        ),
                        bluep,
                    )
                )
    print(bMax)
    # print("a:", sum(bMax.values()))
    print("b:", bMax[1] * bMax[2] * bMax[3])


def a20(f):
    n = [(int(n), c) for c, n in enumerate(f.strip().split("\n"))]
    D = {v: k for k, v in n}
    N = len(n)
    order0 = None
    for order in range(N):
        ind = n.index((D[order], order))
        val = n.pop(ind)
        newInd = (ind + D[order]) % (N - 1)
        if D[order] == 0:
            order0 = order
        if newInd == 0 and D[order] < 0:
            n.append(val)
        else:
            n.insert(newInd, val)
    ind = n.index((D[order0], order0))
    print("a:", n[(ind + 1000) % N][0] + n[(ind + 2000) % N][0] + n[(ind + 3000) % N][0])

    n = [(int(n) * 811589153, c) for c, n in enumerate(f.strip().split("\n"))]
    D = {v: k for k, v in n}
    N = len(n)
    order0 = None
    for i in range(10):
        for order in range(N):
            ind = n.index((D[order], order))
            val = n.pop(ind)
            newInd = (ind + D[order]) % (N - 1)
            if D[order] == 0:
                order0 = order
            if newInd == 0 and D[order] < 0:
                n.append(val)
            else:
                n.insert(newInd, val)
    ind = n.index((D[order0], order0))
    print("b:", n[(ind + 1000) % N][0] + n[(ind + 2000) % N][0] + n[(ind + 3000) % N][0])


def a21(f):
    arr = [tuple(l.split(" ")) for l in f.strip().split("\n")]
    V = {}
    F = {}
    for l in arr:
        if len(l) == 2:
            V[l[0].strip(":")] = int(l[1])
        elif l[2] == "+":
            F[l[0].strip(":")] = (lambda x, y: x + y, l[1], l[3])
        elif l[2] == "*":
            F[l[0].strip(":")] = (lambda x, y: x * y, l[1], l[3])
        elif l[2] == "-":
            F[l[0].strip(":")] = (lambda x, y: x - y, l[1], l[3])
        elif l[2] == "/":
            F[l[0].strip(":")] = (lambda x, y: x // y, l[1], l[3])
    V["humn"] = 3221245824363
    # humn: 0 -> V0
    # humn: 3000000000000 -> V3
    # B: = (V3 - VG) * 3000 / (V0 - V3) ~+-1
    while len(F) > 0:
        delKeys = []
        for k in F.keys():
            f = F[k]
            if f[1] in V and f[2] in V:
                V[k] = f[0](V[f[1]], V[f[2]])
                if k == "root":
                    print("a:", V[k])
                    print("b:", V[f[1]], V[f[2]])  # VG
                    F = {}
                delKeys.append(k)
        for k in delKeys:
            if k in F:
                del F[k]


def a22_move(M, X, Y, R, ord, mX, mY):
    if ord == 0:
        return (X, Y, R)
    x = X + Nx[R]
    y = Y + Ny[R]
    if (x, y) not in M:
        if R == 0:
            x = 0
            while (x, y) not in M:
                x += 1
        if R == 2:
            x = mX
            while (x, y) not in M:
                x -= 1
        if R == 1:
            y = 0
            while (x, y) not in M:
                y += 1
        if R == 3:
            y = mY
            while (x, y) not in M:
                y -= 1
    if M[(x, y)] == ".":
        return a22_move(M, x, y, R, ord - 1, mX, mY)
    if M[(x, y)] == "#":
        return (X, Y, R)


def a22_moveB(M, X, Y, R, ord, mX, mY):
    if ord == 0:
        return (X, Y, R)
    x, y, r = X + Nx[R], Y + Ny[R], R
    if (x, y) not in M:
        Qx = X // 50
        Qy = Y // 50
        x50 = x % 50
        y50 = y % 50
        print("==", Qx, Qy, R)
        if Qx == 1 and Qy == 0 and R == 2:  # 1 left
            x, y, r = 0, 3 * 50 - 1 - y50, 0
        if Qx == 1 and Qy == 0 and R == 3:  # 1 up
            x, y, r = 0, 3 * 50 + x50, 0
        if Qx == 2 and Qy == 0 and R == 0:  # 2 right
            x, y, r = 2 * 50 - 1, 3 * 50 - 1 - y50, 2
        if Qx == 2 and Qy == 0 and R == 1:  # 2 down
            x, y, r = 2 * 50 - 1, 50 + x50, 2
        if Qx == 2 and Qy == 0 and R == 3:  # 2 up
            x, y, r = x50, 4 * 50 - 1, 3
        if Qx == 1 and Qy == 1 and R == 0:  # 3 right
            x, y, r = 2 * 50 + y50, 50 - 1, 3
        if Qx == 1 and Qy == 1 and R == 2:  # 3 left
            x, y, r = y50, 2 * 50, 1
        if Qx == 0 and Qy == 2 and R == 2:  # 4 left
            x, y, r = 50, 50 - 1 - y50, 0
        if Qx == 0 and Qy == 2 and R == 3:  # 4 up
            x, y, r = 50, 50 + x50, 0
        if Qx == 1 and Qy == 2 and R == 0:  # 5 right
            x, y, r = 3 * 50 - 1, 50 - 1 - y50, 2
        if Qx == 1 and Qy == 2 and R == 1:  # 5 down
            x, y, r = 50 - 1, 3 * 50 + x50, 2
        if Qx == 0 and Qy == 3 and R == 0:  # 6 right
            x, y, r = 50 + y50, 3 * 50 - 1, 3
        if Qx == 0 and Qy == 3 and R == 1:  # 6 down
            x, y, r = 2 * 50 + x50, 0, 1
        if Qx == 0 and Qy == 3 and R == 2:  # 6 left
            x, y, r = 50 + y50, 0, 1
    if M[(x, y)] == ".":
        return a22_moveB(M, x, y, r, ord - 1, mX, mY)
    if M[(x, y)] == "#":
        return (X, Y, R)


def a22(f):
    M = {}
    Orders = ""
    X, Y, R = None, None, 0
    maxX, maxY = 0, 0
    for y, l in enumerate(f.split("\n")):
        if Orders is None:
            Orders = re.split(r"(\d+)", l)
            break
        if l == "":
            Orders = None
        for x, c in enumerate(l):
            if c == "." and X is None:
                X = x
                Y = y
            if c == "." or c == "#":
                M[(x, y)] = c
            maxX = max(maxX, x)
        maxY = max(maxY, y)
    for ord in Orders:
        if ord == "":
            continue
        elif ord == "R":
            R += 1
            R %= 4
        elif ord == "L":
            R -= 1
            R %= 4
        else:
            print(X, Y, R, ord)
            # X, Y, R = a22_move(M, X, Y, R, int(ord), maxX, maxY)
            X, Y, R = a22_moveB(M, X, Y, R, int(ord), maxX, maxY)
    print("b:", X, Y, R, 1000 * (Y + 1) + 4 * (X + 1) + R)


def a23(f):
    M = dict()
    for y, l in enumerate(f.split("\n")):
        for x, c in enumerate(l):
            if c == "#":
                M[(x, y)] = 0
    EMPTY = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    CH = [(-1, -1, 0, -1, 1, -1), (-1, 1, 0, 1, 1, 1), (-1, -1, -1, 0, -1, 1), (1, -1, 1, 0, 1, 1)]

    for round in range(1000):
        for ex, ey in M.keys():
            isEmpty = True
            for e in EMPTY:
                if (ex + e[0], ey + e[1]) in M:
                    isEmpty = False
                    break
            if isEmpty is False:
                for i in range(4):
                    ch = CH[(round + i) % 4]
                    if ch:
                        if (
                            (ch[0] + ex, ch[1] + ey) not in M
                            and (ch[2] + ex, ch[3] + ey) not in M
                            and (ch[4] + ex, ch[5] + ey) not in M
                        ):
                            M[ex, ey] = (ch[2] + ex, ch[3] + ey)
                            break
        c = Counter(M.values())
        deleteList = []
        addList = []
        for p, v in M.items():
            if v == 0:
                continue
            if c[v] == 1:
                addList.append(v)
                deleteList.append(p)
            else:
                M[p] = 0
        if len(deleteList) == 0:
            print("b:", round + 1)
            break
        for p in deleteList:
            del M[p]
        for p in addList:
            M[p] = 0
        if round == 9:
            X = sorted([i[0] for i in M.keys()])
            Y = sorted([i[1] for i in M.keys()])
            print("a:", (X[-1] - X[0] + 1) * (Y[-1] - Y[0] + 1) - len(M))


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
