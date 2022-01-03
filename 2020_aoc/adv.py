# https://adventofcode.com/2021
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


def a1(f):
    arr = [int(i) for i in f.split("\n")]
    s = set()
    for i in arr:
        s.add(i)
        if 2020 - i in s:
            print(i * (2020 - i))

    s = set()
    s2 = dict()
    for i in arr:
        s.add(i)
        if 2020 - i in s2:
            print("b:", i * s2[2020 - i][0] * s2[2020 - i][1])
        for j in s:
            s2[i + j] = (i, j)


def a2(f):
    arr = [i.strip() for i in f.split("\n")]
    p1 = 0
    p2 = 0
    for l in arr:
        cond, passw = l.split(": ")
        mn, mx, ch = cond.replace("-", " ").split(" ")
        if int(mn) <= Counter(passw).get(ch, 0) <= int(mx):
            p1 += 1
        if (passw[int(mn) - 1] == ch and passw[int(mx) - 1] != ch) or (
            passw[int(mn) - 1] != ch and passw[int(mx) - 1] == ch
        ):
            p2 += 1
    print(p1)
    print("b:", p2)


def a3(f):
    arr = [i.strip() for i in f.split("\n")]
    print(a3_slope(arr, 3, 1))
    print(
        "b:",
        a3_slope(arr, 1, 1)
        * a3_slope(arr, 3, 1)
        * a3_slope(arr, 5, 1)
        * a3_slope(arr, 7, 1)
        * a3_slope(arr, 1, 2),
    )


def a3_slope(arr, r, d):
    p = [0, 0]
    tree = 0
    depth = len(arr)
    width = len(arr[0])
    while p[1] < depth - d:
        p = [p[0] + r, p[1] + d]
        if arr[p[1]][p[0] % width] == "#":
            tree += 1
    return tree


a4_rules = {
    "byr": re.compile(r"(19[2-9][0-9]|200[0-2])$"),
    "iyr": re.compile(r"20(1\d|20)$"),
    "eyr": re.compile(r"20(2\d|30)$"),
    "hgt": re.compile(r"(1([5-8][0-9]|9[0-3])cm|(59|6\d|7[0-6])in)$"),
    "hcl": re.compile(r"#[a-f0-9]{6}$"),
    "ecl": re.compile(r"(amb|blu|brn|gry|grn|hzl|oth)$"),
    "pid": re.compile(r"\d{9}$"),
}


def a4_validate(p):
    if len(p) != 8:
        return False
    for k in p.keys():
        if k in a4_rules:
            if not a4_rules[k].match(p[k]):
                return False
    return True


def a4(f):
    arr = [i.strip() for i in f.split("\n")]
    data = {"cid": ""}
    ctr = 0
    for line in arr:
        if line.strip() == "":
            if a4_validate(data):
                ctr += 1
            data = {"cid": ""}
            continue
        for part in line.split(" "):
            key, value = part.split(":")
            data[key] = value
    if a4_validate(data):
        ctr += 1
    print("b:", ctr)


def a5(f):
    arr = [i.strip() for i in f.split("\n")]
    sids = [
        int(
            t.replace("F", "0").replace("B", "1").replace("L", "0").replace("R", "1"), 2
        )
        for t in arr
    ]
    print(max(sids))
    ss = set(sids)
    for i in range(max(sids)):
        if i not in ss and i + 1 in ss and i - 1 in ss:
            print("b:", i)


def a6(f):
    arr = [i.strip() for i in f.split("\n")]
    gr = []
    data = ""
    dctr = 0
    for line in arr:
        if line.strip() == "":
            gr.append((Counter(data), dctr))
            data = ""
            dctr = 0
            continue
        data += line
        dctr += 1
    if data:
        gr.append((Counter(data), dctr))
    print(sum([len(i[0].keys()) for i in gr]))
    print("b:", sum([len([k for k in i[0].keys() if i[0][k] == i[1]]) for i in gr]))


def a7_rec(rule, nm):
    total = 1
    if nm not in rule:
        return total
    for bg in rule[nm]:
        total += bg[0] * a7_rec(rule, bg[1])
    return total


def a7(f):
    arr = [i.strip().rstrip(".") for i in f.split("\n")]
    rule = defaultdict(list)
    parents = defaultdict(list)
    for line in arr:
        pr, childs = line.split(" bags contain ")
        for ch in childs.split(", "):
            if ch == "no other bags":
                continue
            rule[pr].append((int(ch.split()[0]), " ".join(ch.split()[1:-1])))
            parents[" ".join(ch.split()[1:-1])].append(pr)

    avails = set()
    proc = ["shiny gold"]
    while proc:
        it = proc.pop()
        for pr in parents[it]:
            avails.add(pr)
            proc.append(pr)
    print(len(avails))
    print("b:", a7_rec(rule, "shiny gold") - 1)


def a8_loop(arr, corrupt):
    ctr = 0
    for lctr, line in enumerate(arr):
        if "nop" in line or "jmp" in line:
            if ctr == corrupt:
                if "nop" in line:
                    arr[lctr] = (line[0].replace("nop", "jmp"), line[1])
                else:
                    arr[lctr] = (line[0].replace("jmp", "nop"), line[1])
            ctr += 1
    ind = 0
    acc = 0
    cmds = set()
    while ind not in cmds and ind != len(arr):
        cmds.add(ind)
        cmd = arr[ind][0]
        val = arr[ind][1]
        if cmd == "nop":
            ind += 1
        if cmd == "acc":
            ind += 1
            acc += val
        if cmd == "jmp":
            ind += val
    if ind == len(arr):
        return True, acc
    return False, acc


def a8(f):
    arr = [(i.strip().split()[0], int(i.strip().split()[1])) for i in f.split("\n")]
    print(a8_loop(arr, -1)[1])
    for i in range(350):
        arr = [(i.strip().split()[0], int(i.strip().split()[1])) for i in f.split("\n")]
        ret = a8_loop(arr, i)
        if ret[0]:
            print("b:", i, ret[1])
            break


def a9_valid(arr, ind):
    base = arr[ind - 25 : ind]
    val = arr[ind]
    for i in range(25):
        for j in range(25):
            if i == j:
                continue
            if base[i] + base[j] == val:
                return True
    return False


def a9(f):
    arr = [int(i) for i in f.split("\n")]
    for i in range(25, len(arr)):
        if a9_valid(arr, i) is False:
            print(arr[i])
            break
    target = arr[i]
    i0, i1 = 0, 0
    ssum = 0
    while ssum != target:
        if ssum > target:
            i0 += 1
            ssum -= arr[i0]
        else:
            i1 += 1
            ssum += arr[i1]
        # ssum = sum(arr[i0:i1])
    # print(i0, i1)
    print("b:", min(arr[i0:i1]) + max(arr[i0:i1]))


def a10(f):
    arr = [int(i) for i in f.split("\n")]
    valid = 0
    sarr = sorted(arr)
    diff = defaultdict(int)
    for ctr, val in enumerate(sarr):
        if val - valid <= 3:
            diff[val - valid] += 1
            valid = val
    diff[3] += 1  # device
    print(diff[1] * diff[3])

    sarr.insert(0, 0)
    sarr.append(sarr[-1] + 3)
    df = [sarr[i] - sarr[i - 1] for i in range(len(sarr))]
    df[0] = 3
    cons = []
    cctr = 0
    for d in df:
        if d == 1:
            cctr += 1
        else:
            if cctr:
                cons.append(cctr)
                cctr = 0
    total = 1
    for i in cons:
        if i == 2:
            total *= 2
        if i == 3:
            total *= 4
        if i == 4:
            total *= 7
    print("b:", total)


def a11(f):
    inp = f.split("\n")
    seats = set()
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            if x == "." or x == "\n":
                continue
            seats.add((lCtr, hCtr))
    lmax = len(inp)
    hmax = len(inp[0])
    changes = 1
    occupied = set()
    while changes != 0:
        changes = 0
        occupiedNext = set()
        for p in seats:
            occ = 0
            for x in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]:
                s = tuple(map(sum, zip(p, x)))  # (p[0] + x[0], p[1] + x[1])
                while s not in seats:
                    s = tuple(map(sum, zip(s, x)))
                    if not (0 <= s[0] <= lmax):
                        break
                    if not (0 <= s[1] <= hmax):
                        break
                if s in occupied:
                    occ += 1
            if p not in occupied:
                if occ == 0:
                    changes += 1
                    occupiedNext.add(p)
            else:
                if occ >= 5:
                    changes += 1
                else:
                    occupiedNext.add(p)
        # print("iteration:", changes)
        occupied = occupiedNext
    print("b:", len(occupied))


def a12(f):
    inp = f.split("\n")
    x, y = 0, 0  # E(x+); N(y+)
    direction = 0
    for line in inp:
        c = line[0]
        num = int(line[1:])
        if c == "N" or (c == "F" and direction == 270):
            y += num
        if c == "S" or (c == "F" and direction == 90):
            y -= num
        if c == "E" or (c == "F" and direction == 0):
            x += num
        if c == "W" or (c == "F" and direction == 180):
            x -= num
        if c == "R":
            direction += num
            direction = direction % 360
        if c == "L":
            direction -= num
            direction = direction % 360
    print("p1", abs(x) + abs(y))

    x, y = 0, 0  # E(x+); N(y+)
    wayX, wayY = 10, 1
    for line in inp:
        c = line[0]
        num = int(line[1:])
        if c == "N":
            wayY += num
        if c == "S":
            wayY -= num
        if c == "E":
            wayX += num
        if c == "W":
            wayX -= num
        if c == "R" or c == "L":
            direction = 0
            if c == "R":
                direction += num
            if c == "L":
                direction -= num
            direction = direction % 360
            if direction == 90:
                wayX, wayY = wayY, -wayX
            if direction == 180:
                wayX, wayY = -wayX, -wayY
            if direction == 270:
                wayX, wayY = -wayY, wayX
        if c == "F":
            x += wayX * num
            y += wayY * num
    print("p2", abs(x) + abs(y))


################## After 2021 AoC ##################
def a13(f):
    tm = None
    arr = []
    for l in f.split("\n"):
        if tm is None:
            tm = int(l)
        else:
            arr = [int(i) if i != "x" else 1 for i in l.split(",")]
    minB = (tm, 0)
    for bus in arr:
        if bus == 1:
            continue
        if minB[0] > bus - (tm % bus):
            minB = (bus - (tm % bus), (bus - (tm % bus)) * bus)
        # print(bus, "coming in", bus - (tm % bus))
    print(minB[1])
    n = 0
    mult = 1
    for ctr, bus in enumerate(arr):
        if bus == 1:
            continue
        # print(n, mult, ctr, bus)
        while n % bus != -ctr % bus:
            n += mult
        mult *= bus
    print("b:", n)


def a14(f):
    M = dict()
    lines = f.split("\n")
    mask = "X" * 36
    for line in lines:
        if line.startswith("mask = "):
            mask = line[7:]
        if line.startswith("mem"):
            ll = line.split(" = ")
            val = format(int(ll[1]), "036b")
            mVal = ""
            for v, m in zip(val, mask):
                if m == "X":
                    mVal += v
                else:
                    mVal += m
            M[ll[0][4:-1]] = int(mVal, 2)
    print(sum(M.values()))

    M = dict()
    mask = "X" * 36
    for lc, line in enumerate(lines):
        if line.startswith("mask = "):
            mask = line[7:]
        if line.startswith("mem"):
            ll = line.split(" = ")
            val = format(int(ll[0][4:-1]), "036b")
            n = mask.count("X")
            for pVal in range(2 ** n):
                perm = format(pVal, "0%db" % n)
                mVal = ""
                x = 0
                for v, m in zip(val, mask):
                    if m == "0":
                        mVal += v
                    elif m == "1":
                        mVal += "1"
                    else:
                        mVal += perm[x]
                        x += 1
                M[mVal] = int(ll[1])
    print("b:", sum(M.values()))


def a15(f):
    d = dict()
    arr = [int(i) for i in f.split(",")]
    for c, i in enumerate(arr):
        d[i] = (c + 1, d.get(i, (None,))[0])
    turn = len(arr) + 1
    last = arr[-1]
    while turn < 30000001:  # 2021
        # pypy3 10sec - python3 30sec
        if turn % 100000 == 0:
            print("processing turn:", turn, last)
        if last in d and d[last][1] is not None:
            last = d[last][0] - d[last][1]
        else:
            last = 0
        d[last] = (turn, d.get(last, (None,))[0])
        if turn == 2020:
            print(turn, last)
        turn += 1
    print("b:", turn - 1, last)


def a16(f):
    rules = []
    S = defaultdict(set)  # 3 -> {'row', 'seat', ...}
    mine = None
    others = []
    pA = 0
    for line in f.split("\n"):
        if " or " in line:
            p = line.split(": ")
            pp = p[1].split(" or ")

            rule = (
                p[0],
                tuple(int(i) for i in pp[0].split("-")),
                tuple(int(i) for i in pp[1].split("-")),
            )
            rules.append(rule)
            for i in range(rule[1][0], rule[1][1] + 1):
                S[i].add(p[0])
            for i in range(rule[2][0], rule[2][1] + 1):
                S[i].add(p[0])
        if "," in line:
            live = [int(i) for i in line.split(",")]
            if mine is None:
                mine = live
                continue
            partA = True
            for i in live:
                if i not in S:
                    pA += i
                    partA = False
            if partA:
                others.append(live)
    print(pA)
    C = [set([i[0] for i in rules]) for i in range(len(rules))]
    for o in others:
        for c, v in enumerate(o):
            for ex in C[c].difference(S[v]):
                C[c].discard(ex)
    changed = True
    index = dict()  # rule -> index
    while changed:
        changed = False
        for c, rl in enumerate(C):
            if len(rl) == 1:
                changed = True
                fixed = rl.pop()
                index[fixed] = c
                for rlE in C:
                    rlE.discard(fixed)
                C[c] = set()
                break
    print(
        "b:",
        reduce(
            lambda x, y: x * y,
            [
                mine[item[1]]
                for item in index.items()
                if item[0].startswith("departure")
            ],
        ),
    )


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
    # After 2021 AoC
    "13": a13,
    "14": a14,
    "15": a15,
    "16": a16,
    # "17": a17,
    # "18": a18,
    # "19": a19,
    # "20": a20,
    # "21": a21,
    # "22": a22,
    # "23": a23,
    # "24": a24,
    # "25": a25,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="advent day")
    parser.add_argument("-e", help="use example set -ee", action="count", default=0)
    args = parser.parse_args()

    if args.d is None:
        args.d = sorted(AoC.keys())[-1]
        print("DAY: ", args.d)
    f = aocd.get_data(year=2020, day=int(args.d))
    if args.e:
        f = (
            open("inp/inp%02de%s" % (int(args.d), str(args.e) if args.e > 1 else ""))
            .read()
            .strip()
        )
    AoC[args.d](f)
