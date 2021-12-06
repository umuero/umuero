# https://adventofcode.com/2020
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, product
import itertools
import re
import argparse


def a1(f):
    arr = [int(i) for i in f.readlines()]
    last = None
    ctr = 0
    for i in arr:
        if last is None:
            last = i
        if i > last:
            ctr += 1
        last = i
    print('p1:', ctr)

    last = None
    ctr = 0
    for i in range(len(arr) - 2):
        s = sum(arr[i:i+3])
        if last is None:
            last = s
        if s > last:
            ctr += 1
        last = s
    print('p2:', ctr)


def a2(f):
    arr = [i.split() for i in f.readlines()]
    hor = 0
    depth = 0
    aim = 0
    for cmd, ctr in arr:
        if cmd == 'forward':
            hor += int(ctr)
            depth += aim * int(ctr)
        if cmd == 'down':
            aim += int(ctr)
        if cmd == 'up':
            aim -= int(ctr)
    print(hor*depth)


def a3(f):
    arr = [i.strip() for i in f.readlines()]
    gamma = ''
    epsilon = ''
    nums = defaultdict(Counter)
    N = len(arr[0])
    for num in arr:
        for ctr, n in enumerate(num):
            nums[ctr][n] += 1

    for i in range(N):
        gamma += nums[i].most_common()[0][0]
        epsilon += nums[i].most_common()[-1][0]

    print(int(gamma, 2) * int(epsilon, 2))

    ox = ''
    co = ''
    ox_C = Counter()
    co_C = Counter()
    for i in range(N):
        for num in arr:
            if num.startswith(ox):
                ox_C[num[i]] += 1
            if num.startswith(co):
                co_C[num[i]] += 1

        ox += sorted(ox_C.items(),
                     key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        ox_C = Counter()
        co += sorted(co_C.items(),
                     key=lambda x: (x[1], x[0]), reverse=False)[0][0]
        co_C = Counter()

    print(int(ox, 2) * int(co, 2))


def a4(f):
    arr = [i.strip().replace('  ', ' ').split(' ') for i in f.readlines()]
    nums = [int(i) for i in arr[0][0].split(',')]
    Bc = []
    Br = []
    for ctr in range(1, len(arr), 6):
        Br.append([[int(i) for i in l] for l in arr[ctr+1:ctr+6]])
        Bc.append([[Br[-1][j][i] for j in range(len(Br[-1]))]
                  for i in range(len(Br[-1][0]))])

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
                    print('a:', sum([sum(i) for i in bc]) * num)
                lastScore = sum([sum(i) for i in bc]) * num
                bc.clear()
    print('b:', lastScore)


def a5(f):
    arr = [[int(k) for n in i.strip().split(' -> ')
            for k in n.split(',')] for i in f.readlines()]
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
                range(cord[0], cord[2] + (1 if cord[2] >=
                      cord[0] else -1), 1 if cord[2] >= cord[0] else -1),
                range(cord[1], cord[3] + (1 if cord[3] >=
                      cord[1] else -1), 1 if cord[3] >= cord[1] else -1)
            ):
                m[(x, y)] += 1
    print('b:', sum([1 for i in m.values() if i > 1]))


def a6(f):
    arr = [int(i) for i in f.read().strip().split(',')]
    c = Counter(arr)
    for d in range(256):
        cN = Counter()
        for k, v in c.items():
            if k == 0:
                cN[8] += v
                cN[6] += v
            else:
                cN[k-1] += v
        c = cN
        if d == 79:
            print('a:', d, sum(cN.values()))
    print('b:', d, sum(cN.values()))


AoC = {
    '1': a1,
    '2': a2,
    '3': a3,
    '4': a4,
    '5': a5,
    '6': a6
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='advent day')
    parser.add_argument('-e', help='use example set', action='store_true')
    args = parser.parse_args()
    if args.d is None:
        args.d = sorted(AoC.keys())[-1]
        print('DAY: ', args.d)
    AoC[args.d](open('inp/inp%02d%s' % (int(args.d), 'e' if args.e else '')))
