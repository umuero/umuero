# https://adventofcode.com/2020
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, product
import itertools
import re


def a1():
    arr = [int(i) for i in open("inp/inp01").readlines()]
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


def a2():
    arr = [i.split() for i in open("inp/inp02").readlines()]
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


def a3():
    arr = [i.strip() for i in open("inp/inp03").readlines()]
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


def a4():
    arr = [i.strip().replace('  ', ' ').split(' ')
           for i in open("inp/inp04").readlines()]
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


a4()
