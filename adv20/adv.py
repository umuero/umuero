# https://adventofcode.com/2020
from collections import defaultdict, deque, Counter
from itertools import permutations, combinations, product
import itertools
import re

def a1():
    arr = [int(i) for i in open("inp/inp01").readlines()]
    s = set()
    for i in arr:
        s.add(i)
        if 2020-i in s:
            print i * (2020-i)

    s = set()
    s2 = dict()
    for i in arr:
        s.add(i)
        if 2020-i in s2:
            print i * s2[2020-i][0] * s2[2020-i][1]
        for j in s:
            s2[i + j] = (i, j)

def a2(p1=False):
    arr = [i.strip() for i in open("inp/inp02").readlines()]
    ctr = 0
    for l in arr:
        cond, passw = l.split(": ")
        mn, mx, ch = cond.replace("-", " ").split(" ")
        if p1:
            if int(mn) <= Counter(passw).get(ch, 0) <= int(mx):
                ctr += 1
        else:
            if (passw[int(mn) - 1] == ch and passw[int(mx) - 1] != ch) or \
               (passw[int(mn) - 1] != ch and passw[int(mx) - 1] == ch):
                ctr += 1
    print ctr

def a3(r=3, d=1):
    arr = [i.strip() for i in open("inp/inp03").readlines()]
    p = [0, 0]
    tree = 0
    depth = len(arr)
    width = len(arr[0])
    while p[1] < depth - d:
        p = [p[0] + r, p[1] + d]
        if arr[p[1]][p[0] % width] == '#':
            tree += 1
    return tree
# a3(1,1) * a3(3,1) * a3(5,1) * a3(7,1) * a3(1,2)

rules = {
    'byr': re.compile(r'(19[2-9][0-9]|200[0-2])$'),
    'iyr': re.compile(r'20(1\d|20)$'),
    'eyr': re.compile(r'20(2\d|30)$'),
    'hgt': re.compile(r'(1([5-8][0-9]|9[0-3])cm|(59|6\d|7[0-6])in)$'),
    'hcl': re.compile(r'#[a-f0-9]{6}$'),
    'ecl': re.compile(r'(amb|blu|brn|gry|grn|hzl|oth)$'),
    'pid': re.compile(r'\d{9}$'),
}
def validate(p):
    if len(p) != 8:
        return False
    for k in p.keys():
        if k in rules:
            if not rules[k].match(p[k]):
                return False
    return True

def a4():
    arr = [i.strip() for i in open("inp/inp04").readlines()]
    ps = []
    data = {'cid': ''}
    ctr = 0
    for line in arr:
        if line.strip() == "":
            if validate(data):
                ctr += 1
            ps.append(data)
            data = {'cid': ''}
            continue
        for part in line.split(" "):
            key, value = part.split(":")
            data[key] = value
    if validate(data):
        ctr += 1
    ps.append(data)
    print ctr

def a5():
    arr = [i.strip() for i in open("inp/inp05").readlines()]
    sids = [int(t.replace('F', '0').replace('B', '1').replace('L', '0').replace('R', '1'), 2) for t in arr]
    print max(sids)
    ss = set(sids)
    for i in range(max(sids)):
        if i not in ss and i+1 in ss and i-1 in ss:
            print i

def a6():
    arr = [i.strip() for i in open("inp/inp06").readlines()]
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
    print sum([len(i[0].keys()) for i in gr])
    print sum([len([k for k in i[0].keys() if i[0][k] == i[1]]) for i in gr])

def a7_rec(rule, nm):
    total = 1
    if nm not in rule:
        return total
    for bg in rule[nm]:
        total += bg[0] * a7_rec(rule, bg[1])
    return total

def a7():
    arr = [i.strip().rstrip(".") for i in open("inp/inp07").readlines()]
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
    proc = ['shiny gold']
    while proc:
        it = proc.pop()
        for pr in parents[it]:
            avails.add(pr)
            proc.append(pr)
    print len(avails)
    print a7_rec(rule, 'shiny gold') - 1

def handleCmd(arr, ind, acc):
    cmd = arr[ind][0]
    val = arr[ind][1]
    if cmd == 'nop':
        return ind + 1, acc
    if cmd == 'acc':
        return ind + 1, acc + val
    if cmd == 'jmp':
        return ind + val, acc
    return ind, acc

def a8(corrupt=-1):
    arr = [(i.strip().split()[0], int(i.strip().split()[1])) for i in open("inp/inp08").readlines()]
    ctr = 0
    for lctr, line in enumerate(arr):
        if 'nop' in line or 'jmp' in line:
            if ctr == corrupt:
                if 'nop' in line:
                    arr[lctr] = (line[0].replace('nop', 'jmp'), line[1])
                else:
                    arr[lctr] = (line[0].replace('jmp', 'nop'), line[1])
            ctr += 1
    ind = 0
    acc = 0
    cmds = set()
    while ind not in cmds and ind != len(arr):
        cmds.add(ind)
        ind, acc = handleCmd(arr, ind, acc)
    if ind == len(arr):
        return True, acc
    return False, acc

# for i in range(350):
#     ret = a8(i)
#     if ret[0]:
#         print ret
#         break

def valid(arr, ind):
    base = arr[ind-25: ind]
    val = arr[ind]
    for i in range(25):
        for j in range(25):
            if i == j:
                continue
            if base[i] + base[j] == val:
                return True
    return False

def a9():
    arr = [int(i) for i in open("inp/inp09").readlines()]
    for i in range(25, len(arr)):
        if valid(arr, i) is False:
            print i, arr[i]
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
    print i0, i1
    print min(arr[i0:i1]) + max(arr[i0:i1])

a9()