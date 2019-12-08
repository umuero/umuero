# https://adventofcode.com/2019

# -- INTCODE A2-A5 ---------------------------------------------------
def intcode_run(mem):
    index = 0
    lastIndex = -1
    while index != lastIndex:
        lastIndex = index
        mem, index = intcode_step(mem, index)
        # print(index, mem)
    return mem

def intcode_step(arr, iPtr):
    opCode = arr[iPtr] % 100
    if opCode == 1:
        arr[arr[iPtr + 3]] = intcode_arg(arr, iPtr, 1) + intcode_arg(arr, iPtr, 2)
        return arr, iPtr + 4
    elif opCode == 2:
        arr[arr[iPtr + 3]] = intcode_arg(arr, iPtr, 1) * intcode_arg(arr, iPtr, 2)
        return arr, iPtr + 4
    elif opCode == 3:
        arr[arr[iPtr + 1]] = int(input("Program input:"))
        return arr, iPtr + 2
    elif opCode == 4:
        print("Program output: ", intcode_arg(arr, iPtr, 1))
        return arr, iPtr + 2
    elif opCode == 5:
        if intcode_arg(arr, iPtr, 1):
            return arr, intcode_arg(arr, iPtr, 2)
        return arr, iPtr + 3
    elif opCode == 6:
        if intcode_arg(arr, iPtr, 1) == 0:
            return arr, intcode_arg(arr, iPtr, 2)
        return arr, iPtr + 3
    elif opCode == 7:
        arr[arr[iPtr + 3]] = 1 if intcode_arg(arr, iPtr, 1) < intcode_arg(arr, iPtr, 2) else 0
        return arr, iPtr + 4
    elif opCode == 8:
        arr[arr[iPtr + 3]] =  1 if intcode_arg(arr, iPtr, 1) == intcode_arg(arr, iPtr, 2) else 0
        return arr, iPtr + 4
    elif opCode == 99:
        print("Program 99")
        return arr, iPtr
    return arr, iPtr + 4

def intcode_arg(arr, iPtr, argNo):
    if arr[iPtr] // (10 ** (argNo + 1)) % 10:
        return arr[iPtr + argNo]
    else:
        return arr[arr[iPtr + argNo]]


def a1():
    inp = open("inp/inp01").readlines()

    mass = [int(i) for i in inp]
    fuel = [i // 3 - 2 for i in mass]
    print(sum(fuel))

    extra = 0
    for f in fuel:
        while f > 0:
            f = f // 3 - 2
            if f > 0:
                extra += f
            else:
                break
    print(sum(fuel) + extra)
def a2():
    arr = [int(i) for i in open("inp/inp02").read().split(",")]
    for n in range(100):
        for v in range(100):
            mem = arr.copy()
            mem[1] = n
            mem[2] = v
            if intcode_run(mem)[0] == 19690720:
                print(n, v)
                break
def a3():
    lines = [i.split(",") for i in open("inp/inp03").readlines()]
    m = dict()
    cross = []
    for lctr, line in enumerate(lines):
        x, y, d = 0, 0, 0
        for cmd in line:
            length = int(cmd[1:])
            if cmd[0] == "R":
                for i in range(1, length + 1):
                    if m.get((x+i, y), (lctr, 0))[0] != lctr:
                        cross.append((x+i, y, m[x+i, y][1], d+i))
                    else:
                        m[(x + i, y)] = (lctr, d + i)
                x += length
            if cmd[0] == "L":
                for i in range(1, length + 1):
                    if m.get((x-i, y), (lctr, 0))[0] != lctr:
                        cross.append((x-i, y, m[x-i, y][1], d+i))
                    else:
                        m[(x - i, y)] = (lctr, d + i)
                x -= length
            if cmd[0] == "U":
                for i in range(1, length + 1):
                    if m.get((x, y+i), (lctr, 0))[0] != lctr:
                        cross.append((x, y+i, m[x, y+i][1], d+i))
                    else:
                        m[(x, y + i)] = (lctr, d + i)
                y += length
            if cmd[0] == "D":
                for i in range(1, length + 1):
                    if m.get((x, y-i), (lctr, 0))[0] != lctr:
                        cross.append((x, y-i, m[x, y-i][1], d+i))
                    else:
                        m[(x, y - i)] = (lctr, d + i)
                y -= length
            d += length
    # print(cross)
    print(sorted([abs(i[0]) + abs(i[1]) for i in cross])[0])
    print(sorted([(i[2] + i[3], abs(i[0]) + abs(i[1])) for i in cross])[0])
def a4(pMin=183564, pMax=657474):
    ctr = 0
    for p in range(pMin, pMax):
        chars = str(p)
        if sorted(chars) != list(chars):
            continue
        # if not sum([chars[i] == chars[i+1] for i in range(len(chars)-1)]): # P1
        if 2 not in [chars.count(i) for i in chars]: # P2
            continue
        ctr += 1
    print(ctr)
def a5():
    arr = [int(i) for i in open("inp/inp05").read().split(",")]
    intcode_run(arr)

def a6(): 
    orb = [i.strip().split(")") for i in open("inp/inp06").readlines()]
    m = {v:k for k,v in orb}
    checksums = {'COM': 0}
    for node in m.keys():
        orbital_fill_checksum(m, checksums, node)
    print("Orbital checksum:", sum(checksums.values()))
    pSanta = orbital_path(m, m['SAN'])
    pYou = orbital_path(m, m['YOU'])
    print (len(pSanta), len(pYou))
    print("Jump:", len(pSanta) + len(pYou) - 2 * len(set(pSanta).intersection(pYou)))

def orbital_fill_checksum(m, checksums, node):
    if node not in checksums:
        checksums[node] = orbital_fill_checksum(m, checksums, m[node]) + 1
    return checksums[node]

def orbital_path(m, node):
    if node not in m:
        return [node]
    return orbital_path(m, m[node]) + [node]


a6()

