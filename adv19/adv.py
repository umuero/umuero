# https://adventofcode.com/2019


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


# --------------------------------------------------------------------
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


def intcode_run(mem):
    index = 0
    lastIndex = -1
    while index != lastIndex:
        lastIndex = index
        mem, index = intcode_step(mem, index)
        # print(index, mem)
    return mem


def intcode_step(arr, iPtr):
    if arr[iPtr] == 1:
        arr[arr[iPtr + 3]] = arr[arr[iPtr + 1]] + arr[arr[iPtr + 2]]
    elif arr[iPtr] == 2:
        arr[arr[iPtr + 3]] = arr[arr[iPtr + 1]] * arr[arr[iPtr + 2]]
    elif arr[iPtr] == 99:
        return arr, iPtr
    return arr, iPtr + 4


# --------------------------------------------------------------------
def a3():
    lines = [i.split(",") for i in open("inp/inp03t").readlines()]
    m = dict()
    cross = []
    for lctr, line in enumerate(lines):
        x, y = 0, 0
        for cmd in line:
            length = int(cmd[1:])
            if cmd[0] == "R":
                for i in range(1, length + 1):
                    if (x + i, y) in m:
                        cross.append((x+i, y))
                        print(x + i, y)
                    m[(x + i, y)] = lctr
                x += length
            if cmd[0] == "L":
                for i in range(1, length + 1):
                    if (x - i, y) in m:
                        cross.append((x-i, y))
                        print(x - i, y)
                    m[(x - i, y)] = lctr
                x -= length
            if cmd[0] == "U":
                for i in range(1, length + 1):
                    if (x, y + i) in m:
                        cross.append((x, y+i))
                        print(x, y + i)
                    m[(x, y + i)] = lctr
                y += length
            if cmd[0] == "D":
                for i in range(1, length + 1):
                    if (x, y - i) in m:
                        cross.append((x, y-i))
                        print(x, y - i)
                    m[(x, y - i)] = lctr
                y -= length
    print(cross)
    print(sorted([abs(i[0]) + abs(i[1]) for i in cross])[0])


a3()

