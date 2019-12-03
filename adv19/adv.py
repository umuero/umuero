#https://adventofcode.com/2019

def a1():
    inp = open("inp/inp01").readlines()

    mass = [int(i) for i in inp]
    fuel = [i//3-2 for i in mass]
    print (sum(fuel))

    extra = 0
    for f in fuel:
        while f > 0:
            f = f // 3 - 2
            if f > 0:
                extra += f
            else:
                break
    print (sum(fuel) + extra)


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
        #print(index, mem)
    return mem

def intcode_step(arr, iPtr):
    if arr[iPtr] == 1:
        arr[arr[iPtr + 3]] = arr[arr[iPtr + 1]] + arr[arr[iPtr + 2]]
    elif arr[iPtr] == 2:
        arr[arr[iPtr + 3]] = arr[arr[iPtr + 1]] * arr[arr[iPtr + 2]]
    elif arr[iPtr] == 99:
        return arr, iPtr
    return arr, iPtr + 4

a2()


