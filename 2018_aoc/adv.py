#https://adventofcode.com/2018
def a1():
    inp = open("inp/adv-1.inp").readlines()
    b = 0
    notFound = True
    olds = set()
    olds.add(b)

    for ctr in range(300):
        for k in inp:
            b += int(k)
            if b in olds:
                notFound = False
                print b, "in loop ctr", ctr
                break
            olds.add(b)
        if not notFound:
            break

def a2():
    inp = [l.strip() for l in open("inp/adv-2.inp").readlines()]
    tot = dict()
    for line in inp:
        l = dict()
        for c in line:
            l[c] = l.get(c, 0) + 1
        for v in set(l.values()):
            if v > 1:
                tot[v] = tot.get(v, 0) + 1
    print tot
    ret = 1
    for vv, v in tot.items():
        ret *= v
    print ret
    vals = dict()
    for ctr, line in enumerate(inp):
        for i in range(len(line)):
            val = "".join(line[:i] + line[i+1:])
            if val in vals:
                print val, ctr, vals[val]
            vals[val] = ctr

def a3():
    import re
    inp = [l.strip() for l in open("inp/adv-3.inp").readlines()]
    tot = dict()
    dupes = dict()
    ids = set()
    dupIds = set()
    r = re.compile(r"#(\d*) @ (\d*),(\d*): (\d*)x(\d*)")
    for line in inp:
        m = r.match(line)
        if not m:
            print "reg exp fail"
        cid, cx, cy, cw, ch = m.groups()
        ids.add(cid)
        cx = int(cx)
        cy = int(cy)
        for x in range(int(cw)):
            for y in range(int(ch)):
                if (cx + x, cy + y) in tot:
                    dupes[(cx + x, cy + y)] = tot[(cx + x, cy + y)] + " " + cid
                    dupIds.add(cid)
                    dupIds.add(tot[(cx + x, cy + y)])
                tot[(cx + x, cy + y)] = cid
    print len(dupes)
    print ids.difference(dupIds)

def a4():
    from collections import defaultdict
    import re
    inp = [l.strip() for l in open("inp/adv-4.inp").readlines()]
    r = re.compile(r"\[(\d+)-(\d+)-(\d+) (\d+):(\d+)\] ([^\n]*)")

    cron = dict()
    for line in inp:
        m = r.match(line)
        if not m:
            print "reg exp fail"
        dyead, dmonth, dday, dhour, dmin, text = m.groups()
        cron[(dmonth, dday, dhour, dmin)] = text

    sk = sorted(cron.keys())
    agents = dict()
    agentActive = "umuero"
    agentStatus = 1 # awake
    lastDate = ("1518", "01", "00", "00", "00")
    for dt in sk:
        text = cron[dt]
        if text.startswith("Guard"):
            agentActive = text.split()[1]
            if agentActive not in agents:
                agents[agentActive] = defaultdict(int)
            agentStatus = 1
        if text == "falls asleep":
            agentStatus = 0
        if text == "wakes up":
            agents[agentActive]['sleep'] += int(dt[-1]) - int(lastDate[-1])
            for i in range(int(dt[-1]) - int(lastDate[-1])):
                agents[agentActive][int(lastDate[-1]) + i] += 1
            agentStatus = 1
        lastDate = dt

    maxSleep = sorted([(agent, ad.get("sleep", 0)) for agent, ad in agents.items()], key=lambda x: x[1], reverse=True)[0]
    maxMin = sorted([(minute, val) for minute, val in agents[maxSleep[0]].items()], key=lambda x: x[1], reverse=True)[1]
    print maxSleep[0], maxMin[0], int(maxSleep[0][1:]) * int(maxMin[0])

    maxMin2 = sorted([(minute, val, agent) for agent in agents.keys() for minute, val in agents[agent].items() if minute != "sleep"], key=lambda x: x[1], reverse=True)[0]
    print maxMin2, int(maxMin2[2][1:]) * int(maxMin2[0])

def a5():
    import copy
    inp = open("inp/adv-5.inp").read().strip()
    stack = []
    rem = dict()
    for c in inp:
        if c.lower() not in rem:
            rem[c.lower()] = copy.copy(stack)
        for rc, rstack in rem.items():
            if c.lower() == rc:
                continue
            if len(rstack) > 0 and rstack[-1].lower() == c.lower() and rstack[-1] != c:
                rstack.pop()
            else:
                rstack.append(c)

        if len(stack) > 0 and stack[-1].lower() == c.lower() and stack[-1] != c:
            stack.pop()
        else:
            stack.append(c)

    print len(stack)
    print sorted([(len(rstack), rc) for rc, rstack in rem.items()])[0]

def a6():
    from collections import Counter
    inp = [l.strip().split(", ") for l in open("inp/adv-6.inp").readlines()]
    points = [(int(i[0]), int(i[1])) for i in inp]
    def near(px, py):
        dists = sorted([(abs(x-px)+abs(y-py), x, y) for x,y in points])
        if dists[0][0] == dists[1][0]:
            return "."
        return dists[0][1], dists[0][2]
    def dsum(px, py):
        return sum([abs(x-px)+abs(y-py) for x,y in points])
    # maxX, maxY -> loop ? tatsiz
    res = dict()
    dCtr = 0
    infs = set()
    maxX = sorted(points, key=lambda x: x[0], reverse=True)[0][0]
    maxY = sorted(points, key=lambda x: x[1], reverse=True)[0][1]
    for x in range(maxX+1):
        for y in range(maxY+1):
            res[x,y] = near(x, y)
            if x == 0 or y == 0 or x == maxX or y == maxY:
                infs.add(res[x,y])
            if dsum(x, y) < 10000:
                dCtr += 1
    stat = sorted(Counter(res.values()).items(), key=lambda x:x[1], reverse=True)
    for p, dist in stat:
        if p not in infs and p != '.':
            print dist, p
            break
    print dCtr

def a7():
    inp = [(l.split()[1],l.split(" ")[7]) for l in open("inp/adv-7.inp").readlines()]
    keys = set()
    req = dict()
    for x, y in inp:
        keys.add(x)
        keys.add(y)
        if y not in req:
            req[y] = []
        req[y].append(x)

    order = []
    def firstOrder(finished):
        for k in sk:
            if k in order:
                continue
            if k not in req:
                return k
            if len(set(req[k]).difference(set(finished))) == 0:
                return k
        return "."

    sk = sorted(keys)
    for i in sk:
        order.append(firstOrder(order))
    print "".join(order)

    sec = 0
    ww = [0] * 5
    order = []
    finish = dict()
    while len(sk) > len(finish):
        for wctr, ws in enumerate(ww):
            if ws <= sec:
                o = firstOrder([i for i in finish.keys() if finish[i] <= sec])
                if o == ".":
                    continue
                order.append(o)
                ww[wctr] = sec + ord(o) - 4 # -64 + 60
                print sec, wctr, o, ww[wctr]
                finish[o] = ww[wctr]
        sec += 1
    print max(finish.values())

def a8():
    inp = [int(l) for l in open("inp/adv-8.inp").read().split()]
    # inp = [int(l) for l in open("inp/adv-8.ex").read().split()]
    def processTree(ind):
        ch, mt = inp[ind], inp[ind + 1]
        msum, val = 0, 0
        chs = dict()
        tctr = 2
        for i in range(ch):
            ret = processTree(ind + tctr)
            chs[i+1] = ret
            tctr += ret[0]
            msum += ret[1]
        for i in range(mt):
            msum += inp[ind + tctr + i]
            # print ch, inp[ind + tctr + i]
            if ch > 0 and inp[ind + tctr + i] <= ch:
                val += chs[inp[ind + tctr + i]][2]
        if ch == 0:
            val = msum
        tctr += mt
        return tctr, msum, val
    print processTree(0)

def a9():
    class N:
        def __init__(self, val=None):
            self.v = val
            self.p = None
            self.n = None
    st = N(0)
    st.n = st
    st.p = st
    head = st
    # numP, lastM = 5, 25 # 32
    # numP, lastM = 10, 1618 # 8317
    # numP, lastM = 13, 7999 # 146373
    numP, lastM = 432, 71019
    # numP, lastM = 432, 7101900
    pl = [0] * numP

    for nextM in range(lastM + 1)[1:]:
        pCtr = nextM % numP
        if nextM % 10000 == 0:
            print nextM
        if nextM % 23 == 0:
            head = head.p.p.p.p.p.p
            pl[pCtr] += head.p.v + nextM
            head.p.p.n = head
            head.p = head.p.p
        else:
            n = N(nextM)
            n.p = head.n
            n.n = head.n.n
            head.n.n = n
            n.n.p = n
            head = n
    print max(pl)

def a10():
    import re
    inp = [l.strip() for l in open("inp/adv-10.inp").readlines()]
    pl = dict()
    r = re.compile(r"position=<([^>]+)>\W*velocity=<([^>]+)>")
    for pctr, line in enumerate(inp):
        m = r.match(line)
        if not m:
            print "reg exp fail", line
        x, v = m.groups()
        pl["%d" % pctr] = (int(x.split(",")[0]),int(x.split(",")[-1]),int(v.split(",")[0]),int(v.split(",")[-1]))

    sec = 0
    cont = True
    while cont:
        print "====", sec
        sk = sorted(pl.keys())
        mult = 1
        if pl["0"][0] > 1000:
            mult = 100
        sec += 1 * mult
        for pn in sk:
            p = pl[pn]
            pl[pn] = (p[0] + p[2] * mult, p[1] + p[3] * mult, p[2], p[3])
        aff = 0
        for p in pl.values():
            for x in pl.values():
                if x == p:
                    continue
                if (abs(p[0] - x[0]) == 1 and p[1] == x[1]) or (abs(p[1] - x[1]) == 1 and p[0] == x[0]):
                    print p, x
                    aff += 1
        print aff, len(pl)
        if aff > len(pl) * 1.5:
            cont = False
            dd = set()
            for p in pl.values():
                dd.add((p[0], p[1]))
            sx = sorted(dd)
            minX = sx[0][0]
            maxX = sx[-1][0]
            sx = sorted(dd, key=lambda x:x[1])
            minY = sx[0][1]
            maxY = sx[-1][1]
            for y in range(minY - 1, maxY + 2):
                for x in range(minX - 1, maxX + 2):
                    if (x, y) in dd:
                        print "X",
                    else:
                        print ".",
                print ""
        print sec


def a11():
    serial = 7347
    def powerL(px, py, ser):
        rackId = px + 10
        return ((((rackId * py) + ser) * rackId) / 100 % 10) - 5
    dt = {(x,y):powerL(x,y,serial) for y in range(1,301) for x in range(1,301)}
    dts = dict()
    for x in range(1,301):
        for y in range(1,301):
            dts[x,y] = dt[x,y] + dts.get((x,y-1),0) + dts.get((x-1,y),0) - dts.get((x-1, y-1), 0)
    max = -10
    curXY = 0,0
    for x in range(1,299):
        for y in range(1,299):
            curS = dts[x+2,y+2] + dts.get((x-1,y-1),0) - dts.get((x+2,y-1),0) - dts.get((x-1,y+2),0)
            if curS > max:
                max = curS
                curXY = x,y
    print curXY, max

    max2 = -10
    curXYD = 0,0,0
    for d in range(1,301):
        for x in range(1,301-d):
            for y in range(1,301-d):
                curS = dts[x+d-1,y+d-1] + dts.get((x-1,y-1),0) - dts.get((x+d-1,y-1),0) - dts.get((x-1,y+d-1),0)
                if curS > max2:
                    max2 = curS
                    curXYD = x,y,d
    print curXYD, max2

def a12():
    inp = [l.strip().split() for l in open("inp/adv-12.inp").readlines()]
    state = inp[0][-1]
    st = {c:i for c,i in enumerate(state) if i=='#'}
    r = {i[0]: i[2] for i in inp[2:] if i[2] == '#'}

    print "0", "".join([st.get(i, '.') for i in range(min(st.keys()) - 1, max(st.keys()) + 2)])
    for gen in range(1,21):
        newSt = dict()
        for i in range(min(st.keys()) - 4, max(st.keys()) + 5):
            ss = "".join([st.get(i + c, '.') for c in range(-2, 3)])
            if ss in r:
                newSt[i] = r[ss]
        st = newSt
    print sum(st.keys())

    state = inp[0][-1]
    st = {c:i for c,i in enumerate(state) if i=='#'}
    r = {i[0]: i[2] for i in inp[2:] if i[2] == '#'}

    print "0", "".join([st.get(i, '.') for i in range(min(st.keys()) - 1, max(st.keys()) + 2)])
    linear = []
    for gen in range(1,4001):
        newSt = dict()
        for i in range(min(st.keys()) - 4, max(st.keys()) + 5):
            ss = "".join([st.get(i + c, '.') for c in range(-2, 3)])
            if ss in r:
                newSt[i] = r[ss]
        st = newSt
        if gen % 100 == 0:
            print gen, sum(st.keys())
            linear.append((gen, sum(st.keys())))

    l1 = linear[-1]
    l2 = linear[-2]
    print l1[1] + (l1[1] - l2[1]) * (50000000000 - l1[0])/ (l1[0] - l2[0])

def a13():
    inp = open("inp/adv-13.inp").readlines()
    # inp = open("inp/adv-13.ex").readlines()
    trans = {">": "-", "<": "-", "v": "|", "^": "|"}
    tr = { # ['L', 'S', 'R']
        (">", "/"): "^",
        ("v", "/"): "<",
        ("<", "/"): "v",
        ("^", "/"): ">",
        (">", "\\"): "v",
        ("v", "\\"): ">",
        ("<", "\\"): "^",
        ("^", "\\"): "<",
        (">", "+"): "^>v",
        ("v", "+"): ">v<",
        ("<", "+"): "v<^",
        ("^", "+"): "<^>"}
    agents = []
    m = dict()
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            if x == ' ' or x == '\n':
                continue
            if x in trans:
                agents.append({"l": lCtr, "h": hCtr, "d": x, "o": 0})
            m[lCtr,hCtr] = trans.get(x, x)

    tick = 0
    while len(agents) > 1:
        ags = sorted([(x['l'], x['h'], i) for i, x in enumerate(agents)])
        xy = [(a[0], a[1]) for a in ags]
        for aIndex in ags:
            ag = agents[aIndex[2]]
            if 'c' in ag:
                continue
            xy.remove((ag['l'], ag['h']))
            if ag['d'] == '>':
                ag['h'] += 1
            if ag['d'] == '<':
                ag['h'] -= 1
            if ag['d'] == '^':
                ag['l'] -= 1
            if ag['d'] == 'v':
                ag['l'] += 1
            if (ag['l'], ag['h']) in xy:
                print (ag['l'], ag['h'])
                for cag in agents:
                    if cag['l'] == ag['l'] and cag['h'] == ag['h']:
                        print "removing", cag
                        cag['c'] = True
            xy.append((ag['l'], ag['h']))
            mc = m[ag['l'], ag['h']]
            if (ag['d'], mc) in tr:
                if mc == '+':
                    ag['d'] = tr[ag['d'], mc][ag['o']]
                    ag['o'] += 1
                    ag['o'] = ag['o'] % 3
                else:
                    ag['d'] = tr[ag['d'], mc]
        agents = [i for i in agents if 'c' not in i]
        tick += 1
        print tick, [(i['l'],i['h'],i['d'],i['o']) for i in agents]


def a14():
    # def prn(st, e1, e2):
    #     pr = ["", ".", ":", "~"]
    #     print " ".join([str(i) + pr[(c==e1) + 2*(c==e2)]  for c,i in enumerate(st)])
    st = list("37")
    e1, e2, llen = 0, 1, 2
    goal = 513401
    while len(st) < 10 + goal:
        score = int(st[e1]) + int(st[e2])
        st.extend("%d" % score)
        e1 = (e1 + 1 + int(st[e1])) % len(st)
        e2 = (e2 + 1 + int(st[e2])) % len(st)
    print "".join(st[goal:goal+10])

    from collections import deque
    # st = list("37")
    # e1, e2, llen = 0, 1, 2
    gl = "513401"
    last = deque(st[-len(gl):])

    while True:
        score = int(st[e1]) + int(st[e2])
        found = False
        for c in str(score):
            st.append(c)
            last.append(c)
            if len(last) > len(gl):
                last.popleft()
            if gl == "".join(last):
                found = True
                break
        if found:
            break
        e1 = (e1 + 1 + int(st[e1])) % len(st)
        e2 = (e2 + 1 + int(st[e2])) % len(st)
        # print " ".join([str(i) + pr[(c==e1) + 2*(c==e2)]  for c,i in enumerate(st)])
    print len(st) - len(gl)


def a15():
    inp, test = open("inp/adv-15.inp").readlines(), "90 * 2555 = 229950"
    # inp, test = open("inp/adv-15.ex").readlines(), "37 * 982 = 36334" # OK
    # inp, test = open("inp/adv-15.ex2").readlines(), "46 * 859 = 39514" # OK
    # inp, test = open("inp/adv-15.ex3").readlines(), "54 * 536 = 28944" # OK
    # inp, test = open("inp/adv-15.ex4").readlines(), "20 * 937 = 18740" # OK
    hp, dmg = 200, 3
    boost = 13
    agents = []
    m = dict()
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            if x == '\n':
                continue
            if x == 'G':
                agents.append({"l": lCtr, "h": hCtr, "type": x, "hp": hp, "dmg": dmg, "ind": len(agents)})
            if x == 'E':
                agents.append({"l": lCtr, "h": hCtr, "type": x, "hp": hp, "dmg": boost, "ind": len(agents)})
            m[lCtr,hCtr] = x

    def pr():
        minX = min(sorted([i[0] for i in m.keys()]))
        maxX = max(sorted([i[0] for i in m.keys()]))
        minY = min(sorted([i[1] for i in m.keys()]))
        maxY = max(sorted([i[1] for i in m.keys()]))
        for x in range(minX, maxX+1):
            print "".join([m.get((x,y), "#") for y in range(minY, maxY+1)])

    def adjacent(p):
        return [(p[0]-1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1), (p[0]+1, p[1])]

    def breadth(ag):
        dm = {(ag['l'], ag['h']): 0}
        level = [(ag['l'], ag['h'])]
        dist = 0
        while level:
            newLevel = set()
            for p in level:
                dm[p[0], p[1]] = dist
                for ap in adjacent(p):
                    if ap not in dm and m[ap] != '#':
                        if m[ap] == '.':
                            newLevel.add(ap)
                        elif m[ap] != ag['type']:
                            # print "found enemy attack location at from", p, ag
                            # dm uzerinde dist-1 arayarak geri sar, dist 1 move location
                            for i in range(dist, 0, -1):
                                for bt in adjacent(p):
                                    if dm.get(bt, 0) == i:
                                        p = bt
                                        break
                            return p
            level = sorted(newLevel)
            dist += 1

    def nearTarget(ag, agents):
        minEn = {'hp': hp + 1, 'ro': -1, 'ind': -1}
        for ap in adjacent((ag['l'], ag['h'])):
            if m[ap] == ('G' if ag['type'] == 'E' else 'E'):
                # esitligi round basi order belli ediyo
                enemy = [x for x in agents if x['l'] == ap[0] and x['h'] == ap[1] and x['hp'] > 0][0]
                if (minEn['hp'], minEn['ro']) > (enemy['hp'], enemy['ro']):
                    minEn = enemy
        return minEn['ind']
    pr()

    tick = 0
    while len(agents) > 1:
        sortedAgents = sorted([(x['l'], x['h'], i) for i, x in enumerate(agents)])
        for roundOrd, ag in enumerate(sortedAgents):
            agents[ag[2]]['ro'] = roundOrd
        for aL, aH, aIndex in sortedAgents:
            ag = agents[aIndex]
            if ag['hp'] == 0:
                continue

            nearEnemyId = nearTarget(ag, agents)
            if nearEnemyId == -1:
                # move - breadth first enemy adjacent - reading order
                nextPos = breadth(ag)
                print "moving", ag, nextPos
                if nextPos is not None:
                    m[ag['l'], ag['h']] = '.'
                    ag['l'] = nextPos[0]
                    ag['h'] = nextPos[1]
                    m[ag['l'], ag['h']] = ag['type']
                    nearEnemyId = nearTarget(ag, agents)

            if nearEnemyId != -1:
                # attack - adjacent min(hp) enemy -- in reading order
                print "attack", ag, agents[nearEnemyId]
                agents[nearEnemyId]['hp'] -= ag['dmg']
                if agents[nearEnemyId]['hp'] <= 0:
                    agents[nearEnemyId]['hp'] = 0
                    m[agents[nearEnemyId]['l'], agents[nearEnemyId]['h']] = '.'
        pr()
        hpE = sum([i['hp'] for i in agents if i['type'] == 'E' and i['hp'] > 0])
        hpG = sum([i['hp'] for i in agents if i['type'] == 'G' and i['hp'] > 0])
        if hpE == 0 or hpG == 0:
            break
        tick += 1

    print test, boost, sum([1 for i in agents if i['type'] == 'E']), sum([1 for i in agents if i['type'] == 'E' and i['hp'] > 0])
    print tick, sum([a['hp'] for a in agents if a['hp'] > 0])
    print tick * sum([a['hp'] for a in agents if a['hp'] > 0])

def opcode(regs, op, a, b, c):
    if op == 'addr':
        regs[c] = regs[a] + regs[b]
    if op == 'addi':
        regs[c] = regs[a] + b
    if op == 'mulr':
        regs[c] = regs[a] * regs[b]
    if op == 'muli':
        regs[c] = regs[a] * b
    if op == 'banr':
        regs[c] = regs[a] & regs[b]
    if op == 'bani':
        regs[c] = regs[a] & b
    if op == 'borr':
        regs[c] = regs[a] | regs[b]
    if op == 'bori':
        regs[c] = regs[a] | b
    if op == 'setr':
        regs[c] = regs[a]
    if op == 'seti':
        regs[c] = a
    if op == 'gtir':
        regs[c] = int(a > regs[b])
    if op == 'gtri':
        regs[c] = int(regs[a] > b)
    if op == 'gtrr':
        regs[c] = int(regs[a] > regs[b])
    if op == 'eqir':
        regs[c] = int(a == regs[b])
    if op == 'eqri':
        regs[c] = int(regs[a] == b)
    if op == 'eqrr':
        regs[c] = int(regs[a] == regs[b])
    return regs

def a16():
    import re
    from collections import defaultdict
    inp = open("inp/adv-16.inp").read()
    opDict = defaultdict(set)
    def tryOps(regs, oplist, nextR):
        ctr = 0
        for op in ['addr','addi','mulr','muli','banr','bani','borr','bori','setr','seti','gtir','gtri','gtrr','eqir','eqri','eqrr']:
            if nextR == opcode([i for i in regs], op, oplist[1], oplist[2], oplist[3]):
                ctr += 1
                opDict[oplist[0]].add(op)
        return ctr

    r = re.compile(r"Before: \[(\d+), (\d+), (\d+), (\d+)\]\n(\d+) (\d+) (\d+) (\d+)\nAfter:  \[(\d+), (\d+), (\d+), (\d+)\]")
    ops = []
    opCtr = 0
    for m in r.findall(inp):
        vl = [int(i) for i in m]
        exc = (vl[:4], vl[4:8], vl[8:])
        ops.append(exc)
        if tryOps(exc[0], exc[1], exc[2]) >= 3:
            opCtr += 1
    print opCtr
    print opDict
    foundOps = dict()
    while len(foundOps) < 16:
        print "loop foundOps", len(foundOps)
        for opId, opSet in sorted(opDict.items(), key=lambda x: len(x[1])):
            opSet = opSet.difference(foundOps.values())
            if len(opSet) == 1:
                foundOps[opId] = list(opSet)[0]

    rr = [0, 0, 0, 0]
    part2Index = inp.find('\n\n\n')
    for opStr in inp[part2Index:].strip().split("\n"):
        op, a, b, c = [int(i) for i in opStr.split()]
        rr = opcode(rr, foundOps[op], a, b, c)
    print rr

def a17():
    import re
    r = re.compile(r"(\w)\=(\d+), \w\=(\d+)\.\.(\d+)")
    inp = open("inp/adv-17.inp").readlines()

    def pr():
        for y in range(maxY+1):
            print "".join([m.get((x,y), ".") for x in range(minX-1, maxX+2)])

    sx, sy = 500, 0
    m = dict()
    for line in inp:
        mg = r.match(line).groups()
        print mg
        for i in range(int(mg[2]), int(mg[3]) + 1):
            if mg[0] == 'x':
                m[int(mg[1]), i] = '#'
            else:
                m[i, int(mg[1])] = '#'

    minX = min(sorted([i[0] for i in m.keys()]))
    maxX = max(sorted([i[0] for i in m.keys()]))
    minY = min(sorted([i[1] for i in m.keys()]))
    maxY = max(sorted([i[1] for i in m.keys()]))

    br = [(sx, sy)]
    active = []
    lastActive = ""
    while len(br):
        p = br.pop(0)
        m[p[0], p[1]] = '|'
        active.append((p[0], p[1]))
        if (p[0], p[1] + 1) not in m or m[p[0], p[1] + 1] == '|':
            # alt bos assagi devam
            if (p[0], p[1] + 1) not in m and p[1] < maxY + 2:
                br.append((p[0], p[1] + 1))
        else:
            # alt su yada, tas (sag sol yayil)
            if (p[0] + 1, p[1]) not in m:
                br.append((p[0] + 1, p[1]))
            if (p[0] - 1, p[1]) not in m:
                br.append((p[0] - 1, p[1]))
        print "======", len(br), p
        # pr()
        if len(br) == 0 and len(active) > 0:
            currActive = "".join([str(i) for i in sorted(active)])
            if currActive == lastActive:
                break
            lastActive = currActive
            # aktif sular durulacak
            for a in active:
                still = True
                ctr = 0
                while True:
                    ctr += 1
                    if (a[0] + ctr, a[1]) not in m:
                        still = False
                        break
                    if m[a[0] + ctr, a[1]] == '#':
                        break
                ctr = 0
                while True:
                    ctr += 1
                    if (a[0] - ctr, a[1]) not in m:
                        still = False
                        break
                    if m[a[0] - ctr, a[1]] == '#':
                        break
                if not still:
                    br.append(a)
                else:
                    m[a[0], a[1]] = "~"
            active = []

    p1 = "".join([m.get((x,y), ".") for x in range(minX-1, maxX+2) for y in range(minY, maxY+1)])
    print p1.count('~') + p1.count("|")
    print p1.count('~')

def a18():
    inp = open("inp/adv-18.inp").readlines()
    # inp = open("inp/adv-18.ex").readlines()
    def conv(mm, xx, yy):
        n = mm.get((xx-1, yy-1), '') + mm.get((xx-1, yy), '') + mm.get((xx-1, yy+1), '') + mm.get((xx, yy-1), '') + mm.get((xx, yy+1), '') + mm.get((xx+1, yy-1), '') + mm.get((xx+1, yy), '') + mm.get((xx+1, yy+1), '')
        if mm[xx, yy] == '.' and n.count('|') >= 3:
            return '|'
        if mm[xx, yy] == '|' and n.count('#') >= 3:
            return '#'
        if mm[xx, yy] == '#' and (n.count('|') == 0 or n.count('#') == 0):
            return '.'
        return mm[xx,yy]

    m = [dict(), dict()]
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            m[0][lCtr,hCtr] = x

    mnt = 0
    print "===", mnt
    for y in range(10):
        print "".join([m[mnt%2][x,y] for x in range(10)])
    while mnt < 10:
        # m[mnt % 2] -> m[mnt % 2 + 1]
        for x,y in m[mnt%2].keys():
            m[(mnt + 1) % 2][(x,y)] = conv(m[mnt%2], x, y)
        mnt += 1
        print "===", mnt
        for x in range(10):
            print "".join([m[mnt%2][x,y] for y in range(10)])
    mp = "".join(m[mnt%2].values())
    print mp.count("#") * mp.count("|")

    olds = dict()
    m = [dict(), dict()]
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            m[0][lCtr,hCtr] = x
    olds["".join([m[mnt%2][x,y] for x in range(10) for y in range(10)])] = mnt

    mnt = 0
    lp = 0
    print "===", mnt
    while mnt < 1000000000:
        # m[mnt % 2] -> m[mnt % 2 + 1]
        for x,y in m[mnt%2].keys():
            m[(mnt + 1) % 2][(x,y)] = conv(m[mnt%2], x, y)
        mnt += 1
        print "===", mnt
        for x in range(10):
            print "".join([m[mnt%2][x,y] for y in range(10)])
        mp = "".join([v for c,v in sorted(m[mnt%2].items()) ])
        if lp == 0 and mp in olds:
            print "found loop in", mnt, olds[mp]
            lp = mnt - olds[mp]
            mnt += ((1000000000 - mnt) / lp) * lp
            print "new mnt ", mnt, lp
        olds[mp] = mnt
    mp = "".join(m[mnt%2].values())
    print mp.count("#") * mp.count("|")

def a19():
    from collections import defaultdict
    inp = open("inp/adv-19.inp").readlines()
    # inp = open("inp/adv-19.ex").readlines()
    cmds = []
    ipp = -1
    for line in inp:
        ps = line.split()
        if ps[0] == "#ip":
            ipp = int(ps[1])
            continue
        cmds.append((ps[0], int(ps[1]), int(ps[2]), int(ps[3])))

    ip = 0
    rr = [0,0,0,0,0,0]
    ctrs = defaultdict(int)
    while ip < len(cmds):
        rr[ipp] = ip
        ctrs[ip] += 1
        opcode(rr, cmds[ip][0], cmds[ip][1], cmds[ip][2], cmds[ip][3])
        # loop ileri sarmaca
        if ip == 10:
            # fast forward
            if rr[1] * rr[3] < rr[5] - rr[1] - 10:
                rr[3] = rr[5] / rr[1]
            # break inner loop
            if rr[1] * rr[3] > rr[5]:
                rr[3] = rr[5]
        print ip, rr, cmds[ip][0], cmds[ip][1], cmds[ip][2], cmds[ip][3]
        ip = rr[ipp]
        ip += 1
    print rr
    #part2: reg5 common denominator toplami
    t, c = 0, 10550400 + rr[5]
    for i in range(1, c+1):
        if c%i==0:
            t += i
    print t

def a20():
    inpStr = open("inp/adv-20.inp").read()
    # inpStr = "^WSSEESWWWNW(S|NENNEEEENN(ESSSSW(NWSW|SSEN)|WSWWN(E|WWS(E|SS))))$"
    # inpStr = "^ENWWW(NEEE|SSE(EE|N))$"
    lenStr = len(inpStr)
    m = {(0, 0): 'X'}
    def consume(pos, index):
        while index < lenStr:
            newPos = set()
            if inpStr[index] == '^':
                index += 1
            if inpStr[index] == '$':
                return pos
            print "index",  index, inpStr[index], pos

            if inpStr[index] == 'N':
                for p in pos:
                    m[p[0], p[1]-1] = '-'
                    newPos.add((p[0], p[1]-2))
            if inpStr[index] == 'S':
                for p in pos:
                    m[p[0], p[1]+1] = '-'
                    newPos.add((p[0], p[1]+2))
            if inpStr[index] == 'W':
                for p in pos:
                    m[p[0]-1, p[1]] = '|'
                    newPos.add((p[0]-2, p[1]))
            if inpStr[index] == 'E':
                for p in pos:
                    m[p[0]+1, p[1]] = '|'
                    newPos.add((p[0]+2, p[1]))
            if inpStr[index] in 'NSWE':
                for p in newPos:
                    m[p[0], p[1]] = '.'
                index += 1
                pos = newPos
                continue

            if inpStr[index] == '|' or inpStr[index] == ')':
                return pos
            if inpStr[index] == '(':
                depth = 0
                brEnd = None
                childs = [index+1]
                for ind in range(index+1, lenStr):
                    if inpStr[ind] == '(':
                        depth += 1
                    if inpStr[ind] == ')':
                        depth -= 1
                    if depth < 0:
                        brEnd = ind + 1
                        break
                    if inpStr[ind] == '|' and depth == 0:
                        childs.append(ind+1)
                print "forking", childs, pos, brEnd
                for ch in childs:
                    chPos = consume(pos, ch)
                    print "fork ret", chPos
                    newPos = newPos.union(chPos)
                index = brEnd
                pos = newPos

    def pr():
        minX = min(sorted([i[0] for i in m.keys()]))
        maxX = max(sorted([i[0] for i in m.keys()]))
        minY = min(sorted([i[1] for i in m.keys()]))
        maxY = max(sorted([i[1] for i in m.keys()]))
        for y in range(minY-1, maxY+2):
            print "".join([m.get((x,y), "#") for x in range(minX-1, maxX+2)])

    consume({(0,0)}, 0)
    p2 = 0
    pos = {(0,0)}
    dist = 0
    while pos:
        newPos = set()
        for p in pos:
            m[p[0], p[1]] = str(dist)
            if dist >= 1000:
                p2 += 1
            if m.get((p[0]-1, p[1]), '') == '|' and m[p[0]-2, p[1]] == '.':
                newPos.add((p[0]-2, p[1]))
            if m.get((p[0]+1, p[1]), '') == '|' and m[p[0]+2, p[1]] == '.':
                newPos.add((p[0]+2, p[1]))
            if m.get((p[0], p[1]-1), '') == '-' and m[p[0], p[1]-2] == '.':
                newPos.add((p[0], p[1]-2))
            if m.get((p[0], p[1]+1), '') == '-' and m[p[0], p[1]+2] == '.':
                newPos.add((p[0], p[1]+2))
        pos = newPos
        dist += 1
    print dist-1, p2
    # pr()

def a21():
    inp = open("inp/adv-21.inp").readlines()
    cmds = []
    ipp = -1
    for line in inp:
        ps = line.split()
        if ps[0] == "#ip":
            ipp = int(ps[1])
            continue
        cmds.append((ps[0], int(ps[1]), int(ps[2]), int(ps[3])))

    ip = 0
    rr = [935350,0,0,0,0,0]
    while ip < len(cmds):
        rr[ipp] = ip
        opcode(rr, cmds[ip][0], cmds[ip][1], cmds[ip][2], cmds[ip][3])
        if ip == 28: # only rr[0] call
            print rr
            break
        print ip, rr, cmds[ip][0], cmds[ip][1], cmds[ip][2], cmds[ip][3]
        ip = rr[ipp]
        ip += 1
    print rr

    # compiled code
    current = 0
    seen = set()
    while True:
        prev = current | 65536
        current = 832312
        while True:
            current = (current + (prev & 255))*65899 & 16777215;
            if prev < 256:
                break
            prev /= 256
        if current in seen:
            break
        seen.add(current)
        print current

def a22():
    d, tx, ty = 8103, 9, 758
    m = {(0,0): 'M', (tx, ty): 'T'}
    g = {}
    def er(x,y):
        return (g[x,y] + d) % 20183
    def gI(x, y):
        if x == 0 and y == 0: return 0
        if x == tx and y == ty: return 0
        if y == 0: return x * 16807
        if x == 0: return y * 48271
        return er(x-1, y) * er(x, y-1)

    part1 = 0
    for y in range(ty+1):
        for x in range(tx+1):
            g[x,y] = gI(x,y)
            tt = er(x,y) % 3
            part1 += tt
            if tt == 0: m[x,y] = '.' # rocky
            if tt == 1: m[x,y] = '=' # wet
            if tt == 2: m[x,y] = '|' # narrow
    print part1

    maxX = 100 # extend margin
    for y in range(0, ty+maxX+1):
        for x in range(0, tx+maxX+1):
            g[x,y] = gI(x,y)
            tt = er(x,y) % 3
            if tt == 0: m[x,y] = '.' # rocky
            if tt == 1: m[x,y] = '=' # wet
            if tt == 2: m[x,y] = '|' # narrow

    for y in range(10):
        print "".join([m[x,y] for x in range(tx+1)])

    ok = {'.': 'TG', '=': 'GN', '|': 'TN'}
    br = [(0, 'T', 0,0)] # min, gear, pos
    prevs = dict()
    while len(br):
        curr = br.pop(0)
        if curr[1] == 'T' and curr[2] == tx and curr[3] == ty:
            print curr
            break
        if (curr[1], curr[2], curr[3]) in prevs:
            continue
        print curr, len(br)
        prevs[curr[1], curr[2], curr[3]] = curr[0]
        if ('T', curr[2], curr[3]) not in prevs and 'T' in ok[m[curr[2], curr[3]]]:
            br.append((curr[0] + 7, 'T', curr[2], curr[3]))
        if ('G', curr[2], curr[3]) not in prevs and 'G' in ok[m[curr[2], curr[3]]]:
            br.append((curr[0] + 7, 'G', curr[2], curr[3]))
        if ('N', curr[2], curr[3]) not in prevs and 'N' in ok[m[curr[2], curr[3]]]:
            br.append((curr[0] + 7, 'N', curr[2], curr[3]))

        if curr[2] < tx + maxX and (curr[1], curr[2] + 1, curr[3]) not in prevs and curr[1] in ok[m[curr[2] + 1, curr[3]]]:
            # prevs.add((curr[1], curr[2] + 1, curr[3]))
            br.append((curr[0] + 1, curr[1], curr[2] + 1, curr[3]))

        if curr[2] > 0 and (curr[1], curr[2] - 1, curr[3]) not in prevs and curr[1] in ok[m[curr[2] - 1, curr[3]]]:
            # prevs.add((curr[1], curr[2] - 1, curr[3]))
            br.append((curr[0] + 1, curr[1], curr[2] - 1, curr[3]))

        if curr[3] < ty + maxX and (curr[1], curr[2], curr[3] + 1) not in prevs and curr[1] in ok[m[curr[2], curr[3] + 1]]:
            # prevs.add((curr[1], curr[2], curr[3] + 1))
            br.append((curr[0] + 1, curr[1], curr[2], curr[3] + 1))

        if curr[3] > 0 and (curr[1], curr[2], curr[3] - 1) not in prevs and curr[1] in ok[m[curr[2], curr[3] - 1]]:
            # prevs.add((curr[1], curr[2], curr[3] - 1))
            br.append((curr[0] + 1, curr[1], curr[2], curr[3] - 1))
        br = sorted(br)

def a23():
    import re
    from itertools import product
    inp = open("inp/adv-23.inp").readlines()
    def p1(m1):
        ctr = 0
        for m2 in ps:
            if abs(m1[0] - m2[0]) + abs(m1[1] - m2[1]) + abs(m1[2] - m2[2]) <= m1[3]:
                ctr += 1
        return ctr
    r = re.compile(r"pos=<([\d-]+),([\d-]+),([\d-]+)>, r=(\d+)")
    ps = []
    for line in inp:
        m = r.match(line)
        if not m:
            print "reg exp fail", line
        ps.append([int(i) for i in m.groups()])

    h = sorted(ps, key=lambda x: x[3], reverse=True)[0]
    print p1(h)

    def p2(m1):
        ctr = 0
        for m2 in ps:
            if abs(m1[0] - m2[0]) + abs(m1[1] - m2[1]) + abs(m1[2] - m2[2]) <= m2[3]:
                ctr += 1
        return ctr
    sortedX = sorted([i[0] for i in ps])
    sortedY = sorted([i[1] for i in ps])
    sortedZ = sorted([i[2] for i in ps])
    rng = h[3]
    sample = product(range(sortedX[0], sortedX[-1]+1, rng), range(sortedY[0], sortedY[-1]+1, rng), range(sortedZ[0], sortedZ[-1]+1, rng))
    while rng > 1:
        print "===", rng
        sampleMax = 0
        goodOnes = []
        for p in sample:
            score = p2(p)
            print p, score
            if score > sampleMax:
                sampleMax = score
                goodOnes = [p]
        print goodOnes, sampleMax
        rng = rng / 2
        theOne = goodOnes[0]
        sample = product(range(theOne[0] - 2*rng, theOne[0] + 1 + 2*rng, rng), range(theOne[1] - 2*rng, theOne[1] + 1 + 2*rng, rng), range(theOne[2] - 2*rng, theOne[2] + 1 + 2*rng, rng))
    print abs(theOne[0]) + abs(theOne[1]) + abs(theOne[2])

def a24():
    import re
    inp = open("inp/adv-24.inp").readlines()
    # inp = open("inp/adv-24.ex").readlines()
    r = re.compile(r"(\d+) units each with (\d+) hit points([^\d]+)with an attack that does (\d+) (\w+) damage at initiative (\d+)")
    def tryWar(bst):
        war = dict()
        army = 'D'
        armyId = 0
        for line in inp:
            line = line.strip()
            if not line:
                continue
            if line == 'Immune System:':
                army = 'D'
                continue
            if line == 'Infection:':
                army = 'A'
                continue
            m = r.match(line)
            mg = m.groups()
            ex = mg[2]
            dmg = int(mg[3])
            if army == 'D':
                dmg += bst
            weak = []
            immune = []
            last = ''
            for extra in ex.strip(" ()").split():
                extra = extra.strip(" ,;")
                if extra == 'to':
                    continue
                if extra in ['weak', 'immune']:
                    last = extra
                else:
                    if last == 'weak':
                        weak.append(extra)
                    elif last == 'immune':
                        immune.append(extra)
                    else:
                        print last, extra, "fail"
            war[armyId] = {'id': armyId, 'army': army, 'num': int(mg[0]), 'hp': int(mg[1]), 'weak': weak, 'imm': immune, 'dmg': dmg, 'attType': mg[4], 'ini': int(mg[5])}
            armyId += 1

        numA, numD = sum([i['num'] for i in war.values() if i['army'] == 'A']), sum([i['num'] for i in war.values() if i['army'] == 'D'])
        while numA != 0 and numD != 0:
            print "======", numA, numD
            sortedTarget = sorted(war.values(), key=lambda x: (x['num'] * x['dmg'], x['ini']), reverse=True)
            for gr in sortedTarget:
                if gr['num'] == 0:
                    continue
                maxDmg = (0, 0)
                for tar in sortedTarget:
                    if tar['army'] == gr['army'] or tar['num'] == 0:
                        continue
                    if 'selected' in tar and tar['selected']: # 2 adam birine saldirmiyomus
                        continue
                    mult = 1
                    if gr['attType'] in tar['weak']:
                        mult = 2
                    if gr['attType'] in tar['imm']:
                        mult = 0
                    dmg = gr['dmg'] * gr['num'] * mult
                    if dmg > maxDmg[0]:
                        maxDmg = (dmg, tar['id'])
                if maxDmg[0] > 0:
                    war[maxDmg[1]]['selected'] = True
                    gr['target'] = maxDmg[1]
            initiative = sorted(war.values(), key=lambda x: x['ini'], reverse=True)
            for gr in initiative:
                if 'target' not in gr:
                    continue
                gr = war[gr['id']]
                tar = war[gr['target']]
                mult = 1
                if gr['attType'] in tar['weak']:
                    mult = 2
                if gr['attType'] in tar['imm']:
                    mult = 0
                dmg = gr['dmg'] * gr['num'] * mult
                # print "killed", min(dmg / tar['hp'], tar['num']), gr['id'], tar['id']
                war[gr['target']]['num'] -= dmg / tar['hp']
                if war[gr['target']]['num'] < 0:
                    war[gr['target']]['num'] = 0
                del tar['selected']
                del gr['target']

            numA2, numD2 = sum([i['num'] for i in war.values() if i['army'] == 'A']), sum([i['num'] for i in war.values() if i['army'] == 'D'])
            if numA == numA2 and numD == numD2:
                print "immune stuck ",
                return numA - numD
            numA = numA2
            numD = numD2
        print "END", numA, numD
        return numA - numD

    print tryWar(0)
    print tryWar(34)
    for boost in range(20, 50):
        # print "==== boost", boost
        if tryWar(boost) < 0:
            print boost
            break

def a25():
    # inp = ["-1,2,2,0", "0,0,2,-2", "0,0,0,-2", "-1,2,0,0", "-2,-2,-2,2", "3,0,2,-1", "-1,3,2,2", "-1,0,-1,0", "0,2,1,-2", "3,0,0,0"] # 4
    # inp = ["1,-1,0,1", "2,0,-1,0", "3,2,-1,0", "0,0,3,1", "0,0,-1,-1", "2,3,-2,0", "-2,2,0,0", "2,-2,0,-1", "1,-1,0,-1", "3,2,0,2"] # 3
    # inp = ["1,-1,-1,-2", "-2,-2,0,1", "0,2,1,3", "-2,3,-2,1", "0,2,3,-2", "-1,-1,1,-2", "0,-2,-1,0", "-2,2,3,-1", "1,2,2,0", "-1,-2,0,-2"] # 8
    inp = open("inp/adv-25.inp").readlines()
    def d(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) + abs(p1[3] - p2[3])

    p = []
    cons = dict()
    for line in inp:
        p.append(tuple([int(i) for i in line.split(",")]))

    for h in p:
        if h not in cons:
            cons[h] = {h}
        for dd in p:
            if d(h, dd) <= 3:
                if dd in cons:
                    for aff in cons[dd]:
                        cons[h].add(aff)
                        cons[aff] = cons[h]
                else:
                    cons[h].add(dd)
                    cons[dd] = cons[h]
    sset = set()
    for c in cons.values():
        sset.add("".join([str(i) for i in sorted(c)]))
    print len(sset)
