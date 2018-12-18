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
    inp = open("inp/adv-15.inp").readlines()
    inp = open("inp/adv-15.ex").readlines()
    trans = {"G": ".", "E": "."}
    ad = 3
    hp = 200
    tr = { # ['L', 'S', 'R']
        (">", "/"): "^",
        ("v", "/"): "<",
        ("<", "/"): "v",
        ("^", "/"): ">"
    }
    agents = []
    m = dict()
    for lCtr, line in enumerate(inp):
        for hCtr, x in enumerate(line):
            if x == '#' or x == '\n':
                continue
            if x in trans:
                agents.append({"l": lCtr, "h": hCtr, "d": x})
            m[lCtr,hCtr] = trans.get(x, x)

    tick = 0
    while len(agents) > 1:
        ags = sorted([(x['l'], x['h'], i) for i, x in enumerate(agents)])
        xy = [(a[0], a[1]) for a in ags]
        for aIndex in ags:
            ag = agents[aIndex[2]]
            if 'c' in ag:
                continue


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

a18()
