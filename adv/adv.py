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
