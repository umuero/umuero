
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
