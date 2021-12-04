import ast
from typing import List
import collections
from copy import deepcopy as copy


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        ret = {()}
        for n in sorted(nums):
            ret = ret.union({l + (n,) for l in ret})
        return [list(i) for i in ret]

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = dict()
        for i, v in enumerate(nums):
            if v not in d:
                d[v] = []
            d[v].append(i)
        for v in d.keys():
            if target - v in d:
                if v == target - v:
                    if len(d[v]) >= 2:
                        return [d[v][0], d[v][1]]
                    else:
                        continue
                return [d[v][0], d[target-v][0]]

    def largestIsland(self, grid: List[List[int]]) -> int:
        N = len(grid)

        def updateCounts(x0, y0):
            stack = [(x0, y0)]
            while stack:
                x, y = stack.pop()
                for x1, y1 in ((x-1, y), (x, y-1), (x+1, y), (x, y+1)):
                    if 0 <= x1 < N and 0 <= y1 < N and grid[x1][y1] == 1:
                        stack.append((x1, y1))
                        grid[x1][y1] == -1
            return len(seen)

        def updateCounts(x, y):
            stack = [(x, y)]

            for x1, y1 in [(x-1, y), (x-1, y-1), (x+1, y), (x+1, y+1)]:
                if x1 < N and y1 < N and grid[x1][y1] == 1:
                    stack.push(x1, y1)

        for x in range(N):
            for y in range(N):
                if grid[x][y] == 1:
                    updateCounts(x, y)

        for x in range(N):
            for y in range(N):
                if grid[x][y] == 1:
                    updateCounts(x, y)

    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        stack = [(root, 0, ())]
        ret = []
        while stack:
            node, val, path = stack.pop()
            if node is None:
                continue
            if node.left is None and node.right is None:
                if node.val + val == targetSum:
                    print(path + (node.val,))
                    ret.append(path + (node.val,))
            else:
                if node.left is not None:
                    stack.append(
                        (node.left, val + node.val, path + (node.val,)))
                if node.right is not None:
                    stack.append(
                        (node.right, val + node.val, path + (node.val,)))
        return ret

    def stoneGame(self, piles: List[int]) -> bool:
        return True
        # s = [0, 0]
        # changed = True
        # while changed:
        #     changed = False
        #     for i in range(len(piles) - 1):
        #         if piles[i] == piles[i+1]:
        #             piles.pop(i)
        #             piles.pop(i)
        #             changed = True
        #             break
        # ctr = 0
        # while piles:
        #     diff = piles[0] - piles[-1]
        #     if len(piles) > 3:
        #         diff += piles[-2] - piles[1]
        #     if diff >= 0:
        #         print('%s start %d %s' % (s, piles[0], piles))
        #         s[ctr % 2] += piles.pop(0)
        #     else:
        #         print('%s end   %d %s' % (s, piles[-1], piles))
        #         s[ctr % 2] += piles.pop(-1)
        #     ctr += 1
        # print('%s' % s)
        # return (s[0] > s[1])

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        ret = []
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if len(ret) == depth:
                ret.append([])
            ret[depth].append(node.val)
            if node.children:
                for child in node.children:
                    stack.insert(0, (child, depth + 1))
        return ret

    def minCut(self, s: str) -> int:
        N = len(s)
        if s is None or len(s) <= 1:
            return 0
        jumps = []
        for i in range(N):
            jumps.append([i + 1])
            for j in range(i + 2, N + 1):
                if s[i:j] == s[i:j][::-1]:
                    jumps[i].insert(0, j)
        # print(jumps)
        stack = [(0, 0)]
        mins = [0 for i in range(N + 1)]
        while stack:
            node, depth = stack.pop()
            if mins[node] == 0:
                mins[node] = depth
            else:
                continue
            # print(f'N {node}\t{depth}')
            if node == N:
                return depth - 1
            for child in jumps[node]:
                stack.insert(0, (child, depth + 1))
        return N - 1

    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        if matrix is None or len(matrix) == 0:
            return []
        X = len(matrix)
        Y = len(matrix[0])
        ret = [[0 for y in range(Y)] for x in range(X)]
        minsX = [(0, None) for i in range(X)]
        minsY = [(0, None) for i in range(Y)]
        vals = sorted([(matrix[x][y], x, y)
                      for x in range(X) for y in range(Y)])

        def getScore(x, y, val):
            scX = minsX[x][0] + (0 if minsX[x][1] == val else 1)
            scY = minsY[y][0] + (0 if minsY[y][1] == val else 1)
            return max(scX, scY)

        def setScore(x, y, val, score):
            ret[x][y] = score
            minsX[x] = max(minsX[x], (score, val))
            minsY[y] = max(minsY[y], (score, val))

        def checkPrevLoop(prevItems, val):
            mScore = max([score for x, y, score in prevItems])
            stack = [(x, y) for x, y, score in prevItems if score != mScore]
            while stack:
                x, y = stack.pop()
                for pX, pY, score in prevItems:
                    if (x == pX or y == pY) and ret[x][y] != ret[pX][pY]:
                        print("p", matrix[x][y], (x, y),
                              ret[x][y], (pX, pY), ret[pX][pY])
                        if ret[x][y] > ret[pX][pY]:
                            score = ret[x][y]
                            setScore(pX, pY, val, score)
                            stack.append((pX, pY))
                        else:
                            score = ret[pX][pY]
                            setScore(x, y, val, score)
                            stack.append((x, y))

        prevVal = None
        prevItems = set()
        for val, x, y in vals:
            if prevVal != val:
                if len(prevItems) > 1:
                    checkPrevLoop(prevItems, prevVal)
                prevItems = set()

            score = getScore(x, y, val)
            setScore(x, y, val, score)

            prevVal = val
            prevItems.add((x, y, score))
        if len(prevItems) > 1:
            checkPrevLoop(prevItems, prevVal)
        return ret

    def matrixRankTransform2(self, matrix: List[List[int]]) -> List[List[int]]:
        m = len(matrix)
        n = len(matrix[0])
        valToPos = collections.defaultdict(list)
        for i in range(m):
            for j in range(n):
                valToPos[matrix[i][j]].append((i, j))

        rank = [0] * (m + n)

        def find(i: int) -> int:
            j = i
            while parent[j] != j:
                j = parent[j]

            while parent[i] != j:
                i, parent[i] = parent[i], j
            return j

        for val in sorted(valToPos):
            parent = list(range(m+n))
            rank2 = rank[:]
            for i, j in valToPos[val]:
                i, j = find(i), find(m+j)
                parent[i] = j
                rank2[j] = max(rank2[j], rank2[i])
            for i, j in valToPos[val]:
                rank[i] = rank[m+j] = matrix[i][j] = rank2[find(i)]+1
        return matrix

    def addStrings(self, num1: str, num2: str) -> str:
        l1 = len(num1)
        l2 = len(num2)
        ret = []
        extra = 0
        for i in range(-1, -max(l1, l2) - 1, -1):
            n1 = 0
            n2 = 0
            if -i <= l1:
                n1 = int(num1[i])
            if -i <= l2:
                n2 = int(num2[i])
            extra = n1 + n2 + extra
            ret.append(str(extra % 10))
            extra = int(extra / 10)
        if extra:
            ret.append(str(extra % 10))
        return "".join(ret[::-1])

    def minFlipsMonoIncr(self, s: str) -> int:
        lhs = 0  # solumdaki 1'ler
        rhs = 0  # sagimdaki 0'lar
        for i in range(len(s)):
            if s[i] == '0':
                rhs += 1
        mn = (0, lhs + rhs)
        print(mn)

        for i in range(1, len(s) + 1):
            if s[i] == '0':
                rhs -= 1
            else:
                lhs += 1
            if rhs + lhs < mn[1]:
                mn = (i, rhs + lhs)
                print(mn)
        return mn[1]

    def canReorderDoubled(self, arr: List[int]) -> bool:
        count = collections.Counter(arr)
        for x in sorted(count, key=abs):
            if count[x] > count[2*x]:
                return False
            count[2*x] -= count[x]
        return True

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        n = collections.defaultdict(list)
        for s in strs:
            n["".join([a[0] + str(a[1])
                      for a in sorted(collections.Counter(s).items())])].append(s)
        return [i for i in n.values()]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        X = len(matrix)
        if X == 0:
            return
        Y = len(matrix[0])
        zX = set()
        zY = set()
        for x in range(X):
            for y in range(Y):
                if matrix[x][y] == 0:
                    zX.add(x)
                    zY.add(y)
        for x in zX:
            for y in range(Y):
                matrix[x][y] = 0
        for y in zY:
            for x in range(X):
                matrix[x][y] = 0

    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        memo = dict()

        def dfs(l, r, k):
            if l > r:
                return 0
            if memo.get((l, r, k), 0) != 0:
                return memo[l, r, k]
            while r > l and boxes[r] == boxes[r-1]:
                r -= 1
                k += 1
            memo[l, r, k] = dfs(l, r-1, 0) + (k+1)**2
            for i in range(l, r):
                if boxes[i] == boxes[r]:
                    memo[l, r, k] = max(memo[l, r, k],
                                        dfs(l, i, k+1) + dfs(i+1, r-1, 0))
            return memo[l, r, k]
        return dfs(0, N-1, 0)

    def find132pattern(self, nums: List[int]) -> bool:
        nMin = None
        nMax = None
        oldVals = []
        for i in nums:
            if nMin is not None and (nMax is None or nMax < i):
                nMax = i
            if nMin is None or nMin > i:
                if nMin and nMax:
                    for pair in oldVals:
                        if nMax < pair[0] or pair[1] < nMin:
                            continue
                        if pair[1] <= nMax:
                            oldVals.remove(pair)
                        if nMax <= pair[0]:
                            nMax = pair[1]
                    oldVals.insert(0, (nMin, nMax))
                nMin = i
                nMax = None
            if nMin < i < nMax:
                return True
            for pair in oldVals:
                if pair[0] < i < pair[1]:
                    return True
                if pair[0] > i:
                    break
        return False


s = Solution()


print("132-pattern")
t14 = [
    ([1, 2, 3, 4], False),
    ([3, 1, 4, 2], True),
    ([-1, 3, 2, 0], True),
    ([11, 13, 6, 8, 12], True),
]

for t in t14:
    ans = s.find132pattern(t[0])
    print(ans, ans == t[-1])

# print("#14-remove boxes")
# t14 = [
#     ([1, 3, 2, 2, 2, 3, 4, 3, 1], 23),
#     ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
#     ([1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1], 139),
#     ([1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2,
#      2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1], 3)
# ]

# for t in t14:
#     ans = s.removeBoxes(t[0])
#     print(ans, ans == t[-1])

# print("#09-array of doubled pairs")
# t11 = [
#     ([1, 2, 4, 16, 8, 4], False),
#     ([4, -2, 2, -4], True),
#     ([2, 1, 2, 6], False),
#     ([3, 1, 3, 6], False)
# ]

# for t in t11:
#     ans = s.canReorderDoubled(t[0])
#     print(ans, ans == t[-1])

# print("#09-add string")
# t9 = [
#     ("456", "77", "533"),
#     ("0", "0", "0"),
# ]

# for t in t9:
#     print(s.addStrings(t[0], t[1]), s.addStrings(t[0], t[1]) == t[-1])


# print("#08-rank transform matrix")
# t8 = [
#     ([[1, 2], [3, 4]], [[1, 2], [2, 3]]),
#     ([[7, 7], [7, 7]], [[1, 1], [1, 1]]),
#     ([[20, -21, 14], [-19, 4, 19], [22, -47, 24], [-19, 4, 19]],
#      [[4, 2, 3], [1, 3, 4], [5, 1, 6], [1, 3, 4]]),
#     ([[7, 3, 6], [1, 4, 5], [9, 8, 2]], [[5, 1, 4], [1, 2, 3], [6, 3, 1]]),
#     ([[-37, -50, -3, 44], [-37, 46, 13, -32], [47, -42, -3, -40], [-17, -
#      22, -39, 24]],
#      [[2, 1, 4, 6], [2, 6, 5, 4], [5, 2, 4, 3], [4, 3, 1, 5]]),
#     ([[-37, -26, -47, -40, -13], [22, -11, -44, 47, -6], [-35, 8, -45, 34, -31], [-16, 23, -6, -43, -20], [47, 38, -27, -8, 43]],
#      [[3, 4, 1, 2, 7], [9, 5, 3, 10, 8], [4, 6, 2, 7, 5], [7, 9, 8, 1, 6], [12, 10, 4, 5, 11]]),
#     ([[-2, -35, -32, -5, -30, 33, -12], [7, 2, -43, 4, -49, 14, 17], [4, 23, -6, -15, -24, -17, 6], [-47, 20, 39, -26, 9, -44, 39], [-50, -47, 44, 43, -22, 33, -36], [-13, 34, 49, 24, 23, -2, -35], [-40, 43, -22, -19, -4, 23, -18]],
#      [[10, 3, 4, 9, 5, 15, 8], [12, 4, 2, 10, 1, 13, 14], [11, 13, 9, 8, 6, 7, 12], [2, 10, 15, 4, 9, 3, 15], [1, 2, 17, 16, 7, 15, 3], [5, 14, 18, 11, 10, 8, 4], [3, 15, 5, 6, 8, 14, 7]]),
#     ([[-24, -9, -14, -15, 44, 31, -46, 5, 20, -5, 34], [9, -40, -49, -50, 17, 40, 35, 30, -39, 36, -49], [-18, -43, -40, -5, -30, 9, -28, -41, -6, -47, 12], [11, 42, -23, 20, 35, 34, -39, -16, 27, 34, -15], [32, 27, -30, 29, -48, 15, -50, -47, -28, -21, 38], [45, 48, -1, -18, 9, -4, -13, 10, 9, 8, -41], [-42, -35, 20, -17, 10, 5, 36, 47, 6, 1, 8], [3, -50, -23, 16, 31, 2, -39, 36, -25, -30, 37], [-48, -41, 18, -31, -48, -1, -42, -3, -8, -29, -2], [17, 0, 31, -30, -43, -20, -37, -6, -43, 8, 19], [42, 25, 32, 27, -2, 45, 12, -9, 34, 17, 32]],
#      [[4, 11, 10, 9, 25, 21, 2, 14, 20, 12, 24], [18, 5, 2, 1, 21, 25, 23, 22, 6, 24, 2], [8, 2, 5, 11, 6, 18, 7, 4, 10, 1, 20], [19, 24, 9, 20, 23, 22, 4, 10, 21, 22, 11], [23, 20, 6, 22, 2, 19, 1, 3, 7, 8, 26], [26, 27, 11, 7, 19, 9, 8, 20, 19, 14, 3], [3, 6, 21, 8, 20, 17, 24, 25, 18, 13, 19], [17, 1, 9, 18, 22, 16, 4, 23, 8, 5, 25], [2, 4, 16, 5, 2, 15, 3, 13, 9, 6, 14], [20, 13, 22, 6, 3, 7, 5, 12, 3, 14, 21], [25, 16, 23, 21, 12, 26, 13, 11, 24, 15, 23]]),
#     ([[-49, -26, 41, 20, 3, -42, 25, 44, -49, -6, 21, -28, 3], [-50, 13, 28, -25, 42, 33, -8, -17, 18, 49, -36, -17, 38], [-11, 40, 43, -22, -43, -48, -5, 6, 13, -28, 19, -38, -7], [24, -45, -38, -19, -44, -37, 46, -3, 20, -1, 38, 41, 28], [23, 18, 5, -4, 47, -18, 29, 12, 3, -2, -7, 16, -5], [-30, -47, 0, -25, -2, -19, 32, -33, 30, 49, 40, -37, 38], [-39, 32, -29, -10, 9, -12, 43, -38, 45, -24, -9, 18, 21], [-12, 35, -46, -31, -24, -1, -22, 5, 48, -13, 2, -27, 28], [-5, -10, 49, -24, 39, 26, 9, 8, 27, -22, 21, 24, 15], [26, -31, -12, -45, 46, 21, 36, 47, -34, 21, -20, 43, 10], [5, -16, -17, 10, 9, 8, -37, 18, -39, 40, 39, -26, 37], [-28, -21, -38, -7, -4, -29, 6, 1, 16, -45, 34, 5, -48], [47, -30, -11, 36, -49, -42, 29, -40, -17, -14, -11, -20, 35]],
#      [[2, 8, 25, 21, 18, 3, 24, 26, 2, 14, 23, 7, 18], [1, 13, 24, 9, 31, 27, 12, 11, 23, 32, 2, 11, 30], [15, 29, 30, 11, 4, 1, 17, 18, 21, 6, 22, 5, 16], [26, 2, 4, 12, 3, 5, 32, 13, 24, 20, 28, 29, 27], [25, 24, 21, 18, 33, 11, 26, 22, 20, 19, 15, 23, 17], [8, 1, 17, 9, 16, 10, 29, 7, 28, 32, 31, 6, 30], [3, 26, 6, 13, 19, 12, 31, 5, 32, 7, 14, 24, 25], [14, 28, 1, 2, 9, 15, 10, 17, 33, 13, 16, 8, 27], [16, 12, 31, 10, 28, 26, 20, 19, 27, 11, 23, 25, 21], [27, 5, 11, 1, 32, 21, 30, 33, 4, 21, 6, 31, 19], [17, 11, 10, 20, 19, 18, 4, 23, 3, 31, 30, 9, 29], [9, 10, 4, 14, 15, 6, 18, 16, 22, 2, 24, 17, 1], [30, 6, 13, 29, 1, 3, 26, 4, 11, 12, 13, 10, 28]]),
#     ([[28, -13, 42, 49, -20, -9, -46, 21, -8, -25, -18, -39, -8, -13, 6, -7, -44, -33], [-30, 9, 4, -45, 14, -7, 44, 47, -30, 9, -40, 43, -50, -15, 36, -1, 46, -35], [-24, 19, 2, -27, -20, 23, 6, -39, 32, 3, -14, -47, -36, 23, -10, 17, 32, 27], [42, 9, 40, 11, -38, -23, 16, -13, -30, -3, -32, -5, -6, -35, 32, -1, 30, 33], [32, -33, 26, 49, -32, -25, 22, -7, 8, -1, -26, 21, -28, 31, 22, 33, 28, 47], [-46, 37, -4, -1, -22, -35, -48, -37, -2, 37, -16, -5, 6, 17, -36, 3, 30, 41], [40, 19, 38, -3, -12, -29, 14, -7, -44, 19, -10, 49, -8, -5, -6, -31, -12, -49], [6, -35, -16, -5, 22, -27, 24, 35, 2, 45, -8, -49, 10, -43, -36, 11, 34, -39], [-4, -21, 18, -7, 24, -21, 26, 1, 12, 15, 46, -35, -20, 7, -26, 1, 36, 39], [-14, 33, -16, 43, 42, 49, 12, -17, -18, 49, -16, 3, -34, 49, -24, -29, -6, -47], [-4, -21, 46, 1, 8, -41, 18, -43, 4, 35, -46, 13, 4, 47, -30, -7, 4, 43], [-18, -11, 4, -21, 38, 1, 32, -49, 10, 37, 12, 19, 2, -27, 32, -33, -46, 33], [-36, 11, -38, 17, -20, 15, 26, -39, -48, -29, -42, -15, -32, 35, -6, 49, 24, -21], [-14, 33, -20, -41, -6, 5, 8, -49, -46, 41, -24, -21, -38, -35, 28, -1, 2, -43], [44, -37, 18, 45, 36, 23, -26, 21, 44, -21, -46, -47, 24, 35, 22, -7, 40, 47], [10, -7, -16, -33, 38, -23, 0, -33, -38, -39, -16, 27, 14, 49, 24, 15, -38, -19], [36, -1, -6, -19, 4, -17, 46, -23, 8, -9, 42, -43, -48, 31, -30, -3, 0, 31], [-10, -23, -20, 15, 30, -7, 16, 43, 18, 17, 32, -29, -38, 17, 24, -13, 6, -27]],
#      [[33, 16, 40, 45, 12, 17, 2, 32, 19, 10, 14, 4, 19, 16, 22, 20, 3, 7], [8, 25, 24, 2, 28, 19, 40, 42, 8, 25, 5, 39, 1, 9, 38, 23, 41, 6], [9, 32, 23, 8, 12, 34, 25, 5, 37, 24, 17, 2, 6, 34, 18, 31, 37, 35], [40, 25, 39, 26, 1, 14, 29, 15, 8, 22, 7, 21, 20, 6, 37, 23, 36, 38], [37, 9, 34, 45, 10, 13, 33, 20, 27, 23, 12, 31, 11, 36, 33, 38, 35, 44], [2, 39, 22, 24, 11, 8, 1, 6, 23, 39, 16, 21, 27, 31, 7, 25, 36, 41], [39, 32, 35, 23, 13, 9, 28, 20, 3, 32, 18, 40, 19, 22, 21, 4, 13, 1], [25, 8, 16, 20, 30, 10, 34, 39, 24, 41, 19, 1, 28, 2, 7, 29, 38, 4], [21, 15, 31, 17, 32, 15, 35, 24, 29, 30, 41, 5, 16, 25, 9, 24, 39, 40], [17, 36, 16, 42, 41, 45, 27, 12, 11, 45, 16, 22, 7, 45, 10, 8, 18, 2], [21, 15, 43, 25, 27, 5, 30, 4, 26, 33, 3, 28, 26, 44, 8, 20, 26, 42], [10, 17, 24, 9, 40, 20, 37, 1, 28, 39, 29, 30, 21, 7, 37, 3, 2, 38], [7, 26, 6, 29, 12, 27, 35, 5, 1, 9, 4, 15, 8, 37, 21, 39, 30, 10], [17, 36, 15, 4, 18, 25, 26, 1, 2, 40, 13, 14, 5, 6, 35, 23, 24, 3], [41, 4, 31, 43, 38, 34, 5, 32, 41, 11, 3, 2, 35, 37, 33, 20, 40, 44], [26, 18, 16, 7, 40, 14, 19, 7, 4, 1, 16, 35, 29, 45, 34, 30, 4, 15], [38, 22, 18, 10, 24, 16, 41, 9, 27, 17, 39, 3, 2, 36, 8, 21, 23, 36], [18, 10, 15, 28, 35, 19, 29, 40, 32, 31, 36, 6, 5, 31, 34, 16, 27, 8]]),
# ]

# for t in t8:
#     ans = s.matrixRankTransform2(t[0])
#     print(ans == t[1])
#     if ans != t[1]:
#         for r in t[0]:
#             print('\t'.join([str(i) for i in r]))
#         print('Ans:')
#         for xc, x in enumerate(t[1]):
#             for yc, y in enumerate(x):
#                 if (ans[xc][yc] == y):
#                     print('%d\t' % y, end='')
#                 else:
#                     print('%d(%d)\t' % (y, ans[xc][yc]), end='')
#             print('')

# print("#07-palindruome partition")
# t7 = [
#     ('aab', 1),
#     ('a', 0),
#     ('abbaba', 2),
#     ('fifgbeajcacehiicccfecbfhhgfiiecdcjjffbghdidbhbdbfbfjccgbbdcjheccfbhafehieabbdfeigbiaggchaeghaijfbjhi', 20),
#     (''.join(['a' for i in range(2000)]), 20)
# ]

# for t in t7:
#     print(s.minCut(t[0]), s.minCut(t[0]) == t[1])


# print("#06-n-ary tree")
# t6 = [
#     (Node(1, [Node(3, [Node(5), Node(6)]), Node(2), Node(4)]),
#      [[1], [3, 2, 4], [5, 6]]),
# ]

# for t in t6:
#     print(s.levelOrder(t[0]))
#     print(s.levelOrder(t[0]) == t[1])

# print("#05-stone game")
# t5 = [([5, 3, 4, 5], True),
#       ([3, 7, 2, 3], True),
#       ([6, 3, 9, 9, 3, 8, 8, 7], True),
#       ([3, 7, 3, 2, 5, 1, 6, 3, 10, 7], True)
#       ]

# for t in t5:
#     print(t)
#     print(s.stoneGame(t[0]) == t[1])

# print("#04-path sum II")
# t4 = [(TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))), TreeNode(8, TreeNode(13), TreeNode(4, TreeNode(5), TreeNode(1)))), 22, [[5, 4, 11, 2], [5, 8, 4, 5]]),
#       (TreeNode(1, TreeNode(2), TreeNode(3)), 5, [])]

# for t in t4:
#     print(s.pathSum(t[0], t[1]), s.pathSum(t[0], t[1]) == t[2])


# print("#03-subsets II")
# t3 = [
#     ([1,2,2], sorted([[],[1],[1,2],[1,2,2],[2],[2,2]])),
#     ([0], sorted([[],[0]])),
# ]

# for t in t3:
#     print(s.subsetsWithDup(t[0]), sorted(s.subsetsWithDup(t[0])) == t[1])

# print("#02-twoSum")
# t2 = [
#     ([2,7,11,15], 9, sorted([0, 1])),
#     ([3,3], 6, sorted([0, 1])),
# ]

# for t in t2:
#     print(s.twoSum(t[0], t[1]), sorted(s.twoSum(t[0], t[1])) == t[2])


# print("#01-largestIsland")
# t1 = [
#     ([[0,1],[1,0]], 3),
#     ([[1,1],[1,0]], 4),
#     ([[1,1],[1,1]], 4),
# ]

# for t in t1:
#     print(s.largestIsland(t[0]), s.largestIsland(t[0]) == t[1])
