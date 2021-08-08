from typing import List


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
        minsX = [0 for i in range(X)]
        minsY = [0 for i in range(Y)]

        vals = {}
        for x in range(X):
            for y in range(Y):
                if matrix[x][y] not in vals:
                    vals[matrix[x][y]] = []
                vals[matrix[x][y]].append((x, y))
        for val in sorted(vals.keys()):
            score = 0
            for x, y in vals[val]:
                scTmp = max(minsX[x], minsY[y]) + 1
                if scTmp > score:
                    score = scTmp
            print(val, score)
            for x, y in vals[val]:
                ret[x][y] = score
                minsX[x] = score
                minsY[y] = score
        return ret


s = Solution()

print("#08-rank transform matrix")
t8 = [
    # ([[1, 2], [3, 4]], [[1, 2], [2, 3]]),
    # ([[7, 7], [7, 7]], [[1, 1], [1, 1]]),
    # ([[20, -21, 14], [-19, 4, 19], [22, -47, 24], [-19, 4, 19]],
    #  [[4, 2, 3], [1, 3, 4], [5, 1, 6], [1, 3, 4]]),
    # ([[7, 3, 6], [1, 4, 5], [9, 8, 2]], [[5, 1, 4], [1, 2, 3], [6, 3, 1]]),
    # ([[-37, -50, -3, 44], [-37, 46, 13, -32], [47, -42, -3, -40], [-17, -
    #  22, -39, 24]],
    #  [[2, 1, 4, 6], [2, 6, 5, 4], [5, 2, 4, 3], [4, 3, 1, 5]]),
    ([[-37, -26, -47, -40, -13], [22, -11, -44, 47, -6], [-35, 8, -45, 34, -31], [-16, 23, -6, -43, -20], [47, 38, -27, -8, 43]],
     [[3, 4, 1, 2, 7], [9, 5, 3, 10, 8], [4, 6, 2, 7, 5], [7, 9, 8, 1, 6], [12, 10, 4, 5, 11]])
]

for t in t8:
    ans = s.matrixRankTransform(t[0])
    print(t)
    print(ans, ans == t[1])

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
