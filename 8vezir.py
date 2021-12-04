from itertools import permutations
results = []

for board in permutations([0, 1, 2, 3, 4, 5, 6, 7], 8):
    if board[0] >= 4:
        continue
    Xs = []
    Xd = []
    for y in range(8):
        if board[y] + y in Xs:
            break
        if board[y] - y in Xd:
            break
        Xs.append(board[y] + y)
        Xd.append(board[y] - y)
    if len(Xs) == 8 and len(Xd) == 8:
        results.append(board)
        results.append(tuple(map(lambda i: 7-i, board)))

# results = set()
# def f(x, X, Xs, Xd):
#     for y in range(8 if x != 0 else 4):
#         if y in X:
#             continue
#         if x + y in Xs:
#             continue
#         if x - y in Xd:
#             continue
#         if x == 7:
#             results.add(X + (y, ))
#             results.add(tuple(map(lambda i: 7-i, X + (y, ))))
#         else:
#             f(x + 1, X + (y, ), Xs + (x + y, ), Xd + (x - y, ))
# f(0, (), (), ())

print(len(results))
print(results.pop())
