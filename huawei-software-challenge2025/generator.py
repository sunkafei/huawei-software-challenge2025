from random import randint, choices, sample
from tqdm import tqdm
output = open("../data/input.in", "w")
T = 86400
M = randint(10, 16)
N = randint(7, 10)
V = randint(10000, 16384)
G = randint(100, 500)
K = randint(0, 100)
print(T, M, N, V, G, K, file=output)
sz = {}
objects = []
total_objs, total_reqs = 0, 0
load = 0
full = int(V * N / 3)
unknown = []
for timestamp in range(1, T + 1 + 105):
    if timestamp % 1000 == 0:
        print(f"{timestamp}... objects({len(objects)}/{total_objs}) load({load / full})")
    print(f"TIMESTAMP {timestamp}", file=output)
    if timestamp < 50000:
        n_delete = randint(0, min(len(objects), 2))
    else:
        n_delete = randint(0, min(len(objects), 3))
    if timestamp > T:
        n_delete = 0
    print(n_delete, file=output)
    deleted = sample(objects, n_delete)
    for id in deleted:
        objects.remove(id)
        load -= sz[id]
        print(id, file=output)
    n_write = randint(0, min(100000 - total_objs, max(int(V * N / 3 * 0.89) - load, 0), 3))
    if timestamp > T:
        n_write = 0
    if timestamp <= 1000:
        n_write += 2
    print(n_write, file=output)
    for i in range(n_write):
        total_objs += 1
        size = randint(1, 5)
        if randint(0, 6) != 0:
            tag = 0
            unknown.append(total_objs)
        else:
            tag = randint(1, M)
        objects.append(total_objs)
        load += size
        sz[total_objs] = size
        print(total_objs, size, tag, file=output)
    n_read = randint(0, min(len(objects), 120))
    if timestamp > T:
        n_read = 0
    print(n_read, file=output)
    for id in choices(objects, k=n_read):
        total_reqs += 1
        print(total_reqs, id, file=output)
    if timestamp % 1800 == 0:
        print("GARBAGE COLLECTION", file=output)
n_incre = len(unknown) // 2
print(n_incre, file=output)
for id in sample(unknown, n_incre):
    tag = randint(1, M)
    print(id, tag, file=output)
for timestamp in range(1, T + 1 + 105):
    print(f"TIMESTAMP {timestamp}", file=output)
output.close()
    