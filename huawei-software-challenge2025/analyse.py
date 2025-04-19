import matplotlib.pyplot as plt
def smooth(vec):
    L = 10
    vec = [0] * L + vec
    ret = [0] * len(vec)
    for i in range(L, len(vec) - L):
        for j in range(-L, 1):
            ret[i] += vec[i + j]
        ret[i] /= L + 1.0
    return ret[L:]

factor = []
used = []
rate = []
with open("data.txt", "r") as fp:
    for line in fp.readlines():
        r, f, u, *o = line.strip().split()
        rate.append(float(r))
        factor.append(float(f))
        used.append(float(u))
smoothing = 1
if smoothing:
    rate = smooth(rate)
    #factor = smooth(factor)
# 数据
x = [(i + 1) for i in range(len(factor))]
plt.plot(x, factor, label='factor')
plt.plot(x, rate, label='rate')
plt.plot(x, used, label='used')
plt.legend()
# 显示图形
plt.show()