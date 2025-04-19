import re
import threading
import os
import shutil
import copy
import collections
import random
import time
import copy
os.system("pkill main.exe")
num_threads = os.cpu_count() - 4
print("num_threads:", num_threads)
filename = "../data/sample_official.in"
with open(filename) as fp:
    data = fp.read().encode()
lines = []
with open("hyperparameters.h", "r") as fp:
    for line in fp:
        if not line.startswith("constexpr"):
            continue
        lines.append(line)
pattern = re.compile(r"constexpr\s*\w+\s*(\w+)\s*=\s*(.+)\s*;.+\[(.+)\].*")
template = ""
hyperparameters = {}
best = {}
for line in lines:
    result = re.match(pattern, line)
    name = result.group(1)
    init = result.group(2)
    choices = result.group(3).replace(' ', '').split(',')
    if name == "SEED":
        line = line.replace(init, "$<SEED>", 1)
    elif len(choices) == 1:
        line = line.replace(init, choices[0], 1)
    else:
        line = line.replace(init, f"$<{name}>", 1)
        hyperparameters[name] = choices
        best[name] = random.choice(choices)
    template += line
if not os.path.exists("run"):
    os.mkdir("run")
def generate(params):
    source = template
    for var, value in params.items():
        source = source.replace(f"$<{var}>", value)
    return source
def func(index, params):
    if not os.path.exists(f"run/{index}"):
        os.mkdir(f"run/{index}")
    for i in ["main.cpp", "hyperparameters.h", "utility.h", "constants.h"]:
        shutil.copyfile(i, f"run/{index}/{i}")
    source = generate(params)
    with open(f"run/{index}/hyperparameters.h", "w") as fp:
        fp.write(source)
    if os.path.exists(f'run/{index}/main.exe'):
        os.remove(f'run/{index}/main.exe')
    if os.path.exists(f'run/{index}/output.txt'):
        os.remove(f'run/{index}/output.txt')
    if os.system(f"g++ run/{index}/main.cpp -o run/{index}/main.exe -O3 -march=native -std=c++17 -D__NATIVE_FINETUNE -D__NATIVE_NOOUTPUT 2>run/{index}/compile.txt") != 0:
        raise
    if os.system(f'run/{index}/main.exe <{filename} 2>run/{index}/output.txt') != 0:
        raise
    try:
        with open(f"run/{index}/output.txt", "r") as fp:
            output = fp.read()
        score = float(output.strip().split()[-1])
        params["score"] = score
        result.append(params)
        if "Error" in result:
            raise
    except Exception as error:
        with open("error.txt", "a") as fp:
            fp.write(source + "\n\n")
        print(error)
        exit(0)
threads = []
for i in range(num_threads):
    result = []
    params = {"SEED": str(random.randint(0, 100000000))}
    for name, choices in hyperparameters.items():
        params[name] = random.choice(choices)
    trd = threading.Thread(target=func, args=(i, params))
    trd.start()
    threads.append(trd)
for trd in threads:
    trd.join()
result.sort(key=lambda d: -d["score"])
best = copy.deepcopy(result[0])
print("Init:", best["score"])
del best["score"]
source = generate(best)
print(source)
    
for iteration in range(128):
    for name, choices in hyperparameters.items():
        start_time = time.time()
        result = []
        combinations = []
        num_seeds = max(1, num_threads // len(choices))
        seeds = [str(random.randint(0, 100000000)) for _ in range(num_seeds)]
        for value in choices:
            for seed in seeds:
                combinations.append((value, seed))
        threads = []
        for i, (value, seed) in enumerate(combinations):
            params = copy.deepcopy(best)
            params[name] = value
            params["SEED"] = seed
            trd = threading.Thread(target=func, args=(i, params))
            trd.start()
            threads.append(trd)
        for trd in threads:
            trd.join()
        statistics = collections.defaultdict(list)
        for params in result:
            statistics[params[name]].append(params["score"])
        selection = (0, 0, "")
        for value, score in sorted(statistics.items()):
            assert len(score) == num_seeds
            average = sum(score) / len(score)
            highest = max(score)
            variance = (sum((i - average) ** 2 for i in score) / len(score)) ** 0.5
            print(f"//{name}[{value}] = {average:.0f} ({variance:.0f})")
            if (average, highest, value) > selection:
                selection = (average, highest, value)
        best[name] = selection[2]
        source = generate(best)
        print(f"//iteration: {iteration + 1}")
        print(f"//time: {time.time() - start_time:.0f}")
        print(f"//max: {selection[1]:.0f}")
        print(f"//avg: {selection[0]:.0f}")
        print(source)
        print("-" * 50)
        with open("result.txt", "w") as fp:
            print(source, file=fp)
        