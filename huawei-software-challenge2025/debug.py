import re
import threading
import os
import shutil
import copy
import collections
import random
import time
os.system("pkill main.exe")
num_threads = os.cpu_count() - 10
print("num_threads:", num_threads)
filename = "../data/sample_practice.in"
lines = []
with open("hyperparameters.h", "r") as fp:
    for line in fp:
        if not line.startswith("constexpr"):
            continue
        lines.append(line)
pattern = re.compile(r"constexpr\s*\w+\s*(\w+)\s*=\s*(.+)\s*;.+\[(.+)\].*")
template = ""
hyperparameters = {}
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
    template += line
if not os.path.exists("run"):
    os.mkdir("run")
counter = 0
def func(index):
    global counter
    if not os.path.exists(f"run/{index}"):
        os.mkdir(f"run/{index}")
    for i in ["main.cpp", "hyperparameters.h", "utility.h", "constants.h"]:
        shutil.copyfile(i, f"run/{index}/{i}")
    while True:
        source = template.replace("$<SEED>", str(random.randint(0, 100000000)))
        for var, values in hyperparameters.items():
            source = source.replace(f"$<{var}>", random.choice(values))
        with open(f"run/{index}/hyperparameters.h", "w") as fp:
            fp.write(source)
        if os.path.exists(f'run/{index}/main.exe'):
            os.remove(f'run/{index}/main.exe')
        if os.path.exists(f'run/{index}/output.txt'):
            os.remove(f'run/{index}/output.txt')
        if os.system(f"g++ run/{index}/main.cpp -o run/{index}/main.exe -O1 -march=native -std=c++17 -D__NATIVE_CHECK  -D__NATIVE_NOOUTPUT -fsanitize=undefined 2>run/{index}/compile.txt") != 0:
            raise
        try:
            if os.system(f'timeout 2400s run/{index}/main.exe <{filename} 2>run/{index}/output.txt') != 0:
                result = "RE or TLE\n"
                raise
            with open(f"run/{index}/output.txt", "r") as fp:
                result = fp.read()
            if "error" in result.lower():
                raise
        except:
            with open("error.txt", "a") as fp:
                fp.write(f"----------\n{result}----------\n")
                fp.write(source + "\n\n")
            print("ERROR!!!!!!!!")
            exit(0)
        counter += 1
        print(f"\rok:{counter}    ", end='')

threads = []
for i in range(num_threads):
    trd = threading.Thread(target=func, args=(i,))
    trd.start()
    threads.append(trd)
for trd in threads:
    trd.join()