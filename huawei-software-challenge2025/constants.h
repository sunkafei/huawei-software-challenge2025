#pragma once
constexpr double FINF = 1e100;
constexpr int INF = 1 << 29;
constexpr int EXTRA_TIME = 105;
constexpr int MAX_T = 86400 + EXTRA_TIME + 2;
constexpr int MAX_M = 16 + 3;
constexpr int MAX_N = 10 + 2;
constexpr int MAX_V = 16384 + 2;
constexpr int MAX_G = 1000 + 2;
constexpr int MAX_OBJECT = 100000 + 2;
constexpr int MAX_REQUEST = 30000000 + 2;
constexpr int DUPLICATE = 3;
constexpr int TIME_INTERVAL_LENGTH = 1800;
constexpr int MAX_INTERVALS = (MAX_T - 1) / TIME_INTERVAL_LENGTH + 1;
constexpr int READ_FACTOR = 8;
constexpr int READ_BASE = 64;
constexpr int MAX_BLOCKS = NUM_BLOCKS + 2;
constexpr int NUM_HEADS = 2;
