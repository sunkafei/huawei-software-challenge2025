#include "utility.h"
#include "hyperparameters.h"
#include "constants.h"
#ifdef __NATIVE_CHECK
#define __NATIVE_FINETUNE
#endif
#ifdef __NATIVE_NOOUTPUT
#define printf(...) ;
#define puts(x) ;
#define fflush(x) ;
#endif
static_assert(!LOCK_DUPLICATE || NUM_BLOCKS % 3 == 0);
using namespace std;
const auto start_time = chrono::steady_clock::now();
mt19937 engine{ SEED };
int T, M, N, V, G, K[2], stage = 0;
int fre_del[MAX_M][MAX_INTERVALS], fre_write[MAX_M][MAX_INTERVALS], fre_read[MAX_M][MAX_INTERVALS];
int timestamp, block_size, num_intervals, tgrank[MAX_M];
int belong[MAX_N][MAX_BLOCKS], disable[MAX_N][MAX_BLOCKS], lst[MAX_N][MAX_BLOCKS];
long long query[MAX_N][MAX_BLOCKS], busy = 0;
double angle[MAX_M][MAX_M], total[2], factor = 0;
bool action[1 << WINDOW_SIZE][READ_FACTOR];
vector<pair<int, int>> group[MAX_N][MAX_BLOCKS];
vector<int> brother[MAX_N][MAX_BLOCKS];
char buffer[MAX_G + MAX_V + 10];
inline auto block(int pos) {
    if constexpr (LOCK_DUPLICATE && NUM_BLOCKS % DUPLICATE == 0) {
        return min(pos / block_size, NUM_BLOCKS - 1);
    }
    else {
        return pos / block_size;
    }
}
enum class status_t {
    idle, deleted, processing, done, future
};
struct range_t {
    struct {
        int begin;
        int end;
        int tail;
    } data[MAX_BLOCKS];
    void init() {
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            data[i].begin = block_size * i;
            data[i].end = data[i].tail = min(block_size * (i + 1), V);
        }
        data[NUM_BLOCKS - 1].tail = V;
    }
    auto operator[] (int index) {
#ifdef __NATIVE_CHECK
        if (index < 0 || index >= NUM_BLOCKS) {
            error();
        }
#endif
        return data[index];
    }
} range;
struct object_t {
    int id;
    int length;
    int tag;
    int continous = true;
    int reference;
    int create_time = INF;
    int del_time;
    int read[MAX_INTERVALS];
    status_t status = status_t::idle;
    array<int, DUPLICATE> replica;
    array<vector<int>, DUPLICATE> unit;
    vector<int> request;
    vector<int> processing;
    vector<int> cancelled;
} object[MAX_OBJECT];
struct request_t {
    inline static vector<int> at[MAX_T];
    int id;
    int obj_id;
    int time;
    status_t status;
} request[MAX_REQUEST];
struct driver_t {
    int cur = -1;
    int head;
    int progress;
    int times;
    bool passby;
    void jmp(int p) {
        cur = -1;
        head = p;
        progress = 0;
        times = 0;
        passby = false;
    }
} driver[MAX_N][NUM_HEADS];
struct expense_t {
    int data[READ_FACTOR] = {};
    expense_t() {
        data[0] = READ_BASE;
        for (int i = 1; i < READ_FACTOR; ++i) {
            data[i] = max(16, (int)ceil(data[i - 1] * 0.8));
        }
    }
    int operator[] (int index) const noexcept {
#ifdef __NATIVE_CHECK
        if (index < 0 || index >= READ_FACTOR) {
            error();
        }
#endif
        return data[index];
    }
} expense;
struct score_t {
    int index = 0;
    int load[MAX_N];
    vector<int> reqs[EXTRA_TIME + 2];
    struct data_t {
        int last = -1;
        int count = 0;
    } data[MAX_OBJECT + 1], backup[MAX_OBJECT + 1];
    static double calculate(int rq) {
        auto f = [](int x) {
            if (x <= 10) {
                return -0.005 * x + 1;
            }
            else if (x <= EXTRA_TIME) {
                return -0.01 * x + 1.05;
            }
            else {
                return 0.0;
            }
            };
        auto g = [](int size) {
            return (size + 1) * 0.5;
            };
        int x = timestamp - request[rq].time;
        return f(x) * g(object[request[rq].obj_id].length);
    }
    static double penalize(int rq) {
        double x = min(timestamp - request[rq].time, EXTRA_TIME);
        double size = object[request[rq].obj_id].length;
        return (x / EXTRA_TIME) * (size + 1) * 0.5;
    }
    void step() {
        index = (index + 1) % EXTRA_TIME;
        for (auto rq : reqs[index]) {
            const int id = request[rq].obj_id;
            if (request[rq].time > data[id].last) {
                data[id].count -= 1;
                for (int i = 0; i < DUPLICATE; ++i) {
                    load[object[id].replica[i]] -= 1;
                }
            }
            if (request[rq].time > backup[id].last) {
                backup[id].count -= 1;
            }
        }
        reqs[index].clear();
    }
    void add(int rq) {
        const int id = request[rq].obj_id;
        reqs[index].push_back(rq);
        data[id].count += 1;
        for (int i = 0; i < DUPLICATE; ++i) {
            load[object[id].replica[i]] += 1;
        }
    }
    void clear(int id) {
        for (int i = 0; i < DUPLICATE; ++i) {
            load[object[id].replica[i]] -= data[id].count;
        }
        backup[id] = data[id];
        data[id].count = 0;
        data[id].last = timestamp;
    }
    void resume(int id) {
        data[id].last = backup[id].last;
        data[id].count += backup[id].count;
        for (int i = 0; i < DUPLICATE; ++i) {
            load[object[id].replica[i]] += backup[id].count;
        }
    }
    int count(int id) {
#ifdef __NATIVE_CHECK
        if (id < 0) {
            error();
        }
#endif
        return data[id].count;
    }
    int disk_load(int x) {
#ifdef __NATIVE_CHECK
        if (x < 0 || x >= N) {
            error();
        }
#endif
        return load[x];
    }
    void init() {
        index = 0;
        memset(load, 0, sizeof(load));
        for (int i = 0; i < size(reqs); ++i) {
            reqs[i].clear();
        }
        for (int i = 0; i < size(data); ++i) {
            data[i] = backup[i] = data_t{};
        }
    }
} score;
struct disk_t {
private:
    int data[MAX_N][MAX_V];
    int statistics[MAX_N][MAX_BLOCKS][MAX_M];
    int fail[MAX_N][MAX_BLOCKS];
public:
    void init() {
        memset(data, -1, sizeof(data));
        memset(statistics, 0, sizeof(statistics));
        for (int x = 0; x < MAX_N; ++x) {
            for (int a = 0; a < MAX_BLOCKS; ++a) {
                fail[x][a] = INF;
            }
        }
    }
    disk_t() {
        init();
    }
    const int* operator[] (int x) const {
#ifdef __NATIVE_CHECK
        if (x < 0 || x >= N) {
            error();
        }
#endif
        return data[x];
    }
    void set(int x, const vector<int>& positions, int value, int tag) {
#ifdef __NATIVE_CHECK
        if (x < 0 || x >= N || positions.empty() || value == -1 || tag < 0 || tag >= M) {
            error();
        }
#endif
        for (auto y : positions) {
#ifdef __NATIVE_CHECK
            if (y < 0 || y > V || data[x][y] != -1 || block(y) >= MAX_BLOCKS) {
                error();
            }
#endif
            data[x][y] = value;
        }
        statistics[x][block(positions.front())][tag] += positions.size();
    }
    void clear(int x, const vector<int>& positions, int tag) {
#ifdef __NATIVE_CHECK
        if (x < 0 || x >= N || positions.empty() || tag < 0 || tag >= M) {
            error();
        }
#endif
        for (auto y : positions) {
#ifdef __NATIVE_CHECK
            if (y < 0 || y > V || data[x][y] == -1 || block(y) >= MAX_BLOCKS) {
                error();
            }
#endif
            data[x][y] = -1;
        }
        const int a = block(positions.front());
        statistics[x][a][tag] -= positions.size();
        fail[x][a] = INF;
#ifdef __NATIVE_CHECK
        if (statistics[x][block(positions.front())][tag] < 0) {
            error();
        }
#endif
    }
    void update(int x, int a, int length) {
        fail[x][a] = min(fail[x][a], length);
    }
    auto get_blocks(int tag, int length) {
        double weight[MAX_N][MAX_BLOCKS];
        for (int x = 0; x < N; ++x) {
            for (int i = 0; i < NUM_BLOCKS; ++i) {
                if (length >= fail[x][i]) {
                    continue;
                }
                double numerator = 0, denominator = 1;
                for (int t = 0; t < M; ++t) {
                    numerator += angle[tag][t] * statistics[x][i][t];
                    denominator += statistics[x][i][t];
                }
                double value = numerator / pow(denominator, DENO_POW);
                for (int t = 0; t < M; ++t) {
                    if (t != tag && statistics[x][i][t] > 0) {
                        value += PENALIZE_MULTI_COLOR;
                    }
                }
                if (statistics[x][i][tag] == 0) {
                    value += PENALIZE_MULTI_COLOR;
                }
                value -= statistics[x][i][tag] * SAME_COLOR_INC;
                if (belong[x][i] == tag) {
                    if constexpr (LEFT_BLOCK_FIRST) {
                        value -= 100 * (NUM_BLOCKS - i + 5);
                    }
                    else {
                        value -= 1e-6;
                    }
                }
                if (belong[x][i] < 0) {
                    value -= abs(belong[x][i]) * 1e-8;
                }
                else {
                    value -= angle[belong[x][i]][tag] * 1e-9;
                }
                weight[x][i] = value;
            }
        }
        vector<tuple<double, int, int>> ret;
        ret.reserve(N * NUM_BLOCKS);
        for (int x = 0; x < N; ++x) {
            for (int a = 0; a < NUM_BLOCKS; ++a) {
                if (length >= fail[x][a]) {
                    continue;
                }
                double sum = 0;
                for (auto [y, b] : group[x][a]) {
                    sum += weight[y][b];
                }
                ret.emplace_back(sum, x, a);
            }
        }
        sort(ret.begin(), ret.end(), less<>{});
        return ret;
    }
    double entropy() {
        double ret = 0;
        for (int x = 0; x < N; ++x) {
            for (int i = 0; i < NUM_BLOCKS; ++i) {
                double sum = 0;
                for (int t = 0; t < M; ++t) {
                    sum += statistics[x][i][t];
                }
                for (int t = 0; t < M; ++t) if (statistics[x][i][t] > 0) {
                    double p = statistics[x][i][t] / sum;
                    ret += -p * log(p);
                }
            }
        }
        return ret / N / NUM_BLOCKS;
    }
} disk;
struct output_t {
    vector<int> del;
    vector<tuple<int, decltype(object->replica), decltype(object->unit)>> create;
    string operation[MAX_N][NUM_HEADS];
    vector<int> finished_reqs;
    vector<int> busy_reqs;
    bool collection;
    vector<pair<int, int>> position[MAX_N];
    void operator() () {
        printf("%d\n", (int)del.size());
        for (auto rq : del) {
            printf("%d\n", rq + 1);
        }
        for (const auto& [id, replica, unit] : create) {
            printf("%d\n", id + 1);
            for (int i = 0; i < DUPLICATE; ++i) {
                printf("%d", replica[i] + 1);
                for (auto p : unit[i]) {
                    printf(" %d", p + 1);
                }
                puts("");
            }
        }
        for (int x = 0; x < N; ++x) for (int y = 0; y < NUM_HEADS; ++y) {
            puts(operation[x][y].c_str());
        }
        printf("%d\n", (int)finished_reqs.size());
        for (auto rq : finished_reqs) {
            printf("%d\n", rq + 1);
        }
        printf("%d\n", (int)busy_reqs.size());
        for (auto rq : busy_reqs) {
            printf("%d\n", rq + 1);
        }
        if (collection) {
            puts("GARBAGE COLLECTION");
            for (int x = 0; x < N; ++x) {
                printf("%d\n", (int)position[x].size());
                for (auto [a, b] : position[x]) {
                    printf("%d %d\n", a + 1, b + 1);
                }
            }
        }
        fflush(stdout);
    }
} output[MAX_T], answer[MAX_T];
auto time_elapsed() {
    auto end_time = chrono::steady_clock::now();
    return chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
}
void retire(vector<int>& reqs) {
    auto iter = remove_if(reqs.begin(), reqs.end(), [](int rq) {
        return request[rq].status != status_t::idle;
        });
    reqs.erase(iter, reqs.end());
    sort(reqs.begin(), reqs.end());
    iter = unique(reqs.begin(), reqs.end());
    reqs.erase(iter, reqs.end());
    for (auto rq : reqs) {
        request[rq].status = status_t::done;
    }
}
void calculate_action() {
    auto calc = [](int s) {
        int dp[WINDOW_SIZE + 1][READ_FACTOR];
        for (int j = 0; j < READ_FACTOR; ++j) {
            dp[WINDOW_SIZE][j] = 0;
        }
        for (int i = WINDOW_SIZE - 1; i >= 0; --i) {
            for (int j = READ_FACTOR - 1; j >= 0; --j) {
                int k = min(j + 1, READ_FACTOR - 1);
                dp[i][j] = dp[i + 1][k] + expense[j];
                if ((~s >> i) & 1) {
                    dp[i][j] = min(dp[i][j], dp[i + 1][0] + 1);
                }
            }
        }
        array<bool, READ_FACTOR> ret;
        for (int j = 0; j < READ_FACTOR; ++j) {
            if ((~s & 1) && dp[0][j] == dp[1][0] + 1) {
                ret[j] = false;
            }
            else {
                ret[j] = true;
            }
        }
        return ret;
        };
    for (int s = 0; s < (1 << WINDOW_SIZE); ++s) {
        auto ret = calc(s);
        for (int i = 0; i < READ_FACTOR; ++i) {
            action[s][i] = ret[i];
        }
    }
}
void calculate_group() {
    for (int x = 0; x < N; ++x) {
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            group[x][i].clear();
            brother[x][i].clear();
        }
    }
    if constexpr (LOCK_DUPLICATE && NUM_BLOCKS % DUPLICATE == 0) {
        map<int, vector<vector<pair<int, int>>>> occurance;
        for (int x = 0; x < N; ++x) {
            vector<pair<int, int>> now;
            for (int i = 0; i < NUM_BLOCKS; ++i) {
                now.emplace_back(x, i);
                if (i + 1 == NUM_BLOCKS || belong[x][i] != belong[x][i + 1]) {
                    occurance[belong[x][i]].push_back(move(now));
                    now.clear();
                }
            }
        }
        for (const auto& [_, vec] : occurance) {
#ifdef __NATIVE_CHECK
            if (vec.size() != DUPLICATE) {
                error(); //todo
            }
            for (int i = 0; i < vec.size(); ++i) if (vec[i].size() != vec[0].size()) {
                error();
            }
#endif
            for (int i = 0; i < vec.size(); ++i) {
                vector<int> tmp;
                for (int j = 0; j < vec[i].size(); ++j) {
                    tmp.push_back(vec[i][j].second);
#ifdef __NATIVE_CHECK
                    if (vec[i][j].first != vec[i][0].first) {
                        error();
                    }
#endif
                }
                for (int j = 0; j < vec[i].size(); ++j) {
                    auto [x, a] = vec[i][j];
                    brother[x][a] = tmp;
                }
            }
            for (int i = 0; i < vec[0].size(); ++i) {
                vector<pair<int, int>> gp;
                for (int j = 0; j < DUPLICATE; ++j) {
                    gp.push_back(vec[j][i]);
                }
                for (int j = 0; j < DUPLICATE; ++j) {
                    auto [x, y] = vec[j][i];
                    group[x][y] = gp;
                }
            }

        }
    }
    else {
        for (int x = 0; x < N; ++x) {
            for (int i = 0; i < NUM_BLOCKS; ++i) {
                group[x][i].emplace_back(x, i);
                brother[x][i].emplace_back(i);
            }
        }
    }
#ifdef __NATIVE_CHECK
    for (int x = 0; x < N; ++x) {
        for (int a = 0; a < NUM_BLOCKS; ++a) {
            if (group[x][a].empty() || brother[x][a].empty()) {
                error();
            }
        }
    }
#endif
}
void remove(const int id, const int start=0) {
    object[id].status = status_t::idle;
    for (int i = start; i < DUPLICATE; ++i) {
        disk.clear(object[id].replica[i], object[id].unit[i], object[id].tag);
        object[id].unit[i].clear();
    }
}
int insert(int id, int x, int a) {
    const auto [L, R, _] = range[a];
    int sum = 0, pos = -1;
    int limit = R - L;
    for (int i = 0; i < limit; ++i) {
        const int y = L + i;
        sum += (disk[x][y] == -1 ? 1 : -sum);
        if constexpr (MAGNETIC) {
            if (sum >= object[id].length) {
                if (pos == -1) {
                    pos = (y - sum + 1 + V) % V;
                }
                if (i + 1 == sum) {
                    break;
                }
                int pid = disk[x][(y - sum + V) % V];
                if (pid != -1 && object[pid].tag == object[id].tag) {
                    pos = (y - sum + 1 + V) % V;
                    break;
                }
            }
        }
        else {
            if (sum >= object[id].length) {
                pos = (y - sum + 1 + V) % V;
                break;
            }
        }
    }
    if (pos == -1) {
        disk.update(x, a, object[id].length);
    }
    return pos;
}
bool insert(const int id, bool exec=true) {
    int counter = DUPLICATE;
    int vis[MAX_N] = {};
    for (auto [w, x, i] : disk.get_blocks(object[id].tag, object[id].length)) {
        if (vis[x]) {
            continue;
        }
        int pos = insert(id, x, i);
        if (pos != -1) {
            vis[x] = true;
            counter -= 1;
            object[id].replica[counter] = x;
            for (int j = 0; j < object[id].length; ++j) {
                const int p = (pos + j) % V;
                object[id].unit[counter].push_back(p);
            }
            disk.set(x, object[id].unit[counter], id, object[id].tag);
        }
        if (counter == 0) {
            break;
        }
    }
    if (counter != 0) {
#ifdef __NATIVE_CHECK
        if (stage == 0) {
            error();
        }
#endif
        remove(id, counter);
        return false;
    }
    if (exec) {
        if (stage == 0) {
            printf("%d\n", id + 1);
            for (int i = 0; i < DUPLICATE; ++i) {
                printf("%d", object[id].replica[i] + 1);
                for (auto p : object[id].unit[i]) {
                    printf(" %d", p + 1);
                }
                puts("");
            }
        }
        else {
            output[object[id].create_time].create.emplace_back(id, object[id].replica, object[id].unit);
        }
    }
    return true;
}
int preallocate(int look_forward=INF) {
    if (stage == 0) {
        return -1;
    }
    auto test = [](int time_limit, bool exec=false) {
        vector<int> indices;
        for (int id = 0; id < MAX_OBJECT; ++id) {
            if (object[id].create_time < timestamp || object[id].create_time >= time_limit) {
                continue;
            }
            indices.push_back(id);
        }
        sort(indices.begin(), indices.end(), [](int x, int y) {
            if constexpr (CMP_TAG_POSITION == 0) {
                if (object[x].tag != object[y].tag) {
                    return tgrank[object[x].tag] < tgrank[object[y].tag];
                }
            }
            if (object[x].del_time != object[y].del_time) {
                return object[x].del_time > object[y].del_time;
            }
            if constexpr (CMP_TAG_POSITION == 1) {
                if (object[x].tag != object[y].tag) {
                    return tgrank[object[x].tag] < tgrank[object[y].tag];
                }
            }
            if (object[x].create_time != object[y].create_time) {
                return object[x].create_time < object[y].create_time;
            }
            if constexpr (CMP_TAG_POSITION == 2) {
                if (object[x].tag != object[y].tag) {
                    return tgrank[object[x].tag] < tgrank[object[y].tag];
                }
            }
            return x < y;
        });
        vector<int> processed;
        for (auto id : indices) {
            if (insert(id, exec)) {
                object[id].status = status_t::future;
                processed.push_back(id);
            }
            else {
                break;
            }
        }
        if (!exec) {
            for (auto id : processed) {
                remove(id);
            }
        }
        return processed.size() == indices.size();
    };
    int L = timestamp, R = min(T + 1, timestamp + look_forward), pos = timestamp;
    if (test(R)) {
        pos = L = R;
    }
    while (L < R) {
        int M = L + (R - L + 1) / 2;
        if (test(M)) {
            pos = L = M;
        }
        else {
            R = M - 1;
        }
    }
    test(pos, true);
    return pos;
}
void preprocess() {
    num_intervals = (int)ceil(1.0 * T / TIME_INTERVAL_LENGTH);
    if constexpr (LOCK_DUPLICATE && NUM_BLOCKS % DUPLICATE == 0) {
        block_size = V / NUM_BLOCKS;
    }
    else {
        block_size = (V - 1) / NUM_BLOCKS + 1;
    }
    range.init();
    if (stage == 1) {
        memset(fre_write, 0, sizeof(fre_write));
        memset(fre_del, 0, sizeof(fre_del));
        memset(fre_read, 0, sizeof(fre_read));
        for (int id = 0; id < MAX_OBJECT; ++id) if (object[id].create_time != INF) {
            const int t = object[id].tag;
            if (t == 0) {
                continue;
            }
#ifdef __NATIVE_CHECK
            if (object[id].create_time == INF || object[id].create_time > object[id].del_time) {
                error();
            }
#endif
            fre_write[t][object[id].create_time / TIME_INTERVAL_LENGTH] += object[id].length;
            if (object[id].del_time != INF) {
                fre_del[t][object[id].del_time / TIME_INTERVAL_LENGTH] += object[id].length;
            }
            for (int j = 0; j < num_intervals; ++j) {
                fre_read[t][j] += object[id].read[j];
            }
        }
        double vec[MAX_M][MAX_INTERVALS] = {}, deno[MAX_M][MAX_INTERVALS] = {};
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < num_intervals; ++j) {
                deno[i][j] += fre_write[i][j] - fre_del[i][j];
            }
            for (int j = 1; j < num_intervals; ++j) {
                deno[i][j] += deno[i][j - 1];
            }
            for (int j = 0; j < num_intervals; ++j) {
                vec[i][j] = fre_read[i][j] / (deno[i][j] + max(1.0, SMOOTH_DENO));
            }
        }
        double importance[MAX_INTERVALS] = {};
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < num_intervals; ++j) {
                importance[j] += fre_read[i][j];
            }
        }
        double sum = accumulate(importance, importance + num_intervals, 1e-9);
        for (int j = 0; j < num_intervals; ++j) {
            importance[j] /= sum;
        }
        for (int id = 0; id < MAX_OBJECT; ++id) if (object[id].create_time != INF) {
            double distance = FINF;
            if (object[id].tag == 0) {
                for (int t = 1; t < M; ++t) {
                    double now = 0;
                    for (int j = 0; j < num_intervals; ++j) {
                        double delta = vec[t][j] - 1.0 * object[id].read[j] / object[id].length;
                        now += pow(fabs(delta), KMEANS_DIS_POW) * pow(importance[j], IMPORTANCE_POW);
                    }
                    if (distance > now) {
                        distance = now;
                        object[id].tag = t;
                    }
                }
                const int t = object[id].tag;
                fre_write[t][object[id].create_time / TIME_INTERVAL_LENGTH] += object[id].length;
                if (object[id].del_time != INF) {
                    fre_del[t][object[id].del_time / TIME_INTERVAL_LENGTH] += object[id].length;
                }
                for (int j = 0; j < num_intervals; ++j) {
                    fre_read[t][j] += object[id].read[j];
                }
            }
        }
    }

    double read[MAX_M][MAX_INTERVALS] = {};
    if constexpr (SMOOTH_DENO != 0.0) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < num_intervals; ++j) {
                read[i][j] += fre_write[i][j] - fre_del[i][j];
            }
            for (int j = 1; j < num_intervals; ++j) {
                read[i][j] += read[i][j - 1];
            }
            for (int j = 0; j < num_intervals; ++j) {
                read[i][j] = 1.0 * fre_read[i][j] / (read[i][j] + SMOOTH_DENO);
            }
        }
    }
    else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < num_intervals; ++j) {
                read[i][j] = fre_read[i][j];
            }
        }
    }
    double importance[MAX_INTERVALS] = {};
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < num_intervals; ++j) {
            importance[j] += read[i][j];
        }
    }
    double sum = accumulate(importance, importance + num_intervals, 1e-9);
    for (int j = 0; j < num_intervals; ++j) {
        importance[j] /= sum;
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < num_intervals; ++j) {
            read[i][j] *= pow(importance[j], IMPORTANCE_POW);
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            double sum = 0, len1 = 1e-9, len2 = 1e-9;
            for (int k = 0; k < num_intervals; ++k) {
                sum += read[i][k] * read[j][k];
                len1 += read[i][k] * read[i][k];
                len2 += read[j][k] * read[j][k];
            }
            double value = sum / sqrt(len1) / sqrt(len2);
            value = clamp(value, -1.0, 1.0);
            angle[i][j] = angle[j][i] = acos(value);
        }
        angle[i][i] = 0;
    }

    auto check = [](int pos, const auto& state) {
        auto& [indices, order] = state;
        array<array<int, MAX_BLOCKS>, MAX_N> ret;
        for (int i = 0; i < MAX_N; ++i) {
            for (int j = 0; j < MAX_BLOCKS; ++j) {
                ret[i][j] = -1;
            }
        }
        int sum[MAX_M] = {}, used[MAX_M] = {};
        int rest[MAX_N] = {};
        for (int i = 0; i < N; ++i) {
            rest[i] = NUM_BLOCKS;
        }
        for (int t = 0; t < M; ++t) {
            double now = 0;
            for (int i = 0; i <= pos; ++i) {
                now += fre_write[t][i];
                if (i > 0) {
                    now -= DELETE_PERCENTAGE * fre_del[t][i - 1];
                }
                sum[t] = max(sum[t], (int)now);
            }
            used[t] = sum[t] * DUPLICATE;
        }
        for (auto t : indices) {
            for (auto x : order[t]) {
                if (used[t] <= 0) {
                    break;
                }
                int chunk = max(min(used[t], sum[t]), 0);
                int needed = (chunk - 1) / block_size + 1;
                int occupied = min(needed, rest[x]);
                if (occupied * block_size < sum[t]) {
                    continue;
                }

                for (int i = NUM_BLOCKS - rest[x]; i < NUM_BLOCKS - rest[x] + occupied; ++i) {
                    ret[x][i] = t;
                }

                rest[x] -= occupied;
                used[t] -= min(occupied * block_size, sum[t]);
            }
            if (used[t] > 0) {
                return pair{ FINF, ret };
            }
        }
        double variance = 0;
        double vec[MAX_N][MAX_INTERVALS] = {}, center[MAX_INTERVALS] = {};
        for (int x = 0; x < N; ++x) {
            for (int i = 0; i < NUM_BLOCKS; ++i) {
                if (i > 0 && ret[x][i] == ret[x][i - 1]) {
                    continue;
                }
                if (ret[x][i] != -1) {
                    for (int j = 0; j < num_intervals; ++j) {
                        vec[x][j] += fre_read[ret[x][i]][j];
                    }
                }
            }
            for (int j = 0; j < num_intervals; ++j) {
                center[j] += vec[x][j];
            }
        }
        for (int j = 0; j < num_intervals; ++j) {
            center[j] /= N;
        }
        for (int x = 0; x < N; ++x) {
            double sum = 0;
            for (int j = 0; j < num_intervals; ++j) {
                sum += (vec[x][j] - center[j]) * (vec[x][j] - center[j]);
            }
            variance += sqrt(sum);
        }
        if constexpr (LOCK_DUPLICATE && NUM_BLOCKS % DUPLICATE == 0) {
            priority_queue<tuple<int, int>> Q;
            for (int x = 0; x < N; ++x) if (rest[x]) {
                Q.emplace(rest[x], x);
            }
            int counter = -1;
            while (Q.size()) {
                if (Q.size() < DUPLICATE) {
                    return pair{ FINF, ret };
                }
                vector<int> indices;
                for (int i = 0; i < DUPLICATE; ++i) {
                    auto [_, x] = Q.top(); Q.pop();
                    indices.push_back(x);
                }
                for (auto x : indices) {
                    ret[x][NUM_BLOCKS - rest[x]] = counter;
                    rest[x] -= 1;
                    if (rest[x]) {
                        Q.emplace(rest[x], x);
                    }
                }
                counter -= 1;
            }
        }
        return pair{ variance, ret };
        };
    auto random = [](auto& state) {
        auto& [indices, order] = state;
        order.resize(M);
        for (int t = 0; t < M; ++t) {
            order[t].resize(N);
            iota(order[t].begin(), order[t].end(), 0);
            shuffle(order[t].begin(), order[t].end(), engine);
        }
        indices.resize(M);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), engine);
        };
    auto transform = [](auto& state) {
        auto& [indices, order] = state;
        uniform_int_distribution<int> gi(0, (int)indices.size() - 1);
        swap(indices[gi(engine)], indices[gi(engine)]);
        uniform_int_distribution<int> go(0, (int)order[0].size() - 1);
        for (int i = 0; i < order.size(); ++i) {
            swap(order[i][go(engine)], order[i][go(engine)]);
        }
        };
    pair<vector<int>, vector<vector<int>>> save;
    random(save);
    int progress = 0;
    for (int i = 0; i < num_intervals; ++i) {
        double best = FINF;
        for (int T = 0; T < SEARCH_ITERATIONS; ++T) {
            pair<vector<int>, vector<vector<int>>> state;
            random(state);
            auto [score, arr] = check(i, state);
            if (score < best) {
                best = score;
                progress = i;
                save = state;
                for (int i = 0; i < MAX_N; ++i) {
                    for (int j = 0; j < MAX_BLOCKS; ++j) {
                        belong[i][j] = arr[i][j];
                    }
                }
                break;
            }
        }
        if (best == FINF) {
            break;
        }
    }
    double best = FINF, current = FINF;
    int last = -1;
    for (int T = 0; T < MAX_ITERATIONS; ++T) {
        pair<vector<int>, vector<vector<int>>> state = save;
        transform(state); //todo: 把transform改成random会变抖，也可能产生更好的结果
        auto [score, arr] = check(progress, state);
        if (current > score) {
            current = score;
            save = state;
            last = T;
        }
        if (best > score) {
            best = score;
            for (int i = 0; i < MAX_N; ++i) {
                for (int j = 0; j < MAX_BLOCKS; ++j) {
                    belong[i][j] = arr[i][j];
                }
            }
        }
        if (T - last > RESTART_ITERATIONS) {
            random(save);
            current = FINF;
        }
    }
    calculate_group();
    for (int i = 0; i < M; ++i) {
        tgrank[i] = i;
    }
    shuffle(tgrank, tgrank + M, engine);
    /*for (int x = 0; x < N; ++x) {
        for (int a = 0; a < NUM_BLOCKS; ++a) {
            fprintf(stderr, "%d ", belong[x][a]);
        }
        fprintf(stderr, "\n");
    }*/
}
void time_align() {
    char buffer[32];
    read_str(buffer);
    int t = read_int();
    printf("TIMESTAMP %d\n", timestamp + 1);
#ifdef __NATIVE_CHECK
    if (t != timestamp + 1) {
        error();
    }
#endif
    fflush(stdout);
}
void object_delete() {
    static int nabort[MAX_REQUEST];
    static vector<int> input[MAX_T];
    if (stage == 0) {
        int n = read_int();
        input[timestamp].reserve(n);
        for (int i = 0; i < n; ++i) {
            int id = read_int();
            id -= 1;
            input[timestamp].emplace_back(id);
            object[id].del_time = timestamp;
        }
    }
    int size = 0;
    for (auto id : input[timestamp]) {
        for (int j = 0; j < DUPLICATE; ++j) {
            for (auto x : object[id].unit[j]) {
#ifdef __NATIVE_CHECK
                if (disk[object[id].replica[j]][x] != id) {
                    error();
                }
#endif
            }
            disk.clear(object[id].replica[j], object[id].unit[j], object[id].tag);
        }
        for (auto rq : object[id].request) {
            nabort[size++] = rq;
        }
        for (auto rq : object[id].processing) {
            nabort[size++] = rq;
        }
        for (auto rq : object[id].cancelled) {
            nabort[size++] = rq;
        }
        object[id].status = status_t::deleted;
        object[id].request.clear();
        object[id].processing.clear();
        object[id].cancelled.clear();
        for (int j = 0; j < DUPLICATE; ++j) {
            int y = object[id].replica[j];
            int b = block(object[id].unit[j].front());
            query[y][b] -= object[id].length * object[id].reference;
            object[id].unit[j].clear();
        }
        score.clear(id);
    }
    auto retired = vector<int>(nabort, nabort + size);
    retire(retired);
    if (stage == 0) {
        printf("%d\n", (int)retired.size());
        for (auto rq : retired) {
            printf("%d\n", rq + 1);
        }
        fflush(stdout);
    }
    else {
        output[timestamp].del = retired;
    }
}
void object_create() {
    static vector<int> input[MAX_T];
    if (stage == 0) {
        int n = read_int();
        input[timestamp].reserve(n);
        for (int i = 0; i < n; ++i) {
            int id = read_int();
            int length = read_int();
            int tag = read_int();
            id -= 1;
            object[id].id = id;
            object[id].length = length;
            object[id].tag = tag;
            input[timestamp].emplace_back(id);
            object[id].create_time = timestamp;
            object[id].del_time = INF;
        }
    }
    for (auto id : input[timestamp]) {
        object[id].status = status_t::idle;
        if (object[id].unit[0].empty()) {
            insert(id);
        }
    }
    fflush(stdout);
}
void lock(int id) {
    object[id].status = status_t::processing;
    object[id].processing = object[id].request;
    object[id].request.clear();
    score.clear(id);
}
void unlock(int id) {
    object[id].processing.clear();
    object[id].status = status_t::idle;
}
auto score_blocks(int x, int y) {
    static long long buffer[MAX_OBJECT], *vis = buffer + 1, counter = 0;
    static int sequence[MAX_V + WINDOW_SIZE + 1] = {};
    const int speculate_length = block_size * SPECULATE_COEF;
    counter += 1;
    double weight[MAX_N] = {};
    if constexpr (DISK_POW_JMP != 0) {
        for (int x = 0; x < N; ++x) {
            weight[x] = score.disk_load(x);
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < NUM_HEADS; ++j) {
            if (x == i && y == j) {
                continue;
            }
            const int head = driver[i][j].head;
            int end = head + speculate_length;
            if constexpr (SPECULATE_COEF == 2.0) {
                end = range[block(head)].tail;
            }
            end %= V;
            for (int p = head; p != end; p = (p + 1 != V ? p + 1 : 0)) {
                vis[disk[i][p]] = counter;
            }
        }
    }
    vis[-1] = counter;
    vector<double> ret(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        const auto [L, R, _] = range[i];
        for (int p = L; p < R; ++p) {
            const int id = disk[x][p];
            if (vis[id] != counter) { // &&object[id].continous
                ret[i] += score.count(id);
            }
        }
        if (ret[i] == 0) {
            continue;
        }
        int state = 0;
        if constexpr (TOKENS_POW != 0) {
            for (int p = L; p < R; ++p) {
                sequence[p] = false;
                const int id = disk[x][p];
                if (vis[id] != counter) { // &&object[id].continous
                    sequence[p] = (bool)score.count(id);
                }
            }
            for (int j = 0; j < WINDOW_SIZE; ++j) {
                sequence[R + j] = false;
                state |= (sequence[L + j] << j);
            }
        }
        int times = 0;
        double tokens = 1;
        for (int p = L; p < R; ++p) {
            if constexpr (TOKENS_POW != 0) {
                bool is_read = action[state][times];
                state = (state >> 1) | (sequence[p + WINDOW_SIZE] << (WINDOW_SIZE - 1));
                if (is_read) {
                    tokens += expense[times];
                    times = min(times + 1, READ_FACTOR - 1);
                }
                else {
                    tokens += 1;
                    times = 0;
                }
            }
        }
        if constexpr (TOKENS_POW != 0) {
            ret[i] /= pow(tokens, TOKENS_POW);
        }
        if (ENABLE_BALANCE || stage == 0) {
            double delta = min<double>(EXTRA_TIME, timestamp - lst[x][i]) / EXTRA_TIME;
            ret[i] *= pow(1 + delta, TIME_POW);
        }
        if constexpr (DISK_POW_JMP != 0) {
            double s = 1;
            for (auto [y, b] : group[x][i]) {
                s += weight[y];
            }
            ret[i] *= pow(s, DISK_POW_JMP);
        }
    }
    return ret;
}
bool jump(int x, int y, bool enforce) {
    auto& [cur, head, progress, times, passby] = driver[x][y];
    auto sc = score_blocks(x, y);
    int init = block(head);
    int index = init;
    auto base = sc[index];
    for (int i = 0; i < sc.size(); ++i) {
        if (sc[i] > base * JUMP_COEF && sc[i] > sc[index]) { //hyperparameter
            index = i;
        }
    }
    if (enforce || index != init) {
        if (cur != -1) {
            object[cur].status = status_t::idle;
            for (auto rq : object[cur].processing) {
                object[cur].request.push_back(rq);
            }
            object[cur].processing.clear();
            score.resume(cur);
        }
        const int L = range[index].begin;
        sprintf(buffer, "j %d", L + 1);
        if (stage == 0) {
            puts(buffer);
        }
        else {
            output[timestamp].operation[x][y] = buffer;
        }
        driver[x][y].jmp(L);
        for (auto [y, b] : group[x][index]) {
            lst[y][b] = timestamp;
        }
        return true;
    }
    return false;
}
auto load_balance() {
    //static ofstream out("data.txt");
    vector<int> ret;
    if (timestamp % TIME_DURATION != TIME_DURATION - 1) {
        return ret;
    }
    static long long vis[MAX_OBJECT], counter = 1; counter += 1;
    decltype(disable) prev;
    memcpy(prev, disable, sizeof(prev));
    memset(disable, 0, sizeof(disable));
    double sum = 1e-80;
    for (int x = 0; x < N; ++x) {
        for (int a = 0; a < NUM_BLOCKS; ++a) {
            sum += query[x][a];
            if (query[x][a] == 0) {
                disable[x][a] = true;
            }
        }
    }
    double rate = 1.0 * busy / sum;
    if (rate <= min(1e-4, BALANCE_POINT)) {
        factor *= DECREASE_FACTOR;
    }
    else if (rate > BALANCE_POINT) {
        factor += rate;
    }
    //综合考虑负载和busy
    factor = clamp(factor, 1e-9, 0.8);
    //print(rate, factor, rate >= 1e-4);
    double threshold = sum * factor;
    for (int i = 0; i < N * NUM_BLOCKS; ++i) {
        pair<int, int> bk{ -1, -1 };
        vector<tuple<double, int, int>> vec;
        for (int x = 0; x < N; ++x) {
            for (int a = 0; a < NUM_BLOCKS; ++a) {
                vec.emplace_back(query[x][a] / (prev[x][a] * CONSISTANT_COEF + 1.0), x, a);
            }
        }
        sort(vec.begin(), vec.end());
        for (auto [_, x, a] : vec) {
            if (query[x][a] > 0 && query[x][a] <= threshold) {
                bk = { x, a };
                break;
            }
        }
        auto [x, a] = bk;
        if (x == -1) {
            break;
        }
        if (query[x][a] <= threshold) {
            threshold -= query[x][a];
            disable[x][a] = true;
            const auto [L, R, _] = range[a];
            for (int i = L; i < R; ++i) {
                int id = disk[x][i];
                if (id == -1) {
                    continue;
                }
                if (disk[x][(i - 1 + V) % V] == id) {
                    continue;
                }
                if (vis[id] != counter) {
                    vis[id] = counter;
                    for (int j = 0; j < DUPLICATE; ++j) {
                        int y = object[id].replica[j];
                        int b = block(object[id].unit[j].front());
                        query[y][b] -= object[id].length * object[id].reference;
                    }
                    if (object[id].status == status_t::idle) {
                        lock(id);
                        for (auto rq : object[id].processing) {
                            ret.push_back(rq);
                        }
                        unlock(id);
                    }
                }
            }
#ifdef __NATIVE_CHECK
            for (auto [y, b] : group[x][a]) {
                if (query[y][b]) {
                    error();
                }
            }
#endif
        }
        else {
            break;
        }
    }
    //out << rate << ' ' << factor << ' ' << (sum * factor - threshold) / sum << ' ' << busy << ' ' << sum << endl;
    busy = 0;
    memset(query, 0, sizeof(query));
    for (int i = 0; i < MAX_OBJECT; ++i) {
        object[i].reference = 0;
    }

    return ret;
}
void object_read() {
    static vector<tuple<int, int>> input[MAX_T];
    if (stage == 0) {
        int n = read_int();
        input[timestamp].reserve(n);
        for (int i = 0; i < n; ++i) {
            int rq = read_int(); rq -= 1;
            int id = read_int(); id -= 1;
            request[rq].id = rq;
            request[rq].obj_id = id;
            input[timestamp].emplace_back(rq, id);
            object[id].read[timestamp / TIME_INTERVAL_LENGTH] += object[id].length;
        }
    }
    vector<int> finished, retired;
    for (auto [rq, id] : input[timestamp]) {
        request[rq].time = timestamp;
        request[rq].status = status_t::idle;
        object[id].reference += 1;
        int disabled = false;
        for (int i = 0; i < DUPLICATE; ++i) {
            const int x = object[id].replica[i];
            const int a = block(object[id].unit[i].front());
            disabled |= disable[x][a];
            query[x][a] += object[id].length;
        }
        if (disabled) {
            retired.push_back(rq);
        }
        else {
            request_t::at[timestamp].push_back(rq);
            object[id].request.push_back(rq);
            score.add(rq);
        }
    }
    for (int x = 0; x < N; ++x) for (int y = 0; y < NUM_HEADS; ++y) {
        auto& [cur, head, progress, times, passby] = driver[x][y];
        times = min(times, READ_FACTOR - 1);
        int start_block = block(head);
        if (passby) {
            passby = false;
            if (jump(x, y, false)) {
                continue;
            }
            for (auto [y, b] : group[x][block(head)]) {
                lst[y][b] = timestamp;
            }
        }
        int skip = -1;
        if (disk[x][head] != cur) {
            cur = -1;
            progress = 0;
        }
        if (cur == -1 && disk[x][head] != -1 && disk[x][head] == disk[x][(head - 1 + V) % V]) {
            skip = disk[x][head];
        }
        auto bad = [](auto id) {
            return !score.count(id) || object[id].status != status_t::idle; // || !object[id].continous
            };
        int sequence[MAX_G + WINDOW_SIZE + 1] = {};
        for (int i = 0; i <= G; ++i) {
            sequence[i] = false;
            const int id = disk[x][(head + i) % V];
            if (id == -1) {
                continue;
            }
            if (id != cur && bad(id)) {
                continue;
            }
            if (id == skip) {
                continue;
            }
            sequence[i] = true;
        }
        int state = 0;
        for (int i = 0; i < WINDOW_SIZE; ++i) {
            state |= (sequence[i] << i);
        }
        char answer[MAX_G + 2] = {};
        int size = 0, tokens = G;
        for (int i = 0; i <= G; ++i) {
            bool is_read = action[state][times];
            state = (state >> 1) | (sequence[i + WINDOW_SIZE] << (WINDOW_SIZE - 1));
            if (!is_read && tokens >= 1) {
                answer[i] = 'p';
                tokens -= 1;
                times = 0;
            }
            else if (is_read && !sequence[i] && tokens >= expense[times]) {
                answer[i] = 'r';
                tokens -= expense[times];
                times = min(times + 1, READ_FACTOR - 1);
            }
            else if (is_read && sequence[i] && tokens >= expense[times]) {
                answer[i] = 'r';
                tokens -= expense[times];
                times = min(times + 1, READ_FACTOR - 1);
                if (cur == -1) {
                    cur = disk[x][head];
                    progress = 0;
                    lock(cur);
                }
                progress += 1;
                if (progress == object[cur].length) {
#ifdef __NATIVE_CHECK
                    if (object[cur].status != status_t::processing) {
                        error();
                    }
#endif
                    for (auto rq : object[cur].processing) {
                        finished.push_back(rq);
                    }
                    unlock(cur);
                    cur = -1;
                    progress = 0;
                }
            }
            else {
                size = i;
                break;
            }
            head = (head + 1) % V;
        }
        if (cur == -1 && size == G) {
            if constexpr (INBLOCK_JMP) {
                for (int i = 0; i < V; ++i) {
                    int p = (head + i) % V;
                    if (block(p) != block(head)) {
                        jump(x, y, true);
                        break;
                    }
                    int id = disk[x][p];
                    if (i == V - 1 || (id != -1 && !bad(id))) {
                        driver[x][y].jmp(p);
                        sprintf(buffer, "j %d", p + 1);
                        if (stage == 0) {
                            puts(buffer);
                        }
                        else {
                            output[timestamp].operation[x][y] = buffer;
                        }
                        break;
                    }
                }
            }
            else {
                jump(x, y, true);
            }
            continue;
        }
        answer[size++] = '#';
        answer[size] = 0;
        int end_block = block(head);
        if (start_block != end_block) {
            passby = true;
        }
        if (stage == 0) {
            printf("%s\n", answer);
        }
        else {
            output[timestamp].operation[x][y] = answer;
        }
    }
#ifdef __NATIVE_FINETUNE
    static double hidden = -1;
#else
    static double hidden = -1;
#endif
    while (!finished.empty() && hidden > 0) {
        auto rq = finished.back();
        finished.pop_back();
        if (request[rq].status == status_t::idle) {
            hidden -= score.calculate(rq) + score.penalize(rq);
            //object[request[rq].obj_id].cancelled.push_back(rq);
            retired.push_back(rq);
        }
    }
    retire(finished);
    if (stage == 0) {
        printf("%d\n", (int)finished.size());
        for (auto rq : finished) {
            printf("%d\n", rq + 1);
        }
    }
    else {
        output[timestamp].finished_reqs = finished;
    }
    if (timestamp - EXTRA_TIME >= 0) {
        for (auto rq : request_t::at[timestamp - EXTRA_TIME]) {
            retired.push_back(rq);
            if (request[rq].status == status_t::idle) {
                busy += object[request[rq].obj_id].length * DUPLICATE;
            }
        }
        request_t::at[timestamp - EXTRA_TIME].clear();
    }
    if (ENABLE_BALANCE || stage == 0) {
        for (auto rq : load_balance()) {
            retired.push_back(rq);
        }
    }
    retire(retired);
    if (stage == 0) {
        printf("%d\n", (int)retired.size());
        for (auto rq : retired) {
            printf("%d\n", rq + 1);
        }
    }
    else {
        for (auto rq : retired) {
            output[request[rq].time].busy_reqs.push_back(rq);
        }
    }
    for (auto rq : finished) {
        total[stage] += score.calculate(rq);
    }
    if (stage == 0) {
        for (auto rq : retired) {
            total[stage] -= score.penalize(rq);
        }
    }
    fflush(stdout);
}
void garbage_collection() {
    if (timestamp % TIME_INTERVAL_LENGTH != TIME_INTERVAL_LENGTH - 1) {
        return;
    }
    if (stage == 0) {
        char buffer[128][2];
        read_str(buffer[0]);
        read_str(buffer[1]);
    }
    vector<pair<int, int>> result[MAX_N];
    vector<tuple<int, int, int>> blocks;
    for (int x = 0; x < N; ++x) {
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            int last = true, chunks = 0;
            const auto [L, R, _] = range[i];
            for (int j = 0; j < R - L; ++j) {
                int now = (disk[x][L + j] != -1);
                if (now != last) {
                    chunks += 1;
                }
                last = now;
            }
            blocks.emplace_back(chunks, x, i);
        }
    }
    sort(blocks.begin(), blocks.end(), greater<>{});
    for (auto [w, x, a] : blocks) {
        const auto [L, R, _] = range[a];
        for (int i = R - 1; i >= L; --i) {
            int id = disk[x][i];
            if (id == -1 || !object[id].continous || object[id].status != status_t::idle) {
                continue;
            }
            if constexpr (MOVE_SAME_COLOR_ONLY) {
                if (belong[x][a] >= 0 && object[id].tag != belong[x][a]) {
                    continue;
                }
            }
            int size = 0;
            for (auto [y, b] : group[x][a]) {
                size = max(size, (int)result[y].size());
            }
            if (object[id].length + size > K[stage]) {
                continue;
            }
            auto r = find(object[id].replica.begin(), object[id].replica.end(), x) - object[id].replica.begin();
#ifdef __NATIVE_CHECK
            if (r == DUPLICATE) {
                error();
            }
#endif
            if (object[id].unit[r].front() != i) {
                continue;
            }

            int pos = -1;
            disk.clear(object[id].replica[r], object[id].unit[r], object[id].tag);
            if constexpr (CROSS_BLOCK_MOVEMENT) {
                for (auto na : brother[x][a]) {
                    pos = insert(id, x, na);
                    if (pos != -1) {
                        break;
                    }
                }
            }
            else {
                pos = insert(id, x, a);
            }
#ifdef __NATIVE_CHECK
            if (pos == -1) {
                error();
            }
#endif
            disk.set(object[id].replica[r], object[id].unit[r], id, object[id].tag);
            if (pos == i) {
                continue;
            }
            const int offset = pos - range[a].begin;
            for (auto [y, b] : group[x][a]) {
                auto r = find(object[id].replica.begin(), object[id].replica.end(), y) - object[id].replica.begin();
#ifdef __NATIVE_CHECK
                if (r == DUPLICATE) {
                    error();
                }
#endif
                const auto tmp = object[id].unit[r];
                const auto base = range[b].begin + offset;
                disk.clear(object[id].replica[r], object[id].unit[r], object[id].tag);
                for (int j = 0; j < object[id].length; ++j) {
                    const int p = (base + j + V) % V;
                    object[id].unit[r][j] = p;
                }
                disk.set(object[id].replica[r], object[id].unit[r], id, object[id].tag);
                const int nb = block(base);
                query[y][b] -= object[id].length * object[id].reference;
                query[y][nb] += object[id].length * object[id].reference;
#ifdef __NATIVE_CHECK
                if (tmp == object[id].unit[r]) {
                    error();
                }
#endif
                for (int j = 0; j < object[id].length; ++j) {
                    result[y].emplace_back(tmp[j], object[id].unit[r][j]);
                }
            }
        }
    }
    if (stage == 0) {
        puts("GARBAGE COLLECTION");
        for (int x = 0; x < N; ++x) {
            printf("%d\n", (int)result[x].size());
            for (auto [a, b] : result[x]) {
                printf("%d %d\n", a + 1, b + 1);
            }
        }
        fflush(stdout);
    }
    else {
        output[timestamp].collection = true;
        for (int x = 0; x < N; ++x) {
            output[timestamp].position[x] = result[x];
        }
    }
}
void reinit() {
    memset(disable, 0, sizeof(disable));
    memset(lst, 0, sizeof(lst));
    memset(query, 0, sizeof(query));
    busy = 0;
    factor = 0;
    for (int i = 0; i < MAX_N; ++i) {
        for (int j = 0; j < NUM_HEADS; ++j) {
            driver[i][j] = driver_t{};
        }
    }
    for (int i = 0; i < MAX_T; ++i) {
        output[i] = output_t{};
    }
    for (int id = 0; id < MAX_OBJECT; ++id) {
        object[id].continous = true;
        object[id].reference = 0;
        object[id].status = status_t::idle;
        object[id].request.clear();
        object[id].processing.clear();
        object[id].cancelled.clear();
        for (int j = 0; j < DUPLICATE; ++j) {
            object[id].unit[j].clear();
            object[id].unit[j].reserve(object[id].length);
        }
    }
    score.init();
    disk.init();
}
void iterate() {
    if (stage == 1) {
        reinit();
    }
    preprocess();
    int last = 0;
    for (timestamp = 0; timestamp < T + EXTRA_TIME; ++timestamp) {
        if (stage == 1) {
            if (last == timestamp) {
                last = preallocate(timestamp == 0 ? INF : LOOK_FORWARD_LENGTH * 1024);
            }
        }
        score.step();
        if (stage == 0) {
            time_align();
        }
        object_delete();
        object_create();
        object_read();
        garbage_collection();
#ifdef __NATIVE_FINETUNE
        if (timestamp == T + EXTRA_TIME - 2) {
            print("[", stage, "] Score:", total[stage]);
        }
#endif
#ifdef __NATIVE_CHECK
        //if (timestamp == 70000) {
        //    render("Entropy:", disk.entropy());
        //}
        if constexpr (LOCK_DUPLICATE && NUM_BLOCKS % DUPLICATE == 0) {
            map<vector<int>, int> cnt;
            for (int x = 0; x < N; ++x) {
                for (int i = 0; i < NUM_BLOCKS; ++i) {
                    const auto [L, R, _] = range[i];
                    vector<int> chunk;
                    chunk.reserve(R - L);
                    for (int j = 0; j < R - L; ++j) {
                        chunk.push_back(disk[x][L + j]);
                    }
                    cnt[chunk] += 1;
                }
            }
            for (const auto& [chunk, c] : cnt) {
                if (c % DUPLICATE != 0) {
                    error();
                }
            }
        }
#endif
    }
}
int main() {
#ifdef __NATIVE_FREOPEN
    freopen("..\\data\\sample_practice_1.in", "r", stdin);
#endif
    calculate_action();
    T = read_int();
    M = read_int(); M += 1;
    N = read_int();
    V = read_int();
    G = read_int();
    K[0] = read_int();
    K[1] = read_int();
    puts("OK");
    fflush(stdout);
    stage = 0;
    iterate();
    stage = 1;
    int n = read_int();
    for (int i = 0; i < n; ++i) {
        int id = read_int(); id -= 1;
        int tag = read_int();
        object[id].tag = tag;
    }
    double best = -FINF;
    do {
        total[stage] = 0;
        iterate();
        if (best < total[stage]) {
            best = total[stage];
            for (int j = 0; j < MAX_T; ++j) {
                answer[j] = move(output[j]);
            }
        }
    } while (time_elapsed() < TIME_LIMIT_SECONDS);
    total[stage] = best;
    for (timestamp = 0; timestamp < T + EXTRA_TIME; ++timestamp) {
        time_align();
        answer[timestamp]();
    }
#ifdef __NATIVE_FREOPEN
    print("Time:", time_elapsed());
#endif
#ifdef __NATIVE_FINETUNE
    cerr << std::fixed << std::setprecision(4);
    cerr << "Score: " << total[0] + total[1] << endl;
    cerr.flush();
#endif
    return 0;
}