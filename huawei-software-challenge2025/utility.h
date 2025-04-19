#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <array>
#include <cstring>
#include <random>
#include <algorithm>
#include <stack>
#include <queue>
#include <tuple>
#include <string>
#include <numeric>
#include <iomanip>
#include <map>
#include <fstream>
#include <cctype>
#define error() report_error(__LINE__)
inline void report_error(int line) {
	std::cerr << "Error at line " << line << std::endl;
	std::exit(1);
}
template<typename T> void render(const T& sth) {
	std::cerr << "\033[1;31m" << sth << "\033[0m" << std::endl;
}
template<typename A, typename ...T> void render(const A& sth, const T& ...args) {
	std::cerr << "\033[1;31m" << sth << " \033[0m";
	render(args...);
}
template<typename T> void print(const T& sth) {
	//cerr << "\033[1;31m" << sth << "\033[0m" << endl;
	std::cerr << sth << std::endl;
}
template<typename A, typename ...T> void print(const A& sth, const T& ...args) {
	//cerr << "\033[1;31m" << sth << " \033[0m";
	std::cerr << sth << " ";
	print(args...);
}
template<typename T> void print(const std::vector<T>& vec) {
	//std::cerr << "\033[1;31m[";
	for (auto x : vec)
		std::cerr << x << ' ';
	std::cerr << std::endl;
	//std::cerr << "]\033[0m\n";
}
template<typename T, int maxsize> struct ModifiablePriorityQueue { // 大根堆
	int data[maxsize * 2];
	int pos[maxsize * 2];
	T value[maxsize * 2];
	int sz;
	ModifiablePriorityQueue() : pos(), sz(0) {}
	void up(int i) { // 值变大
		auto index = data[i];
		auto val = value[index];
		while (i > 1) {
			int fa = i / 2;
			if (val > value[data[fa]]) {
				data[i] = data[fa];
				pos[data[i]] = i;
				i = fa;
			}
			else {
				break;
			}
		}
		data[i] = index;
		pos[index] = i;
	}
	void down(int i) { // 值变小
		auto index = data[i];
		auto val = value[index];
		while (i * 2 <= sz) {
			int child = i * 2;
			if (i * 2 + 1 <= sz && value[data[i * 2 + 1]] > value[data[child]]) {
				child = i * 2 + 1;
			}
			if (val < value[data[child]]) {
				data[i] = data[child];
				pos[data[i]] = i;
				i = child;
			}
			else {
				break;
			}
		}
		data[i] = index;
		pos[index] = i;
	}
	void push(int index, const T& val) {
		sz += 1;
		data[sz] = index;
		value[index] = val;
		up(sz);
	}
	void pop() {
		pos[data[1]] = 0;
		data[1] = data[sz];
		sz -= 1;
		if (sz > 0) {
			down(1);
		}
	}
	void erase(int index) {
		index = pos[index];
		pos[data[index]] = 0;
		data[index] = data[sz];
		sz -= 1;
		if (index <= sz) {
			down(index);
		}
	}
	void modify(int index, const T& val) {
		if (pos[index] == 0) {
			push(index, val);
		}
		else if (val > value[index]) {
			value[index] = val;
			up(pos[index]);
		}
		else {
			value[index] = val;
			down(pos[index]);
		}
	}
	int top() const {
		return data[1];
	}
	T get(int index) const {
		return value[index];
	}
	T maximum() const {
		return value[data[1]];
	}
	bool contains(int index) const {
		return pos[index] != 0;
	}
	int size() const {
		return sz;
	}
	bool empty() const {
		return sz == 0;
	}
};
template<typename T, int maxsize> class darray {
private:
	T data[maxsize];
	int sz;
public:
	constexpr darray() : sz(0) {}
	constexpr void push_back(T val) noexcept {
		data[sz++] = val;
	}
	constexpr auto* begin() noexcept {
		return data;
	}
	constexpr auto* end() noexcept {
		return data + sz;
	}
	constexpr const auto* begin() const noexcept {
		return data;
	}
	constexpr const auto* end() const noexcept {
		return data + sz;
	}
	constexpr auto size() const noexcept {
		return sz;
	}
	constexpr bool empty() const noexcept {
		return sz == 0;
	}
	constexpr bool contains(T val) const noexcept {
		for (int i = 0; i < sz; ++i) {
			if (data[i] == val) {
				return true;
			}
		}
		return false;
	}
	constexpr T front() const noexcept {
		return data[0];
	}
	constexpr T back() const noexcept {
		return data[sz - 1];
	}
	constexpr T& operator[] (int index) noexcept {
		return data[index];
	}
	constexpr const T& operator[] (int index) const noexcept {
		return data[index];
	}
	constexpr void clear() noexcept {
		sz = 0;
	}
};
inline int read_int() { //读取一个正整数
	int x = 0;
	unsigned c = getchar();
	while (c - '0' > 9)
		c = getchar();
	while (c - '0' <= 9) {
		x = x * 10 + c - '0';
		c = getchar();
	}
	return x;
}
inline void read_str(char* buffer) {
	char c = getchar();
	while (isspace(c))
		c = getchar();
	while (!isspace(c)) {
		*buffer++ = c;
		c = getchar();
	}
	*buffer++ = 0;
}
inline int sleep(int seconds) {
	const auto start_time = std::chrono::steady_clock::now();
	for (;;) {
		auto now = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
		if (duration > seconds) {
			break;
		}
	}
	return 0;
}