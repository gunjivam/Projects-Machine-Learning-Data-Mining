#pragma once
#include <vector>
#include "accelerate.h"
#include <omp.h>
#include <cmath>
#include <random>

typedef std::vector<std::vector<float>> mtx;

namespace evo {
	mtx split(std::vector<float> a);
	std::vector<float> get_column(mtx A, int i);
	float dot(std::vector<float> a, std::vector<float> b);
	std::vector<float> add(std::vector<float> a, std::vector<float> b);
	std::vector<float> add(std::vector<float> a, float b);
	std::vector<float> diff(std::vector<float> a, std::vector<float> b);
	std::vector<float> multiply(std::vector<float> a, std::vector<float> b);
	std::vector<float> multiply(std::vector<float> a, float b);
	std::vector<float> empty(int size);
	std::vector<float> vec(int size, int val);
	std::vector<float> random_vec(int size, float max=1.0f, float min=-1.0f);
	float sum(std::vector<float> a);
	
	mtx empty(int m, int n);
	mtx matmul(mtx A, mtx B);
	std::vector<float> matmul(std::vector<float> a, mtx B);
	mtx add(mtx A, mtx B);
	mtx random_mtx(int m, int n, float max=1.0f, float min=-1.0f);

	int argmax(std::vector<float> v);

}