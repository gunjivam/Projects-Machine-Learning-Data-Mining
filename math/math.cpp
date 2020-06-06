#include "math.h"
#include <iostream>

Accel A("C:/Users/ninanpyo/source/repos/Evolette/Evolette/functions.kernel");


float evo::dot(std::vector<float> a, std::vector<float> b) {
	return sum(A.call2v(a, b, "mult"));
}

std::vector<float> evo::add(std::vector<float> a, std::vector<float> b) {
	return A.call2v(a, b, "add");
}

std::vector<float> evo::add(std::vector<float> a, float b) {
	return A.call1v(a, "shift", b);
}
std::vector<float> evo::diff(std::vector<float> a, std::vector<float> b) {
	return A.call2v(a, b, "sub");
}

std::vector<float> evo::multiply(std::vector<float> a, std::vector<float> b) {
	return A.call2v(a, b, "mult");
}

std::vector<float> evo::multiply(std::vector<float> a, float b) {
	return A.call1v(a, "scale", b);
}

std::vector<float> evo::empty(int size) {
	std::vector<float> v(size, 0);
	return v;
}

std::vector<float> evo::vec(int size, int val) {
	return A.vec(val, size);
}

float evo::sum(std::vector<float> a) {
	std::vector<float> b = a;
	float sum = 0;
	while (b.size() > 1) {
		mtx res = split(b);
		b = A.call2v(res[0], res[1], "add");
		sum += res[2][0];
	}
	return sum + b[0];
}

mtx evo::split(std::vector<float> a) {
	int lmt = (int) std::floor(a.size() / 2);
	int lmt2 = a.size();
	int rm = a.size() % 2;
	std::vector<float> v1(lmt), v2(lmt2-lmt-rm), v3(1);

	for (int i = 0; i < lmt; i++) {
		v1[i] = a[i];
	}

	for (int i = lmt; i < lmt2-rm; i++) {
		v2[i-lmt] = a[i];
	}

	(rm) ? v3[0] = (a[lmt2 - 1]) : v3[0] = 0;

	mtx res = { v1, v2, v3 };
	return res;
}

mtx evo::matmul(mtx A, mtx B) {
	mtx C(A.size(), std::vector<float>(B[0].size(), 0));
	int sz = B[0].size();
	int sz1 = A.size();
	for (int j = 0; j < sz; j++) {
		std::vector<float> col = get_column(B, j);
#pragma omp parallel
		for (int i = 0; i < sz1; i++) {
			C[i][j] = dot(A[i], col);
		}
	}
	return C;
}

std::vector<float> evo::matmul(std::vector<float> a, mtx B) {
	int sz = B[0].size();
	std::vector<float> v(sz);
#pragma omp parallel
	for(int i = 0; i < sz; i++){
		v[i] = dot(a, get_column(B, i));
	}
	return v;
}

std::vector<float> evo::get_column(mtx A, int j) {
	std::vector<float> res(A.size());
	int sz = A.size();
#pragma omp parallel
	for (int i = 0; i < sz; i++) {
		res[i] = A[i][j];
	}
	return res;
}

mtx evo::add(mtx A, mtx B) {
	mtx C;
	int sz = A.size();
	for (int i = 0; i < sz; i++) {
		C.push_back(add(A[i], B[i]));
	}
	return C;
}


mtx evo::random_mtx(int m, int n, float max, float min) {
	mtx R(m, std::vector<float>(n, 0));

	for (int i = 0; i < m; i++) {
#pragma omp parallel
		for (int j = 0; j < n; j++) {
			R[i][j] = min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
		}
	}
	return R;
}

std::vector<float> evo::random_vec(int size, float max, float min) {
	std::vector<float> v(size);

#pragma omp parallel
	for (int i = 0; i < size; i++) {
		v[i] = min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));;
	}
	return v;
}

mtx evo::empty(int m, int n) {
	mtx M(m, std::vector<float>(n, 0));
	return M;
}

int evo::argmax(std::vector<float> v) {
	float mx = 0.0f;
	int index = 0;
	(v.size() == 0) ? mx = 0 : mx = v[0];

	for (unsigned int i = 1; i < v.size(); i++) {
		if (mx < v[i]) {
			mx = v[i];
			index = i;
		}
	}
	return index;
}