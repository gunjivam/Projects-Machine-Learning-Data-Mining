#include "Dense.h"


evo::nn:: Dense::Dense(int unit_size, std::string activation, std::vector<float> weight_params, std::vector<float> bias_params,
	bool bias_bool, bool store_vectors, std::string fp, float alpha)
	:out_size(unit_size), in_size(0), activation(activation), weight_params(weight_params), bias_params(bias_params), bias_bool(bias_bool),
	store_vectors(store_vectors), timestamp(1), fp(fp)
{
	act = evo::nn::activation(alpha);
	if (fp == "") {
		initialize_layer();
	}
}

evo::nn::Dense::Dense(int unit_size, int input_size, std::string activation, std::vector<float> weight_params, std::vector<float> bias_params,
	bool bias_bool, bool store_vectors, std::string fp, float alpha)
	:out_size(unit_size), in_size(input_size), activation(activation), weight_params(weight_params), bias_params(bias_params), bias_bool(bias_bool),
	store_vectors(store_vectors), timestamp(1), fp(fp)
{
	act = evo::nn::activation(alpha);
	if (fp == "") {
		initialize_layer();
	}
}

std::vector<float> evo::nn:: Dense::feedforward(std::vector<float> x) {
	vectors.insert(std::pair<std::string, std::vector<float>> ("x" + std::to_string(timestamp), x));
	std::vector<float> h = evo::matmul(x, weight);
	vectors.insert(std::pair<std::string, std::vector<float>>("h" + std::to_string(timestamp), h));
	std::vector<float> y = act.activate(h, activation, "a"+std::to_string(timestamp));
	vectors.insert(std::pair<std::string, std::vector<float>>("y" + std::to_string(timestamp), y));
	timestamp++;
	return y;
}

void evo::nn::Dense::train(std::vector<float> loss_vector, float training_rate) {
	for (int t = timestamp - 1; t > 0; t--) {

		for (int j = 0; j < out_size; j++) {
			float l = loss_vector[j];

			for (int i = 0; i < in_size; i++) {
				gradient(l, i, j, t, training_rate);
			}
		}
	}
}

void evo::nn::Dense::gradient(float error, int i, int j, int timestamp, float training_rate) {
	float dy_dh = act.get_error("a" + std::to_string(timestamp))[j];
	std::string vc = "x" + std::to_string(timestamp);
	std::vector<float> x = vectors[vc];
	float dh_w = 0;
	dh_w = x[i];
	float dh_dx = weight[i][j];
	float cost_w = error * dy_dh * training_rate, cost_b = error* dy_dh* training_rate;
	cost_w *= dh_w;
	weight[i][j] += cost_w;
	bias[j] += cost_b;
}

void evo::nn::Dense::initialize_layer() {
	weight = evo::random_mtx(in_size, out_size, weight_params[0], weight_params[1]);
	bias = evo::random_vec(out_size, weight_params[0], weight_params[1]);
}

template<typename T>
void evo::nn::Dense::print_vector(std::vector<T> a) {
	std::cout << "{ ";
	for (T e : a) {
		std::cout << e << " ";
	}
	std::cout << " }" << std::endl;
}
