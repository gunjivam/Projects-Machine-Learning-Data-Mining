#include "RNNCell.h"

evo::nn::rnn::RNNCell::RNNCell(int hidden_size, int training_iterations, std::string hidden_activation, std::vector<float> weight_params,
	std::vector<float> bias_params, bool bias_bool, std::string fp)
: AbstractRNN::AbstractRNN(0, hidden_size, "", hidden_activation, weight_params, bias_params, bias_bool, training_iterations, 
	alpha, fp), I(hidden_size, 0, "", weight_params, bias_params, 0), H(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool){

}
evo::nn::rnn::RNNCell::RNNCell(int hidden_size, int input_size, int training_iterations, std::string hidden_activation, std::vector<float> weight_params,
	std::vector<float> bias_params, bool bias_bool, std::string fp)
	: AbstractRNN::AbstractRNN(0, hidden_size, "", hidden_activation, weight_params, bias_params, bias_bool, training_iterations,
		alpha, fp), I(hidden_size, input_size, "", weight_params, bias_params, 0), H(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool) {
	if (fp == "") {
		initialize_weights();
	}
}

void evo::nn::rnn::RNNCell::initialize_weights() {
	I.initialize_layer();
	H.initialize_layer();
	dh_dW1 = evo::empty(in_size, out_size);
	dh_dWh = evo::empty(out_size, out_size);
	dh_db = evo::empty(out_size);
}

void evo::nn::rnn::RNNCell::train(std::vector<float> error_vect) {
	int t = 1;
	while (t < timestamp) {
		int max_timestamp = m_min(t + iterations, timestamp);
		while (t < max_timestamp) {
			for (int j = 0; j < out_size; j++) {
				std::vector<float> hp = vectors["h" + std::to_string(t - 1)];
				float h_prv = hp[j];
				float activation_error = act.get_error("h" + std::to_string(t), j);
				float error = error_vect[j];

				H.modify_bias(j, error * dB1(activation_error, H.get_weight(j, j), dh_db[j], j));

#pragma omp parallel
				for (int i = 0; i < in_size; i++) {
					float x = vectors["x" + std::to_string(t)][i];

					I.modify_weight(i, j, error * dW1(activation_error, x, H.get_weight(i, j), dh_dW1[i][j], i, j));
				}

#pragma omp parallel
				for (int j2 = 0; j2 < out_size; j2++) {

					H.modify_weight(j2, j, error * dWh(activation_error, h_prv, H.get_weight(j2, j), dh_dWh[j2][j], j2, j));
				}
			}
			t += 1;
		}
		reset();
	}
}

float evo::nn::rnn::RNNCell::dW1(float ae, float x, float wh, float dh_prv, int i, int j) {
	float er = (x + wh * dh_prv) * ae;
	dh_dW1[i][j] = er;
	return er;
}

float evo::nn::rnn::RNNCell::dWh(float ae, float h_prv, float wh, float dh_prv, int i, int j) {
	float er = ae * (h_prv + wh * dh_prv);
	dh_dWh[i][j] = er;
	return er;
}

float evo::nn::rnn::RNNCell::dB1(float ae, float wh, float dh_prv, int j) {
	float er = ae * (wh * dh_prv + 1);
	dh_db[j] = er;
	return er;
}


void evo::nn::rnn::RNNCell::reset() {
	dh_dW1 = evo::empty(in_size, out_size);
	dh_dWh = evo::empty(out_size, out_size);
	dh_db = evo::empty(out_size);
}

std::vector<float> evo::nn::rnn::RNNCell::feed_forward_one_vect(std::vector<float> x) {
	std::string key = "h" + std::to_string(timestamp-1);
	std::vector<float> h1 = I.feedforward(x);
	std::vector<float> h2 = H.feedforward(vectors[key]);
	std::vector<float> h = evo::add(h1, h2);
	std::vector<float> y = act.activate(h, hidden_activation, ("h" + std::to_string(timestamp)));
	vectors["x" + std::to_string(timestamp)] = x;
	vectors["h" + std::to_string(timestamp)] = y;
	timestamp += 1;
	return y;
}


std::vector<float> evo::nn::rnn::RNNCell::feedforward(mtx v) {
	return AbstractRNN::feedforward(v);
}

template<typename T>
void evo::nn::rnn::RNNCell::print_vector(std::vector<T> a) {
	std::cout << "{ ";
	for (T e : a) {
		std::cout << e << " ";
	}
	std::cout << " }" << std::endl;
}