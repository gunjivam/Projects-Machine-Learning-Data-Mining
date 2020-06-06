#include "AbstractRNN.h"

evo::nn::rnn::AbstractRNN::AbstractRNN(int input_size, int hidden_size, std::string output_activation, std::string hidden_activation,
	std::vector<float> weight_params, std::vector<float> bias_params, bool bias_bool, int training_iterations, float alpha, std::string fp)
	: in_size(input_size), out_size(hidden_size), out_activation(output_activation), hidden_activation(hidden_activation),
	weight_params(weight_params), bias_params(bias_params), bias_bool(bias_bool), alpha(alpha), iterations(training_iterations), fp(fp) {
	vectors["h0"] = evo::empty(hidden_size);
	act = evo::nn::activation(alpha);
}

std::vector<float> evo::nn::rnn::AbstractRNN::feedforward(mtx x) {
	reset_timestamp();
	for (std::vector<float> v : x) {
		feed_forward_one_vect(v);
	}
	return get_output_abs();
}