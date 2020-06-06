#pragma once
#include <vector>
#include <map>
#include "activation.h"
#include "Dense.h"
#include <string>

namespace evo {
	namespace nn {
		namespace rnn {
			class AbstractRNN {
			protected:
				std::map<std::string, std::vector<float>> vectors;
				int timestamp = 1, iterations, in_size, out_size;
				std::string out_activation;
				std::string hidden_activation;
				std::string fp;
				std::vector<float> weight_params, bias_params;
				bool bias_bool;
				evo::nn::activation act;
				float alpha;

				AbstractRNN(int input_size, int hidden_size, std::string output_activation = "softmax", std::string hidden_activation = "tanh",
					std::vector<float> weight_params = { -1, 1 }, std::vector<float> bias_params = { -1, 1 }, bool bias_bool = 1, int training_iterations = 3,
					float alpha=0.1, std::string fp="");

				virtual void initialize_weights() = 0;

				virtual std::vector<float> feed_forward_one_vect(std::vector<float> x) = 0;

				std::vector<float> feedforward(mtx x);

				virtual void train(std::vector<float> error_vect) =  0;

				std::vector<float> get_output_abs() {
					std::string key = "h" + std::to_string(timestamp - 1);
					return vectors[key];
				}

				void reset_timestamp() { timestamp = 1; }
			};
		}
	}
}
