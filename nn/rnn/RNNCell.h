#pragma once
#include "AbstractRNN.h"
#include <omp.h>

namespace evo {
	namespace nn {
		namespace rnn {
			class RNNCell : AbstractRNN {
			private:
				Dense I, H;
				mtx dh_dW1;
				mtx dh_dWh;
				std::vector<float> dh_db;

				inline int m_min(int a, int b) { return (a <= b) ? a : b; }

			public:
				RNNCell(int hidden_size, int training_iterations = 5, std::string hidden_activation = "softmax", std::vector<float> weight_params = { -1, 1 },
					std::vector<float> bias_params = { -1, 1 }, bool bias_bool=1, std::string fp ="");
				RNNCell(int hidden_size, int input_size, int training_iterations = 5, std::string hidden_activation = "softmax", std::vector<float> weight_params = { -1, 1 },
					std::vector<float> bias_params = { -1, 1 }, bool bias_bool = 1, std::string fp = "");
				
				void initialize_weights();

				void train(std::vector<float> error_vect);

				float dW1(float ae, float x, float wh, float dh_prv, int i, int j);

				float dWh(float ae, float h_prv, float wh, float dh_prv, int i, int j);

				float dB1(float ae, float wh, float dh_prv, int j);

				void reset();

				std::vector<float> feedforward(mtx vects);

				inline std::vector<float> get_output() {
					return vectors["h" + std::to_string(timestamp - 1)];
				}

				std::vector<float> feed_forward_one_vect(std::vector<float> x);

				template<typename T>
				void print_vector(std::vector<T> a);
			};
		}
	}
}