#pragma once
#include "math.h"
#include "activation.h"

namespace evo{
	namespace nn {
		class Dense {
			mtx weight;
			std::vector<float> bias;
			int out_size, in_size, timestamp;
			std::string activation; std::string fp;
			std::vector<float> weight_params, bias_params;
			bool bias_bool, store_vectors;
			std::map<std::string, std::vector<float>> vectors;
			evo::nn::activation act;

			void gradient(float error, int i, int j, int timestamp, float training_rate = 0.6);


		public:
			Dense(int unit_size, std::string activation = "softmax", std::vector<float> weight_params = { -1, 1 },
				std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, bool store_vectors = 1, std::string fp = "", float alpha = 0.1);

			Dense(int unit_size, int input_size, std::string activation = "softmax", std::vector<float> weight_params = { -1, 1 },
				std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, bool store_vectors = 1, std::string fp = "", float alpha = 0.1);

			std::vector<float> feedforward(std::vector<float> x);

			void train(std::vector<float> loss_vector, float training_rate = 0.5);

			inline void modify_weight(int i, int j, float val) { weight[i][j] += val; }

			inline void reset_timestamp() { timestamp = 1; }

			inline void modify_bias(int j, float val) { bias[j] += val; }

			inline int get_inputSize() const { return in_size; }

			inline int get_outputSize() const { return out_size; }

			inline float get_weight(int i, int j) const { return weight[i][j]; }

			inline std::string get_activation() const { return activation; }

			inline std::vector<float> get_vector(std::string name, int timestamp = 0) { return vectors[name+std::to_string(timestamp)]; }

			void initialize_layer();

			template<typename T>
			void print_vector(std::vector<T> a);
		};
	}
}