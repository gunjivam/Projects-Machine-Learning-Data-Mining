#pragma once
#include <map>
#include <vector>
#include "math.h"
#include <cmath>
#include <string>
#include <iostream>

namespace evo {

	namespace nn {
		class activation{

			float alpha;

			std::map<std::string, std::vector<float>> errors;
			
			std::map<std::string, int> s2int;

			int str2int(std::string s) {
				return s2int[s];
			}

		public:
			activation(float alpha = 0.01);

			std::vector<float> linear(std::vector<float> v, std::string vect_name);
				
			std::vector<float> sigmoid(std::vector<float> v, std::string vect_name);

			std::vector<float> tanh(std::vector<float> v, std::string vect_name);

			std::vector<float> relu(std::vector<float> v, std::string vect_name);

			std::vector<float> leaky_relu(std::vector<float> v, std::string vect_name);

			std::vector<float> softplus(std::vector<float> v, std::string vect_name);

			std::vector<float> softmax(std::vector<float> v, std::string vect_name);

			inline std::vector<float> get_error(std::string vect_name) { return errors[vect_name]; }

			inline float get_error(std::string vect_name, int j) { return errors[vect_name][j]; }

			std::vector<float> activate(std::vector<float> v, std::string act_function, std::string vect_name = "a");


		};
	}
}