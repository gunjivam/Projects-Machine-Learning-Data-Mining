#pragma once
#include "AbstractRNN.h"
#include <omp.h>
#include <math.h>


namespace evo {
	namespace nn {
		namespace rnn {
			class LSTMCell : AbstractRNN {
			private:
				Dense IG, IH, FG, FH, OG, OH, GG, GH;

				mtx dhp_dWi, dhp_dWf, dhp_dWg, dhp_dWo, dhp_dUi, dhp_dUf, dhp_dUo, dhp_dUg, dc_dWi, dc_dWg, dc_dWf
					, dc_dUi, dc_dUf, dc_dUg;
				std::vector<float> dhp_dBi, dhp_dBg, dhp_dBf, dhp_dBo, dc_dBi, dc_dBf, dc_dBg;

				inline int m_min(int a, int b) { return (a <= b) ? a : b; }

				void gradient_h(std::vector<float> error_vect, int tm);

			public:
				LSTMCell(int hidden_size, int training_iterations = 5, std::string hidden_activation = "softmax", std::string gate_activation="tanh",
					std::vector<float> weight_params = { 0, 1 }, std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, std::string fp = "");

				LSTMCell(int hidden_size, int input_size, int training_iterations = 5, std::string hidden_activation = "softmax", std::string gate_activation = "tanh", std::vector<float> weight_params = { 0, 1 },
					std::vector<float> bias_params = { 0, 1 }, bool bias_bool = 1, std::string fp = "");

				void initialize_weights();

				void train(std::vector<float> error_vect);


				void reset();


				std::vector<float> feedforward(mtx vects);

				inline std::vector<float> get_output() {
					return vectors["h" + std::to_string(timestamp - 1)];
				}

				std::vector<float> feed_forward_one_vect(std::vector<float> x);

				float dhI(int j, int timestamp) {
					return vectors["o" + std::to_string(timestamp)][j] * act.get_error("h" + std::to_string(timestamp), j);
				}
				
				float dcI(int j, int timestamp) {
					return vectors["g" + std::to_string(timestamp)][j] * act.get_error("i" + std::to_string(timestamp), j);
				}

				std::vector<float> dI_w(float dh_dc, float dc_di, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dWi[i][j] + dc_di;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dI_u(float dh_dc, float dc_di, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dUi[i][j] + dc_di;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dI_b(float dh_dc, float dc_di, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dBi[j] + dc_di;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				void dWi(float error, float dh, float dc, int i, int j, int timestamp) {
					float comp = (vectors["x" + std::to_string(timestamp)][i] + IH.get_weight(j, j) * dhp_dWi[i][j]);
					dc = dc*comp;
					dh = dh*comp;
					dc_dWi[i][j] = dc;
					dhp_dWi[i][j] = dh;
					IG.modify_weight(i, j, error * dh);
				}

				void dUi(float error, float dh, float dc, int j1, int j2, int timestamp) {
					float comp = (vectors["h" + std::to_string(timestamp)][j1] + IH.get_weight(j1, j2) * dhp_dUi[j1][j2]);
					dc = dc * comp;
					dh = dh * comp;
					dc_dUi[j1][j2] = dc;
					dhp_dUi[j1][j2] = dh;
					IH.modify_weight(j1, j2, error * dh);
				}

				void dBi(float error, float dh, float dc, int j) {
					float comp = (1 + IH.get_weight(j, j) * dhp_dBi[j]);
					dc = dc * comp;
					dh = dh * comp;
					dc_dBi[j] = dc;
					dhp_dBi[j] = dh;
					IG.modify_bias(j, error * dh);
				}

				float dcF(int j, int timestamp) {
					return vectors["s" + std::to_string(timestamp - 1)][j] * act.get_error("f" + std::to_string(timestamp), j);
				}

				std::vector<float> dF_w(float dh_dc, float dc_df, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dWf[i][j] + dc_df;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dF_u(float dh_dc, float dc_df, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dUf[i][j] + dc_df;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dF_b(float dh_dc, float dc_df, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dBf[j] + dc_df;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				void dWf(float error, float dh, float dc, int i, int j, int timestamp) {
					float comp = vectors["x" + std::to_string(timestamp)][i] + FH.get_weight(j, j) * dhp_dWf[i][j];
					dc *= comp;
					dh *= comp;
					dc_dWf[i][j] = dc;
					dhp_dWf[i][j] = dh;
					FG.modify_weight(i, j, error * dh);
				}

				void dUf(float error, float dh, float dc, int j1, int j2, int timestamp) {
					float comp = (vectors["h" + std::to_string(timestamp)][j1] + FH.get_weight(j1, j2) * dhp_dUf[j1][j2]);
					dc *= comp;
					dh *= comp;
					dc_dUf[j1][j2] = dc;
					dhp_dUf[j1][j2] = dh;
					FH.modify_weight(j1, j2, error * dh);
				}

				void dBf(float error, float dh, float dc, int j) {
					float comp = (1 + FH.get_weight(j, j) * dhp_dBf[j]);
					dc *= comp;
					dh *= comp;
					dc_dBf[j] = dc;
					dhp_dBf[j] = dh;
					FG.modify_bias(j, error * dh);
				}

				float dcG(int j, int timestamp) {
					return vectors["i" + std::to_string(timestamp)][j] * act.get_error("g" + std::to_string(timestamp), j);
				}

				std::vector<float> dG_w(float dh_dc, float dc_dg, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dWg[i][j] + dc_dg;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dG_u(float dh_dc, float dc_dg, int i, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dUg[i][j] + dc_dg;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				std::vector<float> dG_b(float dh_dc, float dc_dg, int j, int timestamp) {
					float dc = vectors["f" + std::to_string(timestamp)][j] * dc_dBg[j] + dc_dg;
					float dh = dh_dc * dc;
					return { dh, dc };
				}

				void  dWg(float error, float dh, float dc, int i, int j, int timestamp) {
					float comp = (vectors["x" + std::to_string(timestamp)][i] + GH.get_weight(j, j) * dhp_dWg[i][j]);
					dc *= comp;
					dh *= comp;
					dc_dWg[i][j] = dc;
					dhp_dWg[i][j] = dh;
					GG.modify_weight(i, j, error * dh);
				}

				void dUg(float error, float dh, float dc, int j1, int j2, int timestamp) {
					float comp = (vectors["h" + std::to_string(timestamp)][j1] + GH.get_weight(j1, j2) * dhp_dUg[j1][j2]);
					dc *= comp;
					dh *= comp;
					dc_dUg[j1][j2] = dc;
					dhp_dUg[j1][j2] = dh;
					GH.modify_weight(j1, j2, error * dh);
				}

				void dBg(float error, float dh, float dc, int j) {
					float comp = (1 + GH.get_weight(j, j) * dhp_dBg[j]);
					dc *= comp;
					dh *= comp;
					dc_dBg[j] = dc;
					dhp_dBg[j] = dh;
					GG.modify_bias(j, error * dh);
				}

				float dO(int j, int timestamp) {
					float dh = std::tanh(vectors["s" + std::to_string(timestamp)][j]) * (act.get_error("o" + std::to_string(timestamp))[j]);
					return dh;
				}

				void dWo(float error, float dh, int i, int j, int timestamp) {
					dh *= (vectors["x" + std::to_string(timestamp)][i] + OH.get_weight(j, j) * dhp_dWo[i][j]);
					dhp_dWo[i][j] = dh;
					OG.modify_weight(i, j, error * dh);
				}

				void dUo(float error, float dh, int j1, int j2, int timestamp) {
					dh *= (vectors["h" + std::to_string(timestamp)][j1] + OH.get_weight(j1, j2) * dhp_dUo[j1][j2]);
					dhp_dUo[j1][j2] = dh;
					OH.modify_weight(j1, j2, error * dh);
				}

				void dBo(float error, float dh, int j) {
					dh *= (1 + OH.get_weight(j, j) * dhp_dBo[j]);
					dhp_dBo[j] = dh;
					OG.modify_bias(j, error * dh);
				}

				template<typename T>
				void print_vector(std::vector<T> a) {
					std::cout << "{ ";
					for (T e : a) {
						std::cout << e << " ";
					}
					std::cout << " }" << std::endl;
				}
			};
		}
	}
}