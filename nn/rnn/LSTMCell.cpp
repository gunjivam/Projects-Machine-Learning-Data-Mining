#include "LSTMCell.h"


evo::nn::rnn::LSTMCell::LSTMCell(int hidden_size, int training_iterations, std::string hidden_activation, std::string gate_activation,
	std::vector<float> weight_params, std::vector<float> bias_params, bool bias_bool, std::string fp) 
	: AbstractRNN::AbstractRNN(0, hidden_size, gate_activation, hidden_activation, weight_params, bias_params, bias_bool, training_iterations, alpha, fp),
	IG(hidden_size, 0, "", weight_params, bias_params, 0), 
	IH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool),
	FG(hidden_size, 0, "", weight_params, bias_params, 0),
	FH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool) ,
	GG(hidden_size, 0, "", weight_params, bias_params, 0),
	GH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool) ,
	OG(hidden_size, 0, "", weight_params, bias_params, 0),
	OH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool) {
	
	vectors["s0"] = evo::empty(hidden_size);

}

evo::nn::rnn::LSTMCell::LSTMCell(int hidden_size, int input_size, int training_iterations, std::string hidden_activation, std::string gate_activation, std::vector<float> weight_params,
	std::vector<float> bias_params, bool bias_bool, std::string fp)
	: AbstractRNN::AbstractRNN(input_size, hidden_size, gate_activation, hidden_activation, weight_params, bias_params, bias_bool, training_iterations, alpha, fp),
	IG(hidden_size, input_size, "", weight_params, bias_params, 0),
	IH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool),
	FG(hidden_size, input_size, "", weight_params, bias_params, 0),
	FH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool),
	GG(hidden_size, input_size, "", weight_params, bias_params, 0),
	GH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool),
	OG(hidden_size, input_size, "", weight_params, bias_params, 0),
	OH(hidden_size, hidden_size, "", weight_params, bias_params, bias_bool) {
	
	vectors["s0"] = evo::empty(hidden_size);
	reset();
	
	if (fp == "") {
		initialize_weights();
	}
}

void evo::nn::rnn::LSTMCell::initialize_weights() {
	IG.initialize_layer(); GG.initialize_layer(); FG.initialize_layer(); GG.initialize_layer(); 
	IH.initialize_layer(); FH.initialize_layer(); GH.initialize_layer(); OH.initialize_layer();
}

void evo::nn::rnn::LSTMCell::train(std::vector<float> error_vect) {
	int t = 1;
	while (t < timestamp) {
		int max_timestamp = m_min(t + iterations, timestamp);
		while (t < max_timestamp) {
			gradient_h(error_vect, t);
			t++;
		}

		reset();
	}
}


void evo::nn::rnn::LSTMCell::gradient_h(std::vector<float> error_vect, int tm) {
	float e, dh, dhO, dci, dcg, dcf, dih, dic, dfh, dfc, dgh, dgc;
	std::vector<float> dhc(2);
#pragma omp parallel
	for (int j = 0; j < out_size; j++) {
		e = error_vect[j];
		dh = dhI(j, tm); dhO = dO(j, tm);
		dci = dcI(j, tm); dcg = dcG(j, tm); dcf = dcF(j, tm);
		/*std::cout << dh << " " << dhO << " " << dci << " " << dcg << " " << dcf << std::endl;
		std::cout << "----------------" << std::endl;*/
		if (bias_bool) {
			dhc = dI_b(dh, dci, j, tm);
			dih = dhc[0], dic = dhc[0];
			dhc = dF_b(dh, dci, j, tm);
			dfh = dhc[0], dfc = dhc[0];
			dhc = dG_b(dh, dci, j, tm);
			dgh = dhc[0], dgc = dhc[0];
			dBi(e, dih, dic, j);
			dBf(e, dfh, dfc, j);
			dBg(e, dgh, dgc, j);
			dBo(e, dhO, j);
		}

		for (int i = 0; i < in_size; i++) {
			dhc = dI_w(dh, dci, i, j, tm);
			dih = dhc[0], dic = dhc[1];
			dhc = dF_w(dh, dcf, i, j, tm);
			dfh = dhc[0], dfc = dhc[1];
			dhc = dG_w(dh, dcg, i, j, tm);
			dgh = dhc[0], dgc = dhc[1];
			dWi(e, dih, dic, i, j, tm);
			dWg(e, dgh, dgc, i, j, tm);
			dWf(e, dfh, dfc, i, j, tm);
			dWo(e, dhO, i, j, tm);
			}

		for (int i = 0; i < out_size; i++) {
			dhc = dI_u(dh, dci, i, j, tm);
			dih = dhc[0], dic = dhc[1];
			dhc = dF_u(dh, dcf, i, j, tm);
			dfh = dhc[0], dfc = dhc[1];
			dhc = dG_u(dh, dcg, i, j, tm);
			dgh = dhc[0], dgc = dhc[1];
			dUi(e, dih, dic, i, j, tm);
			dUg(e, dgh, dgc, i, j, tm);
			dUf(e, dfh, dfc, i, j, tm);
			dUo(e, dhO, i, j, tm);
		}
	}
}

void evo::nn::rnn::LSTMCell::reset() {
	dhp_dUf = evo::empty(out_size, out_size);
	dhp_dUg = evo::empty(out_size, out_size);
	dhp_dUi = evo::empty(out_size, out_size);
	dhp_dUo = evo::empty(out_size, out_size);
	dhp_dWf = evo::empty(in_size, out_size);
	dhp_dWg = evo::empty(in_size, out_size);
	dhp_dWi = evo::empty(in_size, out_size);
	dhp_dWo = evo::empty(in_size, out_size);
	dhp_dBi = evo::empty(out_size);
	dhp_dBo = evo::empty(out_size);
	dhp_dBg = evo::empty(out_size);
	dhp_dBf = evo::empty(out_size);

	dc_dWi = evo::empty(in_size, out_size);
	dc_dWg = evo::empty(in_size, out_size);
	dc_dWf = evo::empty(in_size, out_size);
	dc_dUi = evo::empty(out_size, out_size);
	dc_dUf = evo::empty(out_size, out_size);
	dc_dUg = evo::empty(out_size, out_size);

	dc_dBf = evo::empty(out_size);
	dc_dBi = evo::empty(out_size);
	dc_dBg = evo::empty(out_size);
}

std::vector<float> evo::nn::rnn::LSTMCell::feedforward(mtx vects) {
	return AbstractRNN::feedforward(vects);
}


std::vector<float> evo::nn::rnn::LSTMCell::feed_forward_one_vect(std::vector<float> x) {
	std::vector<float> h_prev = vectors["h" + std::to_string(timestamp - 1)];
	std::vector<float> s_prev = vectors["s" + std::to_string(timestamp - 1)];
	std::vector<float> g = act.activate(evo::add(GG.feedforward(x), GH.feedforward(h_prev)), hidden_activation, "g" + std::to_string(timestamp));
	std::vector<float> i = act.activate(evo::add(IG.feedforward(x), IH.feedforward(h_prev)), out_activation, "i" + std::to_string(timestamp));
	std::vector<float> f = act.activate(evo::add(FG.feedforward(x), FH.feedforward(h_prev)), out_activation, "f" + std::to_string(timestamp));
	std::vector<float> o = act.activate(evo::add(OG.feedforward(x), OH.feedforward(h_prev)), out_activation, "o" + std::to_string(timestamp));
	std::vector<float> s = evo::add(evo::multiply(g, i), evo::multiply(s_prev, f));
	std::vector<float> h = evo::multiply(act.activate(s, hidden_activation, "h" + std::to_string(timestamp)), o);
	/*print_vector(h_prev);
	print_vector(s_prev);
	print_vector(g);
	print_vector(i);
	print_vector(f);
	print_vector(o);
	print_vector(s);
	print_vector(h);
	std::cout << "-------------" << std::endl;*/
	vectors["s" + std::to_string(timestamp)] = s;
	vectors["h" + std::to_string(timestamp)] = h;
	vectors["g" + std::to_string(timestamp)] = g;
	vectors["i" + std::to_string(timestamp)] = i;
	vectors["f" + std::to_string(timestamp)] = f;
	vectors["o" + std::to_string(timestamp)] = o;
	vectors["x" + std::to_string(timestamp)] = x;
	timestamp += 1;
	return h;
}


