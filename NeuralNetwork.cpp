#include "ANN.h"
#include "Matrix.h"
#include <iostream>
#include <cmath>      
#include <vector>
#include <algorithm>

using namespace std;

Matrix nn_utils::relu(const Matrix& m){
	vector<vector<double>> result_data(m.getRows(), vector<double>(m.getCol()));
	for (int i = 0; i < m.getRows(); ++i){
		for (int j = 0; j < m.getCol(); ++j){
			result_data[i][j] = max(0.00,(m.data[i][j]));
		}
	}
	return Matrix(result_data);
}

Matrix nn_utils::drelu(const Matrix& m){
	vector<vector<double>> result_data(m.getRows(), vector<double>(m.getCol()));
	for (int i = 0; i < m.getRows(); ++i){
		for (int j = 0; j < m.getCol(); ++j){
			if (m.data[i][j] > 0.0){
				result_data[i][j] = 1.0; 
			}else{
				result_data[i][j] = 0.0;
			}
		}
	}
	return Matrix(result_data);
}

Matrix nn_utils::softmax(const Matrix& logits) {
    double max_val = -999999;
    for (int i = 0; i < logits.getRows(); i++) {
        if (logits.data[i][0] > max_val) {
            max_val = logits.data[i][0];
        }
    }
    Matrix stable_logits = logits.scalar_subtraction(max_val);
    Matrix exp_logits = stable_logits.exp_multiply(); 
    double sum = 0.0;
    for (int i = 0; i < exp_logits.getRows(); i++) {
        sum += exp_logits.data[i][0];
    }
    Matrix softmax_output(exp_logits.getRows(), 1);
    for (int i = 0; i < softmax_output.getRows(); i++) {
        softmax_output.data[i][0] = exp_logits.data[i][0] / sum;
    }

    return softmax_output;
}

double nn_utils::cross_entropy(const Matrix& predictions, const Matrix& targets) {
    double loss = 0.0;
    double epsilon = 1e-9; 
    for (int i = 0; i < targets.getRows(); i++) {
        if (targets.data[i][0] == 1.0) {
            loss += -log(predictions.data[i][0] + epsilon);
        }
    }
    return loss;
}

int nn_utils::argmax(const Matrix& m){
	if (m.getCol() != 1){cout<< "argmax only supports 1-column matrices!";}
	double max_val = -99999;
	int max_idx = 0;
	for (int i = 0; i < m.getRows(); i++){
		if (m.data[i][0] > max_val){
			max_val = m.data[i][0];
			max_idx = i;
		}
	}
	return max_idx;
}

Layer::Layer() {}

Layer::Layer(int input_size, int output_size, Matrix(*act_func)(const Matrix&), Matrix(*act_deriv)(const Matrix&)){ 
	weights = Matrix(output_size, input_size, true);
	bias = Matrix(output_size, 1, true);
	weight_gradient = Matrix(output_size, input_size); 
	bias_gradient = Matrix(output_size, 1);      
	activation_func = act_func;
	activation_derivative = act_deriv;
}

Matrix Layer::forward(const Matrix& input){
	this->input = input;
	z = (weights * input) + bias;
	return activation_func(z);
}

Matrix Layer::backward(const Matrix& output_gradient){
	Matrix dZ = output_gradient.element_multiply(activation_derivative(z));
	weight_gradient = dZ * input.transpose();
	bias_gradient = dZ; 
	return weights.transpose() * dZ;
}

ANN::ANN(vector<int> layer_sizes, Matrix (*output_activation)(const Matrix&), Matrix (*output_derivative)(const Matrix&)){
	for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i+1];

        Matrix (*act_func)(const Matrix&);
        Matrix (*act_deriv)(const Matrix&);

        if (i == layer_sizes.size() - 2) { // This is the last layer
            act_func = output_activation;
            act_deriv = output_derivative;
        } else { // This is a hidden layer
            act_func = nn_utils::relu;
            act_deriv = nn_utils::drelu;
        }
        layers.emplace_back(input_size, output_size, act_func, act_deriv);

		AdamOptimizer weight_opt;
		weight_opt.initialize(output_size, input_size);
		m_weight_optimizers.push_back(weight_opt);

		AdamOptimizer bias_opt;
		bias_opt.initialize(output_size, 1);
        m_bias_optimizers.push_back(bias_opt);
	}
}

Matrix ANN::feedforward(const Matrix& input){
	Matrix current_activation = input;
	for (size_t i = 0; i < layers.size(); i++){
		current_activation = layers[i].forward(current_activation);
	}
	return current_activation;
}

double ANN::train_step(const Matrix& input, const Matrix& target){
	Matrix predictions = feedforward(input);
	double loss = nn_utils::cross_entropy(predictions, target);
	backward(target, predictions);
	update_parameters();
	return loss;
}

void ANN::backward(const Matrix& target, const Matrix& predictions) {
    Layer& last_layer = layers.back();
    Matrix gradient = predictions - target;
    Matrix last_input = last_layer.getInput(); 
    Matrix last_weights = last_layer.getWeights();
    Matrix weight_grad = gradient * last_input.transpose();
    Matrix bias_grad = gradient;
    last_layer.setWeightGradients(weight_grad);
    last_layer.setBiasGradients(bias_grad);
    gradient = last_weights.transpose() * gradient;
    for (int i = layers.size() - 2; i >= 0; --i) {
        gradient = layers[i].backward(gradient);
    }
}

void AdamOptimizer::initialize(int rows, int cols) {
    m = Matrix(rows, cols);
    v = Matrix(rows, cols);
}

Matrix AdamOptimizer::update(const Matrix& params, const Matrix& grads){
    time++;
    m = m * beta_1 + grads * (1.0 - beta_1);
    v = v * beta_2 + grads.element_multiply(grads) * (1.0 - beta_2);
    Matrix m_hat = m / (1.0 - std::pow(beta_1, time));
    Matrix v_hat = v / (1.0 - std::pow(beta_2, time));
    Matrix sqrt_v_hat = v_hat.element_sqrt();
    Matrix epsilon_matrix(v_hat.getRows(), v_hat.getCol());
    for (int i = 0; i < v_hat.getRows(); i++) {
        for (int j = 0; j < v_hat.getCol(); j++) {
            epsilon_matrix.data[i][j] = epsilon;
        }
    }
    Matrix denominator = sqrt_v_hat + epsilon_matrix;
    Matrix numerator = m_hat * learning_rate;
    Matrix update_term = numerator.element_divide(denominator);
    return params - update_term;
}

void ANN::update_parameters() {
    for (size_t i = 0; i < layers.size(); i++) {
        const Matrix& current_weights = layers[i].getWeights();
        const Matrix& weight_gradients = layers[i].getWeightGradients();
        Matrix new_weights = m_weight_optimizers[i].update(current_weights, weight_gradients);
        layers[i].setWeights(new_weights);
        const Matrix& current_biases = layers[i].getBias();
        const Matrix& bias_gradients = layers[i].getBiasGradients();
        Matrix new_biases = m_bias_optimizers[i].update(current_biases, bias_gradients);
        layers[i].setBias(new_biases);
    }
}
