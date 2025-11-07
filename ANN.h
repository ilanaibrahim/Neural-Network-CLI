
#ifndef ANN_H
#define ANN_H

#include <vector>
#include "Matrix.h"
#include <ctime>

using namespace std;

namespace nn_utils {
	Matrix relu(const Matrix& m); 
	Matrix drelu(const Matrix& m); 
	Matrix softmax(const Matrix& logits);
	double cross_entropy(const Matrix& predictions, const Matrix& targets);
	int argmax(const Matrix& m); 
}

class Layer {
private:
	Matrix weights;
	Matrix bias;
    Matrix weight_gradient;
    Matrix bias_gradient;
	Matrix input;
	Matrix z;
	Matrix(*activation_func)(const Matrix&);
	Matrix(*activation_derivative)(const Matrix&);
public:
	Layer();
    Layer(int input_size, int output_size,Matrix (*act_func)(const Matrix&), Matrix (*act_deriv)(const Matrix&));

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& output_gradient);

    const Matrix& getWeights() const { return weights; }
    const Matrix& getBias() const { return bias; }
    const Matrix& getWeightGradients() const{return weight_gradient;}
    const Matrix& getBiasGradients() const{return bias_gradient;}
    const Matrix& getInput() const { return input; }
    const Matrix& getZ() const { return z; }

    void setWeights(const Matrix& new_weights) { weights = new_weights; }
    void setBias(const Matrix& new_bias) { bias = new_bias; }
    void setWeightGradients(const Matrix& new_grad) { weight_gradient = new_grad; }
    void setBiasGradients(const Matrix& new_grad) { bias_gradient = new_grad; }
};

class AdamOptimizer{
private:
	Matrix m;
	Matrix v;
	double learning_rate;
	double beta_1;
	double beta_2;
	double epsilon;
	int time;
public:
	 AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learning_rate(lr), beta_1(b1), beta_2(b2), epsilon(eps), time(0) {}
	void initialize(int rows, int cols);
	Matrix update(const Matrix& params, const Matrix& grads);

};

class ANN {
private:
	vector<Layer> layers;
	vector<AdamOptimizer> m_weight_optimizers;
	vector<AdamOptimizer> m_bias_optimizers;
	vector<Matrix> activations;  // Store forward pass activations for backpropagation

	void backward(const Matrix& target, const Matrix& predictions);
	void update_parameters();

public:
    ANN(vector<int> layer_sizes, Matrix (*output_activation)(const Matrix&), Matrix (*output_derivative)(const Matrix&));
	Matrix feedforward(const Matrix& input);
	double train_step(const Matrix& input, const Matrix& target);
};

#endif 
