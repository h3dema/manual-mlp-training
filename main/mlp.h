#pragma once

#include <vector>

class MLP {
public:
    MLP(
        int input_size,
        const std::vector<int>& hidden_sizes,
        int output_size,
        double learning_rate);  // needs to be here because MLP does its own backpropagation

    std::vector<std::vector<double>>
    forward(const std::vector<std::vector<double>>& X);

    double computeMSE(
        const std::vector<std::vector<double>>& y_true,
        const std::vector<std::vector<double>>& y_pred);

    void backward(
        const std::vector<std::vector<double>>& X,
        const std::vector<std::vector<double>>& y_true,
        const std::vector<std::vector<double>>& y_pred
    );

private:
    double learning_rate_;

    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;

    std::vector<std::vector<std::vector<double>>> activations_;
    std::vector<std::vector<std::vector<double>>> pre_activations_;

    bool clipping_ = false;

    static double relu(double x);
    static double reluDerivative(double x);

    static std::vector<std::vector<double>> matmul(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B);

    static std::vector<std::vector<double>> transpose(
        const std::vector<std::vector<double>>& A);

    static std::vector<std::vector<double>> addBias(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b);
};
