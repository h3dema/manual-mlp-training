#include "mlp.h"

#include <random>
#include <cmath>
#include <iostream>

double MLP::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double MLP::reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

MLP::MLP(
    int input_size,
    const std::vector<int>& hidden_sizes,
    int output_size,
    double learning_rate)
    : learning_rate_(learning_rate)
{
    std::vector<int> layers;
    layers.push_back(input_size);

    for (auto h : hidden_sizes)
        layers.push_back(h);

    layers.push_back(output_size);

    // create initial matrices sampling values from normal distribution
    // TODO: implement He
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    for (size_t l = 0; l < layers.size() - 1; l++) {

        int in = layers[l];
        int out = layers[l + 1];

        std::vector<std::vector<double>> W(
            in,
            std::vector<double>(out));

        std::vector<double> b(out, 0.0);

        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                W[i][j] = dist(gen);
            }
        }

        weights_.push_back(W);
        biases_.push_back(b);
    }
}

std::vector<std::vector<double>>
MLP::transpose(const std::vector<std::vector<double>>& A)
{
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> T(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            T[j][i] = A[i][j];

    return T;
}

std::vector<std::vector<double>>
MLP::matmul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B)
{
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    std::vector<std::vector<double>> C(m, std::vector<double>(p, 0.0));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < p; j++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

std::vector<std::vector<double>>
MLP::addBias(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b)
{
    auto R = A;

    for (size_t i = 0; i < R.size(); i++)
        for (size_t j = 0; j < b.size(); j++)
            R[i][j] += b[j];

    return R;
}

std::vector<std::vector<double>>
MLP::forward(const std::vector<std::vector<double>>& X)
{
    activations_.clear();
    pre_activations_.clear();
    activations_.push_back(X);
    std::vector<std::vector<double>> current = X;
    for (size_t l = 0; l < weights_.size(); l++) {
        auto Z = addBias(matmul(current, weights_[l]), biases_[l]);
        pre_activations_.push_back(Z);
        bool last_layer = (l == weights_.size() - 1);
        auto A = Z;
        if (!last_layer) {
            for (auto& row : A)
                for (auto& v : row)
                    v = relu(v);
        }
        activations_.push_back(A);  // save activations for backpropagation
        current = A;
    }

    return current;
}

double MLP::computeMSE(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred)
{
    double loss = 0.0;
    int B = y_true.size();
    int M = y_true[0].size();
    for (int i = 0; i < B; i++)
        for (int j = 0; j < M; j++) {
            double e = y_pred[i][j] - y_true[i][j];
            loss += e * e;
        }

    return loss / (B * M);
}

void MLP::backward(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred)
{
    const int B = X.size();

    // output delta
    std::vector<std::vector<double>> delta = y_pred;
    for (size_t i = 0; i < delta.size(); i++) {
        for (size_t j = 0; j < delta[0].size(); j++) {
            delta[i][j] = (2.0 / static_cast<double>(B)) * (y_pred[i][j] - y_true[i][j]);
        }
    }

    // store gradients for all layers
    std::vector<std::vector<std::vector<double>>> gradWs(weights_.size());
    std::vector<std::vector<double>> gradBs(weights_.size());
    for (int l = static_cast<int>(weights_.size()) - 1; l >= 0; --l) {
        const auto& Aprev = activations_[l];  // this was saved in the forward propagation
        // dW = A_prev^T * delta
        auto gradW = matmul(transpose(Aprev), delta);

        std::vector<double> gradB(biases_[l].size(), 0.0);
        for (size_t i = 0; i < delta.size(); i++)
        {
            for (size_t j = 0; j < gradB.size(); j++)
            {
                gradB[j] += delta[i][j];
            }
        }

        gradWs[l] = gradW;
        gradBs[l] = gradB;
        // compute delta for previous layer BEFORE updating weights
        if (l > 0) {
            auto WT = transpose(weights_[l]);
            auto delta_prev = matmul(delta, WT);
            for (size_t i = 0; i < delta_prev.size(); i++) {
                for (size_t j = 0; j < delta_prev[0].size(); j++) {
                    delta_prev[i][j] *= reluDerivative( pre_activations_[l - 1][i][j]);
                }
            }
            delta = std::move(delta_prev);
        }
    }
    
    // ----------------------------------------------------
    // Gradient descent update
    // with gradient clipping
    // ----------------------------------------------------
    constexpr double clip = 1.0;
    for (size_t l = 0; l < weights_.size(); l++) {
        for (size_t i = 0; i < gradWs[l].size(); i++) {
            for (size_t j = 0; j < gradWs[l][0].size(); j++) {
                double g = gradWs[l][i][j];

                if (clipping_) {
                    if (g > clip) g = clip;
                    if (g < -clip) g = -clip;
                }
                weights_[l][i][j] -= learning_rate_ * g;
            }
        }

        for (size_t j = 0; j < gradBs[l].size(); j++) {
            double g = gradBs[l][j];

            if (clipping_) {
                if (g > clip) g = clip;
                if (g < -clip) g = -clip;
            }
            biases_[l][j] -= learning_rate_ * g;
        }
    }
}
