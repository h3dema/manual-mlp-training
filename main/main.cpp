#include "mlp.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }
    // auto rows = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;

        // rows++;
        // std::cout << "Reading row " << rows << std::endl;
        // std::cout << "Line: " << line << std::endl;

        while (std::getline(ss, value, ',')) {
            size_t start = value.find_first_of("+-0123456789.");
            if (start != std::string::npos)
                value = value.substr(start);

            // std::cout << "Value: " << value << " | size " << value.size() << std::endl;
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }

    return data;
}


int main()
{
    // define training hyperparameters
    const int epochs = 10000;
    const float learning_rate = 0.1;

    // you need to define these parameters below manually
    // to reflect the status of the input file
    // it is a comma-separated csv file, where the first
    // `INPUT_DIM` columns belongs to the input X, and the
    // following `OUTPUT_DIM` columns are Y
    const auto data_fname = "regression.csv";
    const int INPUT_DIM = 3;
    const int OUTPUT_DIM = 1;

    std::cout << "Reading " << data_fname << std::endl;
    auto data = readCSV(data_fname);
    std::cout << "Read " << data.size() << " rows" << std::endl;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;

    for (const auto& row : data) {

        std::vector<double> x(row.begin(), row.begin() + INPUT_DIM);
        std::vector<double> y(row.begin() + INPUT_DIM, row.end());

        X.push_back(x);
        Y.push_back(y);
    }

    // (input size, hidden layer size, output size)
    MLP model(INPUT_DIM, {16,16}, OUTPUT_DIM, learning_rate);
    std::cout << "Training model with " << epochs << " epochs and learning rate " << learning_rate << std::endl;

    for (int e = 0; e < epochs; e++) {
        // std::cout << "Training epoch " << e << std::endl;
        auto y_pred = model.forward(X);
        model.backward(X, Y, y_pred);

        if (e % 100 == 0 || e == epochs - 1) {
            double loss = model.computeMSE(Y, y_pred);
            std::cout << "Epoch " << e << " Loss = " << loss << std::endl;
        }
    }

    // -----------------------------------------------
    // plot the results
    // only works if the dimension of the output is 1
    // * you need to have gnuplot installed
    // -----------------------------------------------
    if (OUTPUT_DIM != 1) {
        std::cerr << "Output dimension is not 1, cannot plot results" << std::endl;
        return 1;
    }
    auto y_pred = model.forward(X);
    FILE* g = popen("gnuplot -persist", "w");
    fprintf(g, "set xlabel 'Y'\nset ylabel 'y_{pred}'\n");
    fprintf(g, "plot '-' with points title 'predicted vs true values'\n");
    for (size_t i = 0; i < Y.size(); ++i)
        fprintf(g, "%f %f\n", Y[i][0], y_pred[i][0]);
    fprintf(g, "e\n");
    pclose(g);

    return 0;
}