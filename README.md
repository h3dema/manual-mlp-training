# MLP backpropagation

This project is a simple implementation of a Multi-Layer Perceptron (MLP) in C++.
The MLP is a type of neural network that can be used for tasks such as classification and regression.
The project includes a [mlp.h header file](main/mlp.h) that defines the MLP class and its methods.
`main.cpp` shows how to use the MLP class to read data from a CSV file and train and test the MLP on that data.


## Basic file structure

```
repo/
├── CMakeLists.txt
├── README.md
├── build/
│   └── regression.csv
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── mlp.h
│   └── mlp.cpp
└── notepad/
    └── regression.ipynb
```


> The python notebook `regression.ipynb` is used to train a scikit-learn MLPRegressor on the same data and compare the results with the C++ implementation.


## Compiling & Running

```bash
mkdir build
cd build

cmake ..
make

./main/mlp_example
```

> Notice that:
> ------------
> - you the data file `regression.csv` has to in `build` folder for the command above to work.
> - you need to have `gnuplot` installed to plot the results.
> - there are several hyperparameters that you can change in `main.cpp` to see how they affect the training of the MLP. You have to build the project again after changing the hyperparameters.
