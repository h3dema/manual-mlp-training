# MLP

Consider an MLP with:

- Input: $x \in \mathbb{R}^{d_0}$
- $N$ hidden layers
- ReLU activation in every hidden layer
- Output: $\hat{y}$
- Layer $l$ has weights $W^{(l)}$ and biases $b^{(l)}$

Let the output layer be layer N+1N+1.

### Forward propagation

Define:

$a^{(0)} = x$

For hidden layers $l=1,\ldots,N$:

- $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$
- $a^{(l)} = \operatorname{ReLU}(z^{(l)})$

where

- $\operatorname{ReLU}(z)=\max(0,z)$

For the output layer: $z^{(N+1)} = W^{(N+1)}a^{(N)} + b^{(N+1)}$

If the output is linear: $\hat y = a^{(N+1)} = z^{(N+1)}$

Thus the complete forward pass is

$x \rightarrow z^{(1)} \rightarrow a^{(1)} \rightarrow \cdots \rightarrow z^{(N)} \rightarrow a^{(N)} \rightarrow z^{(N+1)} \rightarrow \hat y$


### Loss

Assume mean squared error:

$L = \frac{1}{2} \|\hat y-y\|^2$

Then

$\frac{\partial L}{\partial \hat y} = \hat y-y$

### Backward propagation

Define

$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$

as the error signal of layer $l$.


### Output layer

Since the output is linear, $\delta^{(N+1)} = \frac{\partial L}{\partial \hat y} = \hat y-y$

#### Gradients:

$\frac{\partial L}{\partial W^{(N+1)}} = \delta^{(N+1)} (a^{(N)})^T$ $\frac{\partial L}{\partial b^{(N+1)}} = \delta^{(N+1)}$

#### ReLU derivative

For hidden layers,

$\operatorname{ReLU}'(z) = \begin{cases} 1 & z>0\\ 0 & z\le 0 \end{cases}$

or equivalently $\operatorname{ReLU}'(z) = \mathbf{1}_{z>0}$,
where $\mathbf{1}$ is the indicator function.

#### Hidden layers

For $l=N,N-1,\ldots,1$: $\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \operatorname{ReLU}'(z^{(l)})$

where $\odot$ denotes element-wise multiplication.

The parameter gradients are:

$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$ $\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$

## Compact vector form

### Forward:

$a^{(0)} = x$

$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$

$a^{(l)} = \begin{cases} \operatorname{ReLU}(z^{(l)}), & l=1,\ldots,N\\ z^{(N+1)}, & l=N+1 \end{cases}$

### Backward:

$\delta^{(N+1)} = \hat y-y$

$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \mathbf{1}_{z^{(l)}>0}$

$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$

$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$

### Gradient descent update

With learning rate $\eta$,

- $W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$
- $b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$

for all $l=1,\ldots,N+1$.
