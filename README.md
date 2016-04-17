lstm: comparing LSTM full gradients across multiple time steps among the following approaches:

1. theano gradient: theano.gradient.grad(E, w) (for calculating derivative of error E w.r.t. weight matrix w)
2. numerical gradient: (E(x+eps) - E(x-eps)) / (2 * eps)
3. Gradient given by Graves and Schmidhuber (2005): dE/dx * dx/dw = delta * y (which is incorrect, because dx/dw ≠ y, because w and y are not in a linear relation, because y depends on w through recurrence
4. New gradient calculation taking into consideration of the non-linearity between w and y

Currently only support gradients on the W matrices (weight for input x, three for the gates and one for the weighted input for the cell). Cmdline arg 0 -- 3 corresponding to the w in the input gate, the forget gate, the output gate, and the weighted input for the cell, respectively.

Dependencies: numpy and theano.