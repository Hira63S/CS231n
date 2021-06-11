from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    use_batchnorm as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer use_batchnorm and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        use_batchnorm=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - use_batchnorm: What type of use_batchnorm the network should use. Valid values
            are "batchnorm", "layernorm", or None for no use_batchnorm (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.cache = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch use_batchnorm, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for i in range(self.num_layers):
            # for the very first layer:
            if i ==0:
                self.params['W' + str(i+1)] = weight_scale * np.random.randn(input_dim, hidden_dims[i])
                self.params['b' + str(i+1)] = np.zeros(hidden_dims[i])
                if self.use_batchnorm:
                    self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])
            # for subsequent layers:
            elif i < self.num_layers - 1:
                self.params['W' + str(i+1)] = weight_scale * np.random.rand(hidden_dims[i-1], hidden_dims[i])
                self.params['b' + str(i+1)] = np.zeros(hidden_dims[i])

                if self.use_batchnorm:
                    self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])

            else:
                self.params['W' + str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
                self.params['b' + str(i+1)] = np.zeros(num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch use_batchnorm we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # use_batchnorm layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch use_batchnorm layer, self.bn_params[1] to the forward
        # pass of the second batch use_batchnorm layer, etc.
        self.bn_params = []
        if self.use_batchnorm == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.use_batchnorm == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.use_batchnorm == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None
        cache = {}
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch use_batchnorm, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch use_batchnorm layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch use_batchnorm #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # fully connected net:
        # first input layer:
        a = {'layer0': X}
        for i in range(self.num_layers):
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            l, l_prev = 'layer' + str(i+1), 'layer'+str(i)

            if mode=='train':
                bn_params={'mode': 'train'}
            else:
                bn_params = {'mode': 'test'}

            if i < self.num_layers-1:
                if self.use_batchnorm and self.use_dropout:
                    gamma, beta = self.params['gamma' + str(i+1)], self.aprams['beta' + str(i+1)]
                    a[l], self.cache[l] = affine_batchnorm_relu_dropout_forward(a[l_prev], W, b, gamma, beta, bn_params, self.dropout_param)
                elif self.use_dropout:
                    a[l], self.cache[l] = affine_relue_dropout_forward(a[l_prev], W, b, self.dropout_param)
                elif self.use_batchnorm:
                    gamma, beta = self.params['gamma' + str(i+1)], self.params['beta'+str(i+1)]
                    a[l], self.cache[l] = affine_batchnorm_relu_forward(a[l_prev], W, b, gamma, beta, bn_params)
                else:
                    a[l], self.cache[l] = affine_relu_forward(a[l_prev], W, b)

            else:
                a[l], self.cache[l] = affine_forward(a[l_prev], W, b)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        scores = a['layer'+str(self.num_layers)]
        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer use_batchnorm, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        lastlayer = self.num_layers
        d = {}
        loss, d_out = softmax_loss(scores, y)

        grads = {}

        # backprop through the last layer:
        w = 'W' + str(lastlayer)
        b = 'b' + str(lastlayer)
        c = 'layer' + str(lastlayer)

        dh, grads[w], grads[b] = affine_backward(d_out, self.cache[c])
        loss += 0.5 * self.reg * np.sum(self.params[w]**2)
        grads[w] += self.reg * self.params[w]

        for i in reversed(range(lastlayer -1)):
            w = 'W' + str(i+1)
            b = 'b' + str(i+1)
            gamma = 'gamma' + str(i+1)
            beta = 'beta' + str(i+1)
            c = 'layer' + str(i+1)

            if self.use_batchnorm and self.use_dropout:
                dh, grads[w], grads[b], grads[gamma], grads[beta] = affine_batchnorm_relu_dropout_backward(dh, self.cache[c])
            elif self.use_dropout:
                dh, grads[w], grads[b] = affine_relu_dropout_backward(dh, self.cache[c])
            elif self.use_batchnorm:
                dh, grads[w], grads[b], grads[gamma], grads[beta] = affine_batchnorm_relu_backward(dh, self.cahce[c])
            else:
                dh, grads[w], grads[b] = affine_relu_backward(dh,self.cache[c])

            loss += 0.5 * self.reg * np.sum(self.params[w]**2)
            grads[w] += self.reg * self.params[w]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
