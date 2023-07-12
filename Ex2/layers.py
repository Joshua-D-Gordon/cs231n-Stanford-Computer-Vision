from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0] # mini batch size of N
    x_reshaped = x.reshape(N, -1)  # Reshape input data to (N, D) tried using flattern but this is simpler
    out = x_reshaped.dot(w) + b   # Perform matrix multiplication and add bias , each one of the mini batches is multipled by W matrix and bias is added

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0] # number of minibatches N
    D = w.shape[0] # mult of d1, d2, ... , d_k = D
    
    dx = dout.dot(w.T).reshape(x.shape) # Gradient of input with respect to loss and reshape to original size
    
    x_reshaped = x.reshape(N, D) # Reshape x to (N,D) where D is mult of d1, d2, ..., dk = D
    
    dw = x_reshaped.T.dot(dout) # Gradient of weights with respect to loss
    
    db = np.sum(dout, axis=0) # Gradient of bias with respect to loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0,x).reshape(x.shape) # relu function on all elements of x, if element < 0 then 0 else the same

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    dx = dout*(x>0) # if x>0 dx at that element is value of dout at that element
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    x_pad = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    #var's for everything
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    #vars for output dimenstion
    H_tag = int(1 + (H + 2 * pad - HH) / stride)
    W_tag = int(1 + (W + 2 * pad - WW) / stride)

    #output dim with zeros
    out = np.zeros((N, F, H_tag, W_tag))

    # Pad the input using np.pad
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # for data point in data points
    for n in range(N):
        # for filter filters
        for f in range(F):
            # for h_tag in hight of output canvas
            for h_t in range(H_tag):
                # Compute the start and end of the current height
                h_start = h_t * stride
                h_end = h_start + HH

                # for w_tag in width of output canvas
                for w_t in range(W_tag):
                    # Compute the start and end of the current width
                    w_start = w_t * stride
                    w_end = w_start + WW

                    # Extract the current field from the padded input
                    x_rf = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # Perform the convolution by element-wise multiplication and summation
                    out[n, f, h_t, w_t] = np.sum(x_rf * w[f]) + b[f]

    # Save the cache for the backward pass
    cache = (x_pad, w, b, conv_param)

    return out, cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #vars for everything
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_out = (H + 2 * pad - HH) // stride - 1
    W_out = (W + 2 * pad - WW) // stride - 1
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    #for each data point
    for n in range(N):
      #for each filter
        for f in range(F):
          #looping through hight and width of the output
            for i in range(H_out):
                for j in range(W_out):
                    # Compute the slice of x_pad and dout that corresponds to the current output pixel
                    vert_start = i * stride
                    vert_end = vert_start + HH
                    horiz_start = j * stride
                    horiz_end = horiz_start + WW
                    x_slice = x[n, :, vert_start:vert_end, horiz_start:horiz_end]
                    dout_slice = dout[n, f, i, j]

                    # Update the gradients
                    dx[n, :, vert_start:vert_end, horiz_start:horiz_end] += w[f] * dout_slice
                    dw[f] += x_slice * dout_slice
                    db[f] += dout_slice

    # Remove the padding from dx
    dx = dx[:, :, pad:-pad, pad:-pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # Compute the output shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Initialize the output
    out = np.zeros((N, C, H_out, W_out))

    # Loop over the output dimensions
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Compute the area of x that corresponds to the current pixel
                    vert_start = i * stride
                    vert_end = vert_start + pool_height
                    horiz_start = j * stride
                    horiz_end = horiz_start + pool_width
                    x_slice = x[n, c, vert_start:vert_end, horiz_start:horiz_end]

                    # Compute the max value of the slice (like in the lecture notes with the 4 colors)
                    out[n, c, i, j] = np.max(x_slice)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #extracting vars
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    
    #output size
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Find the index of the max element in the pooling region
                    vert_start = i * stride
                    vert_end = vert_start + pool_height
                    horiz_start = j * stride
                    horiz_end = horiz_start + pool_width
                    x_pool = x[n, c, vert_start:vert_end, horiz_start:horiz_end]
                    max_idx = np.unravel_index(np.argmax(x_pool), x_pool.shape)
                    
                    # Set the gradient of the max element to the corresponding dout value
                    dx[n, c, vert_start:vert_end, horiz_start:horiz_end][max_idx] = dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
