from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


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
    x_flat = x.reshape(x.shape[0], -1)
    out = x_flat.dot(w)+b

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
    x_flat = x.reshape(x.shape[0], -1)
    dx=dout.dot(w.T)
    dx=dx.reshape(x.shape)

    dw=x_flat.T.dot(dout)

    db=np.sum(dout,axis=0)

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
    out=np.maximum(0,x)

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
    dx = np.where(x > 0, dout, 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)  # shape (D,)
        sample_var = np.var(x, axis=0)   # shape (D,)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        cache = None
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, x_hat, mu, var, gamma, beta, eps = cache
    N, D = dout.shape
    sqrt_var = np.sqrt(var + eps)

    # 1. 对归一化后的值 x_hat 的梯度
    dx_hat = dout * gamma

    # 2. 对方差 var 的梯度
    dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + eps)**(-1.5), axis=0)

    # 3. 对均值 mu 的梯度
    dmu = np.sum(dx_hat * (-1.0 / sqrt_var), axis=0)

    # 4. 对原始输入 x 的梯度
    dx = dx_hat / sqrt_var + dmu / N + dvar * 2 * (x - mu) / N

    # 5. 对 gamma 和 beta 的梯度（最简单）
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, x_hat, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    sqrt_var = np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    #好难，好困
    dx = (gamma / (N * sqrt_var)) * (N * dout - np.sum(dout, axis=0) - x_hat * np.sum(dout * x_hat, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    N, D = x.shape

    # 对每个样本自己做归一化（axis=1）
    mean = np.mean(x, axis=1, keepdims=True)    # (N,1)
    var = np.var(x, axis=1, keepdims=True)      # (N,1)

    # 归一化
    x_norm = (x - mean) / np.sqrt(var + eps)

    # 缩放 + 平移
    out = gamma * x_norm + beta

    # 缓存
    cache = (x, x_norm, mean, var, gamma, beta, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_norm, mean, var, gamma, beta, eps = cache
    N, D = x.shape

    # 计算 dbeta, dgamma
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    # 反向传播到 x_norm
    dx_norm = dout * gamma

    # 反向传播方差和均值
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=1, keepdims=True)
    dmean = np.sum(dx_norm * (-1 / np.sqrt(var + eps)), axis=1, keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=1, keepdims=True) / D

    # 最终 dx
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / D + dmean / D
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p  
        out = x * mask  # 应用 mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
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
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # 获取输入维度
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # 计算输出特征图的高和宽
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # 初始化输出
    out = np.zeros((N, F, H_out, W_out))

    # 对输入进行零填充（不修改原输入 x）
    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    # 遍历每一张图片
    for n in range(N):
        # 遍历每一个滤波器
        for f in range(F):
            # 遍历输出高度
            for i in range(H_out):
                # 遍历输出宽度
                for j in range(W_out):
                    # 计算当前感受野在填充后输入上的位置
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # 取出当前感受野区域: (C, HH, WW)
                    window = x_pad[n, :, h_start:h_end, w_start:w_end]

                    # 卷积计算：点积 + 偏置
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
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
    # 从缓存中取出前向传播的变量
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    # ✅ 关键修复1：对输入x做padding（和前向完全一致）
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    # 获取各维度尺寸
    N, C, H, W = x.shape      # 输入维度
    F, _, HH, WW = w.shape    # 卷积核维度
    _, _, H_out, W_out = dout.shape  # 输出维度

    # 初始化梯度为0（形状和对应变量完全一致）
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # ✅ 关键修复：对输入x做padding（和前向完全一致）
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    dx_pad = np.zeros_like(x_pad)

    # 对输入样本循环
    for n in range(N):
        # 对输出通道（卷积核）循环
        for f in range(F):
            # 对输出高度方向循环
            for h_out in range(H_out):
                # 对输出宽度方向循环
                for w_out in range(W_out):
                    # 计算当前位置在输入上的起始坐标
                    h_start = h_out * stride
                    h_end = h_start + HH
                    w_start = w_out * stride
                    w_end = w_start + WW
                    
                    # 取出当前窗口的输入数据
                    x_window = x_pad[n, :, h_start:h_end, w_start:w_end]
                    
                    # 核心反向传播公式：
                    # 1. dw 累加：dout * 输入窗口
                    dw[f] += x_window * dout[n, f, h_out, w_out]
                    # 2. dx 累加：dout * 卷积核权重
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, h_out, w_out]
    
    # ✅ 关键修复：去掉padding，还原成原始x的梯度dx
    dx += dx_pad[:, :, pad:pad+H, pad:pad+W]

    # 3. db 反向传播：对每个卷积核，在所有位置求和
    db = np.sum(dout, axis=(0, 2, 3))
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

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

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
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    # 计算输出尺寸
    H_out = 1 + (H - pool_h) // stride
    W_out = 1 + (W - pool_w) // stride

    # 初始化输出
    out = np.zeros((N, C, H_out, W_out))

    # 四层循环（朴素实现）
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # 窗口坐标（直接算，不用 pad！）
                    h_start = h_out * stride
                    h_end = h_start + pool_h
                    w_start = w_out * stride
                    w_end = w_start + pool_w

                    # 取窗口最大值
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, h_out, w_out] = np.max(window)
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
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    # 输出尺寸
    _, _, H_out, W_out = dout.shape

    # dx 初始化和 x 一样大，全部为 0
    dx = np.zeros_like(x)

    # 遍历每个位置，把梯度传给前向的最大值位置
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # 找到窗口位置
                    h_start = h_out * stride
                    h_end = h_start + pool_h
                    w_start = w_out * stride
                    w_end = w_start + pool_w

                    # 取出窗口
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # 找到窗口里的最大值的位置
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    
                    # 只把梯度传给最大值的位置！
                    dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, h_out, w_out]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    
    # 1. 把 (N,C,H,W)  reshape 成 (N*H*W, C)
    x_reshaped = x.transpose(0,2,3,1).reshape(-1, C)
    
    # 2. 调用普通 batchnorm
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # 3. 形状变回去
    out = out_reshaped.reshape(N, H, W, C).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    
    # 把梯度 reshape 成 (N*H*W, C)
    dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, C)
    
    # 调用普通 BN 反向
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    
    # 把形状还原
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape

    # 1. 把通道 C 分成 G 组
    # shape 变成 (N, G, C//G, H, W)
    x_group = x.reshape(N, G, C // G, H, W)

    # 2. 在每组内部求均值、方差（对所有维度求，除了 N 和 G）
    # 也就是对 (C//G, H, W) 求平均
    mean = x_group.mean(axis=(2, 3, 4), keepdims=True)
    var = x_group.var(axis=(2, 3, 4), keepdims=True)

    # 3. 归一化
    x_hat = (x_group - mean) / np.sqrt(var + eps)

    # 4. 恢复原来的形状 (N,C,H,W)
    x_hat = x_hat.reshape(N, C, H, W)
    
    # 5. 缩放 + 平移 (gamma, beta shape: (1,C,1,1))
    out = gamma * x_hat + beta

    # 6. 保存反向需要的值
    cache = (x_group, x_hat, mean, var, gamma, beta, G, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x_group, x_hat, mean, var, gamma, beta, G, eps = cache
    N, C, H, W = dout.shape
    _, G, C_G, H, W = x_group.shape

    # 1. 计算 dbeta, dgamma（最简单）
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)

    # 2.  upstream 梯度
    dx_hat = dout * gamma
    dx_hat_group = dx_hat.reshape(N, G, C_G, H, W)

    # 3. 计算均值、方差的梯度（和 BN 公式一样）
    sum_axes = (2, 3, 4)
    N_group = C_G * H * W

    dvar = np.sum(dx_hat_group * (x_group - mean) * -0.5 * (var + eps)**(-1.5), axis=sum_axes, keepdims=True)
    dmean = np.sum(dx_hat_group * -1 / np.sqrt(var + eps), axis=sum_axes, keepdims=True) + dvar * np.sum(-2 * (x_group - mean), axis=sum_axes, keepdims=True) / N_group

    # 4. 计算 dx
    dx_group = dx_hat_group / np.sqrt(var + eps) + dmean / N_group + dvar * 2 * (x_group - mean) / N_group
    dx = dx_group.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


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
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    num_train = x.shape[0]
    
    # 1. Softmax 概率（用新变量，不修改输入 x！）
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted_x)
    prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # 2. 计算损失
    loss = -np.sum(np.log(prob[np.arange(num_train), y])) / num_train
    
    # 3. 计算梯度
    dx = prob.copy()
    dx[np.arange(num_train), y] -= 1
    dx /= num_train   # ✅ 必须除以样本数

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
