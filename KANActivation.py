import paddle


class KANActivation(paddle.nn.Layer):
    """
    KANActivation is a Activation function, this function try to fit a polynomial function with normal distribution.

    Args:
        g (int): Number of Gaussian functions.

    Returns:
        None

    """
    def __init__(self, g):
        super(KANActivation, self).__init__()
        self.a = paddle.create_parameter(
            shape=[1, g],
            dtype=self._dtype,
            is_bias=False
        )
        self.b = paddle.create_parameter(
            shape=[1, g],
            dtype=self._dtype,
            is_bias=False
        )
        self.c = paddle.create_parameter(
            shape=[1, g],
            dtype=self._dtype,
            is_bias=False
        )

    def forward(self, x):
        """
        前向传播函数，用于计算 KANActivation 激活函数的输出。

        Args:
            x (paddle.Tensor): 输入张量，形状为 [batch_size, -1, -1, ...]。

        Returns:
            paddle.Tensor: 经过激活函数处理后的输出张量，形状和输入形状相同。

        """
        x = paddle.unsqueeze(x, [-1])  # [batch_size, in_channels] -> [batch_size, in_channels, 1]
        x = self.a * paddle.exp(-(x - self.b) ** 2 / (2 * self.c ** 2))  # [batch_size, in_channels, g]
        x = paddle.sum(x, axis=[-1])  # [batch_size, in_channels]
        return x

if __name__ == '__main__':
    x = paddle.randn([1, 32])
    # 32 input channels, 64 output channels, 8 normal distribution to fit a polynomial
    layer = KANActivation(8)
    y = layer(x)
    print(y.shape)