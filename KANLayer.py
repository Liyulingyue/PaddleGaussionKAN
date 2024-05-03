import paddle

class KANLayer(paddle.nn.Layer):
    """
    KANLayer: A custom PaddlePaddle Layer using Gaussian functions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        g (int): Number of Gaussian functions.

    Returns:
        paddle.Tensor: Output tensor with shape [batch_size, out_channels].
    """

    def __init__(self, in_channels, out_channels, g):
        super(KANLayer, self).__init__()
        self.a = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False
        )
        self.b = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False
        )
        self.c = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False
        )

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (paddle.Tensor): Input tensor with shape [batch_size, in_channels].

        Returns:
            paddle.Tensor: Output tensor with shape [batch_size, out_channels].
        """
        x = paddle.unsqueeze(x, [-1, -2])  # [batch_size, in_channels] -> [batch_size, in_channels, 1, 1]
        x = self.a * paddle.exp(-(x - self.b) ** 2 / (2 * self.c ** 2))  # [batch_size, in_channels, out_channels, g]
        x = paddle.sum(x, axis=[1, -1])  # [batch_size, out_channels]
        return x

if __name__ == '__main__':
    x = paddle.randn([1, 32])
    # 32 input channels, 64 output channels, 8 normal distribution to fit a polynomial
    layer = KANLayer(32, 64, 8)
    y = layer(x)
    print(y.shape)