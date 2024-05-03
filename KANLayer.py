import paddle

class KANLayer(paddle.nn.Layer):
    # input: a 2D tensor, shape is [batch_size, in_channels]
    # output: a 2D tensor, shape is [batch_size, out_channels]
    # g: number of gaussian functions
    def __init__(self, in_channels, out_channels, g):
        super(KANLayer, self).__init__()
        self.a = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False)
        self.b = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False)
        self.c = paddle.create_parameter(
            shape=[in_channels, out_channels, g],
            dtype=self._dtype,
            is_bias=False)

    def forward(self, x):
        x = paddle.unsqueeze(x, [-1, -2]) # [batch_size, in_channels] -> [batch_size, in_channels, 1, 1]
        x = self.a*paddle.exp(-(x-self.b)**2/(2*self.c**2)) # [batch_size, in_channels, out_channels, g]
        x = paddle.sum(x, axis=[1, -1]) # [batch_size, out_channels]
        return x

if __name__ == '__main__':
    x = paddle.randn([1, 32])
    layer = KANLayer(32, 64, 8)
    y = layer(x)
    print(y.shape)