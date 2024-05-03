# GaussionKAN based on PaddlePaddle
## Introduction
This is a PaddlePaddle implementation of KAN.

In this project, a Gasussion Distribution is used to replace the original B-branches.

## Notes
The demo use PaddlePaddle(CPU), if you want to use GPU, you need uninstall `paddlepaddle` and install `paddlepaddle-gpu`.

## Usage
install paddlepaddle, if you have gpu, please install `paddlepaddle-gpu`.
```bash
pip install paddlepaddle
```

run the demo.
```python
from KANLayer import KANLayer
import paddle

x = paddle.randn([1, 32])
# 32 input channels, 64 output channels, 8 normal distribution to fit a polynomial
layer = KANLayer(32, 64, 8)
y = layer(x)
print(y.shape)
```
