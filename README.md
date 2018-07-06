# tensorflow_conv2d
卷积神经网络

## 弃用API
tf.mul  tf.sub   tf.neg 已经废弃,分别可用tf.multiply  tf.subtract  tf.negative替代.
## 卷积层的返回值形状计算
Output = (W - F + 2P)/S + 1, 计算卷积层的返回值形状，W是输入形状，F是过滤器形状，P是padding的大小，S是步长形状
