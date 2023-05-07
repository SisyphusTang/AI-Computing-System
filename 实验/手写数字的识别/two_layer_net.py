import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 交叉熵损失函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



def sigmoid(x):
    return  1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    #上下同时减去一个常数 整个式子的值并不改变
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad
# # 对一个函数求梯度
# def numerical_gradient(f,x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     for idx in range(x.size):
#         temp_val = x[idx]
#         #计算f（x+h）
#         x[idx] = temp_val + h
#         fxh1 = f(x)

#         # f(x-h)
#         x[idx] = temp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1 - fxh2) / (2*h)
#         x[idx] = temp_val
#     return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        # 输入参数个数 * 隐层参数个数
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 隐层参数个数
        self.params['b1'] = np.zeros(hidden_size)
        # 隐层参数个数 * 输出层参数个数
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 输出的参数个数
        self.params['b2'] = np.zeros(output_size)
    
    # 进行推理，x是输入参数
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 计算识别精度    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # x:输入数据, t:监督数据
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
            return grads
    

# (x_train, t_train), (x_test, t_test) =  load_mnist(normalize=True, one_hot_laobel = True)
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True, one_hot_label=False)
train_loss_list = []

# 超参数
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版!
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(loss)