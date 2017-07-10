#coding:utf-8

import numpy as np

from cnn.Convlayer import ConvLayer


class IdentityActivator(object):
    def forward(self,output):
        return output
    def backward(self,output):
        return 1


def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    cl = ConvLayer(5,5,3,3,2,padding=1,stride=2,activator=IdentityActivator(),learning_rate=0.001)
    cl.filter[0].weights = np.array(
        [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    cl.filter[0].bias=1
    cl.filter[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    return a, b, cl
def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)
    # 求取sensitivity too many indices for arraymap，是一个全1数组
    sensitivity_array = np.ones(cl.output.shape,
                                dtype=np.float64)
    # 计算梯度
    cl.mbackward(a, sensitivity_array,
                  IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filter[0].w_grad.shape[0]):
        for i in range(cl.filter[0].w_grad.shape[1]):
            for j in range(cl.filter[0].w_grad.shape[2]):
                cl.filter[0].w[d,i,j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output)
                cl.filter[0].w[d,i,j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filter[0].w[d,i,j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filter[0].w_grad[d,i,j]))

# gradient_check()