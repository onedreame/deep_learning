import numpy as np

from lstm.Activator import IdentityActivator
from lstm.LstmLayer import LstmLayer


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d
def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    lstm = LstmLayer(3, 2, 1e-3)
    # 计算forward值
    x, d = data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])
    # 求取sensitivity map
    sensitivity_array = np.ones(lstm.h_list[-1].shape,
                                dtype=np.float64)
    # 计算梯度
    lstm.mbackward(x[1], sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    # for i in range(lstm.wf.shape[0]):
    #     for j in range(lstm.wf.shape[1]):
    #         lstm.wf[i,j] += epsilon
    #         lstm.reset_state()
    #         lstm.forward(x[0])
    #         lstm.forward(x[1])
    #         err1 = error_function(lstm.h_list[-1])
    #         lstm.wf[i,j] -= 2*epsilon
    #         lstm.reset_state()
    #         lstm.forward(x[0])
    #         lstm.forward(x[1])
    #         err2 = error_function(lstm.h_list[-1])
    #         expect_grad = (err1 - err2) / (2 * epsilon)
    #         lstm.wf[i,j] += epsilon
    #         print('weights(%d,%d): expected - actural %.4e - %.4e' % (
    #             i, j, expect_grad, lstm.wf_grad[i,j]))
    check(lstm,lstm.wf,x,epsilon,error_function,lstm.wf_grad)
    check(lstm,lstm.wi,x,epsilon,error_function,lstm.wi_grad)
    check(lstm, lstm.wo, x, epsilon, error_function, lstm.wo_grad)
    check(lstm, lstm.wct, x, epsilon, error_function, lstm.wct_grad)
    print('bf check')
    check(lstm, lstm.bf, x, epsilon, error_function, lstm.bf_grad)
    return lstm

def check(lstm,w,x,epsilon,error,w_grad):
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i,j]+=epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1=error(lstm.h_list[-1])
            w[i,j]-=2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2=error(lstm.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            w[i, j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                i, j, expect_grad, w_grad[i, j]))
gradient_check()