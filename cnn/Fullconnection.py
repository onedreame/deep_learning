#coding:utf-8

import numpy as np

class FullConnection(object):

    def __init__(self,input_size,output_size,activator,learning_rate):
        '''
        构造函数
        :param input_size:输入向量维度
        :param output_size: 输出向量维度
        :param activator: 激活函数
        '''
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator
        self.learning_rate=learning_rate

        self.w=np.random.uniform(-0.1,0.1,[output_size,input_size])
        self.b=np.zeros((output_size,1))
        self.output=np.zeros((output_size,1))

    def forward(self,inputs):
        '''
        前向传播
        :param inputs: 输入向量
        '''
        self.input_array=inputs.copy()
        self.output=np.dot(self.w,self.input_array)
        # outt=output.transpose()
        # o1=output.T
        # print('output.shape is ',output.shape)
        # print('self.b shape ',self.b.shape)
        # o1=output+self.b
        if self.activator is not None:
            self.element_wise_op(self.output,self.activator.forward)

    @staticmethod
    def element_wise_op(inputs, op):
        for i in np.nditer(inputs, op_flags=['readwrite']):
            i[...] = op(i)

    def backward(self,delta_array):  #delta_array shape为[...],input的shape是[[..],[..]...]
        '''
        反向传播计算
        :param delta_array: 反向传播下一层向量
        '''
        # print('delta_array is ',delta_array.shape)
        # print('self.w.T is ',self.w.T.shape)
        # print('self.w is ',self.w.shape)
        # print('self.output is ',self.output.shape)
        # print('self.input is ',self.input.shape)
        # self.delta=np.dot(delta_array,self.w)*self.activator.backward(self.input).T
        self.delta_array=np.dot(delta_array,self.w)
        if self.activator is not None:
            inputs=self.input_array.copy()
            self.element_wise_op(inputs,self.activator.backward)
            self.delta_array = inputs.T*self.delta_array
        # self.w_grad=self.input*delta_array
        self.w_grad = delta_array.T * self.input_array.T  #delta_array [1,10],self.input [300,1],相乘结果却是[300,10]
        self.b_grad=delta_array.T
        # old_w=self.w.copy()
        old_b=self.b.copy()
        self.update(self.learning_rate)
        # res=(old_w!=self.w)
        # resb=(old_b!=self.b)
        # print('Fullconnection,update %d weights,update %d bias'
        #       %(res.astype(np.int32).sum(),resb.astype(np.int32).sum()))
        # print('self.w_grad is ',self.w_grad.shape)
        # print('self.b_grad is ',self.b_grad.shape)

    def update(self,learning_rate):

        self.w-=learning_rate*self.w_grad
        self.b-=learning_rate*self.b_grad


# conn=FullConnection(1,1,None)
# print(conn.__class__.__name__)
