#coding:utf-8
import numpy as np
class SigmoidActivator(object):
    def forward(self,inputs):
        return 1.0/(1+np.exp(-inputs))
    def backward(self,inputs):
        return inputs*(1-inputs)

class FullConnectedLayer(object):
    def __init__(self,input_size,output_size,activator):
        '''
        构造函数
        :param input_size:输入向量维度
        :param output_size: 输出向量维度
        :param activator: 激活函数
        '''
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator

        self.w=np.random.uniform(-0.1,0.1,[output_size,input_size])
        self.b=np.zeros((output_size,1))
        self.output=np.zeros((output_size,1))

    def forward(self,inputs):
        '''
        前向传播
        :param inputs: 输入向量
        '''
        self.input=inputs
        output=np.dot(self.w,self.input)
        outt=output.transpose()
        o1=output.T
        # print('output.shape is ',output.shape)
        # print('self.b shape ',self.b.shape)
        o1=output+self.b
        self.output=self.activator.forward(output+self.b)

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
        self.delta = self.activator.backward(self.input).T*np.dot(delta_array, self.w)
        # self.w_grad=self.input*delta_array
        self.w_grad = delta_array.T * self.input.T  #delta_array [1,10],self.input [300,1],相乘结果却是[300,10]
        self.b_grad=delta_array.T
        # print('self.w_grad is ',self.w_grad.shape)
        # print('self.b_grad is ',self.b_grad.shape)

    def update(self,learning_rate):

        self.w+=learning_rate*self.w_grad
        self.b+=learning_rate*self.b_grad
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        i=0
        # print('layer %d,sample shape is ' %i,sample.shape)
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
            # i+=1
            # print('layer %d,sample shape is ' % i, output.shape)
        return output
    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d:d+1],
                    data_set[d:d+1].T, rate)
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)
    def calc_gradient(self, label):
        # delta = self.layers[-1].activator.backward(
        #     self.layers[-1].output
        # ) * (label - self.layers[-1].output)
        # print(label.shape)
        # print(self.layers[-1].output.shape)
        delta = self.layers[-1].activator.backward(    #这里label的数组格式为[[..],[..]...]
            self.layers[-1].output.T                #因而delta的格式也是[[..],[..],...]
        ) * (label - self.layers[-1].output.T)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    def update_weight(self, rate):
        tot=.0;i=0
        for layer in self.layers:
            ori_weight=layer.w.copy()
            ori_b=layer.b.copy()
            layer.update(rate)
            print('第{}层: sum w_grad is {},sum b_grad is {}'.format(i,np.sum(layer.w_grad),np.sum(layer.b_grad)))
            print('weight 更新个数：%d,b更新个数：%d.' % (
                                                np.sum((ori_weight != layer.w).astype(np.int32)),
                                                np.sum((ori_b != layer.b).astype(np.int32))))
            i+=1