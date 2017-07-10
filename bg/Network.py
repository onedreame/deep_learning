#coding:utf-8
from functools import reduce

from bg.Connection import Connection
from bg.Connections import Connections
from bg.layer import layer


class Network(object):
    '''
    搭建网络结构的类
    '''
    def __init__(self,layers):
        '''
        构建网络结构
        :param layers:每层网络的神经元个数
        '''
        self.connections=Connections()
        self.layers=[]
        for i in range(len(layers)):
            self.layers.append(layer(i,layers[i]))
        for i in range(len(layers)-1):
            connections=[Connection(down,up)
                         for up in self.layers[i].nodes
                         for down in self.layers[i+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    def train(self,labels,data_set,rate,iteration):
        '''
        训练模型
        :param labels:样本标记
        :param data_set: 数据集
        :param rate: 学习率
        :param iteration: 训练次数
        '''
        for i in range(iteration):
            for vec in range(len(data_set)):
                self.train_one_sample(labels[vec],data_set[vec],rate)

    def train_one_sample(self,label,data,rate):
        '''
        训练单个样本
        :param label: 样本标记
        :param data: 样本向量
        :param rate: 学习率
        '''
        self.predict(data)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self,date):
        '''
        预测结果
        :param date:样本向量
        '''
        self.layers[0].set_output(data=date)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output,self.layers[-1].nodes[:-1])

    def calc_delta(self,label):
        '''
        计算输出层的delta
        :param label: 标签
        '''
        output_nodes=self.layers[-1].nodes
        for i in range(len(output_nodes[:-1])):
            output_nodes[i].calc_output_layer_delta(label[i])
        for i in range(len(self.layers)-1,0,-1):
            for node in self.layers[i].nodes[:-1]:
                node.calc_hidden_layer_delta()

    def update_weight(self,rate):
        '''
        更新边的权重
        :param rate:学习率
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        计算每条边的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self,label,samplt):
        '''
        获取网络的一个样本下，每个连接上的权重
        :param label: 样本标签
        :param samplt: 样本向量
        :return: 每个连接上的权重
        '''
        self.predict(date=samplt)
        self.calc_delta(label)
        self.calc_gradient()

    def gradient_check(self,label,sample):
        '''
        梯度检查
        :param label:样本标签
        :param sample: 样本向量
        '''
        #计算网络误差，这里采用的是MSE
        network_error=lambda v1,v2: 0.5*reduce(lambda a,b:a+b,
                                               map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(v1,v2)))
        self.get_gradient(label=label,samplt=sample)
        for conn in self.connections.connections:
            actual_gradient=conn.get_gradient()
            epsition=0.0001
            conn.weight+=epsition
            error1=network_error(self.predict(sample),label)
            conn.weight-=2*epsition
            error2=network_error(self.predict(sample),label)
            expected_gradient=(error2-error1)/(2*epsition)
            print('expected gradient:\t%f\nactual gradient:\t%f' %(expected_gradient,actual_gradient))

    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()
