#coding:utf-8
from bg.ConstNode import ConstNode
from bg.node import node


class layer(object):
    '''
    代表神经网络的每一层，由该类来管理每一层的神经元节点
    '''
    def __init__(self,layer_index,node_count):
        self.layer_index=layer_index
        self.nodes=[]
        for i in range(node_count):
            self.nodes.append(node(layer_index=layer_index,node_index=i))
        self.nodes.append(ConstNode(node_index=node_count,layer_index=layer_index))

    def set_output(self,data):
        '''
        设置这一层节点的输出值
        :data:要设置的初始值列表
        '''
        for i in range(len(data)-1):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        计算每个节点的输出
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        输出这一层所有节点的信息
        :return:
        '''
        for node in self.nodes:
            print(node)