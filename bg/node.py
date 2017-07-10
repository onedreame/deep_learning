#coding:utf-8
from functools import reduce
import math

class node(object):
    '''
    代表神经网中的神经元节点
    '''

    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0

    def append_downstream_connection(self,node):
        self.downstream.append(node)

    def append_upstream_connection(self,node):
        self.upstream.append(node)

    def set_output(self,output):
        self.output=output

    def sigmod(self,output):
        '''激活函数'''
        return 1.0/(1+math.exp(-output))

    def calc_output(self):
        '''计算该节点的输出'''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output=self.sigmod(output)

    def calc_hidden_layer_delta(self):
        '''计算隐藏层的偏导项'''
        output=reduce(lambda ret,conn:ret+conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*output

    def calc_output_layer_delta(self, label):
        '''
        节点属于输出层时，根据式3计算delta
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str