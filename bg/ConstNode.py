#coding:utf-8
from functools import reduce


class ConstNode(object):
    '''
    作为偏置项，输出为1
    '''
    def __init__(self,layer_index,node_index):
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.delta=0.0
        self.output=1.0

    def append_downstream_connection(self,node):
        self.downstream.append(node)

    def calc_hidden_layer_delta(self):
        self.delta=reduce(lambda ret,conn:ret+conn.downstream_node.output*conn.weight,self.downstream,0.0)

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str