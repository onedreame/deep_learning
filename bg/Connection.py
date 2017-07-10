#coding:utf-8
import numpy as np
class Connection(object):
    '''
    相邻神经层之间两个神经元的连边
    '''
    def __init__(self,downstream_node,upstream_node):
        self.downstream_node=downstream_node
        self.upstream_node=upstream_node
        self.gradient=0.0
        self.weight=np.random.uniform(-0.1,0.1)

    def get_gradient(self):
        '''

        :return: 梯度
        '''
        return self.gradient

    def update_weight(self,rating):
        '''
        更新这条边的权重
        :return:
        '''
        self.calc_gradient()
        self.weight+=rating*self.gradient

    def calc_gradient(self):
        '''计算该边的梯度'''
        self.gradient=self.upstream_node.output*self.downstream_node.delta

    def __str__(self):
        '''打印结果'''
        return "(%u-%u)-->(%u-%u):%f" \
               %(self.upstream_node.layer_index,
                self.upstream_node.node_index,
                self.downstream_node.layer_index,
                self.downstream_node.node_index,
                 self.weight)
        # return '(%u-%u) -> (%u-%u) = %f' % (
        #     self.upstream_node.layer_index,
        #     self.upstream_node.node_index,
        #     self.downstream_node.layer_index,
        #     self.downstream_node.node_index,
        #     self.weight)
