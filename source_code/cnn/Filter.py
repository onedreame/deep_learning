#coding:utf-8

import numpy as np

class Filter(object):

    def __init__(self,filter_size,channel_num):
        self.w=np.random.uniform(-0.1,0.1,(channel_num,filter_size,filter_size))
        self.b=0
        self.w_grad=np.zeros((channel_num,filter_size,filter_size))
        self.b_grad=0

    def get_weights(self):
        return self.w

    def get_bias(self):
        return self.b

    def update(self,learning_rate):
        # old_w = self.w.copy()
        # old_b = self.b
        self.w-=learning_rate*self.w_grad
        self.b-=learning_rate*self.b_grad
        # res=(old_w!=self.w)
        # resb=(old_b!=self.b)
        # if resb: resb=1
        # else: resb=0
        # print('Filter,update %d weights,update %d bias'
        #       % (res.astype(np.int32).sum(), resb))