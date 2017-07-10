#!/usr/bin/env python
# coding=utf-8

import numpy as np

from cnn.Filter import Filter


class ConvLayer(object):

    def __init__(self,input_width,input_height,channel_num,
                 filter_size,filter_num,learning_rate,padding,stride,activator):
        self.input_height=input_height
        self.input_width=input_width
        self.channel_num=channel_num
        self.filter_height=filter_size
        self.filter_width=filter_size
        self.filter_num=filter_num
        self.learning_rate=learning_rate
        self.stride=stride
        self.padding=padding
        self.output_width=self.calculate_output_size(input_width,filter_size,stride,padding)
        self.output_height=self.calculate_output_size(input_height,filter_size,stride,padding)
        self.output=np.zeros((filter_num,self.output_height,self.output_width))
        self.filter=[]
        for i in range(filter_num):
            self.filter.append(Filter(filter_size,channel_num))
        self.activator=activator

    @staticmethod
    def calculate_output_size(input_size,filter_size,stride,padding):
        return int((input_size-filter_size+2*padding)/stride)+1

    def forward(self,inputs):
        self.input_array=inputs
        extend_array=self.padding_array(inputs,self.padding)
        self.padded_input_array=extend_array
        for i in range(self.filter_num):
            # for j in range(self.channel_num):
            self.conv(extend_array,self.filter[i].get_weights(),self.output[i],self.stride,self.filter[i].get_bias())
        if self.activator is not None:
            self.element_wise_op(self.output,self.activator.forward)


    def padding_array(self,inputs,zp):
        # if zp==0: return inputs
        n=len(inputs.shape)
        zp=int(zp)
        if n==3:
            channel=inputs.shape[0]
            height=inputs.shape[1]
            width=inputs.shape[2]
            output_array=np.zeros((channel,height+2*zp,width+2*zp))
            for i in range(channel):
                output_array[i,zp:zp+height,zp:zp+width]=inputs[i,:,:]
            return output_array
        else:
            height=inputs.shape[0]
            width=inputs.shape[1]
            output_array=np.zeros((height+2*zp,width+2*zp))
            output_array[zp:zp+height,zp:zp+width]=inputs[:,:]
            return output_array
    @staticmethod
    def conv(inputs,mfilter,output,stride,bias):
        m,n=output.shape[0],output.shape[1]
        for i in range(m):
            for j in range(n):
                # print('output shape is{},inputs shape is {},mfilter shape is {}'.format(output.shape,
                #                                 inputs.shape,
                #                                  mfilter.shape))
                output[i][j]=(ConvLayer.get_batch(inputs,i,j,stride,mfilter.shape[1])*mfilter).sum()+bias

    @staticmethod
    def element_wise_op(inputs,op):
        for i in np.nditer(inputs,op_flags=['readwrite']):
            i[...]=op(i)

    @staticmethod
    def get_batch(inputs,i,j,stride,filter_size):
        m=len(inputs.shape)
        if m==3:
            return inputs[:,i*stride:i*stride+filter_size,j*stride:j*stride+filter_size]
        else:
            return inputs[i*stride:i*stride+filter_size,j*stride:j*stride+filter_size]

    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        cnn网的反向传播
        :param sensitivity:本层的sensitivity map
        :activator: 上一层的激活函数
        '''
        expanded_array=self.expand_sensitivity_map(sensitivity_array)
        expanded_width=expanded_array.shape[2]
        zp=(self.input_width+self.filter_width-1-expanded_width)//2
        padded_array=self.padding_array(expanded_array,zp)
        self.delta_array=self.create_delta_array()
        for i in range(self.filter_num):
            delta_array=self.create_delta_array()
            filter_array=np.array(list(map(lambda l:np.rot90(l,2),self.filter[i].get_weights())))
            # print('padded_array:{},filter_array:{},delta_array:{}'.format(padded_array.shape,
            #                                                               filter_array.shape,
            #                                                               delta_array.shape))
            for j in range(delta_array.shape[0]):
                self.conv(padded_array[i],filter_array[j],delta_array[j],1,0)
            self.delta_array+=delta_array
        # print('before element op,input_array is ')
        derived_array=self.input_array.copy()
        self.element_wise_op(derived_array,activator.backward)
        # derived_array=activator.backward(self.input_array)
        # print('derived_array is ',derived_array)
        self.delta_array*=derived_array#.astype(np.float32)

    def create_delta_array(self):
        return np.zeros((self.channel_num,self.input_height,self.input_width))

    def expand_sensitivity_map(self,sensitivity_map):
        depth=sensitivity_map.shape[0]
        expanded_width=self.input_width-self.filter_width+2*self.padding+1
        expanded_height=self.input_height-self.filter_height+2*self.padding+1
        expanded_array=np.zeros((depth,expanded_height,expanded_width))
        m,n=sensitivity_map.shape[1],sensitivity_map.shape[2]
        for i in range(m):
            for j in range(n):
                expanded_array[:,i*self.stride,j*self.stride]=sensitivity_map[:,i,j]
        return expanded_array

    def bg_gradient(self,sensitivity_map):
        expanded_array=self.expand_sensitivity_map(sensitivity_map)
        for i in range(self.filter_num):
            n=self.filter[i].get_weights().shape[0]
            # print('input_array shape is {},expanded is {},filter[i].'
            #       'w_grad[j] is {}'.format(self.input_array.shape,expanded_array.shape,self.filter[i].w_grad.shape))
            for j in range(n):
                self.conv(self.padded_input_array[j],expanded_array[i],self.filter[i].w_grad[j],1,0)
            self.filter[i].b_grad=expanded_array[i].sum()

    def update(self):
        for i in range(self.filter_num):
            self.filter[i].update(self.learning_rate)

    def backward(self,sensitivity_map,activator):
        self.bp_sensitivity_map(sensitivity_map,activator)
        self.bg_gradient(sensitivity_map)
        self.update()

    def mbackward(self,inputs,sensitivity_map,activator):
        self.bg_gradient(sensitivity_map)
        self.bp_sensitivity_map(sensitivity_map,activator)
        self.update()

# if __name__=='__main__':
#     x=np.arange(9).reshape((1,3,3))
#     output=np.zeros((3,3))
#     ConvLayer.conv(x,np.array([[1,0],[1,0]]),output,1,0)
#     print(x)
#     print(output)

