#coding:utf-8
import numpy as np

class MaxPoolingLayer(object):

    def __init__(self,input_width,input_height,channel_num,filter_size,stride):
        self.input_width=input_width
        self.input_height=input_height
        self.channel_num=channel_num
        self.filter_size=filter_size
        self.stride=stride
        self.output_width=int((input_width-filter_size)/stride)+1
        self.output_height=int((input_height-filter_size)/stride)+1
        self.output=np.zeros((channel_num,self.output_height,self.output_width))

    def forward(self,inputs):
        self.input_array=inputs.copy()
        for d in range(self.channel_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output[d][i][j]=self.get_batch(d,inputs,i,j,self.filter_size,self.stride).max()


    def get_batch(self,d,inputs,i,j,filter_size,stride):
        if d!=-1:
            return inputs[d,i*stride:i*stride+filter_size,j*stride:j*stride+filter_size]
        else:
            return inputs[i*stride:i*stride+filter_size,j*stride:j*stride+filter_size]

    def backward(self,input_array,sensitivity_map):
        self.delta_array=np.zeros(input_array.shape)
        for d in range(self.channel_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array=self.get_batch(d,input_array,i,j,self.filter_size,self.stride)
                    maxi,maxj=self.get_max_index(patch_array)
                    self.delta_array[d,i*self.stride+maxi,j*self.stride+maxj]=sensitivity_map[d,i,j]

    def get_max_index(self,inputs):
        m,n=inputs.shape[0],inputs.shape[1]
        maxval=-np.inf;maxi=0;maxj=0
        for i in range(m):
            for j in range(n):
                if inputs[i][j]>maxval:
                    maxval=inputs[i][j]
                    maxi=i;maxj=j
        return maxi,maxj

