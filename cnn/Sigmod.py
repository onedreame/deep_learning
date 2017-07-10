#coding:utf-8
import numpy as np
class Relu(object):

    def forward(self,output):
        return output if output>0 else 0

    def backward(self,output):
        return 1 if output>0 else 0

class SigmoidActivator(object):
    def forward(self,inputs):
        return 1.0/(1+np.exp(-inputs))
    def backward(self,inputs):
        return inputs*(1-inputs)