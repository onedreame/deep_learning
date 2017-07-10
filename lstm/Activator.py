#coding:utf-8
import numpy as np
class SigmoidActivator(object):

    def forward(self,inputs):
        return 1.0/(1+np.exp(-inputs))

    def backward(self,output):
        return output*(1-output)

class TanhActivator(object):

    def forward(self,inputs):
        return 2.0/(1+np.exp(-2.*inputs))-1.

    def backward(self,output):
        return 1-output*output

class IdentityActivator(object):
    def forward(self,output):
        return output
    def backward(self,output):
        return 1

# x=np.array([[1,0],[1,0]])
# acti=TanhActivator()
# y=acti.forward(x)
# print(y)