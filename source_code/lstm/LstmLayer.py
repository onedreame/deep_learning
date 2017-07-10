#coding:utf-8

import numpy as np

from lstm.Activator import SigmoidActivator, TanhActivator
debug=False

class LstmLayer(object):

    def __init__(self,input_state,state_num,learning_rate):
        self.input_state=input_state
        self.state_num=state_num
        self.learning_rate=learning_rate
        self.h_list=self.init_state_list()
        self.c_list=self.init_state_list()
        self.f_list=self.init_state_list()
        self.i_list=self.init_state_list()
        self.o_list=self.init_state_list()
        self.ct_list=self.init_state_list()
        self.wi,self.bi=self.init_gate_mat()
        self.wf,self.bf=self.init_gate_mat()
        self.wct,self.bct=self.init_gate_mat()
        self.wo,self.bo=self.init_gate_mat()
        self.wix=self.init_input_mat()
        self.wfx=self.init_input_mat()
        self.wctx=self.init_input_mat()
        self.wox=self.init_input_mat()
        self.gate_activator=SigmoidActivator()
        self.output_activator=TanhActivator()
        self.input_array=None
        self.times=0
        self.f=open('log.txt','w+')

    def init_state_list(self):
        c_list=[]
        c_list.append(np.zeros((self.state_num,1)))
        return c_list

    def init_gate_mat(self):
        w=np.random.uniform(-0.1,0.1,(self.state_num,self.state_num))
        b=np.zeros((self.state_num,1))
        return w,b
    def init_input_mat(self):
        w=np.random.uniform(-0.1,0.1,(self.state_num,self.input_state))
        return w

    def forward(self,inputs):
        '''
        这里发生了溢出错误，
        :param inputs:输入样本向量
        '''
        self.times+=1
        h_prev=self.h_list[self.times-1]
        self.input_array=inputs.copy()
        f=np.dot(self.wf,h_prev)+np.dot(self.wfx,inputs)+self.bf
        self.f_list.append(self.gate_activator.forward(f))
        i=np.dot(self.wi,h_prev)+np.dot(self.wix,inputs)+self.bi
        self.i_list.append(self.gate_activator.forward(i))
        o=np.dot(self.wo,h_prev)+np.dot(self.wox,inputs)+self.bo
        self.o_list.append(self.gate_activator.forward(o))
        ct=np.dot(self.wct,h_prev)+np.dot(self.wctx,inputs)+self.bct
        self.ct_list.append(self.output_activator.forward(ct))
        # self.f.write('forward,ct：{},f:{},i:{}'.format(self.ct_list[-1],self.f_list[-1],self.i_list[-1]))
        c=self.f_list[-1]*self.c_list[self.times-1]+self.i_list[-1]*self.ct_list[self.times]
        self.c_list.append(c)
        # self.f.write('forward,h,c is {}'.format(c))
        # print('forward,h,c:{}'.format(c))
        # assert c.max()>-400
        #这里产生了爆炸现象，也就是说c的值变的非常大
        h=self.o_list[-1]*self.output_activator.forward(c)
        # print('forward,h:{}'.format(h))
        self.h_list.append(h)

    def backward(self,delta_array):
        self.cal_delta(delta_array)
        self.cal_gradient()
        self.update()

    def cal_delta(self,delta_array):
        self.h_delta=self.init_delta()
        self.c_delta=self.init_delta()
        self.f_delta=self.init_delta()
        self.i_delta=self.init_delta()
        self.ct_delta=self.init_delta()
        self.o_delta=self.init_delta()
        self.h_delta[-1]=delta_array
        # print('cal_delta_k,tanhct')
        for i in range(self.times,0,-1):
            self.cal_delta_k(i)

    def cal_delta_k(self,k):
        tanhct=self.output_activator.forward(self.c_list[k])
        deltak=self.h_delta[k]
        o_delta=deltak*tanhct*self.o_list[k]*(1-self.o_list[k])
        # print('shape,deltak:{},tanhct:{},o_list:{},o_delta:{}'.format(deltak.shape,tanhct.shape,
        #                                                     self.o_list[k].shape,o_delta.shape))
        self.o_delta[k]=o_delta
        i_delta=deltak*self.o_list[k]*(1-tanhct*tanhct)*self.ct_list[k]*self.i_list[k]*(1-self.i_list[k])
        self.i_delta[k]=i_delta
        ct_delta=deltak*self.o_list[k]*(1-tanhct*tanhct)*self.i_list[k]*(1-self.ct_list[k]*self.ct_list[k])
        self.ct_delta[k]=ct_delta
        f_delta=deltak*self.o_list[k]*(1-tanhct*tanhct)*self.c_list[k-1]*self.f_list[k]*(1-self.f_list[k])
        self.f_delta[k]=f_delta
        self.h_delta[k-1]=(np.dot(o_delta.T,self.wo)+np.dot(i_delta.T,self.wi) \
                          +np.dot(ct_delta.T,self.wct)+np.dot(f_delta.T,self.wf)).transpose()
        # print('k-1:{}'.format(self.h_delta[k-1].shape))
        if debug:
            print('o_delta:{}'.format(type(o_delta)))
            print('o_delta:{},i_delta:{},ct_delta:{},f_delta:{}'.format(o_delta,i_delta,ct_delta,f_delta))


    def cal_gradient(self):
        self.wf_grad=np.zeros((self.state_num,self.state_num));self.bf_grad=np.zeros((self.state_num,1))
        self.wi_grad=np.zeros((self.state_num,self.state_num));self.bi_grad=np.zeros((self.state_num,1))
        self.wo_grad=np.zeros((self.state_num,self.state_num));self.bo_grad=np.zeros((self.state_num,1))
        self.wct_grad=np.zeros((self.state_num,self.state_num));self.bct_grad=np.zeros((self.state_num,1))
        for i in range(1,self.times+1):
            # print('self.wf_grad:{},self.f_delta:{},self.h_list:{}'.format(self.wf_grad.shape,
            #                                                               self.f_delta[i].shape,
            #                                                               self.h_list[i-1].shape))
            self.wf_grad+=self.f_delta[i]*self.h_list[i-1].T
            self.bf_grad+=self.f_delta[i]
            self.wi_grad+=self.i_delta[i]*self.h_list[i-1].T
            self.bi_grad+=self.i_delta[i]
            self.wo_grad+=self.o_delta[i]*self.h_list[i-1].T
            self.bo_grad+=self.o_delta[i]
            self.wct_grad+=self.ct_delta[i]*self.h_list[i-1].T
            self.bct_grad+=self.ct_delta[i]
        self.wfx_grad = np.zeros((self.state_num, self.input_state))
        self.wix_grad = np.zeros((self.state_num, self.input_state))
        self.wox_grad = np.zeros((self.state_num, self.input_state))
        self.wctx_grad = np.zeros((self.state_num, self.input_state))
        # self.wfx_grad=np.dot(self.f_delta[-1],x.T)
        # self.wix_grad = np.dot(self.i_delta[-1] , x.T)
        # self.wox_grad = np.dot(self.o_delta[-1] , x.T)
        # self.wctx_grad = np.dot(self.ct_delta[-1] , x.T)
        # self.wfx_grad = self.f_delta[-1] * x.T
        # self.wix_grad = self.i_delta[-1] * x.T
        # self.wox_grad = self.o_delta[-1] * x.T
        # self.wctx_grad = self.ct_delta[-1] * x.T
        self.wfx_grad=self.f_delta[-1]*self.input_array.T
        self.wix_grad=self.i_delta[-1]*self.input_array.T
        self.wox_grad=self.o_delta[-1]*self.input_array.T
        self.wctx_grad=self.ct_delta[-1]*self.input_array.T
        # for i in range(1,self.times+1):
        #
        #     self.wfx_grad += self.f_delta[i] * x.T
        #     self.wix_grad += self.i_delta[i] * x.T
        #     self.wox_grad += self.o_delta[i] * x.T
        #     self.wctx_grad += self.ct_delta[i] * x.T

    def init_delta(self):
        c_list=[]
        for i in range(self.times+1):
            c_list.append(np.zeros((self.state_num,1)))
        return c_list

    def update(self):
        debug=False
        if debug:
            tmp=(self.wi_grad!=np.zeros(self.wi_grad.shape))
            print('lstm.wi_grad:{},shape is {}'.format(tmp.astype(np.int32).sum(),tmp.shape))
            # print('lstm.wi_grad:{}'.format(self.wi_grad))
        self.wi-=self.learning_rate*self.wi_grad
        self.bi-=self.learning_rate*self.bi_grad
        self.wo-=self.learning_rate*self.wo_grad
        self.bo-=self.learning_rate*self.bo_grad
        self.wct-=self.learning_rate*self.wct_grad
        self.bct-=self.learning_rate*self.bct_grad
        self.wf-=self.learning_rate*self.wf_grad
        self.bf-=self.learning_rate*self.bf_grad
        self.wix -= self.learning_rate * self.wix_grad
        self.wox -= self.learning_rate * self.wox_grad
        self.wctx -= self.learning_rate * self.wctx_grad
        self.wfx -= self.learning_rate * self.wfx_grad

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_list()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_list()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_list()
        # 各个时刻的输入门i
        self.i_list = self.init_state_list()
        # 各个时刻的输出门o
        self.o_list = self.init_state_list()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_list()

    def mbackward(self,x,delta_array,activator):
        self.cal_delta(delta_array)
        self.cal_gradient()
        # self.update()