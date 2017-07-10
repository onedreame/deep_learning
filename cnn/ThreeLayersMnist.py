# import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime

from cnn.Convlayer import ConvLayer
from cnn.Fullconnection import FullConnection
from cnn.MaxPoolingLayer import MaxPoolingLayer
from cnn.Sigmod import Relu, SigmoidActivator

LAYER_NODE=500
INPUT_NODE=784
OUTPUT_NODE=10
BATCH_SIZE=100
TRAINING_STEPS=300
CNN_LAYERS=2
FULLCONNECTION_LAYERS=2
debug=False
filename='log.txt'
def load(infile,tot1,tot2):
    train_count={}
    test_count={}
    with open(infile) as f:
        train_dataset=[];train_labels=[]
        test_dataset=[];test_labels=[]
        for i,line in enumerate(f.readlines()):
            # print(line)
            line=line.split(',')
            line=list(map(int,line))
            label = [0] * 10
            label[int(line[-1])] = 1
            if i<tot1:
                train_labels.append(label)
                train_count[line[-1]]=train_count.get(line[-1],0)+1
                train_dataset.append(line[:-1])
            elif i<tot1+tot2:
                test_dataset.append(line[:-1])
                test_count[line[-1]] = test_count.get(line[-1], 0) + 1
                test_labels.append(label)
            else:
                return train_count,test_count,np.array(train_dataset),np.array(train_labels)\
                    ,np.array(test_dataset),np.array(test_labels)

def train(train_file,test_file):
    mnist=input_data.read_data_sets('/tmp/data', one_hot=True)
    # train_count,test_count,train_data_set,train_labels,test_data_set,test_labels=load(train_file,50,3)
    # print('train_data_set:{},train_labels:{}'.format(train_data_set[0].shape,train_labels[0].shape))
    # print('test_data_set:{},test_labels:{}'.format(test_data_set[0].shape,test_labels[0].shape))
    # train_data_set,train_labels=mnist.validation.images.tolist(),mnist.validation.labels.tolist()
    # test_data_set,test_labels=mnist.test.images.tolist(),mnist.test.labels.tolist()
    train_data_set, train_labels = mnist.validation.images[:1000], mnist.validation.labels[:1000]
    train_count=count_lable(train_labels)
    test_data_set, test_labels= mnist.test.images[:150], mnist.test.labels[:150]
    test_count=count_lable(test_labels)
    print('train_count:\n',train_count,'\ntest_count:\n',test_count)
    if debug:
        print('train_data_set entry:{},train_lables entry:{}'.format(train_data_set[0].shape,train_labels[0].shape))
    layers=[]
    f=open(filename,'w+')
    cnn_layers=[32,64]
    layers.append(ConvLayer(28,28,1,3,32,0.01,1,1,Relu()))  #该步以后输入尺寸变为28*28*16
    layers.append(MaxPoolingLayer(28,28,32,2,2))        #该步以后输入尺寸变为14*14*16
    # layers.append(ConvLayer(14,14,16,3,32,0.1,1,1,Relu()))
    # layers.append(MaxPoolingLayer(14,14,32,2,2))    #该步以后变为7*7*64
    layers.append(FullConnection(32*14*14,10,Relu(),0.01))
    # layers.append(FullConnection(50,10,None,0.1))
    # layers.append(ConvLayer(7,7,32,7,10,0.1,0,1,None)) #该步以后就变成了onehot矩阵
    old_error_rate=1.0
    for i in range(TRAINING_STEPS):
        for k in range(len(train_data_set)):
            # print('begin {} example,time:{}'.format(k,datetime.now()))
            inputs=train_data_set[k].reshape((1,28,28))
            for j in range(len(layers)):
                if debug:
                    print('begin {} conv,inputs shape:'.format(j,inputs.shape))
                if layers[j].__class__.__name__=='FullConnection':
                    inputs=inputs.reshape((-1,1))
                layers[j].forward(inputs)
                inputs=layers[j].output
                f.write("第%d层\n" %(j))
                f.write("input:\n");f.write(str(layers[j].input_array))
                f.write("output:\n");f.write(str(layers[j].output))
            label=train_labels[k].reshape(inputs.shape)
            # print('before softmax,inputs:{}'.format(inputs))
            inputs=softmax(inputs)
            # print('after softmax,inputs:{}'.format(inputs))
            delta=(inputs-label)
            if layers[-1].__class__.__name__=='FullConnection':
                delta=delta.reshape((1,-1))
            # print('inputs:{},label:{},may invalid'.format(inputs,label))
            for j in range(len(layers)-1,-1,-1):
                if debug:
                    print('delta shape:{},layer:{}'.format(delta.shape,j))
                if layers[j].__class__.__name__=='ConvLayer': #如果不是第一层卷积层
                    layers[j].backward(delta,layers[j].activator)
                    delta=layers[j].delta_array
                elif layers[j].__class__.__name__=='FullConnection':
                    layers[j].backward(delta)
                    delta=layers[j].delta_array
                    if j>0 and layers[j-1].__class__.__name__!='FullConnection':
                        delta=delta.reshape(layers[j-1].output.shape)
                elif j>0:                                       #如果是第一层卷积层或者池化层
                    layers[j].backward(layers[j-1].output,delta)
                    delta=layers[j].delta_array
        if i%2==0: #and i:
            error_count=0
            for test in range(len(test_data_set)):
                inputs=test_data_set[test].reshape((1,28,28))
                for j in range(len(layers)):
                    if layers[j].__class__.__name__ == 'FullConnection':
                        # print('before reshape ,inputs is {}'.format(inputs))
                        inputs = inputs.reshape((-1, 1))
                        # print('after reshape,inputs is {}'.format(inputs))
                    layers[j].forward(inputs)
                    inputs=layers[j].output
                logit=inputs.reshape((-1)).argmax()
                # print('inputs is {},logit is {},label is {}'.format(inputs,logit,test_labels[test].argmax(axis=-1)))
                if logit!=test_labels[test].argmax(axis=-1):
                    error_count+=1
            print('now:{},iterations:{},error_ratio: {}'.format(datetime.now(),i,error_count/len(test_labels)))
            if error_count/len(test_labels)<0.05: break
            if error_count/len(test_labels)> old_error_rate:
                layers[-1].learning_rate*=0.9
                layers[0].learning_rate*=0.9
            old_error_rate=error_count/len(test_labels)

def softmax(inputs):
    return np.exp(inputs)/np.exp(inputs).sum()

def count_lable(inputs):
    count_table={}
    for item in inputs:
        count_table[item.argmax()]=count_table.get(item.argmax(),0)+1
    return count_table

if __name__=='__main__':
    train('/media/yk/d/projects/python/deep learning/cnn/mnist/train.format','/media/yk/d/projects/python/deep learning/cnn/mnist/test.format')
    # d,l=load('/media/wenzt/d/project/cnn/mnist/train.format',10)
    # print(d)
    # print(l)
