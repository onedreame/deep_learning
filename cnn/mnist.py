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
    train_data_set, train_labels = mnist.validation.images[:800], mnist.validation.labels[:800]
    train_count=count_lable(train_labels)
    test_data_set, test_labels= mnist.test.images[:100], mnist.test.labels[:100]
    test_count=count_lable(test_labels)
    print('train_count:\n',train_count,'\ntest_count:\n',test_count)
    if debug:
        print('train_data_set entry:{},train_lables entry:{}'.format(train_data_set[0].shape,train_labels[0].shape))
    layers=[]
    f=open(filename,'w+')
    cnn_layers=[32,64]
    layers.append(ConvLayer(28,28,1,3,16,0.1,1,1,Relu()))  #该步以后输入尺寸变为28*28*32
    layers.append(MaxPoolingLayer(28,28,16,2,2))        #该步以后输入尺寸变为14*14*32
    layers.append(ConvLayer(14,14,16,3,32,0.1,1,1,Relu()))
    layers.append(MaxPoolingLayer(14,14,32,2,2))    #该步以后变为7*7*64
    layers.append(FullConnection(32*7*7,50,Relu(),0.1))
    layers.append(FullConnection(50,10,Relu(),0.1))
    # layers.append(ConvLayer(7,7,32,7,10,0.1,0,1,None)) #该步以后就变成了onehot矩阵
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
                if layers[j].__class__.__name__=='ConvLayer' and j>0: #如果不是第一层卷积层
                    layers[j].backward(delta,layers[j-2].activator)
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
            if error_count/len(test_labels)<0.05:break

    # for i in range(CNN_LAYERS):
    #     layers.append(ConvLayer(28,28,3,3,cnn_layers[i],0.1,1,cnn_layers[i]//32,SigmoidActivator()))
    # for i in range(CNN_LAYERS):
    #     layers[i].forward(train_data_set)
    # output=layers[-1].output.reshape((-1))
    # layers.append(FullConnection(output.shape[0],LAYER_NODE,SigmoidActivator()))
    # layers.append(FullConnection(LAYER_NODE,10,SigmoidActivator()))
    # layers.append(FullConnectedLayer())
    # for i in range(TRAINING_STEPS):
    #     for k in range(len(train_data_set)):
    #         inputs=train_data_set[i].reshape((1,28,28))
    #         for j in range(len(layers)):
    #             if layers[j].__class__.__name__=='FullConnection':
    #                 inputs=inputs.reshape((-1,1))
    #             layers[j].forward(inputs)
    #         if debug:
    #             print('total %d layers' %(len(layers)))
    #             print('the last class is %s' %(layers[-1].__class__.__name__))
    #         logits=layers[-1].ori_output
    #         logits=softmax(logits)
    #         delta=np.argmax(logits,axis=1)*np.log(np.array(train_labels[k])+0.1)
    #         for j in range(len(layers),-1,-1):
    #             if j+1<len(layers) and layers[j].__class__.__name__=='ConvLayer' and layers[j+1].__class__.__name__=='FullConnection':
    #                 derived_array=layers[j].padded_input_array.copy()
    #                 layers[j].element_wise_op(derived_array,layers[j].activator.backward)
    #                 delta=delta*derived_array.reshape((-1,1))
    #                 delta=delta.reshape((layers[j].filter_num,layers[j].filter_size,layers[j].filter_size))
    #             layers[j].backward(delta)
    #             delta=layers[j].delta_array
    #     if i %10==0 and i>0:
    #         error_count=0
    #         for k in range(len(test_data_set)):
    #             inputs=test_data_set[k]
    #             for j in range(len(layers)):
    #                 if layers[i].__class__.__name__ == 'FullConnection':
    #                     inputs = inputs.reshape((-1, 1))
    #                 layers[i].forward(inputs)
    #             logits = layers[-1].ori_output
    #             if np.argmax(logits,axis=1)!=test_labels[k]:
    #                 error_count+=1
    #         print("after %d iteration,error_ratio is %f" %(i,error_count/len(test_labels)))

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
