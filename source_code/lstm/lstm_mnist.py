# import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from datetime import datetime

from cnn.Fullconnection import FullConnection
from cnn.Sigmod import Relu
from lstm.LstmLayer import LstmLayer

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
    train_data_set, train_labels = mnist.validation.images[:100], mnist.validation.labels[:100]
    train_count=count_lable(train_labels)
    test_data_set, test_labels= mnist.test.images[:10], mnist.test.labels[:10]
    test_count=count_lable(test_labels)
    print('train_count:\n',train_count,'\ntest_count:\n',test_count)
    if debug:
        print('train_data_set entry:{},train_lables entry:{}'.format(train_data_set[0].shape,train_labels[0].shape))
    layers=[]
    f=open(filename,'w+')
    layers.append(LstmLayer(28,32,0.01))
    layers.append(FullConnection(32*28,50,Relu(),0.001))
    layers.append(FullConnection(50,10,Relu(),0.001))
    for i in range(TRAINING_STEPS):
        for k in range(len(train_data_set)):
            # print('begin {} example,time:{}'.format(k,datetime.now()))
            # layers[0].f.write('begin {} example,time:{}'.format(k,datetime.now()))
            inputs=train_data_set[k].reshape((28,28,1))
            layers[0].reset_state()
            for batch in range(inputs.shape[0]):
                # layers[0].f.write('总批次为:%d,当前批次为:%d' %(len(range(inputs.shape[0])),batch))
                layers[0].forward(inputs[batch])
            # inputs=layers[0].h_list[-1]
            inputs=np.concatenate(layers[0].h_list[1:]).reshape((-1,1))
            for j in range(1,len(layers)):
                layers[j].forward(inputs)
                inputs=layers[j].output
            # f.write("第%d层\n" %(j))
            # f.write("input:\n");f.write(str(layers[j].input_array))
            # f.write("output:\n");f.write(str(layers[j].output))
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
                layers[j].backward(delta)
                if j:
                    delta=layers[j].delta_array
                    if layers[j-1].__class__.__name__=='LstmLayer':
                        # print('delta[-32:]:{},delta:{}'.format(delta[:,-32:].shape,delta.shape))
                        delta=delta[:,-32:].reshape(layers[j-1].h_list[-1].shape)
        if i%2==0: #and i:
            error_count=0
            for test in range(len(test_data_set)):
                inputs=test_data_set[test].reshape((28,28,1))
                layers[0].reset_state()
                for batch in range(inputs.shape[0]):
                    layers[0].forward(inputs[batch])
                # inputs = layers[0].h_list[-1]
                inputs = np.concatenate(layers[0].h_list[1:]).reshape((-1, 1))
                for j in range(1, len(layers)):
                    layers[j].forward(inputs)
                    inputs = layers[j].output
                logit=inputs.reshape((-1)).argmax()
                print('inputs is {},logit is {},label is {}'.format(inputs,logit,test_labels[test].argmax(axis=-1)))
                if logit!=test_labels[test].argmax(axis=-1):
                    error_count+=1
            print('now:{},iterations:{},error_ratio: {}'.format(datetime.now(),i,error_count/len(test_labels)))
            if error_count/len(test_labels)<0.05:break

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
