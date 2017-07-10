#coding:utf-8
from bg import *
import struct
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# from bg.Network import Network
from bg.FullConnection import Network

class Loader(object):
    def __init__(self,path,count):
        '''
        初始化加载器
        :param path:数据文件路径
        :param count: 文件中样本个数
        '''
        self.path=path
        self.count=count

    def get_file_content(self):
        '''
        获得输入的数据
        :return: 指定文件内容
        '''
        f=open(self.path,'rb')
        content=f.readlines()
        f.close()
        return content

    def to_int(self,byte):
        '''
        将unsigned byte字符转化为整数
        '''
        print('byte is ',byte)
        return struct.unpack('B',byte)[0]

class ImageLoader(Loader):
    def get_picture(self,content,index):
        '''
        内部函数，从文件中获取图像
        '''
        start=index*28*28#+16
        picture=[]
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start+i*28+j]))
        return picture

    def get_one_sample(self,picture):
        '''
        将图片转化为向量
        '''
        sample=[]
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获取全部样本的输入向量
        '''
        content=self.get_file_content()
        print('content len is ',len(content))
        print('content[0] is ',len(content[0]))
        data_set=[]
        for index in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content,index)))
        return data_set

class LableLoader(Loader):
    def load(self):
        '''
        加载数据文件，获取全部样本的标签向量
        '''
        content=self.get_file_content()
        labels=[]
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels
    def norm(self,label):
        '''
        one-hot处理标签
        '''
        label_vec=[]
        label_value=self.to_int(label)
        for i in range(10):
            if label_value!=i:
                label_vec.append(0.1)
            else:
                label_vec.append(0.9)
        return label_vec

def get_training_data_set():
    '''
    加载训练数据
    '''
    input_data.read_data_sets('/tmp/data',one_hot=True)
    image_loader=ImageLoader('/tmp/data/train-images.idx3-ubyte',60000)
    label_loader=LableLoader('/tmp/data/train-labels.idx1-ubyte',60000)
    return image_loader.load(),label_loader.load()

def get_test_data_set():
    '''
    加载测试数据
    '''
    input_data.read_data_sets('/tmp/data',one_hot=True)
    image_loader=ImageLoader('t10k-images.idx3-ubyte',10000)
    label_loader=LableLoader('t10k-labels.idx1-ubyte',10000)
    return image_loader.load(),label_loader.load()

def get_result(label):
    '''
    获得最大概率的索引
    '''
    return np.argmax(label)

def evaluate(network,test_data_set,test_labels):
    error=0
    total=len(test_data_set)
    for i in range(total):
        label=get_result(test_labels[i])
        predict=get_result(network.predict(test_data_set[i]))
        if label!=predict:
            error+=1

    return float(error)/float(total)
def train_and_evaluate():
    last_error_ratio=1.0
    epoch=0
    # train_data_set,train_labels=get_training_data_set()
    # test_data_set,test_labels=get_test_data_set()
    mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
    # train_data_set,train_labels=mnist.validation.images.tolist(),mnist.validation.labels.tolist()
    # test_data_set,test_labels=mnist.test.images.tolist(),mnist.test.labels.tolist()
    train_data_set, train_labels = mnist.validation.images, mnist.validation.labels
    test_data_set, test_labels = mnist.test.images, mnist.test.labels
    network= Network([784, 300, 10])
    print('begin train')
    while True:
        epoch+=1
        network.train(train_labels,train_data_set,-0.003,1)
        print('%s epoch %d finished' %(datetime.now(),epoch))
        if epoch%10==0:
            error_ratio=evaluate(network,test_data_set,test_labels)
            print('%s after epoch %d,error ratio is %f' %(datetime.now(),epoch,error_ratio))
            if error_ratio>last_error_ratio:
                break
            else:
                last_error_ratio=error_ratio

if __name__=='__main__':
    train_and_evaluate()
