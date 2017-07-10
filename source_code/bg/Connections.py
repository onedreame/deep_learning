#coding:utf-8

class Connections(object):
    '''
    管理所有连接的类
    '''
    def __init__(self):
        self.connections=[]

    def add_connection(self,conn):
        '''
        添加连边
        :param conn: 要添加的连边
        '''
        self.connections.append(conn)

    def dump(self):
        '''
        打印保存的连边信息
        '''
        for conn in self.connections:
            print(conn)