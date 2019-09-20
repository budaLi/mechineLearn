import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    def __init__(self):
        """
        逻辑回归虽然名字有回归 但是它其实是用来做分类的 其主要思想是：根据现有数据
        对分类边界线建立回归公式 以此进行分类

        假设现在有一些数据点 我们用一条直线对这些点进行拟合  这条线称为最佳拟合直线
        这个拟合的过程就叫做回归

        单位阶越函数 # return 1.0 / (1 + e的-z次幂)
        """
        path = r"C:\Users\lenovo\PycharmProjects\mechineLearn\logisticRegression\data"

        #默认情况下pd会将数据的第一行定为标题 在这里将header指定为空 通过names参数自定义标题
        self.data = pd.read_table(path,header=None,names=["Exam1",'Exam2',"Res"])

    def show_pic(self):
        """
        展示初始数据散点图
        :return:
        """
        #标记出数据集中的正例及负例  会将res为1 的数据筛选出来
        positive = self.data[self.data["Res"]==1]
        negative = self.data[self.data["Res"]==0]

        #指定画图大小  长宽
        fig,ax =plt.subplots(figsize = (10,5))

        ax.scatter(positive["Exam1"],positive["Exam2"],s=30,c="b",marker = "o",label = "positive")
        ax.scatter(negative["Exam1"],negative["Exam2"],s=30,c="r",marker = "x",label = "negative")
        ax.legend()
        ax.set_xlabel("Exam 1 Score")
        ax.set_ylabel("Exam 2 Score")

        #在这里要指定plt 使用fig会一闪就退出
        plt.show()

    def sigmoid(self,z):
        """
        sigmoid映射函数  将函数值映射到0-1之间
        :param z:
        :return:
        """
        return 1/(1+np.exp(-z))

    def show_sigmoid(self):
        """
        展示sigmoid函数的曲线
        :return:
        """
        nums = np.arange(-10,10)
        fig,ax = plt.subplots(figsize =(10,5))
        ax.plot(nums,self.sigmoid(nums),"r")
        plt.show()

    def prepare_data(self):
        #在数据的第一列插入1
        self.data.insert(0,"Ones",1)
        #将数据装换为矩阵形式
        self.data = np.array(self.data)
        #获取数据的列数
        col =  self.data.shape[1]
        #需要进行运算的数据
        data = self.data[:,:col-1]
        #数据的标签
        label = self.data[:,col-1:col]

        return data,label



if __name__=="__main__":
    logistic = LogisticRegression()
    logistic.show_pic()
    logistic.show_sigmoid()
    data,label = logistic.prepare_data()
    print(data)
    print(label)
