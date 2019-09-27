# @Time    : 2019/9/23 9:37
# @Author  : Libuda
# @FileName: linerregression.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class  LinerRegreeeion:
    def __init__(self):
        self.path=r"C:\Users\lenovo\PycharmProjects\mechineLearn\linerRegression\data"
        self.data = self.loadData(self.path)
    def loadData(self,path):
        """
        加载数据
        :param path:文件目录
        :return:
        """
        data = pd.read_table(path,header=None,names=["one","two","res"])
        return data

    def showDataPic(self):
        """
        展示原始数据
        :return:
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.data['two'],self.data['res'],s=30,c="b",marker = "o")
        plt.show()


if __name__=="__main__":
    Line = LinerRegreeeion()
    Line.showDataPic()
