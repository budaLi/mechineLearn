import math


class DecisionTree:
    def __init__(self):
        pass

    def createDataSet(self):
        """
        创建数据集
        :return:
        """
        dataSet = [[1, 1, 'y'],
                   [1, 1, 'y'],
                   [1, 0, 'n'],
                   [0, 1, 'n'],
                   [0, 1, 'n']]
        labels = ["surfacing without water ", "has flippers"]

        return dataSet, labels

    def calShannonEnt(self, dataSet):
        """
        计算给定的数据集的香农熵
        :param dataSet:
        :return:
        """
        length = len(dataSet)

        # 存储数据集中类别出现的次数
        labeldic = {}
        for one in dataSet:
            if one[-1] not in labeldic:
                labeldic[one[-1]] = 1
            else:
                labeldic[one[-1]] += 1


        # 香农熵之和
        shannonEnt = 0.0
        for value in labeldic.values():
            percentage = float(value / length)
            shannonEnt -= percentage * math.log2(percentage)

        return shannonEnt




if __name__ == "__main__":
    de = DecisionTree()
    dataSet,label = de.createDataSet()
    res = de.calShannonEnt(dataSet)
    print(res)
