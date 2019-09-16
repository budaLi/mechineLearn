import numpy


class Beyesian:
    """
    p(A|B1, B2...) = P(B1|A)*P(B2|A)...*P(A)  /  P(B)
    """

    def __init__(self):
        self.dataSet, self.labels = self.createDataSet()
        self.wordList = self.createWordList()
        self.vector = self.createVector()

    def createDataSet(self):
        """
        创建数据集
        :return:
        """
        dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        labels = [0, 1, 0, 1, 0, 1]  # 0代表非侮辱性的

        # dataSet = [
        #     ['dapenti', "hushi"],
        #     ['dapenti', "nongfu"],
        #     ['touten', "jianzhugongren"],
        #     ['touten', "jianzhugongren"],
        #     ['dapenti', "jiaoshi"],
        #     ['touten', "jiaoshi"],
        # ]
        #
        # # 1 感冒  2 过敏  3 脑震荡
        # labels = [1, 2, 3, 1, 1, 3]

        return dataSet, labels

    def createWordList(self):
        """
        从数据集中构建词向量 生成词汇表
        :return:
        """
        wordList = []
        for data in self.dataSet:
            for one in data:
                if one not in wordList:
                    wordList.append(one)

        return wordList

    def createVector(self):
        """
        生成根据词汇表构建的向量矩阵
        :return:
        """
        vector = numpy.tile([0], (len(self.labels), len(self.wordList)))
        for i in range(len(self.dataSet)):
            for j in range(len(self.dataSet[i])):
                vector[i][self.wordList.index(self.dataSet[i][j])] = 1
        return vector

    def getPercentageByTemandLael(self, tem, label):
        """
        求出某种情况下某个特征发生的概率 比如在感冒的情况下打喷嚏发生的概率
        :param label:情况
        :return:特征
        """
        res = 0
        count = 0
        index = self.wordList.index(tem)
        for i in range(len(self.labels)):
            if self.labels[i] == label:
                count += 1
                res += self.vector[i][index]
        return round(float(res / count),3)

    def getPercentageByLabel(self, label):
        """
        求出某个情况发生的概率 如感冒发生的概率 在这里label是int类型
        :param label:
        :return:
        """
        return round(float(self.labels.count(label) / len(self.labels)),3)

    def getPercentageByTezheng(self, t):
        """
        求出某个特征发生的概率 如打喷嚏发生的概率
        :param t:
        :return:
        """
        index = self.wordList.index(t)
        count = 0
        for i in range(len(self.vector)):
            count += self.vector[i][index]

        return round(float(count / len(self.labels)),3)

    def main(self, inputWord, label):
        """
        求出在多种特征下某种情况发生的概率  如打喷嚏的建筑工人 感冒的概率
            P(感冒|打喷嚏x建筑工人)
        　　　　= P(打喷嚏|感冒) x P(建筑工人|感冒) x P(感冒)
        　　　　/ P(打喷嚏) x P(建筑工人)
        :param inputWord:
        :param label:
        :return:
        """
        percentage = 1.0

        # P(打喷嚏|感冒) x P(建筑工人|感冒)/ P(打喷嚏) x P(建筑工人)
        for one in inputWord:
            percentage *= self.getPercentageByTemandLael(one, label) / self.getPercentageByTezheng(one)

        # P(打喷嚏|感冒) x P(建筑工人|感冒) x P(感冒)
        percentage *= self.getPercentageByLabel(label)

        return round(percentage,3)

    def beiyesian(self,inputWord):
        """
        求出最大概率的情况
        :param inputWord:
        :return:
        """
        res= 0
        for label in self.labels:
            res = max(res,self.main(inputWord,label))
        return res


if __name__ == "__main__":
    beyesian = Beyesian()
    inputWord = ["dapenti","jianzhugongren"]
    inputWord2 = ['stupid', 'garbage']
    print("wordList", beyesian.wordList)
    print("vector", beyesian.vector)
    print("p1", beyesian.getPercentageByLabel(1))
    print("p2", beyesian.getPercentageByTezheng("my"))
    print("p3", beyesian.getPercentageByTemandLael("my", 1))
    print("res0",beyesian.main(inputWord2,0))
    print("res1",beyesian.main(inputWord2,1))
    print("s",beyesian.beiyesian(inputWord2))
    # print(beyesian.beiYeSian(beyesian.dataSet[1]))
    # print(0.66 * 0.33 * 0.5)
