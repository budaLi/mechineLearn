import numpy
import matplotlib.pyplot as plt


class Knn:
    def __init__(self):
        pass

    def fileToMatrix(self, filename):
        """
        得到数据的特征矩阵及标签矩阵
        :param filename:
        :return:
        """
        with open(filename, "r") as f:
            length = len(f.readlines())
            resMat = numpy.zeros((length, 3))
            labelMat = []
            with open(filename, "r") as f:
                alldata = f.readlines()
                for index, data in enumerate(alldata):
                    data = data.strip().split("\t")
                    resMat[index:] = data[0:3]
                    labelMat.append(data[-1])

                return resMat, labelMat

    def drawpicture(self, resMat):
        fig = plt.figure()
        # 将画布且分为一行一列
        ax = fig.add_subplot(111)
        ax.scatter(resMat[:, 0], resMat[:, 1])
        plt.show()

    def autoNorm(self, dataSet):
        """
        将数据集归一化
        Y = （X - Xmin)/(Xmax - Xmin)  可以将数字特征变为0-1的区间

        min(0)  每一列的最小值
        min(1)  每一行的最小值 max同理

        np.shape(data)  返回data有几行几列 data.shape[0]  行数  data.shape[1]  列数
        tile(a,(m,n))  将矩阵a 延x,y两个方向叠加
        :param dataSet:
        :return:
        """
        minvals = dataSet.min(0)  # 由于有三列  返回结果为 [x1,x2,x3]这种形式 表示每一列的最小值
        maxvals = dataSet.max(0)

        ranges = maxvals - minvals  # 相当于公式中的Xmax -Xmin
        line = dataSet.shape[0]  # 得出数据集的行数 shape[1] 表示得出数据集的列数

        # （X - Xmin)/(Xmax - Xmin)  需要将矩阵通过tile转化成相同规模的矩阵
        normData = (dataSet - numpy.tile(minvals, (line, 1))) / (numpy.tile(ranges, (line, 1)))

        return normData

    def getKnn(self, testData, dataSet, labelSet, k):
        """
        knn 伪代码
        1.计算需要分类的数据与数据集之间的欧式距离 返回的结果为[x1,x2,x3]形式
        2.将距离从小到大排序
        3.选取前k个最短距离
        4.选取这k个数据中最多的分类类别 返回
        :param testData:
        :param dataSet:
        :param labelSet:
        :param k:
        :return:
        """
        m = dataSet.shape[0]  # 数据集的行数
        temData = dataSet - numpy.tile(testData, (m, 1))  # 先将测试数据和数据集做减法
        temData = temData ** 2  # 平方
        distince = temData.sum(axis=1)  # 对每一行数据求和  0表示按列求和 1表示按行求和
        distince = distince ** 0.5  # 开根后即为每一条数据与测试数据的欧式距离

        # 将数据从小到大排序  但是返回的是从小到大排序后的索引值 而不是具体的值
        sortData = distince.argsort()

        # 选取k个最短距离中分类最多的项
        classCount = {}
        for i in range(k):
            tem = labelSet[sortData[i]]
            if tem not in classCount:
                classCount[tem] = 1
            else:
                classCount[tem] += 1
        # 以values的大小进行逆向排序
        res = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
        return res[0][0]  # ('2', 97) 类似这种形式


if __name__ == "__main__":
    knn = Knn()
    resMat, labelMat = knn.fileToMatrix("data.txt")
    # normData = knn.autoNorm(resMat)

    testData = numpy.array([7000, 12.03333, 0.7])
    res = knn.getKnn(testData, resMat, labelMat, 20)

    knn.drawpicture(resMat)

    print(res)
