#朴素贝叶斯算法假设所有特征的出现相互独立互不影响，每一特征同等重要，
#又因为其简单，而且具有很好的可解释性一般。相对于其他精心设计的更复杂的分类算法，
#朴素贝叶斯分类算法是学习效率和分类效果较好的分类器之一。
#朴素贝叶斯算法一般应用在文本分类，垃圾邮件的分类，信用评估，钓鱼网站检测等。

# 打开数据集，获取邮件内容，
# spam为垃圾邮件，ham为正常邮件
import numpy as np
import random
from gensim.models import Word2Vec
from pandas.io.parsers import TextParser as textParse

def loadData():
    # 选取一部分邮件作为测试集
    testIndex = random.sample(range(1, 25), 5)

    dict_word_temp = []

    testList = []
    trainList = []
    testLabel = []
    trainLabel = []

    for i in range(1, 26):
        wordListSpam = textParse(open('./email/spam/%d.txt' % i, 'r').read())
        wordListHam = textParse(open('./email/ham/%d.txt' % i, 'r').read())
        dict_word_temp = dict_word_temp + wordListSpam + wordListHam

        if i in testIndex:
            testList.append(wordListSpam)
            # 用1表示垃圾邮件
            testLabel.append(1)
            testList.append(wordListHam)
            # 用0表示正常邮件
            testLabel.append(0)
        else:
            trainList.append(wordListSpam)
            # 用1表示垃圾邮件
            trainLabel.append(1)
            trainList.append(wordListHam)
            # 用0表示正常邮件
            trainLabel.append(0)

    # 去重得到词字典
    dict_word = list(set(dict_word_temp))
    trainData = tranWordVec(dict_word, trainList)
    testData = tranWordVec(dict_word, testList)

    return trainData, trainLabel, testData, testLabel

def train(trainData, trainLabel):
    trainMatrix = np.array(trainData)

    # 计算训练的文档数目
    trainNum = len(trainMatrix)
    # 计算每篇文档的词条数
    wordNum = len(trainMatrix[0])
    # 文档属于垃圾邮件类的概率
    ori_auc = sum(trainLabel) / float(trainNum)

    # 拉普拉斯平滑
    # 分子+1
    HamNum = np.ones(wordNum)
    SpamNum = np.ones(wordNum)
    # 分母+2
    HamDenom = 2.0
    SpamDenom = 2.0

    for i in range(trainNum):
        # 统计属于垃圾邮件的条件概率所需的数据，即P(x0|y=1),P(x1|y=1),P(x2|y=1)···
        if trainLabel[i] == 1:
            SpamNum += trainMatrix[i]
            SpamDenom += sum(trainMatrix[i])
        else:
            # 统计属于正常邮件的条件概率所需的数据，即P(x0|y=0),P(x1|y=0),P(x2|y=0)···
            HamNum += trainMatrix[i]
            HamDenom += sum(trainMatrix[i])
    # 取对数，防止下溢出
    SpamVec = np.log(SpamNum / SpamDenom)
    HamVec = np.log(HamNum / HamDenom)
    # 返回属于正常邮件类的条件概率数组，属于垃圾邮件类的条件概率数组，文档属于垃圾邮件类的概率
    return HamVec, SpamVec, ori_auc

# 预测函数
def predict(testDataVec, HamVec, SpamVec, ori_auc):
    predictToSpam = sum(testDataVec * SpamVec) + np.log(ori_auc)
    predictToHam = sum(testDataVec * HamVec) + np.log(1.0 - ori_auc)
    if predictToSpam > predictToHam:
        return 1
    else:
        return 0