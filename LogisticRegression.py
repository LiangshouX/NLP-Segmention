import numpy as np
from utils import process_text_list, build_freqs, loadData
import time


def sigmoid(z):
    """
    将输入“z”映射到一个介于0到1之间的值，因此它可以被视为一个概率
    :param z: 可以是标量或者数组
    :return: sigmoid(z)
    """
    h = 1 / (1 + np.exp(-z))
    return h


def list_to_str(lst):
    return "".join(lst)


class LogisticRegression:
    def __init__(self, train_pos, train_neg, test_x, test_y, freqs, alpha=1e-7, num_epochs=10000):
        """

        :param train_pos: 积极情绪的训练集单词列表，已处理过stopword
        :param train_neg: 消极情绪的训练集单词列表
        :param test_x:
        :param test_y:
        :param alpha:     学习率
        :param num_epochs: 迭代次数
        :param freqs:      词频字典
        """
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_pos + train_neg
        self.train_y = np.mat(np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg)))))
        self.train_y = self.train_y.T
        self.alpha = alpha  # 学习率
        self.num_epochs = num_epochs
        self.freqs = freqs  # 存储每个(单词， 标签)的频率的字典
        self.theta = np.zeros((3, 1))  # (n+1，1)的权重向量
        self.b = 0.

    def feature_vectors(self, sentence_list):
        """

        :param sentence_list:    单个语句的单词列表
        :return: 单个语句对应的特征向量，一个1x3的向量
        """
        vector = np.zeros((1, 3))
        # 设置偏置项
        vector[0, 0] = 1

        for word in sentence_list:
            if isinstance(word, list):
                word = list_to_str(word)
            if (word, 1) in self.freqs:
                vector[0, 1] += self.freqs[(word, 1)]
            if (word, 0) in self.freqs:
                vector[0, 2] += self.freqs[(word, 0)]
        assert (vector.shape == (1, 3))
        return vector

    def feature_mat(self):
        """

        :return: mx3维的，所有训练集、测试集中的文本特征矩阵
        """
        m = len(self.train_x)
        mat_X = np.zeros((m, 3))
        for i in range(m):
            mat_X[i, :] = self.feature_vectors(self.train_x[i])
        return mat_X

    def train(self):
        m = len(self.train_x)
        J = 0.
        X = self.feature_mat()
        Y = self.train_y

        # print("theta.shape:", self.theta.shape, self.theta)  # theta:(3,1)
        for i in range(self.num_epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            J = -(1 / m) * (np.dot(np.transpose(Y), np.log(h)) +
                            np.dot(np.transpose(1 - Y), np.log(1 - h)))

            # print("z.shape ", z.shape, '\n', "h.shape ", h.shape, '\n',
            #       "X.shape", X.shape, '\n', "self.theta.shape ", self.theta.shape, '\n',
            #       "h.shape", h.shape, '\n', "Y.shape", Y.shape, '\n',
            #       "(h-Y).shape", (h - Y).shape, '\n')

            self.theta = self.theta - (self.alpha / m) * (np.dot(X.T, (h - Y)))

            if i % 100 == 0:
                # 每训练100轮打印一次
                print("epoch: %2d \t test acc: %.6f \t loss:%.6f..." % (i, self.test_logistic_regression(), J))
        # print(self.theta)
        return self.test_logistic_regression()

    def predict(self, sentence):
        sentence = process_text_list(sentence)
        # print(sentence)
        feature_vector = self.feature_vectors(sentence)
        y_pred = sigmoid(np.dot(feature_vector, self.theta))  # 居然是个二维的列表？有点奇怪
        # print(y_pred[0, 0])
        return y_pred[0, 0] + self.b

    def test_logistic_regression(self):
        """

        :return:
        """
        # print("Start to test on test Set...")
        y_hat = []
        for text in self.test_x:
            y_pred = self.predict(text)
            if y_pred > 0.5:
                y_hat.append(1)
            else:
                y_hat.append(0)
        y_hat = np.array(y_hat)
        test_y = np.squeeze(self.test_y)
        count = 0

        for i in range(len(test_y)):
            if y_hat[i] == test_y[i]:
                count += 1
            # if i % 100 == 0:
            #     accuracy = count / len(test_y)
            #     print("epoch %2d  acc:%.5f" % (i, accuracy))
        accuracy = count / len(test_y)
        # print("Predict over! Accuracy = %.5f" % accuracy)
        return accuracy


def main():
    t1 = time.time()
    all_pos, all_neg = loadData()
    freq1 = build_freqs(all_pos, np.ones((len(all_pos), 1)))
    freq2 = build_freqs(all_neg, np.zeros((len(all_neg), 1)))
    freq1.update(freq2)

    train_pos = all_pos[:800]
    train_neg = all_neg[:800]
    test_pos = all_pos[801:-1]
    test_neg = all_neg[801:-1]
    test_x = test_pos + test_neg
    test_y = np.hstack((np.ones(len(test_pos)) + np.zeros(len(test_neg))))
    # print(len(test_x))
    # print(len(test_y))

    model = LogisticRegression(train_pos, train_neg, test_x, test_y, freq1)

    t2 = time.time()

    test_acc = model.train()
    t3 = time.time()
    print("测试准确率：%.6f, \t训练用时：%.3f" % (test_acc, t3 - t2))
    # model.test_logistic_regression(all_pos + all_neg)

    while 1:
        sentence = input("\ntest over，yours(00000 to exit):")
        if sentence == '00000':
            print("see you!")
            break
        res = model.predict(sentence)
        print(res, end='\t')
        if res > 0.5:
            print("Positive!")
        else:
            print("Negative!")


if __name__ == '__main__':
    main()
