from utils import loadData, process_text_list, build_freqs, lookup
from LogisticRegression import LogisticRegression
import numpy as np
from time import time


class NaiveBayes(LogisticRegression):
    def __init__(self, train_pos, train_neg, test_x, test_y, freqs, alpha=1e-9, num_epochs=100000):
        """

        :param train_pos: 积极情绪的训练集单词列表，已处理过stopword
        :param train_neg: 消极情绪的训练集单词列表
        :param test_pos:
        :param test_neg:
        :param alpha:     学习率
        :param num_epochs: 迭代次数
        :param freqs:      词频字典
        """
        super().__init__(train_pos, train_neg, test_x, test_y, freqs, alpha, num_epochs)
        self.likelihood = {}
        self.logprior = 0

    def train(self):
        """

        :return:
            logprior:先验概率的对数值
            loglikelihood:贝叶斯方程的对数似然
        """
        p_w_pos_sum, p_w_neg_sum = 0, 0

        # print("train start...\n\n", self.train_y, '\n', len(self.train_y), '\n', len(self.train_x))
        # 计算词汇表中唯一单词的数量V
        vocab = set([pair[0] for pair in self.freqs.keys()])
        V = len(vocab)

        # 计算N_pos和N_neg
        N_pos, N_neg = 0, 0
        for pair in self.freqs.keys():
            if pair[1] > 0:
                N_pos += self.freqs[pair]
            else:
                N_neg += self.freqs[pair]
        D = len(self.train_y)
        D_pos = sum(self.train_y)[0, 0]

        D_neg = D - D_pos
        # print(D, type(D_pos), D_neg)

        self.logprior = np.log(D_pos) - np.log(D_neg)

        for word in vocab:
            freq_pos = lookup(self.freqs, word, 1)
            freq_neg = lookup(self.freqs, word, 0)

            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            p_w_pos_sum += p_w_pos
            p_w_neg_sum += p_w_neg

            self.likelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)
        # print(self.logprior, '\t', self.likelihood)
        print("Train over!\n")

        return self.logprior, self.likelihood

    def predict(self, sentence):
        word1 = process_text_list(sentence)
        # 概率
        p = 0
        p += self.logprior
        for word in word1:
            if isinstance(word, list):
                word = "".join(word)
            if word in self.likelihood:
                p += self.likelihood.get(word)
        return p

    def test_naive_bayes(self):
        """

        :return:
        """
        accuracy, value = 0, 0
        y_hat = []
        for text in self.test_x:
            if self.predict(text) > 0:
                y_hat_i = 1
            else:
                y_hat_i = 0
            y_hat.append(y_hat_i)
        y_hat = np.array(y_hat)
        # print(len(y_hat))       # 396

        test_y = np.squeeze(self.test_y)
        print(len(test_y))      # 198

        for i in range(len(test_y)):
            error = np.abs(y_hat[i] - test_y[i]) / len(y_hat)
            value = value + error
            if i % 10 == 0 or i == len(y_hat):
                accuracy = 1 - value
                print("epoch %2d  acc:%.5f" % (i, accuracy))
        accuracy = 1 - value
        return accuracy


def main():
    t1 = time()
    all_pos, all_neg = loadData()
    freq1 = build_freqs(all_pos, np.ones((len(all_pos), 1)))
    freq2 = build_freqs(all_neg, np.zeros((len(all_neg), 1)))
    freq1.update(freq2)

    freq = build_freqs(all_pos + all_neg, np.vstack((
        np.ones((len(all_pos), 1)),
        np.zeros((len(all_neg), 1)))
    ))
    train_pos = all_pos[:800]
    train_neg = all_neg[:800]
    test_pos = all_pos[801:-1]
    test_neg = all_neg[801:-1]
    test_x = test_pos + test_neg
    test_y = np.hstack((np.ones(len(test_pos)) + np.zeros(len(test_neg))))

    model = NaiveBayes(train_pos, train_neg, test_x, test_y, freq)
    # print(model.predict("我是谁我在那我要干什么"))
    # print(freq1)

    a, b = model.train()
    # print(a, b)

    t3 = time()
    print("训练用时：%.3f" % (t3 - t1))
    model.test_naive_bayes()

    while 1:
        sentence = input("\ntest over，yours(00000 to exit):")
        if sentence == '00000':
            print("see you!")
            break
        res = model.predict(sentence)
        print(res, end='\t')
        if 1 - res > 0:
            print("Positive!")
        else:
            print("Negative!")


if __name__ == '__main__':
    main()
