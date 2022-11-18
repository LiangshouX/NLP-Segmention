
import numpy as np
import jieba

f = open("./data/vocabulary/stopwords_utf8.txt", mode='r', encoding='utf-8')
stopwords_list = f.read().splitlines()
stopwords_list.append('\n')
stopwords_list.append("")
stopwords_list.append(" ")


def loadData():
    """

    :return:
    """
    all_positive_comment, all_negative_comment = [], []

    for i in range(1000):
        f_pos = open("./data/positive/pos." + str(i) + ".txt", mode='r', encoding='utf-8')
        all_positive_comment.append(f_pos.read())
        # all_positive_comment[i] = process_text(all_positive_comment[i])
        f_pos.close()
    for i in range(1000):
        f_neg = open("./data/negative/neg." + str(i) + ".txt", mode='r', encoding='utf-8')
        all_negative_comment.append(f_neg.read())
        # all_negative_comment[i] = process_text(all_negative_comment[i])
        f_neg.close()

    all_positive_comment = process_text_list(all_positive_comment)

    all_negative_comment = process_text_list(all_negative_comment)
    return all_positive_comment, all_negative_comment


def process_text_list(text_list):
    """

    :param text_list: 包含多条语句的列表
    :return:
    """
    all_text_clean = []
    for text in text_list:
        text = process_text(text)   # 一个单词列表
        all_text_clean.append(text)
    return all_text_clean


def process_text(text):
    """
    :param text: 一条评论
    :return: 处理过stopword的单词列表
    """
    text_clean = []
    text = jieba.lcut(text)
    for word in text:
        if word not in stopwords_list:
            text_clean.append(word)
    return text_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.

    yslist = np.squeeze(ys).tolist()
    # print(yslist)

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_text_list(tweet):
            # print("word:", "".join(word))
            if isinstance(word, list):
                word = "".join(word)
            pair = (word, y)
            # print(pair, freqs)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    # freqs.update(pre_freq())

    return freqs


def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n


def pre_freq():
    freq = {}
    fi = open("./data/vocabulary/full_pos_dict_sougou.txt", mode='r', encoding='utf-8')
    pos_list = fi.read().splitlines()
    fi.close()
    # print(pos_dic)
    y_pos = np.ones(len(pos_list))

    for word, y in zip(pos_list, y_pos):
        pair = (word, y)
        if pair in freq:
            freq[pair] += 1
        else:
            freq[pair] = 1

    freq_temp = {}
    fi = open("./data/vocabulary/full_neg_dict_sougou.txt", mode='r', encoding='utf-8')
    neg_list = fi.read().splitlines()
    fi.close()
    y_neg = np.ones(len(neg_list))
    for word, y in zip(neg_list, y_neg):
        pair = (word, y)
        if pair in freq_temp:
            freq_temp[pair] += 1
        else:
            freq_temp[pair] = 1
    freq.update(freq_temp)
    return freq


