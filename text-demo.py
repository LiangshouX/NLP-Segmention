import jieba
from utils import process_text, loadData, process_text_list
import  numpy as np

a = "你是谁？你为什么 在这里\n"
b = "我是你的二大爷,你个小兔崽子，去死吧你"
print(jieba.lcut(a))
print(jieba.lcut(b))
c = [a, b]
print("============================")
# print(process_text_list([a,b]))
a = process_text(a)
b = process_text(b)

print(b)


e, d = loadData()
# e = np.array(e)
print(e)