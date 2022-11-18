import numpy as np
from utils import process_text_list, build_freqs, loadData, process_text
from time import time


def main():
    t1 = time()
    all_pos, all_neg = loadData()

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

    while True:
        sentence = input("enter your sentence here:")
        if sentence == "00000":
            print("see you!")
            break
        sentence = process_text(sentence)

        N_pos, N_neg = 0, 0
        for word in sentence:
            if (word, 1) in freq:
                if (word, 0) in freq:
                    N_pos += freq[(word, 1)] / (freq[(word, 1)] + freq[(word, 0)])
                else:
                    N_pos += freq[(word, 1)]
            elif (word, 0) in freq:
                if (word, 1) in freq:
                    N_neg += freq[(word, 0)] / (freq[(word, 1)] + freq[(word, 0)])
                else:
                    N_neg += freq[(word, 0)]
        print(sentence)
        print(N_pos, '\t', N_neg)
        if N_pos > N_neg:
            print("Positive!")
        else:
            print("Negative!")


if __name__ == '__main__':
    main()
