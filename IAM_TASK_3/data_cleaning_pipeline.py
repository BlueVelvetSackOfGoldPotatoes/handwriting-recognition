import numpy as np
import json
import collections
import re
import pandas as pd
from matplotlib import pyplot as plt

"""
Count classe representation. Verify if label is correct?
"""

def main():
    with open('data/IAM-data/dic/sentence_no_processing_dictionary.json', 'r') as f:
        dic = json.load(f)

    sentences = list(dic.values())
    word_labels = []
    singular_words = []

    for words in sentences:
        # print(words)
        sep_word = [c.strip() for c in re.split('(\W+)', words) if c.strip() != '']
        for word in sep_word:
            if word not in singular_words:
                singular_words.append(word)
            # print(f"Adding {word} to list --------")
            word_labels.append(word)

    counts = pd.Series(word_labels).value_counts().sort_values(ascending=False)
    # print(counts)
    # print(type(counts))
    
    counts_above_50 = counts[counts > 50]
    counts_above_50 = counts_above_50.to_dict()

    counts_dic = counts.to_dict()

    with open("class_analysis.txt", "w") as f:
        f.write("Each token word is a class, that is, all instances of 'the' are the same class, namely the class 'the'.\n\n(1) Total number of classes: " + str(len(counts.keys())) + "\n(2) Total number of classes that have more than 50 occurences: " + str(len(counts_above_50.keys())))
        f.write("\nAll class and their occurrences:\n")

        for key, value in zip(counts_dic.keys(), counts_dic.values()):
            f.write(key + " : " + str(value) + "\n---------------\n")

    # plt.hist(counts_above_50)
    plt.bar(counts_above_50.keys(), counts_above_50.values(), edgecolor='black', color='b')
    plt.title('Histogram of Class distribution')
    plt.xlabel('Tokens')
    plt.ylim(0,max(counts_above_50.values()))
    plt.xticks(range(len(counts_above_50.keys())), counts_above_50.keys(), size='small', rotation = 'vertical')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    main()