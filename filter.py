import os
import imp
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix

def count_words(train_dir):
    emails_txt = [os.path.join(train_dir,file) for file in os.listdir(train_dir)]
    words = []
    for email in emails_txt:
        with open(email) as mail:
            for line in mail:
                words_line = line.split()
                words += words_line
    return Counter(words)

def clean_words(count_words):
    strange_words = []
    for word in list(count_words):
        if len(word) == 1:
            del count_words[word]
        # elif not word.isalpha():
        #     strange_words.append(word)
        #     del count_words[word]
    return count_words

def create_x_matrix(counter_of_words,columns, mail_dir):
    emails = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    x_matrix = np.zeros((len(emails), columns))
    email_ID = 0
    for k,file in enumerate(emails):
        print("{}% - Estou no {} de {} emails...".format(k/len(emails), k, len(emails), ))
        with open(file) as text:
            for i,line in enumerate(text):
                words = line.split()
                for word in words:
                    word_ID = 0
                    for i, key in enumerate(counter_of_words):
                        if word == key[0]: #key[0] = word || key[1] = count
                            word_ID = i
                            x_matrix[email_ID, word_ID] = words.count(word)
        email_ID += 1
    return x_matrix

def main():
    train_dir = "/home/ferragut/Workspace/SpamFilter/ling-spam/train-mails"
    test_dir = "/home/ferragut/Workspace/SpamFilter/ling-spam/test-mails"
    num = 5000

    counter_of_words = clean_words(count_words(train_dir)).most_common(num)
    train_matrix =  create_x_matrix(counter_of_words, num, train_dir)
    test_matrix = create_x_matrix(counter_of_words, num, test_dir)
    train_label_vector = np.zeros(702)
    train_label_vector[351:702] = 1
    test_label_vector = np.zeros(260)
    test_label_vector[130:260] = 1

    model = BernoulliNB()
    model2 = MultinomialNB()
    model.fit(train_matrix,train_label_vector)
    model2.fit(train_matrix,train_label_vector)
    result = model.predict(test_matrix)
    result2 = model2.predict(test_matrix)

    print(confusion_matrix(result, test_label_vector))
    print(confusion_matrix(result2, test_label_vector))
    


if __name__ == '__main__':
    main()