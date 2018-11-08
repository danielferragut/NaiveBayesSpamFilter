# Naive Bayes Spam Filter
This is my first personal project related to Data Science/Machine Learning. Basically, it is a email spam filter using Naive Bayes. The program is written in Python and uses the scikit-learn library to implement Machine Learning.

 The filter.py program does a simple training over emails that are known to be spam or not("ham"), utilizing some basic text analysis, the program counts the words over all emails and trains a Naive Bayes model over these words and the amount of times they show in spams and hams.

 As of November 8 of 2018, the model needs to be adjusted a little more. Testing only over "ling-spam", the program predicts with an accuracy of ~60% using a Bernoulli based Naive Bayes method, which can certainly be improved on.
