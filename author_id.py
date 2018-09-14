#!/usr/bin/python

""" 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

t0 = time()
classifier = GaussianNB()
classifier.fit (features_train, labels_train)
print "\n\nnaive bayes training time: " , round(time() - t0 , 3), "s"

t0 = time()
results = classifier.predict(features_test)
print "naive bayes prediction time: " , round(time() - t0 , 3), "s"

#accuracy

print "Number of predicted emails that Chris sent is (Using naive bayes): " , list(results).count(1)
print "Number of predicted emails that Sara sent is (Using naive bayes): " , list(results).count(0)
print "naive bayes results accuracy: " , accuracy_score(results , labels_test)

"""
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
features_train_svm = features_train[:len(features_train)/100]  #using 1% percent of the dataset to improve the performance
labels_train_svm = labels_train[:len(labels_train)/100]		   #using 1% percent of the dataset to improve the performance
t0 = time()
classifier = SVC(C = 10000.0 , kernel = "rbf")
classifier.fit(features_train_svm , labels_train_svm)
print "\n\nSVM training time(only 1% of the dataset): " , round(time() - t0 , 3), "s"

t0 = time()
results = classifier.predict(features_test)
print "SVM prediction time(only 1% of the dataset): " , round(time() - t0 , 3), "s"

#accuracy

print "Number of predicted emails that Chris sent is (Using SVM): " , list(results).count(1)
print "Number of predicted emails that Sara sent is (Using SVM): " , list(results).count(0)
print "SVM results accuray by using 1% of the dataset: " , accuracy_score(labels_test , results)

"""
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
t0 = time()
classifier = DecisionTreeClassifier(min_samples_split = 50)
classifier.fit (features_train , labels_train)
print "\n\nDecision Tree training time: " , round(time() - t0 , 3), "s"

t0 = time()
results = classifier.predict(features_test)
print "Decision Tree training time: " , round(time() - t0 , 3), "s"

#accuracy

print "Number of predicted emails that Chris sent is (Using Decision Tree): " , list(results).count(1)
print "Number of predicted emails that Sara sent is (Using Decision Tree): " , list(results).count(0)
print "Decision Tree results accuracy: " ,accuracy_score(labels_test , results)
