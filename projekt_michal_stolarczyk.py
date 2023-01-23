# -*- coding: utf-8 -*-
"""
Created on Tue Jan 3 19:09:10 2023

@author: michal
"""
"""kNN with leave-one-out model"""

import math
import matplotlib.colors as mcolors
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from tabulate import tabulate


DATA_PATH = r"heart_disease.dat"
matrix = []
matrix_decision_class = []

nr_pos_class = 0
nr_neg_class = 0

mean_fpr = np.linspace(0, 1, 1000)

positive_class = None
negative_class = None

tp_CV = 0
fn_CV = 0
fp_CV = 0
tn_CV = 0
no_p_CV = 0
no_n_CV = 0
roc_CV = []
tpr_CV = []

with open(DATA_PATH, "r") as file:
    for row in file:
        row = row.rstrip()
        tmp_row = []
        tmp_decission = []
        separator = " "
        sep_row = row.split(separator)
        count = 1
        for attrib_no in range(0,len(sep_row)):
            if count == len(sep_row):
                matrix_decision_class.append(int(sep_row[attrib_no]))
            else:
                try:
                    if(attrib_no==len(sep_row)-1):
                        tmp_row.append(int(sep_row[attrib_no]))
                    else:
                        tmp_row.append(float(sep_row[attrib_no]))
                except:
                    if sep_row[attrib_no] == " ":
                        tmp_row.append(None)
                    else:
                        tmp_row.append(str(sep_row[attrib_no]))
                count +=1
            
        matrix.append(tmp_row)  

# print("________________________________")
# nr_objects = len(matrix)
# print("nr of objects: {}".format(nr_objects))
# print("matrix_decision_class: {}".format(matrix_decision_class))

def split_data(matrix,matrix_decision_class ):
    global positive_class
    global negative_class
    row=0
    column=-1
    matrix = np.array(matrix)
    matrix_decision_class = np.array(matrix_decision_class)
    print("________________________________")
    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(matrix)
    print("Number of splits: {}".format(n_splits))
    print("================================")   
    for train_index, test_index in loo.split(matrix):
        column=column+1
        if(column==5):
            row=row+1
            column=0
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = matrix[train_index], matrix[test_index]
        y_train, y_test = matrix_decision_class[train_index], matrix_decision_class[test_index]
        kNN_classifier(y_train, y_test,  x_train, x_test, column, row, matrix_decision_class)  
        roc_score = float(sum(roc_CV))
        try:
            roc_score = round(roc_score/len(roc_CV),5)
        except:
            roc_score = 0       
    
    return y_train, y_test,  x_train, x_test

def learn_model(data,no_p,no_n):
    print("________________________________")
    confussion_matrix = data
    #print(confussion_matrix)
    t_p =confussion_matrix[0][0]
    t_n =confussion_matrix[1][1]
    f_p =confussion_matrix[1][0]
    f_n =confussion_matrix[0][1]
    if (t_p + f_p) > (t_n + f_n):
        t_p = confussion_matrix[1][1]
        t_n = confussion_matrix[0][0]
        f_p = confussion_matrix[1][0]
        f_n = confussion_matrix[0][1]
    print("================================")
    print("Confusion matrix: ")
    print("True positive: {}".format(t_p))
    print("True negative: {}".format(t_n))
    print("False positive: {}".format(f_p))
    print("False negative: {}".format(f_n))

    try:
        precision = round(t_p / (t_p + f_n), 5) #precision = metrics.precision_score()
    except:
        precision = float(1)

    try:
        specifity = round( t_n / (t_n + f_p), 5)
    except:
        specifity = float(1)

    try:
        total_acc = round((t_n + t_p) / (t_p + f_n + t_n + f_p), 5)
    except:
        total_acc = float(1)

    try:
        balance_acc = round((specifity + precision)/2, 5)
    except:
        balance_acc = float(0)

    try:
        recall = round(t_p / (t_p + f_p), 5)
    except:
        recall = float(1)

    try:
        true_neg_rate = round(t_n / (t_n + f_n), 5)
    except:
        true_neg_rate = float(0)

    try:
        cover_pos = round((t_p + f_n) / no_p, 5)
    except:
        cover_pos = float(1)

    try:
        cover_neg = round((t_n + f_p) / no_n, 5)
    except:
        cover_neg = float(1)

    try:
        total_cover = round((t_p + f_n + t_n + f_p) / (no_p + no_n), 5)
    except:
        total_cover = float(1)

    try:
        f1_score = round((precision * recall * 2) / (recall + precision), 5)
    except:
        f1_score = float(0)

    try:
        g_mean = round(math.sqrt(recall * specifity), 5)
    except:
        g_mean = float(0)
    print("================================")
    print("Precision:           {}".format(precision))
    print("Specifity:           {}".format(specifity))
    print("Total accuracy:      {}".format(total_acc))
    print("Balance accuracy:    {}".format(balance_acc))
    print("recall:              {}".format(recall))
    print("True negative rate:  {}".format(true_neg_rate))
    print("Coverage positive:   {}".format(cover_pos))
    print("Coverage negative:   {}".format(cover_neg))
    print("Total coverage:      {}".format(total_cover))
    print("F1 score:            {}".format(f1_score))
    print("================================")
    return t_p, t_n, f_p, f_n, precision, specifity, total_acc, balance_acc, recall, true_neg_rate, cover_neg, cover_pos, total_cover, f1_score, g_mean

def kNN_classifier(y_train, y_test,  x_train, x_test, column, row, matrix_decision_class):
    global positive_class
    global negative_class
    print("________________________________")
    for i in range(1, 2):
        classifier = KNeighborsClassifier(metric="euclidean", n_neighbors=i)
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        no_1 = 0
        no_2 = 0
            
        for ele in y_test:
            if ele==matrix_decision_class[0]:
                no_1 +=1
            else:
                no_2 +=1

        if(no_1>=no_2):
            positive_class = matrix_decision_class[1]
            negative_class = matrix_decision_class[0]
        else:
            positive_class = matrix_decision_class[0]
            negative_class = matrix_decision_class[1]

        no_positive_class = 0
        no_negative_class = 0
        
        for ele in y_test:
            if ele==positive_class:
                no_positive_class +=1
            else:
                no_negative_class +=1
        print(f"y_test: {y_test}")
        print(f"y_pred: {y_pred}")
        # cm = metrics.confusion_matrix(y_test, y_pred)
        cm = [[0,0],[0,0]]
        no_p = 0
        no_n = 0
        if y_test == positive_class & y_pred == positive_class:
            cm[0][0] = 1
            no_p += 1
        elif y_test == negative_class & y_pred == negative_class:
            cm[1][1] = 1
            no_n +=1
        elif y_test == positive_class & y_pred == negative_class:
            cm[0][1] = 1
            no_p += 1
        else:
            cm[1][0] = 1
            no_n +=1
        t_p, t_n, f_p, f_n, precision, specifity, total_acc, balance_acc, recall, true_neg_rate, cover_neg, cover_pos, total_cover, f1_score, g_mean = learn_model(cm,no_p,no_n)

        
        global tp_CV
        global fn_CV
        global fp_CV
        global tn_CV
        global no_p_CV
        global no_n_CV
        tp_CV += t_p
        fn_CV+=f_n
        fp_CV+=f_p
        tn_CV+=t_n
        no_p_CV+=no_positive_class
        no_n_CV+=no_negative_class
        

    

y_train, y_test,  x_train, x_test = split_data(matrix,matrix_decision_class)
cm = [[tp_CV,fn_CV],[fp_CV,tn_CV]]
t_p, t_n, f_p, f_n, precision, specifity, total_acc, balance_acc, recall, true_neg_rate, cover_neg, cover_pos, total_cover, f1_score, g_mean = learn_model(cm, no_p_CV, no_n_CV)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("________________________________")
nr_objects = len(matrix)
print("nr of objects: {}".format(nr_objects))
print("matrix_decision_class: {}".format(matrix_decision_class))
print("================================")
print("positive_class: {}".format(positive_class))
print("negative_class: {}".format(negative_class))
print("================================")
print("Precision:           {}".format(precision))
print("Specifity:           {}".format(specifity))
print("Total accuracy:      {}".format(total_acc))
print("Balance accuracy:    {}".format(balance_acc))
print("recall:              {}".format(recall))
print("True negative rate:  {}".format(true_neg_rate))
print("Coverage positive:   {}".format(cover_pos))
print("Coverage negative:   {}".format(cover_neg))
print("Total coverage:      {}".format(total_cover))
print("F1 score:            {}".format(f1_score))
print("================================")
print("Confusion matrix for ALL folds:")
conf_matrix=[
["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
[positive_class, t_p, f_n, no_p_CV, precision, cover_pos],
[negative_class, f_p, t_n, no_n_CV, specifity, cover_neg]
]
print(tabulate(conf_matrix))