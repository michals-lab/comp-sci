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
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import operator
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve
from tabulate import tabulate


DATA_PATH = r"heart_disease.dat"
matrix = []
matrix_decision_class = []

# number_of_folds = 10

nr_pos_class = 0
nr_neg_class = 0

mean_fpr = np.linspace(0, 1, 1000)

positive_class = None
negative_class = None

tp_CV = []
fn_CV = []
fp_CV = []
tn_CV = []
no_p_CV = []
no_n_CV = []
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

print("________________________________")
nr_objects = len(matrix)
print("nr of objects: {}".format(nr_objects))
print("matrix_decision_class: {}".format(matrix_decision_class))
print("**********************************")

def split_data(matrix,matrix_decision_class ):
    positive_class = None
    negative_class = None
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
        kNN_classifier(y_train, y_test,  x_train, x_test, column, row, matrix_decision_class,positive_class,negative_class)  

        tp=float(sum(tp_CV))
        tp=tp/len(tp_CV)
        fn=float(sum(fn_CV))
        fn=fn/len(fn_CV)
        fp=float(sum(fp_CV))
        fp=fp/len(fp_CV)
        tn=float(sum(tn_CV))
        tn=tn/len(tn_CV)
        no_positive_class = float(sum(no_p_CV))
        no_positive_class = no_positive_class/len(no_p_CV)
        no_negative_class = float(sum(no_n_CV))
        no_negative_class = no_negative_class/len(no_n_CV)
        roc_score = float(sum(roc_CV))
        try:
            roc_score = round(roc_score/len(roc_CV),5)
        except:
            roc_score = 0
        # all_fpr = mean_fpr
        # all_tpr = np.mean(tpr_CV, axis=0)
        # all_tpr[0] = 0.0
        # all_tpr[-1] = 1.0
        # cm = [[tp,fn],[fp,tn]]   
        count = 0  
        count += 1
        print(f"let's see whats in here: {count}")         
    
    return y_train, y_test,  x_train, x_test



# no_rows = math.ceil((number_of_folds+1)/5)
# fig, ax = plt.subplots(no_rows, 5, figsize=(11,2*no_rows),num='ROC',constrained_layout=True)
# for x in range(0,no_rows):
#     for y in range(0,5):
#         ax[x,y].set_axis_off()
l_colors = [(k, v) for k, v in mcolors.TABLEAU_COLORS.items()]
def get_color(i:int):
    i = i%len(l_colors)
    return l_colors[i][1]



def learn_model(data):
    print("________________________________")
    confussion_matrix = data
    print(confussion_matrix)
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
    print("================================")

    precision = round(t_p / (t_p + f_n), 5) #precision = metrics.precision_score()
    if math.isnan(precision):
        precision = float(0)

    specifity = round( t_n / (t_n + f_p), 5)
    if math.isnan(specifity):
        specifity = float(0)

    total_acc = round((t_n + t_p) / (t_p + f_n + t_n + f_p), 5)
    if math.isnan(total_acc):
        total_acc = float(0)

    balance_acc = round((specifity + precision)/2, 5)
    if math.isnan(balance_acc):
        balance_acc = float(0)

    recall = round(t_p / (t_p + f_p), 5)
    if math.isnan(recall):
        recall = float(0)

    true_neg_rate = round(t_n / (t_n + f_n), 5)
    if math.isnan(true_neg_rate):
        true_neg_rate = float(0)

    cover_pos = round((t_p + f_n) / nr_pos_class, 5)
    if math.isnan(cover_pos):
        cover_pos = float(0)

    cover_neg = round((t_n + f_p) / nr_neg_class, 5)
    if math.isnan(cover_neg) or np.isinf(cover_neg):
        cover_neg = float(0)

    total_cover = round((t_p + f_n + t_n + f_p) / (nr_pos_class + nr_neg_class), 5)
    if math.isnan(total_cover) or np.isinf(total_cover):
        total_cover = float(0)

    f1_score = round((precision * recall * 2) / (recall + precision), 5)
    if math.isnan(f1_score):
        f1_score = float(0)

    g_mean = round(math.sqrt(recall * specifity), 5)
    if math.isnan(g_mean):
        g_mean = float(0)

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
    print("**********************************")
    return t_p, t_n, f_p, f_n, precision, specifity, total_acc, balance_acc, recall, true_neg_rate, cover_neg, cover_pos, total_cover, f1_score, g_mean

def kNN_classifier(y_train, y_test,  x_train, x_test, column, row, matrix_decision_class,positive_class,negative_class):
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
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(f"cm: {cm}")
        t_p, t_n, f_p, f_n, precision, specifity, total_acc, balance_acc, recall, true_neg_rate, cover_neg, cover_pos, total_cover, f1_score, g_mean = learn_model(cm)
        y_pred_proba = classifier.predict_proba(x_test)[::,1]

        # try:
        #     ax[row,column].plot(mean_fpr, mean_fpr, "--")
        #     ax[row,column].set_title("ROC fold "+str(x+1)+".")
        #     ax[row,column].set_ylabel('True Positive Rate')
        #     ax[row,column].set_axis_on()
        # except:
        #     ax[column].plot(mean_fpr, mean_fpr, "--")
        #     ax[column].set_title("ROC fold "+str(x+1)+".")
        #     ax[column].set_ylabel('True Positive Rate')
        #     ax[column].set_axis_on()
        tp_CV.append(t_p)
        fn_CV.append(f_n)
        fp_CV.append(f_p)
        tn_CV.append(t_n)
        no_p_CV.append(no_positive_class)
        no_n_CV.append(no_negative_class)
        # tpr = np.interp(mean_fpr,fpr,tpr)
        # tpr_CV.append(tpr)
        print(f"positive_class: {positive_class}")
        print(f"negative_class: {negative_class}")
        print("Confusion matrix for "+str(2137)+". fold:")
        conf_matrix=[
        ["", positive_class,negative_class,"No. of objects","Accuracy", "Coverage"],
        [int(positive_class), t_p, f_n, no_positive_class, precision, cover_pos],
        [int(negative_class), f_p, t_n, no_negative_class, specifity, cover_neg]
        ]
        print(tabulate(conf_matrix))
    print("**********************************")

    

y_train, y_test,  x_train, x_test = split_data(matrix,matrix_decision_class)

