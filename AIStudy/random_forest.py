from numpy import *
from math import log
from sklearn.tree import *
import matplotlib as mpl
import matplotlib.pyplot as plt
def load_data():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataset, labels


def calc_shano_ent(dataset):
    num = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        label = feat_vec[-1]
        label_counts[label] = label_counts.get(label,0)+1
    shano_ent = 0.0
    for key in label_counts:
        prod = float(label_counts[key]/num)
        shano_ent -= prod * log(prod,2)
    return shano_ent

def calc_gini_ent(dataset):
    num = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        label = feat_vec[-1]
        label_counts[label] = label_counts.get(label,0)+1
    gini_ent = 1.0
    for key in label_counts:
        prod = (label_counts[key]/num)**2
        gini_ent -= prod
    return gini_ent

def split_dataset(dataset,axis,val):
    result_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == val:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis+1:])
            result_dataset.append(reduce_feat_vec)
    return result_dataset

def choose_best_feat(dataset,model=1):
    num = len(dataset[0])-1
    best_feature = -1
    best_shano_ent = 10.0
    best_gini_ent = 10.0
    for i in range(num):
        feat_list = [example[i] for example in dataset]
        feat_set = set(feat_list)
        new_shano_ent = 0.0
        new_gini_ent = 0.0
        for feat_type in feat_set:
            sub_dataset = split_dataset(dataset,i,feat_type)
            prod = (float)(len(sub_dataset)/len(dataset))
            new_shano_ent += prod * calc_shano_ent(sub_dataset)
            new_gini_ent += prod * calc_gini_ent(sub_dataset)
        if new_shano_ent < best_shano_ent and model == 1:
            best_shano_ent = new_shano_ent
            best_feature = i
        if new_gini_ent < best_gini_ent and model ==2:
            best_gini_ent = new_gini_ent
            best_feature = i
    return best_feature

def create_tree(dataset,labels):
    result_list = [example[-1] for example in dataset]
    if len(set(result_list)) == 1:
        return result_list[0]
    if dataset[0] == 1:
        return result_list[0]
    best_feature = choose_best_feat(dataset)
    best_label = labels[best_feature]
    my_tree = {best_label:{}}
    del(labels[best_feature])
    feat_values = [example[best_feature] for example in dataset]
    feat_values_set = set(feat_values)
    for feat in feat_values_set:
        sub_labels = labels[:]
        my_tree[best_label][feat] = create_tree(split_dataset(dataset,best_feature,feat),sub_labels)
    return my_tree

dataset,labels = load_data()
print(create_tree(dataset,labels))