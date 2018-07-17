# Simple first attempt to predict the propability of demand
# Not using the image info so far and only taking simple 
# categorical features into account
#
import numpy as np
import pandas as pd
import math
import time
import os.path 
import gc
import random

import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.neural_network import MLPRegressor

# Version 128: Score: 0.230 +
# Version 114: Score: 0.231
# Version 68: Score: 0.231
# Version 66: Score: 0.232
# Version 65: Score: 0.236


def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time
    
# quick way of calculating a numeric hash for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

# hash a complete column of a pandas dataframe    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

def text_to_words(raw_text):
    if type(raw_text) == float:
        return []
    return raw_text.split() 
    
def word_count(raw_text):
    if type(raw_text) == float:
        return 0
    return len(raw_text.split()) 
    
def count_chars(raw_text):
    if type(raw_text) == float:
        return 0
    return len(raw_text)

def replace_sub_zero(value):
    if value < 0:
        return 0
    elif value > 1:
        return 1
    return value 

def main():
    start_time = time.time()
    # create a xgboost model
    model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    
    # load the training data
    train = pd.read_csv('../input/train.csv')
    train['title'] = train['title'].fillna(value="")
    train['description'] = train['description'].fillna(value="")
    train['param_1'] = train['param_1'].fillna(value="")
    train['param_2'] = train['param_2'].fillna(value="")
    train['param_3'] = train['param_3'].fillna(value="")
    test = pd.read_csv('../input/test.csv')
    train['title'] = train['title'].fillna(value="")
    test['description'] = test['description'].fillna(value="")
    test['param_1'] = test['param_1'].fillna(value="")
    test['param_2'] = test['param_2'].fillna(value="")
    test['param_3'] = test['param_3'].fillna(value="")
    
    
    # train a model for the text
    tv = CountVectorizer(max_features=2000, min_df=5)
    # title
    train_text_features_title = tv.fit_transform(train['title'])
    test_text_features_title = tv.transform(test['title'])
    start_time = print_duration (start_time, "Finished get title feature matrix")  
    # description
    train_text_features_descr = tv.fit_transform(train['description'])
    test_text_features_descr = tv.transform(test['description'])
    start_time = print_duration (start_time, "Finished get description feature matrix")  
    # param 1
    train_text_features_param1 = tv.fit_transform(train['param_1'])
    test_text_features_param1 = tv.transform(test['param_1'])
    start_time = print_duration (start_time, "Finished get param_1 feature matrix")  
    # param 2
    train_text_features_param2 = tv.fit_transform(train['param_2'])
    test_text_features_param2 = tv.transform(test['param_2'])
    start_time = print_duration (start_time, "Finished get param_2 feature matrix")  
    # param 3
    train_text_features_param3 = tv.fit_transform(train['param_3'])
    test_text_features_param3 = tv.transform(test['param_3'])
    start_time = print_duration (start_time, "Finished get param_3 feature matrix")  
    
    # predict
    text_model = MLPRegressor(solver='adam', hidden_layer_sizes=(50,50), random_state=7)
    train_text_features = hstack((train_text_features_title, train_text_features_descr, train_text_features_param1, train_text_features_param2, train_text_features_param3)).tocsr()
    text_model.fit(train_text_features, train['deal_probability'])
    test_text_features = hstack((test_text_features_title, test_text_features_descr, test_text_features_param1, test_text_features_param2, test_text_features_param3)).tocsr()
    text_pred = text_model.predict(test_text_features)
    start_time = print_duration (start_time, "Finished train and predict mlp text")  
    
    ## end text prediction
    
    print(train.groupby(['item_seq_number']).image_top_1.count())
    # *** Start the numeric feature prediction
    # calculate consistent numeric hashes for any categorical features 
    train['date_hash'] = train.apply (lambda row: hash_column (row, 'activation_date'),axis=1)
    train['region_hash'] = train.apply (lambda row: hash_column (row, 'region'),axis=1)
    train['city_hash'] = train.apply (lambda row: hash_column (row, 'city'),axis=1)
    train['parent_category_name_hash'] = train.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    train['category_name_hash'] = train.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    train['user_type_hash'] = train.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    # for the beginning I use only the information if there is an image or not 
    train['image_exists'] = train['image'].isnull().astype(int)
    train['title_len'] = train['title'].apply(count_chars) 
    train['title_w_count'] = train['title'].apply(word_count) 
    train['descr_len'] = train['description'].apply(count_chars) 
    train['descr_w_count'] = train['description'].apply(word_count) 
    train['param_1_len'] = train['param_1'].apply(count_chars) 
    train['param_1_w_count'] = train['param_1'].apply(word_count) 
    train['param_2_len'] = train['param_2'].apply(count_chars) 
    train['param_2_w_count'] = train['param_2'].apply(word_count) 
    train['param_3_len'] = train['param_3'].apply(count_chars) 
    train['param_3_w_count'] = train['param_3'].apply(word_count) 
    
    #print(train.groupby(['image_top_1']).image_top_1.count())
    #print(train['image_exists'])
    start_time = print_duration (start_time, "Finished reading")   
    # start training
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.replace(np.nan, 0)
    train_X = train.as_matrix(columns=['item_seq_number',
                                        'date_hash',
                                        'image_top_1', 
                                        'price', 
                                        'region_hash', 
                                        'city_hash', 
                                        'parent_category_name_hash', 
                                        'category_name_hash', 
                                        'user_type_hash', 
                                        'image_exists',
                                        'title_len',
                                        'descr_len',
                                        'param_1_len',
                                        'param_2_len',
                                        'param_3_len',
                                        'title_w_count',
                                        'descr_w_count',
                                        'param_1_w_count',
                                        'param_2_w_count',
                                        'param_3_w_count'
                                        ])
    
    model.fit(train_X, train['deal_probability'])
    
    # read test data set
    test['date_hash'] = test.apply (lambda row: hash_column (row, 'activation_date'),axis=1)
    test['region_hash'] = test.apply (lambda row: hash_column (row, 'region'),axis=1)
    test['city_hash'] = test.apply (lambda row: hash_column (row, 'city'),axis=1)
    test['parent_category_name_hash'] = test.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    test['category_name_hash'] = test.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    test['user_type_hash'] = test.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    test['image_exists'] = test['image'].isnull().astype(int)
    test['title_len'] = test['title'].apply(count_chars)
    test['title_w_count'] = test['title'].apply(word_count) 
    test['descr_len'] = test['description'].apply(count_chars) 
    test['descr_w_count'] = test['description'].apply(word_count) 
    test['param_1_len'] = test['param_1'].apply(count_chars) 
    test['param_1_w_count'] = test['param_1'].apply(word_count) 
    test['param_2_len'] = test['param_2'].apply(count_chars) 
    test['param_2_w_count'] = test['param_2'].apply(word_count) 
    test['param_3_len'] = test['param_3'].apply(count_chars) 
    test['param_3_w_count'] = test['param_3'].apply(word_count) 
    
    test = test.replace([np.inf, -np.inf], np.nan)
    test = test.replace(np.nan, 0)
    test_X = test.as_matrix(columns=['item_seq_number',
                                        'date_hash',
                                        'image_top_1', 
                                        'price', 
                                        'region_hash', 
                                        'city_hash', 
                                        'parent_category_name_hash', 
                                        'category_name_hash', 
                                        'user_type_hash', 
                                        'image_exists',
                                        'title_len',
                                        'descr_len',
                                        'param_1_len',
                                        'param_2_len',
                                        'param_3_len',
                                        'title_w_count',
                                        'descr_w_count',
                                        'param_1_w_count',
                                        'param_2_w_count',
                                        'param_3_w_count'
                                        ])
    
    
    start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
    pred = model.predict(test_X)
    
    start_time = print_duration (start_time, "Finished prediction, start store results")    
    submission = pd.read_csv("../input/sample_submission.csv")
    # factors for applying the numeric prediction result and the text prediction result
    F_NUMERIC = 0.7
    F_TEXT = 0.3
    submission['deal_probability'] = np.add(np.multiply(pred, F_NUMERIC), np.multiply(text_pred, F_TEXT))
    submission['deal_probability'] = submission['deal_probability'].apply(replace_sub_zero)
    submission.to_csv("submission.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")
    
if __name__ == '__main__':
    main()
    
    