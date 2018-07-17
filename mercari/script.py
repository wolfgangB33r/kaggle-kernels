# This kernel collects all numeric features and transforms all textual features into
# term counts or term frequency counts. 
# The resulting feature vector is then used to train a neural network in a partial 
# fit fashion in order to not overuse memory. 
# To not run longer than the kernel limit, the training loop has an exit condition
# set to bail out after 2500 seconds. 
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
import random
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_selection import SelectPercentile, f_classif

### feature engineering section
def n_hash(s):
    random.seed(hash(s))
    return random.random()

# 1. feature engineering: convert category to single subcategory strings
def label_category (row, index):
    if 'category_name' in row:
        cats = str(row['category_name']).split('/')
    if len(cats) > index:
        return cats[index]
    return 'none'
    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')  

# evaluation
def eval_error (d):
    sum = 0.0;
    for index, row in d.iterrows():
        sum = sum + (math.log(row['price']+1) - math.log(row['real_price']+1))**2
    return math.sqrt(sum / (index + 1))

def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time
    
def fill_empty_values(train, test):
    test['category_name'].fillna(value='none', inplace=True)
    test['brand_name'].fillna(value='none', inplace=True)
    test['item_description'].fillna(value='none', inplace=True)
    train['category_name'].fillna(value='none', inplace=True)
    train['brand_name'].fillna(value='none', inplace=True)
    train['item_description'].fillna(value='none', inplace=True)
    return train, test

def main():
    start_time = time.time()
    ### Read data
    train = pd.read_csv('../input/train.tsv', sep='\t')
    test = pd.read_csv('../input/test.tsv', sep='\t')
    # sample for testing purposes
    #train = train.sample(frac=0.01, random_state=1)
    #test = test.sample(frac=0.01, random_state=1)
    
    result = test[['test_id']].copy()
    result['test_id'] = result['test_id'].astype(int) 
    # delete rows where price is zero
    train = train[train['price'] > 0]
    train['price'].dropna()
    y = np.log1p(train['price']) # get the result to fit for
    start_time = print_duration(start_time, "Finished to read data")
    
    # 1. feature engineering: convert category to single subcategory strings
    for i in range(0, 5):
        train['cat_' + str(i)] = train.apply (lambda row: label_category (row, i),axis=1)
        test['cat_' + str(i)] = test.apply (lambda row: label_category (row, i),axis=1)
    start_time = print_duration(start_time, "Finished to split subcategories")
    
    # fill empty values
    train, test = fill_empty_values(train, test)
    start_time = print_duration(start_time, "Finished to fill missing values")
    
    tv = TfidfVectorizer(max_features=200000, ngram_range=(1, 3), stop_words='english')
    train_text_features = tv.fit_transform(train['item_description'])
    test_text_features = tv.transform(test['item_description'])
    start_time = print_duration(start_time, "Finished transform product description")
    
    tv = TfidfVectorizer(max_features=200000, ngram_range=(1, 3), stop_words='english')
    train_X_name = tv.fit_transform(train['name'])
    test_X_name = tv.transform(test['name'])
    start_time = print_duration(start_time, "Finished vectorize product name")
    
    # feature selection step
    s = SelectPercentile(f_classif, percentile=20)
    s_train_text_features = s.fit_transform(train_text_features, y)
    s_test_text_features = s.transform(test_text_features)
    print(s_train_text_features.shape)
    
    s = SelectPercentile(f_classif, percentile=20)
    s_train_X_name = s.fit_transform(train_X_name, y)
    s_test_X_name = s.transform(test_X_name)
    print(s_train_X_name.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat0 = cv.fit_transform(train['cat_0'])
    test_X_cat0 = cv.transform(test['cat_0'])
    start_time = print_duration(start_time, "Finished vectorize cat_0")
    print(train_X_cat0.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat1 = cv.fit_transform(train['cat_1'])
    test_X_cat1 = cv.transform(test['cat_1'])
    start_time = print_duration(start_time, "Finished vectorize cat_1")
    print(train_X_cat1.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat2 = cv.fit_transform(train['cat_2'])
    test_X_cat2 = cv.transform(test['cat_2'])
    start_time = print_duration(start_time, "Finished vectorize cat_2")
    print(train_X_cat2.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat3 = cv.fit_transform(train['cat_3'])
    test_X_cat3 = cv.transform(test['cat_3'])
    start_time = print_duration(start_time, "Finished vectorize cat_3")
    print(train_X_cat3.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat4 = cv.fit_transform(train['cat_4'])
    test_X_cat4 = cv.transform(test['cat_4'])
    start_time = print_duration(start_time, "Finished vectorize cat_4")
    print(train_X_cat4.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_brand = cv.fit_transform(train['brand_name'])
    test_X_brand = cv.transform(test['brand_name'])
    start_time = print_duration(start_time, "Finished vectorize brand")
    print(train_X_brand.shape)
    
    cv = CountVectorizer(min_df=10)
    train_X_cat = cv.fit_transform(train['category_name'])
    test_X_cat = cv.transform(test['category_name'])
    start_time = print_duration(start_time, "Finished vectorize full category name")
    print(train_X_cat.shape)
    
    # convert to sparse matrix
    # 'brand_name_i', 'cat', 'cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 
    train_other_features = train.as_matrix(columns=['shipping', 'item_condition_id'])
    X = hstack((train_other_features, train_X_cat, train_X_brand, s_train_X_name, s_train_text_features, train_X_cat0, train_X_cat1, train_X_cat2, train_X_cat3, train_X_cat4)).tocsr()
    
    test_other_features = test.as_matrix(columns=['shipping', 'item_condition_id'])
    X_test = hstack((test_other_features, test_X_cat, test_X_brand, s_test_X_name, s_test_text_features, test_X_cat0, test_X_cat1, test_X_cat2, test_X_cat3, test_X_cat4)).tocsr()
    start_time = print_duration(start_time, "Finished to stack and create sparse matrix")
    
    # try partial fit
    model = MLPRegressor(solver='adam', hidden_layer_sizes=(20,20), random_state=7)
    batch_size = 10000
    total_rows = X.shape[0]
    duration = 0
    start_train = time.time()
    pos = 0
    while duration < 2500 and pos < total_rows:
        if pos+batch_size > total_rows:
            batch_size = total_rows - pos
        print("Pos %d/%d duration %d" % (pos, total_rows, duration))
        X_p = X[pos:pos+batch_size]
        y_p = y[pos:pos+batch_size]
        model.partial_fit(X_p, y_p)
        pos = pos + batch_size
        duration = time.time() - start_train # how long did we train so far?
    # end test partial fit  

    start_time = print_duration(start_time, "Finished to training the model")
    
    predictions = model.predict(X_test)
    start_time = print_duration(start_time, "Finished to predict result")
    
    # write result
    result['price'] = np.expm1(predictions)
    result.to_csv('submission.csv', encoding='utf-8', index=False)
    start_time = print_duration(start_time, "Finished to store result")
    

if __name__ == '__main__':
    main()