import numpy as np
import pandas as pd
import math
import time
import os.path

import xgboost as xgb

# all attributed = 0 means 0.5 score

def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time

def main():
    start_time = time.time()
    # create a xgboost model
    model = xgb.XGBClassifier(max_depth=3, n_estimators=600, learning_rate=0.05)
    
    # already prepared for chunked learning in case that full training data
    # set does not fit into memory
    # fit the model
    chunksize = 10000000 
    chunk_idx = 0 
    chunk_max = 1
    for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize):
        # Extracting new features
        chunk['hour'] = pd.to_datetime(chunk.click_time).dt.hour.astype('uint8')
        chunk['day'] = pd.to_datetime(chunk.click_time).dt.day.astype('uint8')
        train_X = chunk.as_matrix(columns=['ip', 'app', 'device', 'os', 'channel', 'hour', 'day'])
        if chunk_idx > 0: # not load in first run
            model.fit(train_X, chunk['is_attributed'], xgb_model=model)
        else:
            model.fit(train_X, chunk['is_attributed'])
        print("Processed chunk %d" % chunk_idx)
        chunk_idx = chunk_idx + 1
        if chunk_idx >= chunk_max:
            break
        
    # read test data set
    test = pd.read_csv('../input/test.csv')
    test['hour'] = pd.to_datetime(test.click_time).dt.hour.astype('uint8')
    test['day'] = pd.to_datetime(test.click_time).dt.day.astype('uint8')
    
    test_X = test.as_matrix(columns=['ip', 'app', 'device', 'os', 'channel', 'hour', 'day'])
    start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
    pred = model.predict_proba(test_X)
    
    start_time = print_duration (start_time, "Finished prediction, start store results")    
    submission = pd.read_csv("../input/sample_submission.csv")
    submission['is_attributed'] = pred[:,1]
    submission.to_csv("submission.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")
    
    
if __name__ == '__main__':
    main()
    
    