import numpy as np
import pandas as pd
import math
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn import preprocessing as pre
from sklearn.feature_selection import SelectPercentile, f_classif
from gensim.models import word2vec
from keras.preprocessing import text
import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
  
def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time
    
def text_to_words(raw_text):
    words = text.text_to_word_sequence(raw_text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    return words 
    
def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in model: 
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # Initialize a counter
    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%100000 == 0:
           print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs
    
def loadGloveModel(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. %d words loaded!" % len(model))
    return model
    
def prepare_text(raw_text):
	# https://www.fastcompany.com/3026596/140-characters-of-fck-sht-and-ss-how-we-swear-on-twitter
	wAliases = { 
		'f***' : 'fuck',
		'f**k' : 'fuck', 
		'f*ck' : 'fuck',
		's**t' : 'shit',
		'sh!t' : 'shit'
	}
	
	t = raw_text.lower()
	for key, val in wAliases.items():
		t = t.replace(key, val)
	return t

# Use sigmoid activation on the last layer to get probabilities in a multi label problem.
# Softmax is to optimize for single class output. 
#
def fitPredictKerasModel(X_train, y, X_test, epochs):
    config = tf.ConfigProto(intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        model = Sequential()
        model.add(Dense(60, input_shape=(X_train.shape[1],), kernel_initializer ='uniform', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(60, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(60, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        # train
        model.fit(x=X_train, y=y, batch_size=1000, epochs=epochs, verbose=0)
        # eval
        score, acc = model.evaluate(X_train, y, batch_size=1000, verbose=0)
        print("score: %f acc: %f" % (score, acc))
        # predict
        return model.predict(X_test)[:, 0]
        
def fitPredictMLPModel(X_train, y, X_test, epochs):
    m = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30,30), random_state=1)
    batch_size = 10000
    total_rows = X_train.shape[0]
    while epochs > 0: 
        epochs = epochs - 1
        duration = 0
        start_train = time.time()
        pos = 0
        classes = [0,1]
        while duration < 2500 and pos < total_rows:
            if pos+batch_size > total_rows:
                batch_size = total_rows-pos
            X_p = X_train[pos:pos+batch_size]
            y_p = y[pos:pos+batch_size]
            m.partial_fit(X_p, y_p, classes)
            pos = pos + batch_size
            duration = time.time() - start_train # how long did we train so far?
            #print("Pos %d/%d duration %d" % (pos, total_rows, duration))
        # end test partial fit  
    pred = m.predict_proba(X_test)[:,1]
    del m
    return pred

def main():
    start_time = time.time()
    ### Read data
    train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
    test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
    train['comment_text'].fillna(value='', inplace=True)
    test['comment_text'].fillna(value='', inplace=True)
    train['comment_text'] = train['comment_text'].apply(prepare_text)
    test['comment_text'] = test['comment_text'].apply(prepare_text)
    
    start_time = print_duration(start_time, "Finished read and prepare texts")
    # --
    
    sentences_train = train['comment_text'].apply(text_to_words)
    sentences_test = test['comment_text'].apply(text_to_words)
    start_time = print_duration(start_time, "Finished tokenizing sentences")
    
    # load glove model
    wm = loadGloveModel('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')
    num_features = 200
    f_matrix_train_word = getAvgFeatureVecs(sentences_train, wm, num_features)
    f_matrix_test_word = getAvgFeatureVecs(sentences_test, wm, num_features)
    
    # check
    where_are_NaNs = np.isnan(f_matrix_train_word)
    f_matrix_train_word[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(f_matrix_test_word)
    f_matrix_test_word[where_are_NaNs] = 0
    
    tv = TfidfVectorizer(max_features=6000, ngram_range=(1, 3), analyzer='word', stop_words='english')
    train_X_tfidf = tv.fit_transform(train['comment_text'])
    test_X_tfidf = tv.transform(test['comment_text'])
    
    tvChar = TfidfVectorizer(max_features=6000, ngram_range=(1, 3), analyzer='char', stop_words='english')
    train_X_tfidf_char = tvChar.fit_transform(train['comment_text'])
    test_X_tfidf_char = tvChar.transform(test['comment_text'])
    
    # combine both feature matrices
    train_X = hstack((f_matrix_train_word, train_X_tfidf, train_X_tfidf_char)).tocsr()
    test_X = hstack((f_matrix_test_word, test_X_tfidf, test_X_tfidf_char)).tocsr()
    print(train_X.shape)
    print(test_X.shape)
    
    start_time = print_duration(start_time, "Finished create feature vector")
    
    y = [train['toxic'], train['severe_toxic'], train['obscene'], train['threat'], train['insult'], train['identity_hate']]
    start_time = print_duration(start_time, "Finished to read data")
    # predict
    start_time = print_duration(start_time, "Finished to predict result")
    submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
    start_time = print_duration(start_time, "Begin predict toxic")
    submission['toxic'] = fitPredictMLPModel(train_X, y[0], test_X, 2)
    start_time = print_duration(start_time, "Finished predict toxic")
    submission['severe_toxic'] = fitPredictMLPModel(train_X, y[1], test_X, 2) * submission['toxic'] 
    start_time = print_duration(start_time, "Finished predict severe_toxic")
    submission['obscene'] = fitPredictMLPModel(train_X, y[2], test_X, 2) * submission['toxic'] 
    start_time = print_duration(start_time, "Finished predict obscene")
    submission['threat'] = fitPredictMLPModel(train_X, y[3], test_X, 2) * submission['toxic'] 
    start_time = print_duration(start_time, "Finished predict threat")
    submission['insult'] = fitPredictMLPModel(train_X, y[4], test_X, 2) * submission['toxic'] 
    start_time = print_duration(start_time, "Finished predict insult")
    submission['identity_hate'] = fitPredictMLPModel(train_X, y[5], test_X, 2) * submission['toxic'] * submission['insult']
    start_time = print_duration(start_time, "Finished predict identity_hate")
    submission.to_csv("submission.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")

if __name__ == '__main__':
    main()
    