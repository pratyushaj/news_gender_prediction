### Project Baseline ### 
'''
1. Read in data
2. Separate out text of comments (features) and gender of employee (labels)
3. Basic feature engineering (unigrams and bigrams) 
4. Tune hyperparameters with grid search 
5. Train and evaluate Logistic Regression and SVM (linear and RBF kernels)

'''
#Fucking unicode
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#Libraries we'll need 
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, neighbors, linear_model
from sklearn.svm import SVC
import scipy.stats
from sklearn.feature_extraction import DictVectorizer
import numpy as np 
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#initially reading in data
df = pd.read_csv('articles_train_degendered_filtered.csv',encoding='utf8')

#selecting features and labels 
data = df[['sentence','gender']]

#preprocessing: eliminating punctuation
print "Cleaning data..."
def remove_punctuations(text):
	for punctuation in string.punctuation:
		text = str(text).encode('utf-8').replace(punctuation, ' ')
	text = re.sub( '\s+', ' ', text ).encode('utf-8').strip()
	return text

data.loc[:, 'sentence'] = data.loc[:, 'sentence'].apply(remove_punctuations)

print data.size
#feature engineering: unigrams alone 

print "Featurizing..."
def unigrams(text):
	words = text.split(' ')
	return dict(Counter(words))

#converting samples into features 
feat_dicts = []
labels = []
for index, row in data.iterrows():
	text = row['sentence']
	label = row['gender']
	feature_dict = unigrams(text)
	feat_dicts.append(feature_dict)
	labels.append(label)

vectorizer = DictVectorizer(sparse=True)
features = vectorizer.fit_transform(feat_dicts)

print "Splitting into train and test..."
#Train test split: 70-30
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)

print X_train.shape[0],len(y_train),X_test.shape[0],len(y_test)

#Model: Logistic Regression right out of the box 
#knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(verbose=3)
#svm = SVC()

print "Training the model..."
logistic_model = logistic.fit(X_train, y_train)
#knn_model = knn.fit(X_train,y_train)
#svm_model = svm.fit(X_train,y_train)

logistic_predictions = logistic.predict(X_test)
#knn_predictions = knn.predict(X_test)
#svm_predictions = svm.predict(X_test)

print "Logistic Regression: \n"
print classification_report(y_test, logistic_predictions) 

'''print("KNN: \n")
print(classification_report(y_test, knn_predictions)) 

print("SVM: \n")
print(classification_report(y_test, svm_predictions))'''













