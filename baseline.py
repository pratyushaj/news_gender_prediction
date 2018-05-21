### Project Baseline ### 
'''
1. Read in data
2. Separate out text of comments (features) and gender of employee (labels)
3. Basic feature engineering (unigrams and bigrams) 
4. Tune hyperparameters with grid search 
5. Train and evaluate Logistic Regression and SVM (linear and RBF kernels)

'''

#Libraries we'll need 
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, neighbors, linear_model
from sklearn.svm import SVC
import scipy.stats
from sgd_classifier import BasicSGDClassifier
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np 
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#initially reading in data
df = pd.read_csv('dct_data.csv')

#selecting features and labels 
data = df[['Clean Section Comment','Employee Gender']]

#preprocessing: eliminating punctuation
def remove_punctuations(text):
	for punctuation in string.punctuation:
		text = str(text).replace(punctuation, ' ')
	text = re.sub( '\s+', ' ', text ).strip()
	return text

data.loc[:, 'Clean Section Comment'] = data.loc[:, 'Clean Section Comment'].apply(remove_punctuations)

#preprocessing: gendered pronouns
def remove_gender(text):
	gender = ['he','him','his','she','her','hers','himself','herself']
	for pronoun in gender:
		text = str(text).replace(pronoun, ' ')
	text = re.sub( '\s+', ' ', text ).strip()
	return text

data.loc[:, 'Clean Section Comment'] = data.loc[:, 'Clean Section Comment'].apply(remove_punctuations)

#feature engineering: unigrams alone 

def unigrams(text):
	words = text.split(' ')
	return dict(Counter(words))

#converting samples into features 
feat_dicts = []
labels = []
for index, row in data.iterrows():
	text = row['Clean Section Comment']
	label = row['Employee Gender']
	feature_dict = unigrams(text)
	feat_dicts.append(feature_dict)
	labels.append(label)

vectorizer = DictVectorizer(sparse=False)
features = vectorizer.fit_transform(feat_dicts)


#Train test split: 70-30
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)

print(len(X_train),len(X_test),len(y_train),len(y_test))

#Model: Logistic Regression right out of the box 
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
svm = SVC()

logistic_model = logistic.fit(X_train, y_train)
knn_model = knn.fit(X_train,y_train)
svm_model = svm.fit(X_train,y_train)

logistic_predictions = logistic.predict(X_test)
knn_predictions = knn.predict(X_test)
svm_predictions = svm.predict(X_test)

print("Logistic Regression: \n")
print(classification_report(y_test, logistic_predictions)) 

print("KNN: \n")
print(classification_report(y_test, knn_predictions)) 

print("SVM: \n")
print(classification_report(y_test, svm_predictions)) 













