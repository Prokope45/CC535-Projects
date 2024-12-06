from sklearn import svm from sklearn.metrics import accuracy_score, log_loss from sklearn.neighbors import KNeighborsClassifier from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier from sklearn.naive_bayes import GaussianNB from sklearn.discriminant_analysis import LinearDiscriminantAnalysis from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis from pandas import DataFrame, read_csv import pandas as pd


df=pd.read_csv('C:/data_file.csv')
df=shuffle(df)
df=df.drop(['Path'],axis=1)

test=df.iloc|:300,:]
train=df.iloc[300:,:]

train_labels=train['Class'].tolist()
train=train.drop(['Class'],axis=1)

train samples=train.values.tolist

test_labels=test['Class'].tolist()
test=test.drop(['Class'],axis=1)
test_samples=test.values.tolist)

for i in range (len(train_labels)): if train labels[i]=='class1': train labelslil=0
if train labels[i]=='class2':
train labels[il=1
for i in range (len(test labels)): if test labels[il=='class1':
test labels[i]=0
if test labelslil=='class2':
test_labels[i]=1

classifiers = [
KNeighborsClassifier(3),
SVC(kernel="rbf", C=0.025, probability=True),
NuSVC(probability=True),
Decision TreeClassifier), RandomForestClassifier), AdaBoostClassifier),
GradientBoostingClassifier), GaussianNB),
LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()] for clf in classifiers:
clf.fit(train_samples, train_labels) res=clf.predict(test_samples)
acc = accuracy_score(test_labels, res)
print (clf._class__name__+" Accuracy: "+str(acc))
correct_samples=0
for i in range (len(test_labels)): if test_labels[i]==res(i]:
correct_samples=correct_samples+1
print(correct_samples/len(test_labels))