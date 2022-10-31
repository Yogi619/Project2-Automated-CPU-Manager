import pandas as pd
# reading given csv file
# and creating dataframe
websites = pd.read_csv("topfile.txt", header = None)
# adding column headings
websites.columns = ['PID', 'USER', 'PR', 'NI', 'VIRT', 'RES', 'SHR', 's', '%CPU', '%MEM', 'TIME+', 'COMMAND']
websites
websites.nunique()
websites = websites.dropna()
websites = websites.drop_duplicates()
websites.nunique()
websites
websites = websites[["PR","NI","%CPU","%MEM","COMMAND"]]
#print(websites)
websites.dtypes
#1
from sklearn.preprocessing import LabelEncoder
from numpy import array
from numpy import argmax

#21-60

websites1 = websites[["PR", "NI", "%CPU", "%MEM", "COMMAND"]]
# 25-32
# importing pandas library
import pandas as pd

# reading given csv file
# and creating dataframe
websites2 = pd.read_csv("Final1.csv"
                        , header = None)

websites2.columns = ["PR", "NI", "%CPU", "%MEM", "COMMAND", "CLASS"]

websites2
#print(websites2)
websites2 = websites2[["CLASS"]]
X = websites1[["%CPU", "%MEM"]]
y = websites2
print(X)
print(y)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC()
clf.fit(train_features, train_labels)

pred = clf.predict(test_features)
from sklearn.metrics import accuracy_score

accuracy_score(pred, test_labels)
import joblib

filename = 'finalized_model.sav'
joblib.dump(clf, filename)
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(test_features, test_labels)
print(result)

pred = loaded_model.predict(test_features)
print(pred)
