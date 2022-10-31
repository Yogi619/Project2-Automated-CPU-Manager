import pandas as pd
# reading given csv file and creating dataframe
processes = pd.read_csv("topfile.txt", header = None)
# adding column headings
processes.columns = ['PID', 'USER', 'PR', 'NI', 'VIRT', 'RES', 'SHR', 's', '%CPU', '%MEM', 'TIME+', 'COMMAND']
processes
processes.nunique()
processes = processes.dropna()
processes = processes.drop_duplicates()
processes.nunique()
processes
processes = processes[["PR", "NI", "%CPU", "%MEM", "COMMAND"]]
processes.dtypes
processes1 = processes[["PR", "NI", "%CPU", "%MEM", "COMMAND"]]
# importing pandas library
import pandas as pd
# reading given csv file and creating dataframe
processes2 = pd.read_csv("Final2.csv", header=None)
processes2.columns = ["%CPU", "%MEM", "CLASS"]
processes2
processes2 = processes2[["CLASS"]]
processes2 = processes2.values.ravel()
X = processes1[["%CPU", "%MEM"]]
y = processes2
print(y)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

from sklearn.svm import SVC
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
