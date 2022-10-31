import pandas as pd
import numpy as np
import os, signal
import joblib
# reading given csv file
# and creating dataframe
process = pd.read_csv("topfile.txt", header=None, error_bad_lines=False)
# adding column headings
process.columns = ['PID', 'USER', 'PR', 'NI', 'VIRT', 'RES', 'SHR', 's', '%CPU', '%MEM', 'TIME+', 'COMMAND']
process = process.dropna()
print(process)
process1 = process[["PR", "NI", "%CPU", "%MEM", "COMMAND"]]
#process2 = process2[["PR", "NI", "%CPU", "%MEM", "COMMAND"]]
process2 = process1[["%CPU", "%MEM"]]
filename = "finalized_model.sav"
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.predict(process2)
print(result)
print(np.where(result == 1))
finalResult = process.iloc[np.where(result == 1)]
finalResult = finalResult.drop_duplicates(subset=['PID'])
pid_list = finalResult.iloc[:, 0].tolist()
print(pid_list)
#for i in pid_list:
    #os.kill(i, signal.SIGKILL)

