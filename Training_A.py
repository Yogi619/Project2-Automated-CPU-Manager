import pandas as pd
# reading given csv file
# and creating dataframe
websites = pd.read_csv("topfile.txt",header = None)
# adding column headings
websites.columns = ['PID', 'USER', 'PR','NI','VIRT','RES','SHR','s','%CPU','%MEM','TIME+','COMMAND']
websites
websites.nunique()
websites = websites.dropna()
websites.value_counts()
websites = websites.drop_duplicates()
websites.nunique()
websites
websites = websites[["%CPU","%MEM"]]
websites
websites.dtypes
websites.to_csv("Final.csv")
