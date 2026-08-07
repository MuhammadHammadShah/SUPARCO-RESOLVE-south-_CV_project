import pandas as pd
df = pd.read_excel("data/minerals.xlsx")
df.to_csv("data/minerals.csv",index=False)