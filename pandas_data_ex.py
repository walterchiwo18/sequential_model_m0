import pandas as pd

df = pd.read_csv("./train.csv")

print("Shape of dataset", df.shape)

df.head(5)
