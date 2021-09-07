import pandas as pd


O_PATH = "./split_0"
I_PATH = "./train.csv"

df = pd.read_csv(I_PATH)
df["gap"] = 0

df = df[["SMILES", "gap"]]
df.to_csv(O_PATH + "/data_test.txt", index=False, header=False, sep=" ")
