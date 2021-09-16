import pandas as pd
import os

I_PATH = "./test.csv"

df = pd.read_csv(I_PATH)
df["gap"] = 0

df = df[["SMILES", "gap"]]
for split in range(5):
    O_PATH = f"./split_{split}_2"
    os.makedirs(O_PATH, exist_ok=True)

    df.to_csv(O_PATH + "/data_test.txt", index=False, header=False, sep=" ")
