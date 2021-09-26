import pandas as pd
import os

I_PATH = "./test.csv"

df = pd.read_csv(I_PATH)

# df = df[["SMILES", "S1_energy(eV)", "T1_energy(eV)"]]


# set mu : 4, std : 0.8

df[["S1_energy(eV)", "T1_energy(eV)"]] = 0.0
df["gap"] = df["S1_energy(eV)"] - df["T1_energy(eV)"]
# from tqdm import tqdm
# for idx in tqdm(df.index):


print(len(df))

# k fold
from sklearn.model_selection import KFold

# fold
kf = KFold(n_splits=5, shuffle=True, random_state=731)
# train, test index
for i, (train_index, test_index) in enumerate(kf.split(df)):
    O_PATH = f"./split_sdf_{i}"
    os.makedirs(O_PATH, exist_ok=True)

    train_df = df
    train_df.to_csv(O_PATH + "/data_test.txt", index=False, header=False, sep=" ")
