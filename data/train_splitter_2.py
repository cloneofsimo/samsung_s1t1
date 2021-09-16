import pandas as pd
import os

O_PATH = "./split_0"
I_PATH = "./train.csv"

df = pd.read_csv(I_PATH)

df = df[["SMILES", "S1_energy(eV)", "T1_energy(eV)"]]

df_stat = df[["S1_energy(eV)", "T1_energy(eV)"]].values
mu = df_stat.mean(axis=0)
std = df_stat.std(axis=0)
print(mu, std)

# set mu : 4, std : 0.8

df[["S1_energy(eV)", "T1_energy(eV)"]] = (
    df[["S1_energy(eV)", "T1_energy(eV)"]] - 4
) / 0.9

# k fold
from sklearn.model_selection import KFold

# fold
kf = KFold(n_splits=5, shuffle=True, random_state=731)
# train, test index
for i, (train_index, test_index) in enumerate(kf.split(df)):
    O_PATH = f"./split_{i}_2"
    os.makedirs(O_PATH, exist_ok=True)

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    train_df.to_csv(O_PATH + "/data_train.txt", index=False, header=False, sep=" ")
    test_df.to_csv(O_PATH + "/data_val.txt", index=False, header=False, sep=" ")
