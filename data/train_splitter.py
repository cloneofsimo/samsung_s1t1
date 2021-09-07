import pandas as pd


O_PATH = "./split_0"
I_PATH = "./train.csv"

df = pd.read_csv(I_PATH)
df["gap"] = df["S1_energy(eV)"] - df["T1_energy(eV)"]

df = df[["SMILES", "gap"]]

# k fold
from sklearn.model_selection import KFold

# fold
kf = KFold(n_splits=5, shuffle=True)
# train, test index
for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    train_df.to_csv(O_PATH + "/train.txt", index=False, header=False, sep=" ")
    test_df.to_csv(O_PATH + "/val.txt", index=False, header=False, sep=" ")
    break

# df.to_csv(O_PATH, index=False, header=False, sep=" ")
