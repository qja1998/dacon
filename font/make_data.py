import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv", encoding="utf-8")
df = df.drop("id", axis=1)

for i, path in enumerate(df.img_path):
    df.iloc[i].img_path = path.split('/')[-1]

print(df.head())

df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

for f in os.listdir("./deep-text-recognition-benchmark-master/data/train"):
    os.remove(os.path.join("./deep-text-recognition-benchmark-master/data/train", f))
for f in os.listdir("./deep-text-recognition-benchmark-master/data/valid"):
    os.remove(os.path.join("./deep-text-recognition-benchmark-master/data/valid", f))

for train_data in df_train.img_path:
    shutil.copyfile("./data/train/" + train_data, "./deep-text-recognition-benchmark-master/data/train/" + train_data)
for val_data in df_val.img_path:
    shutil.copyfile("./data/train/" + train_data, "./deep-text-recognition-benchmark-master/data/valid/" + val_data)

df_train.to_csv("deep-text-recognition-benchmark-master/data/train.tsv", sep='\t', encoding="utf-8", index=False)
df_val.to_csv("deep-text-recognition-benchmark-master/data/valid.tsv", sep='\t', encoding="utf-8", index=False)