import pandas as pd
import re

diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')

def remove_diacritics(text):
    if isinstance(text, str):
        return re.sub(diacritics_pattern, '', text)
    return text


df = pd.read_csv("/content/train_quqa.csv")
df["question"] = df["question"].apply(remove_diacritics)
df["answer"] = df["answer"].apply(remove_diacritics)

df = df.sample(n=50, random_state=42)


df[["question", "answer"]].to_csv(
    "test_dataset_V1.csv",
    index=False,
    encoding="utf-8-sig"
)
