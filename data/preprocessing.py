import pandas as pd
from tqdm import tqdm
import random
# QUQA

quqa = pd.read_excel("QUQA.xlsx")

def process_quqa(quqa_df):
    # passage_id
    quqa_df["passage_id"] = (
        quqa_df["Chapter_Number"].astype(str) + ":" +
        quqa_df["Verses_Number_Start"].astype(str) + "-" +
        quqa_df["Verses_Number_End"].astype(str)
    )

    quqa_df = quqa_df[quqa_df["Quran_Full_Verse_Answer"].notna()]
    quqa_df = quqa_df[quqa_df["Full_Answer"].notna()]

    processed_rows = []

    for _, row in tqdm(quqa_df.iterrows(), total=len(quqa_df), desc="Building pairs"):
        question = row["Question_Text"]
        pos_pid = row["passage_id"]
        pos_passage = row["Quran_Full_Verse_Answer"]
        answer=row["Full_Answer"]
        ansid=row["Answer_ID"]
        qid = row["Question_Id"]


        processed_rows.append({
            "question_id": qid,
            "passage_id": pos_pid,
            "answer_id":ansid,
            "question": question,
            "passage": pos_passage,
            "answer":answer,
            "label": 1
        })

    
    return pd.DataFrame(processed_rows)

train_quqa = process_quqa(quqa)

train_quqa.to_csv("train_quqa.csv", index = False)

# HAQA

haqa = pd.read_csv("HAQA.csv")

def process_haqa(haqa_df):
    haqa_df = haqa_df.copy()

    haqa_df = haqa_df[haqa_df["Hadith_Full_Answer"].notna()].reset_index(drop=True)
    haqa_df["passage_id"] = ["HADITH#{:05d}".format(i) for i in range(len(haqa_df))]

    processed_rows = []

    for _, row in tqdm(haqa_df.iterrows(), total=len(haqa_df), desc="Processing HAQA"):
        question = row["Question_Text"]
        pos_pid = row["passage_id"]
        pos_passage = row["Hadith_Full_Answer"]
        qid = str(row["Question_Id"])
        ansid=str(row["Answer_ID"])
        answer=row["Full_Answer"]

        processed_rows.append({
            "question_id": qid,
            "passage_id": pos_pid,
            "question": question,
            "passage": pos_passage,
            "answer_id":ansid,
            "answer":answer,
            "label": 1
        })

        

    return pd.DataFrame(processed_rows)

haqa_train = process_haqa(haqa, num_negatives=5)

haqa_train.to_csv("haqa_train.csv", index=False)