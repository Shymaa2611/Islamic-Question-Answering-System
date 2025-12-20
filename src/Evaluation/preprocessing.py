import pandas as pd
import json

def process__train_dataset():
    rows = []

    with open("/content/qrcd_v1.1_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            rows.append({
                "id": item.get("pq_id", ""),
                "question": item.get("question", ""),
                "context": item.get("passage", ""),
                "answer": item["answers"][0]["text"] if item.get("answers") else ""
            })

    return pd.DataFrame(rows)

df = process__train_dataset()
df.to_csv("qrcd_train.csv", index=False, encoding="utf-8-sig")


def process__test_dataset():
    rows = []

    with open("/content/qrcd_v1.1_dev.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            rows.append({
                "id": item.get("pq_id", ""),
                "question": item.get("question", ""),
                "context": item.get("passage", ""),
                "answer": item["answers"][0]["text"] if item.get("answers") else ""
            })

    return pd.DataFrame(rows)

df_test = process__test_dataset()
df_test.to_csv("qrcd_test.csv", index=False, encoding="utf-8-sig")
