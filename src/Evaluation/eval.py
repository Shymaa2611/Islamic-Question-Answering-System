# ===== FIX matplotlib backend (MUST be first) =====
import os
os.environ["MPLBACKEND"] = "Agg"

# ===== Imports =====
import json
import string
import pandas as pd
from bert_score import BERTScorer
from QA import QA

# ===== Initialize BERTScore =====
scorer = BERTScorer(
    model_type="aubmindlab/bert-base-arabertv02",
    num_layers=12,
    lang="ar"
)

# ===== Load CSV data =====
def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    data = []

    for _, row in df.iterrows():
        data.append({
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", ""))
        })

    return data

# ===== Text normalization =====
def normalize_text(text):
    if not isinstance(text, str):
        return ""

    stop_words = {'من', 'الى', 'إلى', 'عن', 'على', 'في', 'حتى'}
    punctuation = set(string.punctuation) | {'،', '؛', '؟'}

    tokens = [t for t in text.split() if t not in stop_words]
    text = " ".join(tokens)
    text = "".join(ch for ch in text if ch not in punctuation)

    return " ".join(text.split())

# ===== BERTScore computation =====
def compute_bert_score(truth, prediction):
    truth = normalize_text(truth)
    prediction = normalize_text(prediction)

    if truth == "" or prediction == "":
        return 0.0, 0.0, 0.0

    P, R, F1 = scorer.score([prediction], [truth])

    return (
        float(P.mean().item()),
        float(R.mean().item()),
        float(F1.mean().item())
    )

# ===== Main evaluation loop =====
def main():
    eval_data = load_data_csv("/content/qrcd_test.csv")

   # P_scores, R_scores, F1_scores = [], [], []
    print(len(eval_data))
    for item in eval_data[35:50]:
        question = item["question"]
        truth = item["answer"]

        prediction = QA(question)
        prediction = prediction if isinstance(prediction, str) else ""

        p, r, f1 = compute_bert_score(truth, prediction)
        print(f"Precision: {p}")
        print(f"Recall:    {r}")
        print(f"F1 Score:  {f1}")
    print("length of data = ",len(eval_data))

if __name__ == "__main__":
    main()
