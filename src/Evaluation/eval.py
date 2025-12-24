""" # ===== FIX matplotlib backend (MUST be first) =====
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
 """



import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy,answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
import pandas as pd
from QA import QA_model,QA_with_in_context_learning_model


os.environ["GOOGLE_API_KEY"] = "AIzaSyCATU-Rx5oB4GYk60vfSFST8bm4SkXmT_4"

evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    data = []

    for _, row in df.iterrows():
        data.append({
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),     
        })

    return data

def main():
    eval_data = load_data_csv("/content/test_dataset_V1.csv")
    for item in eval_data[13:]:
        question = item["question"]
        ground_truth = item["answer"]
        prediction,context =QA_with_in_context_learning_model(question)
        prediction = prediction if isinstance(prediction, str) else ""
        print("Question",question)
        print("answer",prediction)
        print("ground_truth",ground_truth)
        print("context",context)
        data_samples = {
       "question":[question],
       "answer": [prediction],
       "contexts": [context],
       "ground_truth": [ground_truth]
    }

        dataset = Dataset.from_dict(data_samples)
        result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

        print(result)


if __name__ == "__main__":
    main()



