# !pip install faiss-gpu-cu11==1.10.0
# !pip install --upgrade sentence_transformers

import pandas as pd
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login, hf_hub_download
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import ast
import random
import os
import gzip
import re
import google.generativeai as genai
import re
from huggingface_hub import snapshot_download
from run import template
snapshot_download(
    repo_id="SeragAmin/NAMAA-retriever-cosine-final_60-90",
    repo_type="model",
    local_dir="retriever_model",
    allow_patterns="NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985/*"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD RETRIEVAL MODEL
retrieval_model = SentenceTransformer("retriever_model/NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985")
retrieval_tokenizer = retrieval_model.tokenizer
retrieval_model.to(device)
retrieval_model.eval()

# EMBEDS FUNCTION
def get_embedding(text):
    with torch.no_grad():
        emb = retrieval_model.encode(text, convert_to_numpy=True, device=device)
    return emb

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

model = CrossEncoder("yoriis/GTE-tydi-quqa-haqa")

#test_df = pd.read_csv("/content/IslamicEval2025/data/Task Data/data/QH-QA-25_Subtask2_ayatec_v1.3_test.tsv", sep="\t", names=["question_id", "question"])

diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670]')

quran_passages = []
with open("/content/Islamic-Question-Answering-System/data/QH-QA-25_Subtask2_QPC_v1.1.tsv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            passage_id = parts[0]
            passage_text = parts[1]
            quran_passages.append({"text": passage_text, "source": "quran", "id": passage_id})

hadith_passages = []
with open("/content/Islamic-Question-Answering-System/data/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = ast.literal_eval(line.strip())
            cleaned_text = diacritics_pattern.sub('', item['hadith'])
            hadith_passages.append({
                  "text": cleaned_text,
                  "source": "hadith",
                  "id": item['hadith_id']
            })
        except Exception as e:
            print(f"Skipping invalid line: {e}")

all_passages = quran_passages + hadith_passages
print(f" Loaded total passages: {len(all_passages)}")

quran_texts = [p["text"] for p in quran_passages]
hadith_texts = [p["text"] for p in hadith_passages]

# Encode
quran_embeddings = retrieval_model.encode(
    quran_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

hadith_embeddings = retrieval_model.encode(
    hadith_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
quran_index = build_faiss_index(quran_embeddings)
hadith_index = build_faiss_index(hadith_embeddings)

def search(query, k_quran=50, k_hadith=20):
    query_emb = get_embedding(query)

    # Search separately
    D_q, I_q = quran_index.search(np.array([query_emb]), k_quran)
    D_h, I_h = hadith_index.search(np.array([query_emb]), k_hadith)

    results = []

    for i, score in zip(I_q[0], D_q[0]):
        passage = quran_passages[i]
        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "quran",
            "text": passage["text"]
        })

    for i, score in zip(I_h[0], D_h[0]):
        passage = hadith_passages[i]
        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "hadith",
            "text": passage["text"]
        })

    # Optionally, sort by score (before reranking)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results

"""
def predict_test_rerank_crossencoder(test_df, model, search_fn, k_retrieve=70, score_threshold=0.15, max_returned=20):
    test_df["question_id"] = test_df["question_id"].astype(str)

    qid_to_question = test_df.drop_duplicates("question_id")[["question_id", "question"]].set_index("question_id")["question"].to_dict()

    all_results = []

    for qid, question in tqdm(qid_to_question.items(), desc="Questions"):
        retrieved = search_fn(question)
        candidate_texts = [r["text"] for r in retrieved]
        candidate_ids = [r["id"] for r in retrieved]

        reranked = model.rank(query=question, documents=candidate_texts)

        filtered = [item for item in reranked if item['score'] >= score_threshold]
        filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:max_returned]

        if not filtered:
            all_results.append({
                "question_id": qid,
                "passage_id": "-1",
                "score": 0,
            })
            continue

        for item in filtered:
            all_results.append({
                "question_id": qid,
                "passage_id": candidate_ids[item['corpus_id']],
                "score": item['score']
            })

    return pd.DataFrame(all_results)

test_results = predict_test_rerank_crossencoder(test_df, model, search_fn=search, k_retrieve=70)

test_results.to_csv("run_03_crossenc.csv", index = False)
"""

"""
def trec_formatter(df):
    tag = "nur_run03"

    # Rank passages within each question_id
    df["rank"] = df.groupby("question_id")["score"].rank(ascending=False, method="first").astype(int)

    # Create TREC-format DataFrame
    trec_df = pd.DataFrame({
        "qid": df["question_id"],
        "Q0": "Q0",
        "pid": df["passage_id"],
        "rank": df["rank"],
        "score": df["score"].round(4),
        "tag": tag
    })

    # Save as TSV
    return trec_df


trec_df = trec_formatter(test_results)

trec_df.to_csv("ranked_03.tsv", sep="\t", index=False, header=False)

"""

# Predict Question RElevant Passages
def predict_Question_rerank_crossencoder(question, model, search_fn, k_retrieve=70, score_threshold=0.15, max_returned=20):
    all_results = []
    retrieved = search_fn(question)
    candidate_texts = [r["text"] for r in retrieved]
    #candidate_ids = [r["id"] for r in retrieved]

    # rerank step
    reranked = model.rank(query=question, documents=candidate_texts)
    # filter and sort
    filtered = [item for item in reranked if item['score'] >= score_threshold]
    filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:max_returned]
    # check if zero answer
    if not filtered:
            all_results.append({
               
                "لا توجد اجابة"
            })
        
    # collect top texts
    for item in filtered:
        corpus_id = item['corpus_id']
        if corpus_id < len(candidate_texts):
            all_results.append(candidate_texts[corpus_id])

    return all_results

def QA(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    genai.configure(api_key="AIzaSyBQYT-wo_y1ZmUbwgWkpwQmcp_z5ZZIkwM")
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt=template(question,context)
    response = model2.generate_content(prompt)
    return response.text

# Perprocessing For Answer Format


def clean_and_format(text):
    # Remove patterns like (Passage 4), Passage 4, [Passage 4], etc.
    text = re.sub(r'\(*\s*Passage\s*\d+\s*\)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(*\s*P\s*\d+\s*\)*', '', text, flags=re.IGNORECASE)

    # Remove Markdown **text**
    text = re.sub(r'\*{2}(.*?)\*{2}', r'\1', text)

    # Remove numbers (Arabic and English)
    text = re.sub(r'[0-9٠-٩]+', '', text)

    # Remove list numbers like "1. "
    text = re.sub(r'\s*\d+\.\s*', '', text)

    # Remove remaining asterisks
    text = text.replace("*", "")

    # Remove repeated commas and spaces around commas
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'(,\s*){2,}', ', ', text)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text


""" 

#print(get_relevant_passages("اريد حديث عن عدم التحدث بسوء"))
Questions=[
 "من هو أول إنسان خلقه الله؟",
 "من هو النبي الذي بنى الكعبة؟",
 "لماذا حرّم الله شرب الخمر في القرآن؟",
 "كيف وصف القرآن العلاقة بين الزوج والزوجة؟",
 "ماذا أمر الله بشأن الأيتام في القرآن؟",
 "ما العقوبة التي ذكرها القرآن للسرقة؟",
 "كيف وصف القرآن يوم القيامة؟",
 "ماذا قال القرآن عن مساعدة الفقراء؟",
 "ماذا قال النبي ﷺ عن النية؟",
 "أي حديث يصف أركان الإسلام؟",
 "ماذا قال النبي ﷺ عن حسن الخلق؟",
 "ما الحديث الذي يصف الإيمان؟",
 "ما هى الدلائل التي تشير بأن الانسان مخير؟",
 "ما هو النبي المعروف بالصبر؟",
 "من هم الملائكة المذكورون فى القرأن",
 "ما هو الإتقان؟",
 "ما هي وصايا الله ورسوله في معاملة الوالدين؟",
 "ما الأعمال التي قد تحجب الرزق؟",
 "ما هي الصفات السلبية لطبيعة النفس الإنسانية؟",
 "ما الحكمة في عدد ركعات كل صلاة بالتحديد؟",
 "لست مقتنعًا بالصلاة، فهل يمكنك أن تذكر لي فوائدها أو الحكمة من أدائها؟",
 "هل يعتبر الاعتراف بالذنب من الأعمال الصالحة؟",
 "هل تدبر القرآن فرض؟",
 "هل للصدقة علاقة بنمو المال؟",
 "ما هي وصايا لقمان لابنه؟",
 "ما هي شجرة الزقوم؟",
 "كم مدة عدة الأرملة؟"

]

for question in Questions:
    print (f"=============== السؤال :  {question} ====================")
    answer= QA(question)
    #answer=clean_and_format(answer)
    print("\n")
    print("الاجابة : ",answer)
    print("\n\n")

 """


