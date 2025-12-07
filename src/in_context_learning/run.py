# !pip install faiss-gpu-cu11==1.10.0
# !pip install --upgrade sentence_transformers

import pandas as pd
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

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
model = CrossEncoder("yoriis/GTE-tydi-quqa-haqa")

def get_embedding(text):
    with torch.no_grad():
        emb = retrieval_model.encode(text, convert_to_numpy=True, device=device)
    return emb

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def load_passages_csv(file_path, source_name):
    df = pd.read_csv(file_path)
    passages = []
    for _, row in df.iterrows():
        passages.append({
            "id": row["passage_id"],
            "text": row["passage"],
            "question": row["question"],
            "answer": row["answer"] if "answer" in row else "",
            "source": source_name
        })
    return passages

quran_passages = load_passages_csv("/content/Islamic-Question-Answering-System/data/QUQA/train_quqa.csv", "quran")
hadith_passages = load_passages_csv("/content/Islamic-Question-Answering-System/data/HAQA/haqa_train.csv", "hadith")
all_passages = quran_passages + hadith_passages

print(f"Total passages loaded: {len(all_passages)}")

quran_texts = [p["text"] for p in quran_passages]
hadith_texts = [p["text"] for p in hadith_passages]

quran_embeddings = retrieval_model.encode(quran_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
hadith_embeddings = retrieval_model.encode(hadith_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

quran_index = build_faiss_index(quran_embeddings)
hadith_index = build_faiss_index(hadith_embeddings)


def search(query, k_quran=50, k_hadith=20):
    query_emb = get_embedding(query)
    D_q, I_q = quran_index.search(np.array([query_emb]), k_quran)
    D_h, I_h = hadith_index.search(np.array([query_emb]), k_hadith)

    results = []
    for i, score in zip(I_q[0], D_q[0]):
        passage = quran_passages[i]
        results.append({"score": float(score), "id": passage["id"], "source": "quran", "text": passage["text"]})
    for i, score in zip(I_h[0], D_h[0]):
        passage = hadith_passages[i]
        results.append({"score": float(score), "id": passage["id"], "source": "hadith", "text": passage["text"]})

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results

def predict_Question_rerank_with_QA(question, model, search_fn, all_passages, k_retrieve=70, score_threshold=0.15, max_returned=5):
    all_results = []
    retrieved = search_fn(question)
    candidate_texts = [r["text"] for r in retrieved]
    candidate_ids = [r["id"] for r in retrieved]

    reranked = model.rank(query=question, documents=candidate_texts)
    filtered = [item for item in reranked if item['score'] >= score_threshold]
    filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:max_returned]

    if not filtered:
        return [{"question": question, "answer": "لا توجد اجابة", "text": "", "source": "", "score": 0}]

    for item in filtered:
        corpus_id = item['corpus_id']
        if corpus_id < len(candidate_ids):
            passage_id = candidate_ids[corpus_id]
            row = next((p for p in all_passages if p["id"] == passage_id), None)
            if row:
                all_results.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "text": row["text"],
                    "source": row["source"],
                    "score": item['score']
                })
    return all_results

def template(query,new_context):
     new_question = query
     results = predict_Question_rerank_with_QA(query, model, search, all_passages)
     few_shot_examples = ""
     for r in results:
        question = r.get("question", "")
        context = r.get("text", "")
        answer = r.get("answer", "")
        
        if question and context and answer: 
            few_shot_examples += f"Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
  
     prompt = few_shot_examples + f"Question: {new_question}\nContext: {new_context}\nAnswer:"
     return prompt






