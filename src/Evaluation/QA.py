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
import requests
import json
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


all_passages = quran_passages 
print(f" Loaded total passages: {len(all_passages)}")

quran_texts = [p["text"] for p in quran_passages]


# Encode
quran_embeddings = retrieval_model.encode(
    quran_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


quran_index = build_faiss_index(quran_embeddings)

def search(query, k_quran=50, k_hadith=20):
    query_emb = get_embedding(query)

    # Search separately
    D_q, I_q = quran_index.search(np.array([query_emb]), k_quran)
   
    results = []

    for i, score in zip(I_q[0], D_q[0]):
        passage = quran_passages[i]
        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "quran",
            "text": passage["text"]
        })

 
    # Optionally, sort by score (before reranking)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results


# Predict Question RElevant Passages
def predict_Question_rerank_crossencoder(question, model, search_fn, k_retrieve=70, score_threshold=0.15, max_returned=3):
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

def QA_with_in_context_learning(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    genai.configure(api_key="API_KEY")
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt=template(question,context)
    response = model2.generate_content(prompt)
    return response.text

def QA(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    genai.configure(api_key="API_KEY")
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt = f"""
      You are a question answering system.
      Question: {question}
      Context passages:{context}
      Give a concise short answer using only the information from the passages.
     """
    response = model2.generate_content(prompt)
    return response.text

def QA_model(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt = f"""
        You are a precise question answering system.

        Strict rules:
        - Answer with ONE sentence only.
        - Use ONLY information explicitly stated in the context.
        - Do NOT infer, explain, paraphrase, or add details.
        Question:
        {question}

        Context:
        {context}

        answer:
        """

    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer API_KEY",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>", 
    },
    data=json.dumps({
        "model": "mistralai/devstral-2512:free",
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ]
    })
    )
    response.raise_for_status()  
  
    return response.json()["choices"][0]["message"]["content"],candiated_passages

   
def QA_with_in_context_learning_model(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt=template(question,context)
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer API_KEY",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>", 
        "X-Title": "<YOUR_SITE_NAME>", 
    },
    data=json.dumps({
        "model": "mistralai/devstral-2512:free",
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ]
    })
    )
    response.raise_for_status()  
  
    return response.json()["choices"][0]["message"]["content"],candiated_passages

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


