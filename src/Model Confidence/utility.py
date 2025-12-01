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
genai.configure(api_key="AIzaSyC59FWVuHXHvxbnLNRYJPXjw9bQtWCL5xM")
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

diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670]')

quran_passages = []
with open("/content/IslamicEval2025/data/Task Data/data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            passage_id = parts[0]
            passage_text = parts[1]
            quran_passages.append({"text": passage_text, "source": "quran", "id": passage_id})

hadith_passages = []
with open("/content/IslamicEval2025/data/Task Data/data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", "r", encoding="utf-8") as f:
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
   # genai.configure(api_key="AIzaSyAq9XC0gM_UQ4Ra79kOD25LlC8kD9UoDDw")
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt = f"""
      You are a question answering system.
      Question: {question}
      Context passages:{context}
      Give a concise answer using only the information from the passages.
     """
    response = model2.generate_content(prompt)
    return response.text , context

def QA_Confidence(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
   # genai.configure(api_key="AIzaSyAVKQuV5iGyZXZ4RQ83DontdcBNRKts8wc")
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt = f"""
    You are a question answering system.
    Question: {question}
    Context passages: {context}

    Answer the question using only the information from the passages. 
    After giving your answer, provide a confidence score from 0 to 1 indicating how confident you are in your answer.

    Use the following techniques:

    1. **Vanilla**: Provide a direct answer and confidence.
    2. **Chain of Thought**: Explain your reasoning step by step before the answer.
    3. **Multi-Step**: Break the question into smaller steps, answer each, then give a final answer and confidence.
    4. **Top-K**:Generate 5 possible answers. For each answer, provide a confidence score based on agreement and reasoning. Identify the answer with the **highest confidence** and mark it as the final answer.


    Format your response like this:

    Technique: <Technique Name>
    Answer: <Answer>
    Confidence: <0-1>
    """

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



def self_probing_technique(question):
      answer,context=QA(question)
      model2 = genai.GenerativeModel("gemini-2.5-flash")
      prompt_confidence = f"""
        You are a question answering system.
        Question: {question}
        Context:{context}
        Given the following answer:
        Answer: {answer}

        Based on the context and your knowledge, assess how confident you are that this answer is correct. 
        Provide a confidence score from 0 to 1.
        Confidence:
        """
      response = model2.generate_content(prompt_confidence)

      return response.text,answer


def self_random(question):
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    model2 = genai.GenerativeModel("gemini-2.5-flash")
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
    prompt =f"""
    You are a question answering system.
    Question: {question}
    Context passages: {context}

    Generate 5 different possible answers to this question. 
    Each answer should be as independent and random as possible while still being valid.
    After generating the answers, identify the answer that appears most consistent across your generations. 
    Provide a final answer and a confidence score from 0 to 1 based on how many of the 5 answers agree.

    Format your response like this:

    Technique: Self-Random
    Question: {question}

    Answer 1: <text>
    Answer 2: <text>
    Answer 3: <text>
    Answer 4: <text>
    Answer 5: <text>

    Final Answer: <most consistent answer>
    Final Confidence: <0-1 based on agreement>
    """
    response = model2.generate_content(prompt)
    return response.text

def prompting(question):
        model2 = genai.GenerativeModel("gemini-2.5-flash")
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
        prompt = f"""
        You are a question answering system that can also paraphrase questions.
        Original Question: {question}
        Context passages: {context}

        Step 1: Generate 5 different paraphrased versions of the question. 
        - Each paraphrase should preserve the original meaning but use different wording and sentence structure.

        Step 2: For each paraphrased question, generate one valid answer based on the context.

        Step 3: After generating all answers, identify the answer that appears most consistent across your generations. 
        - Provide a final answer and a confidence score from 0 to 1 based on how many of the 5 answers agree.

        Format your response like this:

        Technique: Self-Random + Question Paraphrasing
        Original Question: {question}
        Paraphrased Question 1: ...
        Answer 1: ...
        Paraphrased Question 2: ...
        Answer 2: ...
        Paraphrased Question 3: ...
        Answer 3: ...
        Paraphrased Question 4: ...
        Answer 4: ...
        Paraphrased Question 5: ...
        Answer 5: ...
        Final Answer: ...
        Confidence: ...
        """
        response = model2.generate_content(prompt)
        return response.text


for question in Questions:
    print (f"=============== السؤال :  {question} ====================")
    answer= QA_Confidence(question)
    confidence,answer2= self_probing_technique(question)
    answer3=self_random(question)
    answer4=prompting(question)
    #answer=clean_and_format(answer)
    print("\n")
    print(answer)
    print("5.self-Proping: ",answer2,'\n',confidence)
    print("6.self-random: ",answer3)
    print(answer4)
    print("\n\n")
  





