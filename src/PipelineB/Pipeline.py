import pandas as pd
import numpy as np
import ast
import os
import json
import time
import re
from tqdm import tqdm
from CrossEncoderNotebook import search 

from google import genai
from google.genai import types


# Gemini Configuration

client = genai.Client(api_key="AIzaSyAq9XC0gM_UQ4Ra79kOD25LlC8kD9UoDDw")

def safe_generate(prompt, temperature=0.0):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}. Retrying after 60s...")
        time.sleep(60)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature)
            )
            return response.text.strip()
        except Exception as e2:
            print(f"Second failure: {e2}")
            return ""


def parse_ids(model_output):
    pattern = r"\d+:\d+(?:-\d+)?"
    ids = re.findall(pattern, model_output)
    if not ids:
        return [-1]
    return ids

def filter_results_gemini(question):
    retrieved = search(question)
    context_ids = [r["id"] for r in retrieved]

    PROMPT = f"""
    Given a question in Modern Standard Arabic (MSA) and a list of Quranic and Hadith verses (each with an associated ID), identify the IDs of the verses that contain the answer to the question.
    Instructions:
      - Return only the IDs of the extremely relevant verses in a list, ordered from most relevant to least relevant.
      - Do not explain your answer.
      - If the answer is not found or you are unsure, return [-1].
      - Use the verse ID exactly as provided.
      - Format strictly as a Python list, e.g. [2:4-6, 5:11] or [-1].

    Question: {question}
    Verses: {context_ids}
    """.strip()

    model_output = safe_generate(PROMPT, temperature=0.2)
    return [{"model_output": model_output}]


def get_relevant_passages(question):
 
    results = filter_results_gemini(question)
    model_output = results[0]["model_output"]
    ids = parse_ids(model_output)

    if ids == [-1]:
        return "لا توجد اجابة"
    retrieved = search(question)
    passages = []

    for pid in ids:
        for item in retrieved:
            if item["id"] == pid:
                passages.append(item["text"])

    if not passages:
        return "No matching passages found."

  
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])

    prompt = f"""
      You are a question answering system.
      Question: {question}
      Context passages:{context}
      Give a concise answer using only the information from the passages.
     """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text




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
 "ما حكم التصوير بالأشعة فوق البنفسجية لأغراض الكشف الطبي؟",
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
    print("الاجابة : ",get_relevant_passages(question))
    print("\n\n")





