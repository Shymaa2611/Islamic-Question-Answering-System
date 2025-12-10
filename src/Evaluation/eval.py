import json
from src.in_context_learning.QA import QA

with open("/content/Islamic-Question-Answering-System/data/tydiqa-goldp-v1.1-dev.json", "r", encoding="utf-8") as f:
    data = json.load(f)

eval_data = []
for item in data["data"]:
    for paragraph in item["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"] if qa["answers"] else ""
            
            eval_data.append({
                "question": question,
                "context": context,
                "answer": answer
            })


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)


def main():
    Exact_match=[]
    F1_list=[]
    for item in eval_data:
        question=item['question']
        truth=item['answer']
        prediction=QA(question)
        EM=compute_exact_match(prediction, truth)
        Exact_match.append(EM)
        F1=compute_f1(prediction, truth)
        F1_list.append(F1)
    print("EM",sum(Exact_match)/len(eval_data)*100 )
    print("F1 = ",sum(F1_list)/len(eval_data)*100)


if __name__=="main":
  main()
        

