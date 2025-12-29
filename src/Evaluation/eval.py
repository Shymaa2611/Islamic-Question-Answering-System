import os
from ragas import evaluate
from ragas.metrics import answer_relevancy,answer_correctness
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
import pandas as pd



os.environ["GOOGLE_API_KEY"] = "API_KEY"

evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    data = []

    for _, row in df.iterrows():
        data.append({
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),  
            "generatedAnswer": str(row.get("generatedAnswer", "")),     
        })

    return data

def main():
    eval_data = load_data_csv("/content/test_dataset_V1.csv")
    for item in eval_data:
        question = item["question"]
        ground_truth = item["answer"]
        prediction=item["generatedAnswer"]
        print("Question",question)
        print("answer",prediction)
        print("ground_truth",ground_truth)
        data_samples = {
       "question":[question],
       "answer": [prediction],
       "ground_truth": [ground_truth]
    }

        dataset = Dataset.from_dict(data_samples)
        result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            answer_correctness
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

        print(result)
 
if __name__ == "__main__":
    main()



