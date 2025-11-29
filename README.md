# Islamic Question Answering System 

The Islamic Question Answering System is designed to provide precise and contextually relevant answers to questions posed in Modern Standard Arabic (MSA) by leveraging two primary Islamic textual resources: the Holy Qur’an and Sahih Al-Bukhari Hadith collections. The system operates in two main stages:

## Passage Retrieval:
- Given a free-text question posed in MSA, a collection of Qur'anic passages (that cover the Holy Qur'an) and a collection of Hadiths from Sahih Bukhari, a system is required to retrieve a ranked list of up-to 20 answer-bearing Qur'anic passages or Hadiths (i.e., Islamic sources that potentially enclose the answer(s) to the given question) from the two collections. The question can be a factoid or non-factoid question. To make the task more realistic (thus challenging), some questions may not have an answer in the Holy Qur'an and Sahih Al-Bukhari. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 20 answer-bearing sources.
## Answer Extraction:
- Uses Gemini to extract accurate answers from the retrieved passages, supporting both fact-based and explanatory questions.

## Pipeline A – CrossEncoder + Gemini
- Overview: Pipeline A is a full question-answering system using a CrossEncoder for retrieval and Gemini for answer extraction.

- Retrieval: The CrossEncoder ranks Qur’anic and Hadith passages according to their relevance to the input MSA question, returning up to 20 candidate passages.
- Answer Extraction: Gemini processes the retrieved passages to extract precise, context-aware answers.

## Pipeline B – Gemini Retrieval + Gemini Extraction
- Overview: Pipeline B is an end-to-end system where Gemini performs both retrieval and answer extraction.

- Retrieval: Gemini identifies relevant Qur’anic and Hadith passages directly from the collections, leveraging deep contextual understanding.
- Answer Extraction: Gemini extracts the final answer from the retrieved passages.

## Installation & Requirements

### Requirements
- Python 3.10
- Gemini API
- Additional libraries:
``bash
!pip install huggingface_hub==0.13.4
!pip install -U sentence-transformers
!pip install faiss-cpu
!pip install -U google-generativeai
!pip install -U google-genai

```

### Installation
1- Clone the repository:

```bash

git clone https://github.com/Shymaa2611/Islamic-Question-Answering-System.git
cd Islamic-Question-Answering-System

```

2- Create and activate a virtual environment (optional but recommended):
```bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```
3- Install dependencies:
```bash

pip install -r requirements.txt

```

5- Run System:
 - RUN USING PIPELINE A
 ```bash

!python src/PipelineA/Pipeline.py

```

 - RUN USING PIPELINE B
 ```bash

!python src/PipelineB/Pipeline.py

```