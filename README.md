# Islamic Question Answering System 

The Islamic Question Answering System is designed to provide precise and contextually relevant answers to questions posed in Modern Standard Arabic (MSA) by leveraging two primary Islamic textual resources: the Holy Qur’an and Sahih Al-Bukhari Hadith collections. The system operates in two main stages:

Passage Retrieval:
- Given a free-text question posed in MSA, a collection of Qur'anic passages (that cover the Holy Qur'an) and a collection of Hadiths from Sahih Bukhari, a system is required to retrieve a ranked list of up-to 20 answer-bearing Qur'anic passages or Hadiths (i.e., Islamic sources that potentially enclose the answer(s) to the given question) from the two collections. The question can be a factoid or non-factoid question. To make the task more realistic (thus challenging), some questions may not have an answer in the Holy Qur'an and Sahih Al-Bukhari. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 20 answer-bearing sources.
Answer Extraction:
Uses Gemini to extract accurate answers from the retrieved passages, supporting both fact-based and explanatory questions.