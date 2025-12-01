from utility import *

class BlackBoxConfidenceTechniques:
    def __init__(self,api_key):
        genai.configure(api_key=api_key)
        self.model2 = genai.GenerativeModel("gemini-2.5-flash")

    def Vanilla(self,question):
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
        prompt = f"""
        You are a question answering system.
        Question: {question}
        Context passages: {context}

        Answer the question using only the information from the passages. 
        After giving your answer, provide a confidence score from 0 to 1 indicating how confident you are in your answer.

        Use the following technique:
       
        **Vanilla**: Provide a direct answer and confidence.
         
         Format your response like this:

        Technique: <Technique Name>
        Answer: <Answer>
        Confidence: <0-1>
        """

        response = self.model2.generate_content(prompt)
        return response.text

    def Chain_of_thought(self,question):
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
        prompt = f"""
        You are a question answering system.
        Question: {question}
        Context passages: {context}

        Answer the question using only the information from the passages. 
        After giving your answer, provide a confidence score from 0 to 1 indicating how confident you are in your answer.

        Use the following technique:

        **Chain of Thought**: Explain your reasoning step by step before the answer.
    
        Format your response like this:

        Technique: <Technique Name>
        Answer: <Answer>
        Confidence: <0-1>
        """
        response = self.model2.generate_content(prompt)
        return response.text
            

    def self_probing(self,question):
      answer,context=QA(question)
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
      response = self.model2.generate_content(prompt_confidence)

      return response.text,answer
       
    def multi_step(self,question):
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
        prompt = f"""
        You are a question answering system.
        Question: {question}
        Context passages: {context}

        Answer the question using only the information from the passages. 
        After giving your answer, provide a confidence score from 0 to 1 indicating how confident you are in your answer.

        Use the following technique:

         **Multi-Step**: Break the question into smaller steps, answer each, then give a final answer and confidence.
       
        Format your response like this:

        Technique: <Technique Name>
        Answer: <Answer>
        Confidence: <0-1>
        """

        response = self.model2.generate_content(prompt)
        return response.text


    def top_k(self,question):
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])
        prompt = f"""
        You are a question answering system.
        Question: {question}
        Context passages: {context}

        Answer the question using only the information from the passages. 
        After giving your answer, provide a confidence score from 0 to 1 indicating how confident you are in your answer.

        Use the following technique:

        **Top-K**: Generate top 5 possible answers and pick the most likely one; provide confidence based on agreement.

        Format your response like this:

        Technique: <Technique Name>
        Answer: <Answer>
        Confidence: <0-1>
        """

        response = self.model2.generate_content(prompt)
        return response.text


    def self_random(self,question):
        candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
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
        response = self.model2.generate_content(prompt)
        return response.text



    def prompting(self,question):
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
        response = self.model2.generate_content(prompt)
        return response.text

    def misleading(self,question):
        pass





