
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class TransactionQA:
    def __init__(self, model_name="all-MiniLM-L6-v2", qa_model_path="./qa_model"):
        self.embedding_model = SentenceTransformer(model_name)
        self.qa_pipeline = pipeline("question-answering", model=qa_model_path)
        self.faiss_index = faiss.read_index("transaction_faiss_index.bin")
        self.transaction_data = pd.read_csv("transaction_data_with_text.csv")

    def get_relevant_transactions(self, query, k=5):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.faiss_index.search(query_embedding, k)
        relevant_indices = I[0]
        return self.transaction_data.iloc[relevant_indices]["text_data"].tolist()

    def answer_question(self, question, context):
        QA_input = {
            "question": question,
            "context": context
        }
        res = self.qa_pipeline(QA_input)
        return res["answer"]

if __name__ == "__main__":
    qa_system = TransactionQA()
    question = "What is the transaction type for 0x10927...b039478?"
    relevant_transactions = qa_system.get_relevant_transactions(question)
    context = "\n".join(relevant_transactions)
    answer = qa_system.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


