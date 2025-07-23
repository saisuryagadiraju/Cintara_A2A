import os
from huggingface_hub import login

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token)


import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import faiss
import numpy as np
import os

# Load the dataset
df = pd.read_csv('Transactions_cleaned.csv')

# Combine relevant columns for embedding
df['text_data'] = df.apply(lambda row: f"Transaction Hash: {row['txn_hash']}, Type: {row['type']}, Block: {row['block']}, From: {row['from_address']}, To: {row['to_address']}, Timestamp: {row['timestamp']}, Value: {row['value_cint']}, Status: {row['status']}, Direction: {row['direction']}", axis=1)

# Initialize the Sentence Transformer model
# Using a smaller, faster model for demonstration. For better accuracy, consider 'all-MiniLM-L6-v2' or 'all-mpnet-base-v2'
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(df['text_data'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# Save embeddings and text data for later use
np.save('transaction_embeddings.npy', embeddings)
df[['txn_hash', 'text_data']].to_csv('transaction_data_with_text.csv', index=False)

print("Embeddings and text data saved.")

# Create a FAISS index
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, 'transaction_faiss_index.bin')
print("FAISS index created and saved.")

# Initialize a question-answering pipeline from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Save the model and tokenizer (optional, for offline use)
qa_pipeline.save_pretrained("./qa_model")
print("Question-answering model downloaded and saved.")

print("Phase 1 complete: Data analysis, embedding generation, and model setup are done.")


