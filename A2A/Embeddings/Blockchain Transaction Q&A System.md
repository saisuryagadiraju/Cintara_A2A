# Blockchain Transaction Q&A System

A comprehensive AI-powered question-answering system for blockchain transaction data using embeddings, FAISS vector search, and Hugging Face transformers.

## 🚀 Features

- **Semantic Search**: Uses sentence transformers to create embeddings for transaction data
- **Vector Database**: FAISS for efficient similarity search
- **Question Answering**: DistilBERT model for extracting answers from relevant transactions
- **Interactive UI**: Streamlit web interface with sample questions and visualizations
- **Real-time Processing**: Fast retrieval and answering of questions about blockchain transactions

## 📊 System Architecture

```
Transaction Data → Sentence Transformers → FAISS Index
                                              ↓
User Question → Embedding → Similarity Search → Relevant Transactions
                                              ↓
                           DistilBERT QA Model → Answer + Confidence
```

## 🛠️ Components

### 1. Data Processing (`main.py`)
- Loads and preprocesses blockchain transaction data
- Generates embeddings using `all-MiniLM-L6-v2` model
- Creates FAISS index for efficient vector search
- Downloads and saves DistilBERT QA model

### 2. Model Integration (`model_integration.py`)
- `TransactionQA` class for handling the complete pipeline
- Methods for retrieving relevant transactions and generating answers
- Configurable number of relevant transactions to consider

### 3. Streamlit UI (`streamlit_app.py`)
- Interactive web interface
- Sample questions for easy testing
- Real-time answer generation with confidence scores
- Dataset overview with transaction statistics
- Expandable view of relevant transactions used for answering

## 📋 Requirements

```
streamlit
sentence-transformers
transformers
torch
faiss-cpu
pandas
numpy
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install streamlit sentence-transformers transformers torch faiss-cpu pandas numpy
   ```

2. **Prepare Data**:
   - Place your `Transactions_cleaned.csv` file in the project directory
   - Run the data processing script:
   ```bash
   python main.py
   ```

3. **Launch the UI**:
   ```bash
   python -m streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```

4. **Access the Application**:
   - Open your browser and navigate to `http://localhost:8501`
   - Click "Initialize/Reload Models" to load the AI models
   - Start asking questions about your blockchain transactions!

## 💡 Sample Questions

- "What is the transaction type for hash 0x10927...b039478?"
- "Show me contract creation transactions"
- "What transactions happened on block 100055?"
- "Find transactions with status Success"
- "What is the value of transaction 0x09cf8...b2bace6?"

## 🔧 Configuration

### Model Settings
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **QA Model**: `distilbert-base-cased-distilled-squad`
- **Vector Database**: FAISS with L2 distance

### Customization Options
- Adjust the number of relevant transactions to consider (1-10)
- Modify embedding model for different performance/accuracy trade-offs
- Change QA model for domain-specific improvements

## 📈 Performance

- **Dataset Size**: 1,030 transactions
- **Embedding Generation**: ~12 seconds for full dataset
- **Query Response Time**: <2 seconds per question
- **Memory Usage**: ~500MB for models and embeddings

## 🔍 How It Works

1. **Preprocessing**: Transaction data is combined into text format including hash, type, block, addresses, timestamp, value, status, and direction.

2. **Embedding Generation**: Each transaction is converted to a 384-dimensional vector using sentence transformers.

3. **Indexing**: FAISS creates an efficient index for similarity search.

4. **Query Processing**: 
   - User question is embedded using the same model
   - FAISS finds the most similar transactions
   - Relevant transactions are combined as context

5. **Answer Generation**: DistilBERT processes the question and context to extract the most relevant answer with a confidence score.

## 📊 Dataset Overview

The system processes blockchain transaction data with the following fields:
- `txn_hash`: Transaction hash identifier
- `type`: Transaction type (Contract_Creation, Coin_Transfer, etc.)
- `block`: Block number
- `from_address`: Sender address
- `to_address`: Recipient address
- `timestamp`: Transaction timestamp
- `txn_fee_cint`: Transaction fee
- `value_cint`: Transaction value
- `status`: Transaction status (Success/Failed)
- `direction`: Transaction direction

## 🚀 Deployment

The system is designed to be easily deployable:
- Streamlit provides a web interface accessible via browser
- All models are downloaded and cached locally
- FAISS index is saved for quick loading
- No external API dependencies for inference

## 🔮 Future Enhancements

- Support for larger datasets with distributed processing
- Advanced filtering and search capabilities
- Integration with real-time blockchain data feeds
- Multi-language support for questions
- Custom fine-tuned models for blockchain-specific terminology
- Export functionality for search results and answers

## 📝 License

This project is open source and available under the MIT License.

