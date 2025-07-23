import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Set page configuration
st.set_page_config(
    page_title="Blockchain Transaction Q&A",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .context-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
    st.session_state.loading = False

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
        return self.transaction_data.iloc[relevant_indices]["text_data"].tolist(), D[0]

    def answer_question(self, question, context):
        QA_input = {
            "question": question,
            "context": context
        }
        res = self.qa_pipeline(QA_input)
        return res["answer"], res["score"]

# Main app
def main():
    st.markdown('<h1 class="main-header">🔗 Blockchain Transaction Q&A System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        st.info("This system uses AI to answer questions about blockchain transactions.")
        
        st.header("🔧 Model Details")
        st.write("**Embedding Model:** all-MiniLM-L6-v2")
        st.write("**QA Model:** DistilBERT")
        st.write("**Vector Database:** FAISS")
        
        if st.button("🔄 Initialize/Reload Models"):
            st.session_state.loading = True
            st.rerun()

    # Initialize models
    if st.session_state.qa_system is None or st.session_state.loading:
        if st.session_state.loading:
            with st.spinner("Loading models... This may take a moment."):
                try:
                    st.session_state.qa_system = TransactionQA()
                    st.session_state.loading = False
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
                    st.stop()
        else:
            st.warning("⚠️ Please initialize the models using the sidebar button.")
            st.stop()

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Sample questions
        st.subheader("📝 Sample Questions")
        sample_questions = [
            "What is the transaction type for hash 0x10927...b039478?",
            "Show me contract creation transactions",
            "What transactions happened on block 100055?",
            "Find transactions with status Success",
            "What is the value of transaction 0x09cf8...b2bace6?"
        ]
        
        selected_sample = st.selectbox("Choose a sample question:", [""] + sample_questions)
        
        # Question input
        question = st.text_area(
            "Enter your question about the blockchain transactions:",
            value=selected_sample if selected_sample else "",
            height=100,
            placeholder="e.g., What is the transaction type for hash 0x10927...b039478?"
        )
        
        # Number of relevant transactions to retrieve
        k_value = st.slider("Number of relevant transactions to consider:", 1, 10, 5)
        
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Searching for relevant transactions and generating answer..."):
                    try:
                        # Get relevant transactions
                        relevant_transactions, distances = st.session_state.qa_system.get_relevant_transactions(question, k=k_value)
                        context = "\n".join(relevant_transactions)
                        
                        # Get answer
                        answer, confidence = st.session_state.qa_system.answer_question(question, context)
                        
                        # Display results
                        st.markdown('<div class="question-box">', unsafe_allow_html=True)
                        st.write(f"**Question:** {question}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.write(f"**Answer:** {answer}")
                        st.write(f"**Confidence Score:** {confidence:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show relevant transactions
                        with st.expander("📄 View Relevant Transactions", expanded=False):
                            st.markdown('<div class="context-box">', unsafe_allow_html=True)
                            for i, (transaction, distance) in enumerate(zip(relevant_transactions, distances)):
                                st.write(f"**Transaction {i+1}** (Distance: {distance:.4f})")
                                st.write(transaction)
                                st.write("---")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.header("📈 Dataset Overview")
        
        # Load and display dataset info
        try:
            df = pd.read_csv("transaction_data_with_text.csv")
            st.metric("Total Transactions", len(df))
            
            # Load original data for more stats
            original_df = pd.read_csv('Transactions_cleaned.csv')
            
            st.subheader("📊 Transaction Types")
            type_counts = original_df['type'].value_counts()
            st.bar_chart(type_counts)
            
            st.subheader("✅ Status Distribution")
            status_counts = original_df['status'].value_counts()
            for status, count in status_counts.items():
                st.write(f"**{status}:** {count}")
                
            st.subheader("🔄 Direction Distribution")
            direction_counts = original_df['direction'].value_counts()
            for direction, count in direction_counts.items():
                st.write(f"**{direction}:** {count}")
                
        except Exception as e:
            st.error(f"Error loading dataset info: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

