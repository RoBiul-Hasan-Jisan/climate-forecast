import streamlit as st
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load embedding model and FAISS index
# -----------------------------
st.title("Terra PDF Chatbot (Local)")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Update path where your files are stored
DATA_DIR = "D:/game/Terre_chatbot/notebook/model_data"

embeddings_array = np.load(f"{DATA_DIR}/embeddings.npy")
all_chunks = joblib.load(f"{DATA_DIR}/chunks.pkl")
index = faiss.read_index(f"{DATA_DIR}/faiss.index")
faiss.normalize_L2(embeddings_array)

st.write("✅ Model and FAISS index loaded!")

# -----------------------------
# Q&A function
# -----------------------------
def get_answer(question, top_k=3, max_chars=500):
    q_emb = model.encode([question])
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, top_k)

    # Take the most relevant chunk only
    best_chunk = all_chunks[indices[0][0]]

    # Truncate if too long
    if len(best_chunk) > max_chars:
        best_chunk = best_chunk[:max_chars] + "..."

    return best_chunk

# -----------------------------
# Streamlit UI
# -----------------------------
question = st.text_input("Ask a question about Terra:")

if question:
    with st.spinner("🔍 Searching..."):
        answer = get_answer(question)
    st.subheader("Answer (from PDFs):")
    st.write(answer)
