import streamlit as st
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
import os

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("Terra PDF Chatbot (Local)")

# -----------------------------
# Load embedding model and FAISS index
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# Paths for model data (relative to app.py)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "model_data")  # model_data folder at project root

embeddings_array = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
all_chunks = joblib.load(os.path.join(DATA_DIR, "chunks.pkl"))
index = faiss.read_index(os.path.join(DATA_DIR, "faiss.index"))
faiss.normalize_L2(embeddings_array)

st.success("✅ Model and FAISS index loaded!")


def get_answer(question, top_k=2, max_chars=250):
    q_emb = model.encode([question])
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, top_k)

    # Take the most relevant chunk only
    best_chunk = all_chunks[indices[0][0]].strip()

    # Truncate for brevity
    if len(best_chunk) > max_chars:
        best_chunk = best_chunk[:max_chars].rsplit(" ", 1)[0] + "..."

    return best_chunk


question = st.text_input("Ask a question about Terra:")

if question:
    with st.spinner(" Searching..."):
        answer = get_answer(question)
    st.subheader("Answer (from PDFs):")
    st.write(answer)
