import streamlit as st
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

# --- Configuration ---
genai.configure(api_key="AIzaSyD38xnKP0Qj30ZEu1PKKpFBZH5TsH1RESg")
model = genai.GenerativeModel("models/gemini-2.5-flash")
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI ---
st.set_page_config(page_title="Next-Level RAG Demo", layout="wide")
st.title("🔍 Graph-RAG + Multi-Hop Reasoning App")

uploaded_files = st.file_uploader("📂 Upload project files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
texts = []

if uploaded_files:
    import fitz, docx
    for f in uploaded_files:
        text = ""
        if f.name.endswith(".pdf"):
            with fitz.open(stream=f.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
        elif f.name.endswith(".docx"):
            d = docx.Document(f)
            text = "\n".join([p.text for p in d.paragraphs])
        else:
            text = f.read().decode("utf-8")
        texts.append(text)

    st.success(f"✅ Loaded {len(texts)} documents.")

    # Chunk text
    chunks = []
    for t in texts:
        for i in range(0, len(t), 500):
            c = t[i:i+500].strip()
            if c:
                chunks.append(c)
    st.write(f"📑 Total chunks: {len(chunks)}")

    # Embeddings + FAISS
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Graph building
    G = nx.DiGraph()
    for t in chunks:
        doc = nlp(t)
        ents = [e.text for e in doc.ents]
        for i in range(len(ents) - 1):
            G.add_edge(ents[i], ents[i+1], relation="related_to")
    st.write(f"🧠 Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges.")

    # Query input
    query = st.text_input("🔎 Enter your question", "Which author proposed Method B and which dataset did they evaluate it on?")
    if st.button("Run Query"):
        # --- Baseline Retrieval ---
        q_emb = embedder.encode([query], convert_to_tensor=False)
        D, I = index.search(np.array(q_emb), k=3)
        context = "\n".join([chunks[i] for i in I[0]])

        baseline_prompt = f"Using the context below, answer the question precisely.\n\n{context}\n\nQuestion: {query}"
        baseline_answer = model.generate_content(baseline_prompt).text.strip()

        # --- Graph-RAG Retrieval ---
        qdoc = nlp(query)
        qents = [e.text for e in qdoc.ents]
        neighborhood = []
        for e in qents:
            if e in G:
                for n in G.neighbors(e):
                    for t in chunks:
                        if n in t:
                            neighborhood.append(t)
        graph_ctx = "\n".join(neighborhood or chunks)

        graph_prompt = f"Answer the question using graph reasoning:\n\n{graph_ctx}\n\nQuestion: {query}"
        graph_answer = model.generate_content(graph_prompt).text.strip()

        # --- Display results ---
        st.subheader("🧩 Baseline RAG Answer")
        st.write(baseline_answer)
        st.subheader("🕸 Graph-RAG Answer")
        st.write(graph_answer)

        # --- Graph visualization ---
        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw(G, with_labels=False, node_color="skyblue", node_size=40, edge_color="gray", ax=ax)
        st.pyplot(fig)
