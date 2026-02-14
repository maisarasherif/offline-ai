import streamlit as st
import os
import time
import json
import requests
import pdfplumber
import re
from typing import List, Generator
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "porto_marine_kb"  # Renamed for your company
LLM_MODEL = "mistral:7b" 

# REPLACE THIS WITH YOUR ACTUAL LOGO URL
# You can use a local file path (e.g., "logo.png") or a public URL
LOGO_URL = "https://i.ibb.co/VYBW0brC/logo-default-slim.png" 

# --- Page Config ---
st.set_page_config(
    page_title="Porto Marine Services - AI",
    page_icon="⚓",
    layout="wide"
)

# --- CSS Styling (Fixed for Dark Mode & Branding) ---
st.markdown("""
<style>
    /* Force chat text to be black even in Dark Mode */
    .stChatMessage {
        background-color: #f0f2f6; 
        border-radius: 10px;
        color: black !important; /* This fixes the white text issue */
    }
    
    /* Ensure user message bubble is distinct */
    div[data-testid="stChatMessage"] {
        background-color: #f0f2f6;
        color: black !important;
    }
    
    /* Specific styling for the user side if needed */
    .stChatMessage.user {
        background-color: #e8f0fe !important;
    }

    /* Hide the 'Deploy' button */
    .stDeployButton {display:none;}
    
    /* Add a subtle border to the sidebar logo */
    [data-testid="stSidebar"] img {
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend Logic (Same as before) ---
@st.cache_resource
def get_vector_engine():
    return CPUVectorEngine()

class CPUVectorEngine:
    def __init__(self):
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(url=QDRANT_URL)
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def search(self, query: str, limit: int = 4):
        query_vec = list(self.embedder.embed([query]))[0]
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=limit,
            with_payload=True
        )
        return results.points

    def index_text(self, text: str, source: str):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = []
        length = 0
        
        for sent in sentences:
            current.append(sent)
            length += len(sent)
            if length > 500:
                chunks.append(" ".join(current))
                current = []
                length = 0
        if current: chunks.append(" ".join(current))
        
        embeddings = list(self.embedder.embed(chunks))
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=int(time.time()*1000) + i,
                vector=vector,
                payload={"text": chunk, "source": source}
            ))
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

def generate_stream(prompt: str):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.3}
            },
            stream=True,
            timeout=60
        )
        for line in response.iter_lines():
            if line:
                json_resp = json.loads(line)
                if "response" in json_resp:
                    yield json_resp["response"]
    except Exception as e:
        yield f"Error: {e}"

# --- UI Layout ---

# Sidebar for controls
with st.sidebar:
    # --- BRANDING SECTION ---
    try:
        # Display Logo with UPDATED parameter
        st.image(LOGO_URL, use_container_width=True) 
    except:
        st.header("Porto Marine Services")
    
    st.markdown("### ⚓ Knowledge Base")
    st.divider()
    
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Index"):
            engine = get_vector_engine()
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with pdfplumber.open(file) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        full_text += page.extract_text() or ""
                    
                    engine.index_text(full_text, file.name)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"Indexed {len(uploaded_files)} files!")
            time.sleep(1)
            st.rerun()

    st.divider()
    st.caption("Powered by Local AI")

# Main Chat Interface
st.title("🚢 Operations Assistant")
st.caption("Porto Marine Services Internal Doc Search")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about safety protocols, schedules, or reports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    engine = get_vector_engine()
    search_results = engine.search(prompt)
    
    if search_results:
        context_text = "\n\n".join([f"[{r.payload['source']}]: {r.payload['text']}" for r in search_results])
        system_prompt = f"""You are an assistant for Porto Marine Services. Use the context below to answer.
        
Context:
{context_text}

Question: {prompt}
Answer:"""
        
        with st.expander("🔎 View Retrieved Sources"):
            for r in search_results:
                st.markdown(f"**{r.payload['source']}**")
                st.caption(f"...{r.payload['text'][:200]}...")
                st.divider()
    else:
        system_prompt = f"Question: {prompt}\nAnswer:"
        context_text = ""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in generate_stream(system_prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})