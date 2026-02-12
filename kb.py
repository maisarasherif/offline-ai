import os
import time
import json
import requests
import pdfplumber
import re
from typing import List, Dict, Generator
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding  # CPU-based embeddings

# Configuration
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "optimized_rag"
LLM_MODEL = "mistral:7b"  # Or "llama3.2" for even more speed

@dataclass
class Chunk:
    text: str
    metadata: Dict

class PDFProcessor:
    """optimized PDF extraction with sentence boundary detection."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        # Normalize whitespace
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def load_pdf(file_path: str) -> List[Chunk]:
        print(f"   📄 Parsing {os.path.basename(file_path)}...")
        chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    # Simple sentence chunking for speed/coherence
                    # Split by .!? but keep the punctuation
                    sentences = re.split(r'(?<=[.!?]) +', text)
                    
                    current_chunk = []
                    current_len = 0
                    target_chunk_size = 500  # Characters
                    
                    for sentence in sentences:
                        clean_sent = PDFProcessor.clean_text(sentence)
                        if not clean_sent:
                            continue
                            
                        current_chunk.append(clean_sent)
                        current_len += len(clean_sent)
                        
                        if current_len >= target_chunk_size:
                            chunk_text = " ".join(current_chunk)
                            chunks.append(Chunk(
                                text=chunk_text,
                                metadata={
                                    "source": os.path.basename(file_path),
                                    "page": i + 1
                                }
                            ))
                            # Keep last sentence as overlap for next chunk
                            current_chunk = [current_chunk[-1]]
                            current_len = len(current_chunk[0])
                            
                    # Add remaining text
                    if current_chunk:
                        chunks.append(Chunk(
                            text=" ".join(current_chunk),
                            metadata={"source": os.path.basename(file_path), "page": i+1}
                        ))
                        
        except Exception as e:
            print(f"Error reading PDF: {e}")
            
        return chunks

class CPUVectorEngine:
    """
    Handles Embeddings (CPU) and Storage (Qdrant).
    Uses FastEmbed to avoid touching GPU VRAM.
    """
    def __init__(self):
        print("   ⚙️  Initializing CPU Embedding Model (BAAI/bge-small-en-v1.5)...")
        # This runs ON CPU. Lightweight and fast.
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(url=QDRANT_URL)
        
        # Initialize collection
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                # BGE-Small uses 384 dimensions
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def index_documents(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        print(f"   🧠 Embedding {len(chunks)} chunks on CPU...")
        texts = [c.text for c in chunks]
        
        # FastEmbed generates embeddings locally
        embeddings = list(self.embedder.embed(texts))
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            # Create a deterministic ID based on content hash or simple index
            # Using simple index + timestamp for this example
            point_id = int(time.time() * 1000) + i
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"]
                }
            ))
            
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"   ✅ Indexed {len(chunks)} chunks successfully.")

    def search(self, query: str, limit: int = 4) -> List[Dict]:
        # Embed query on CPU
        query_vec = list(self.embedder.embed([query]))[0]
        
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "text": p.payload.get("text"),
                "source": p.payload.get("source"),
                "page": p.payload.get("page")
            }
            for p in results.points
        ]

class GPULLMClient:
    """Handles LLM Generation (GPU) via Ollama."""
    
    @staticmethod
    def generate_stream(prompt: str) -> Generator[str, None, None]:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": True,  # Critical for perceived speed
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 4096 
                    }
                },
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_resp = json.loads(line)
                        if "response" in json_resp:
                            yield json_resp["response"]
                    except:
                        pass
        except Exception as e:
            yield f"Error: {str(e)}"

def main():
    print("🚀 Starting Hybrid RAG System (CPU-Embed / GPU-LLM)")
    print("-" * 50)
    
    # 1. Initialize Engines
    vector_engine = CPUVectorEngine()
    
    # 2. Ingestion Loop (Optional)
    while True:
        choice = input("\n[1] Index a PDF folder\n[2] Ask a question\n[q] Quit\nSelect: ").strip()
        
        if choice == '1':
            folder = input("Enter PDF folder path: ").strip()
            if os.path.exists(folder):
                all_chunks = []
                files = [f for f in os.listdir(folder) if f.endswith('.pdf')]
                for f in files:
                    all_chunks.extend(PDFProcessor.load_pdf(os.path.join(folder, f)))
                vector_engine.index_documents(all_chunks)
            else:
                print("❌ Folder not found.")
                
        elif choice == '2':
            query = input("\n🔍 Question: ").strip()
            if not query: continue
            
            # A. Search (Fast, CPU)
            print("   🔎 Searching...")
            start_time = time.time()
            results = vector_engine.search(query)
            search_time = time.time() - start_time
            
            if not results:
                print("   ⚠️ No relevant documents found.")
                continue

            # B. Construct Prompt
            context_str = "\n\n".join(
                [f"[Source: {r['source']}, Page {r['page']}]\n{r['text']}" for r in results]
            )
            
            full_prompt = f"""You are a helpful assistant. Answer the user question based ONLY on the context below.
            
Context:
{context_str}

Question: {query}
Answer:"""

            # C. Generate (GPU, Streaming)
            print(f"   ⚡ Found info in {search_time:.2f}s. Generating answer...\n")
            print("-" * 50)
            
            # Stream the response
            for token in GPULLMClient.generate_stream(full_prompt):
                print(token, end="", flush=True)
            print("\n" + "-" * 50)
            
        elif choice.lower() == 'q':
            break

if __name__ == "__main__":
    main()