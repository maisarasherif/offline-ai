"""
Enhanced RAG System for PDF Documents
Combines best practices from multiple implementations:
- pdfplumber for better table extraction (from new_rag.py)
- Proper text cleaning and sentence-boundary chunking (from new_rag_v1.py)
- Qdrant vector database for production-ready persistence
- Page-level metadata tracking for better citations
- Error handling and retry logic
- Configurable chunking strategies
"""

import requests
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pdfplumber  # Better table handling than pypdf


@dataclass
class DocumentChunk:
    """Structured chunk with metadata."""
    id: str
    text: str
    metadata: Dict


class PDFLoader:
    """Enhanced PDF loader with table preservation using pdfplumber."""
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with layout preservation for tables.
        Returns list of pages with metadata.
        """
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract with layout=True to preserve table structure
                    text = page.extract_text(layout=True)
                    
                    if text and text.strip():
                        pages.append({
                            'text': text,
                            'metadata': {
                                'source': os.path.basename(pdf_path),
                                'page': i + 1,
                                'type': 'pdf',
                                'total_pages': len(pdf.pages)
                            }
                        })
        except Exception as e:
            print(f"⚠️  Error loading PDF {pdf_path}: {e}")
        
        return pages


class TextChunker:
    """Smart text chunking with sentence boundaries and cleaning."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline (paragraph breaks)
        text = re.sub(r'\n\n+', '\n\n', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\n\(\)\[\]\/\%\$]', '', text)
        return text.strip()
    
    @staticmethod
    def chunk_by_sentences(
        text: str, 
        chunk_size: int = 800, 
        overlap: int = 150,
        metadata: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """
        Split text at sentence boundaries for better semantic coherence.
        Increased chunk_size (800) to fit more table context.
        """
        text = TextChunker.clean_text(text)
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to find sentence boundary near the end
            if end < len(text):
                # Look back up to 20% of chunk size for sentence end
                search_start = end - int(chunk_size * 0.2)
                sentence_end = max(
                    text.rfind('.', search_start, end),
                    text.rfind('!', search_start, end),
                    text.rfind('?', search_start, end)
                )
                
                # If found a sentence boundary, use it
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = chunk_index
                chunk_id = f"{chunk_metadata.get('source', 'unknown')}_p{chunk_metadata.get('page', 0)}_{chunk_index}"
                
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_metadata
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text) - overlap:
                break
        
        return chunks


class EmbeddingService:
    """Handles embedding generation with retry logic."""
    
    def __init__(self, model_name: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def embed_text(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Generate embeddings with retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": text},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["embeddings"][0]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Embedding attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Embedding failed after {max_retries} attempts: {e}")
                    return None


class LLMService:
    """LLM interface with improved prompting."""
    
    def __init__(self, model_name: str = "mistral:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response with configurable temperature."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error generating response: {e}"


class RAGSystem:
    """Complete RAG system with Qdrant and Ollama."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        qdrant_url: str = "http://localhost:6333",
        embedding_model: str = "mxbai-embed-large",
        llm_model: str = "mistral:7b"
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = EmbeddingService(model_name=embedding_model)
        self.llm = LLMService(model_name=llm_model)
        
        # Initialize collection if needed
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the current collection."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                'points_count': collection_info.points_count,
                'exists': True
            }
        except:
            return {'points_count': 0, 'exists': False}
    
    def ingest_pdfs(
        self, 
        pdf_directory: str,
        chunk_size: int = 800,
        overlap: int = 150,
        batch_size: int = 10
    ):
        """
        Process and index all PDFs from a directory.
        
        Args:
            pdf_directory: Path to directory containing PDFs
            chunk_size: Size of text chunks (increased for tables)
            overlap: Overlap between chunks
            batch_size: Number of chunks to process before showing progress
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            return
        
        print(f"📚 Found {len(pdf_files)} PDF files. Starting ingestion...\n")
        
        point_id = 0
        total_chunks = 0
        
        for pdf_file in pdf_files:
            print(f"📄 Processing: {pdf_file.name}")
            
            # Extract pages from PDF
            pages = PDFLoader.extract_from_pdf(str(pdf_file))
            print(f"   Extracted {len(pages)} pages")
            
            # Chunk each page
            file_chunks = []
            for page in pages:
                page_chunks = TextChunker.chunk_by_sentences(
                    page['text'],
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata=page['metadata']
                )
                file_chunks.extend(page_chunks)
            
            print(f"   Created {len(file_chunks)} chunks")
            
            # Embed and store chunks in batches
            for i, chunk in enumerate(file_chunks):
                embeddings = self.embedder.embed_text(chunk.text)
                
                if embeddings is None:
                    print(f"   ⚠️  Skipping chunk {i+1} due to embedding failure")
                    continue
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=[PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload={
                            "text": chunk.text,
                            "source": chunk.metadata['source'],
                            "page": chunk.metadata['page'],
                            "chunk_index": chunk.metadata['chunk_index']
                        }
                    )]
                )
                point_id += 1
                
                # Show progress every batch_size chunks
                if (i + 1) % batch_size == 0:
                    print(f"   ⏳ Indexed {i + 1}/{len(file_chunks)} chunks...")
            
            total_chunks += len(file_chunks)
            print(f"   ✅ Completed {pdf_file.name}\n")
        
        print(f"🎉 Successfully indexed {total_chunks} chunks from {len(pdf_files)} PDFs!")
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        temperature: float = 0.3
    ) -> str:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            temperature: LLM temperature (lower = more focused)
        """
        # Create search-optimized query embedding
        search_query = f"Represent this sentence for searching relevant passages: {question}"
        query_embedding = self.embedder.embed_text(search_query)
        
        if query_embedding is None:
            return "Error: Failed to generate query embedding."
        
        # Search for relevant chunks
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            with_payload=True,
            limit=top_k
        )
        
        if not search_results.points:
            return "No relevant information found in the indexed documents."
        
        # Format context with citations
        context_parts = []
        for i, point in enumerate(search_results.points):
            source = point.payload.get('source', 'Unknown')
            page = point.payload.get('page', '?')
            text = point.payload.get('text', '')
            context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = f"""You are a technical assistant analyzing documents. Answer the question based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Base your answer ONLY on the provided context
- If tables are present, analyze them carefully row by row
- Cite specific sources when referencing information
- If the context doesn't contain enough information, say so clearly
- Be precise and technical in your response

Answer:"""
        
        return self.llm.generate(prompt, temperature=temperature)
    
    def clear_collection(self):
        """Clear all data from the collection."""
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"✅ Collection '{self.collection_name}' cleared")


def main():
    """Main interactive loop."""
    print("="*60)
    print("🚀 Enhanced RAG System for PDF Documents")
    print("="*60 + "\n")
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name="documents",
        embedding_model="mxbai-embed-large",
        llm_model="mistral:7b"  # Can change to "deepseek-r1:8b" or "llama3.2"
    )
    
    # Check collection status
    stats = rag.get_collection_stats()
    
    if stats['points_count'] == 0:
        print("📭 Collection is empty. Let's index your PDFs first.\n")
        pdf_dir = input("Enter the directory path containing your PDF files: ").strip()
        
        if not os.path.exists(pdf_dir):
            print(f"❌ Error: Directory '{pdf_dir}' does not exist!")
            return
        
        rag.ingest_pdfs(pdf_dir, chunk_size=800, overlap=150)
        print("\n" + "="*60 + "\n")
    else:
        print(f"📊 Found {stats['points_count']} indexed chunks in the database.\n")
        reindex = input("Do you want to re-index PDFs? (y/n): ").strip().lower()
        
        if reindex == 'y':
            rag.clear_collection()
            pdf_dir = input("Enter the directory path containing your PDF files: ").strip()
            
            if os.path.exists(pdf_dir):
                rag.ingest_pdfs(pdf_dir, chunk_size=800, overlap=150)
                print("\n" + "="*60 + "\n")
            else:
                print(f"❌ Error: Directory '{pdf_dir}' does not exist!")
                return
    
    # Interactive query loop
    print("💬 You can now ask questions about your PDFs!")
    print("📝 Commands: 'quit'/'exit' to stop, 'stats' for collection info\n")
    
    while True:
        question = input("\n🔍 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.lower() == 'stats':
            stats = rag.get_collection_stats()
            print(f"\n📊 Collection Stats:")
            print(f"   - Total chunks: {stats['points_count']}")
            continue
        
        if not question:
            continue
        
        print("\n💭 Analyzing documents...")
        answer = rag.query(question, top_k=5, temperature=0.3)
        print(f"\n✨ Answer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()