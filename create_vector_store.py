"""
Create Vector Store from Maintenance Manuals

This script loads PDF documents from the maintenance_manuals directory,
processes them, creates embeddings, and saves a vector store for the RAG system.

Usage:
    python create_vector_store.py
"""

import os
from src.rag_system import load_maintenance_pdfs, chunk_documents, VectorStore

def main():
    print("="*80)
    print("Creating Vector Store for RAG System")
    print("="*80)
    
    # Load PDFs from maintenance manuals
    print("\nStep 1: Loading PDF documents...")
    documents = load_maintenance_pdfs()
    
    if not documents:
        print("\n✗ No PDF documents found!")
        print("\nPlease add PDF files to: maintenance_manuals/pdfs/")
        return
    
    # Chunk documents
    print("\nStep 2: Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=1000, overlap=200)
    
    # Create vector store
    print("\nStep 3: Creating vector store with embeddings...")
    vector_store = VectorStore()
    
    # Extract texts and metadata
    texts = [chunk.text for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    
    # Add to vector store
    vector_store.add_documents(texts, metadata)
    
    # Save the vector store
    print("\nStep 4: Saving vector store...")
    vector_store.save_index(path='models/rag/vector_store')
    
    print("\n" + "="*80)
    print("✓ Vector store created successfully!")
    print("="*80)
    print(f"Total documents: {len(chunks)}")
    print(f"Saved to: models/rag/vector_store.faiss")
    print("\nYou can now use the RAG Q&A feature in the Streamlit app!")

if __name__ == "__main__":
    main()

