"""
Vector store builder for creating and managing vector databases
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import hashlib

# Vector store imports
import faiss
import chromadb
from chromadb.config import Settings

# Embedding model
from sentence_transformers import SentenceTransformer

# Text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """
    Main class for building vector stores from complaint data
    """
    
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 vector_store_type: str = 'both',  # 'faiss', 'chroma', or 'both'
                 persist_dir: str = '../vector_store'):
        """
        Initialize the vector store builder
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            vector_store_type: Type of vector store to create
            persist_dir: Directory to persist the vector stores
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_type = vector_store_type
        self.persist_dir = Path(persist_dir)
        
        # Initialize components
        self.embedding_model = None
        self.text_splitter = None
        self.faiss_index = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VectorStoreBuilder initialized with:")
        logger.info(f"  Model: {embedding_model_name}")
        logger.info(f"  Chunk size: {chunk_size} characters")
        logger.info(f"  Chunk overlap: {chunk_overlap} characters")
        logger.info(f"  Store type: {vector_store_type}")
        logger.info(f"  Persist dir: {persist_dir}")
    
    def load_embedding_model(self):
        """
        Load the embedding model
        """
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self.embedding_dim}")
        
        return self.embedding_model
    
    def create_text_splitter(self):
        """
        Create the text splitter for chunking
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        logger.info(f"Text splitter created with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}")
        
        return self.text_splitter
    
    def create_stratified_sample(self, 
                                 df: pd.DataFrame, 
                                 sample_size: int = 12000,
                                 stratify_col: str = 'product_category') -> pd.DataFrame:
        """
        Create a stratified sample from the dataframe
        
        Args:
            df: Input dataframe
            sample_size: Target sample size
            stratify_col: Column to stratify by
            
        Returns:
            Stratified sample dataframe
        """
        logger.info(f"Creating stratified sample of size {sample_size}")
        
        if stratify_col not in df.columns:
            raise ValueError(f"Stratify column '{stratify_col}' not found in dataframe")
        
        # Calculate proportions
        category_counts = df[stratify_col].value_counts()
        category_proportions = category_counts / len(df)
        
        # Calculate target sizes per category
        target_sizes = (category_proportions * sample_size).round().astype(int)
        
        # Adjust for rounding
        total_allocated = target_sizes.sum()
        if total_allocated != sample_size:
            largest_category = target_sizes.idxmax()
            target_sizes[largest_category] += sample_size - total_allocated
        
        # Sample from each category
        sampled_dfs = []
        for category, size in target_sizes.items():
            category_df = df[df[stratify_col] == category]
            
            # If we need more than available, take all
            n_samples = min(size, len(category_df))
            
            if n_samples > 0:
                category_sample = category_df.sample(n=n_samples, random_state=42)
                sampled_dfs.append(category_sample)
        
        # Combine
        sample_df = pd.concat(sampled_dfs, ignore_index=True)
        
        logger.info(f"Created stratified sample with {len(sample_df)} records")
        
        # Log distribution
        sample_dist = sample_df[stratify_col].value_counts()
        for category, count in sample_dist.items():
            logger.info(f"  {category}: {count} records ({(count/len(sample_df)*100):.1f}%)")
        
        return sample_df
    
    def chunk_dataframe(self, 
                        df: pd.DataFrame, 
                        text_column: str = 'cleaned_narrative',
                        id_column: str = 'complaint_id') -> pd.DataFrame:
        """
        Chunk all documents in the dataframe
        
        Args:
            df: Input dataframe
            text_column: Column containing text to chunk
            id_column: Column containing document IDs
            
        Returns:
            Dataframe with all chunks
        """
        if self.text_splitter is None:
            self.create_text_splitter()
        
        logger.info(f"Chunking {len(df)} documents...")
        
        all_chunks = []
        
        for idx, row in df.iterrows():
            # Extract metadata
            complaint_id = row.get(id_column, f"complaint_{idx}")
            text = row.get(text_column, '')
            
            if not text or len(str(text).strip()) == 0:
                continue
            
            # Prepare metadata
            metadata = {
                'complaint_id': complaint_id,
                'product_category': row.get('product_category', 'Unknown'),
                'product': row.get('Product', row.get('product', 'Unknown')),
                'issue': row.get('Issue', 'Unknown'),
                'sub_issue': row.get('Sub-issue', 'Unknown'),
                'company': row.get('Company', 'Unknown'),
                'state': row.get('State', 'Unknown'),
                'date_received': row.get('Date received', 'Unknown'),
                'original_text_length': len(str(text))
            }
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk records
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"{complaint_id}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk,
                    'chunk_size_chars': len(chunk),
                    'chunk_size_words': len(chunk.split())
                })
                all_chunks.append(chunk_metadata)
        
        # Convert to DataFrame
        chunks_df = pd.DataFrame(all_chunks)
        
        logger.info(f"Created {len(chunks_df)} chunks from {len(df)} documents")
        logger.info(f"Average chunks per document: {len(chunks_df)/len(df):.2f}")
        
        return chunks_df
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> 'VectorStoreBuilder':
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            
        Returns:
            Self for chaining
        """
        logger.info("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.faiss_metadata = metadata
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        
        return self
    
    def save_faiss_index(self, index_name: str = 'faiss_index'):
        """
        Save FAISS index to disk
        
        Args:
            index_name: Base name for the index files
        """
        if self.faiss_index is None:
            raise ValueError("No FAISS index built")
        
        # Save index
        index_path = self.persist_dir / f"{index_name}.faiss"
        faiss.write_index(self.faiss_index, str(index_path))
        
        # Save metadata
        metadata_path = self.persist_dir / f"{index_name}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.faiss_metadata, f)
        
        logger.info(f"FAISS index saved to {index_path}")
        logger.info(f"FAISS metadata saved to {metadata_path}")
        
        return str(index_path)
    
    def build_chroma_collection(self, 
                                embeddings: np.ndarray, 
                                metadatas: List[Dict],
                                documents: List[str],
                                collection_name: str = 'complaint_chunks') -> 'VectorStoreBuilder':
        """
        Build ChromaDB collection
        
        Args:
            embeddings: Numpy array of embeddings
            metadatas: List of metadata dictionaries
            documents: List of document texts
            collection_name: Name of the ChromaDB collection
            
        Returns:
            Self for chaining
        """
        logger.info(f"Building ChromaDB collection: {collection_name}")
        
        # Initialize ChromaDB client
        chroma_dir = self.persist_dir / 'chroma_db'
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Generate IDs
        ids = [f"chunk_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.chroma_collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
            documents=documents
        )
        
        logger.info(f"ChromaDB collection built with {self.chroma_collection.count()} documents")
        
        return self
    
    def save_chroma_info(self, info_name: str = 'chromadb_info'):
        """
        Save ChromaDB information to file
        
        Args:
            info_name: Base name for the info file
        """
        if self.chroma_collection is None:
            raise ValueError("No ChromaDB collection built")
        
        info = {
            'collection_name': self.chroma_collection.name,
            'collection_count': self.chroma_collection.count(),
            'persist_directory': str(self.persist_dir / 'chroma_db'),
            'save_date': pd.Timestamp.now().isoformat()
        }
        
        info_path = self.persist_dir / f"{info_name}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"ChromaDB info saved to {info_path}")
        
        return str(info_path)
    
    def build_from_dataframe(self, 
                             df: pd.DataFrame,
                             sample_size: Optional[int] = None,
                             text_column: str = 'cleaned_narrative') -> Dict:
        """
        Complete pipeline: sample, chunk, embed, and build vector stores
        
        Args:
            df: Input dataframe
            sample_size: If provided, create stratified sample of this size
            text_column: Column containing text to process
            
        Returns:
            Dictionary with paths to created files
        """
        logger.info("Starting complete vector store build pipeline...")
        
        # Step 1: Create sample if requested
        if sample_size and sample_size < len(df):
            logger.info(f"Creating stratified sample of size {sample_size}")
            df = self.create_stratified_sample(df, sample_size)
        else:
            logger.info(f"Using full dataset with {len(df)} records")
        
        # Step 2: Chunk the data
        chunks_df = self.chunk_dataframe(df, text_column)
        
        # Save chunks for reference
        chunks_path = self.persist_dir.parent / 'data' / 'processed' / 'text_chunks.csv'
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_df.to_csv(chunks_path, index=False)
        logger.info(f"Text chunks saved to {chunks_path}")
        
        # Step 3: Generate embeddings
        texts = chunks_df['chunk_text'].tolist()
        embeddings = self.generate_embeddings(texts)
        
        # Step 4: Prepare metadata
        metadatas = []
        for _, row in chunks_df.iterrows():
            metadata = {
                'complaint_id': str(row['complaint_id']),
                'product_category': str(row['product_category']),
                'product': str(row['product']),
                'issue': str(row['issue']),
                'sub_issue': str(row['sub_issue']),
                'company': str(row['company']),
                'state': str(row['state']),
                'date_received': str(row['date_received']),
                'chunk_index': int(row['chunk_index']),
                'total_chunks': int(row['total_chunks'])
            }
            metadatas.append(metadata)
        
        # Step 5: Build vector stores
        created_files = {}
        
        if self.vector_store_type in ['faiss', 'both']:
            # Build FAISS
            self.build_faiss_index(embeddings, metadatas)
            faiss_index_path = self.save_faiss_index()
            created_files['faiss_index'] = faiss_index_path
        
        if self.vector_store_type in ['chroma', 'both']:
            # Build ChromaDB
            self.build_chroma_collection(embeddings, metadatas, texts)
            chroma_info_path = self.save_chroma_info()
            created_files['chroma_info'] = chroma_info_path
        
        # Step 6: Save configuration
        config = {
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': embeddings.shape[1],
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(chunks_df),
            'total_documents': len(df),
            'vector_stores_created': self.vector_store_type,
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        config_path = self.persist_dir / 'build_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        created_files['config'] = str(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Create summary
        summary = f"""
        VECTOR STORE BUILD SUMMARY:
        ---------------------------
        Input documents: {len(df):,}
        Text chunks created: {len(chunks_df):,}
        Embedding dimension: {embeddings.shape[1]}
        Vector stores created: {self.vector_store_type}
        
        Files created:
        - Text chunks: {chunks_path}
        - FAISS index: {created_files.get('faiss_index', 'Not created')}
        - ChromaDB: {created_files.get('chroma_info', 'Not created')}
        - Configuration: {created_files.get('config')}
        """
        
        summary_path = self.persist_dir / 'build_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(summary)
        logger.info("Vector store build completed successfully!")
        
        return created_files

# Usage example
if __name__ == "__main__":
    # Example usage
    builder = VectorStoreBuilder(
        embedding_model_name='all-MiniLM-L6-v2',
        chunk_size=500,
        chunk_overlap=50,
        vector_store_type='both',
        persist_dir='../vector_store'
    )
    
    # Load your cleaned data
    # df = pd.read_csv('../data/processed/filtered_complaints.csv')
    
    # Build vector stores (with sampling for Task 2)
    # created_files = builder.build_from_dataframe(df, sample_size=12000)