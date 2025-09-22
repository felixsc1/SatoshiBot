from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
import glob
from pathlib import Path
import logging
from urllib.parse import urljoin
from typing import List, Dict, Any
from langchain_core.documents import Document
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatoshiDocumentIngestion:
    def __init__(self, 
                 source_dir: str = "nakamotoinstitute_files",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embeddings_model: str = "granite-embedding:30m"):
        self.source_dir = source_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OllamaEmbeddings(model=embeddings_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", ". ", " ", ""]
        )
        
    def get_html_files(self) -> List[str]:
        """Get all HTML files from the source directory"""
        html_files = []
        
        # Get files from subdirectories
        for subdir in ['emails', 'posts', 'quotes']:
            pattern = os.path.join(self.source_dir, subdir, '*.html')
            html_files.extend(glob.glob(pattern))
        
        # Get main files  (not useful)
        # main_pattern = os.path.join(self.source_dir, '*.html')
        # html_files.extend(glob.glob(main_pattern))
        
        logger.info(f"Found {len(html_files)} HTML files to process")
        return html_files
    
    def extract_metadata_from_filename(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from filename and path"""
        path = Path(filepath)
        filename = path.stem
        
        metadata = {
            'source_file': filepath,
            'filename': filename,
        }
        
        # Determine content type from path
        if 'emails' in path.parts:
            metadata['content_type'] = 'email'
            metadata['collection'] = 'emails'
        elif 'posts' in path.parts:
            metadata['content_type'] = 'post'
            metadata['collection'] = 'posts'
        elif 'quotes' in path.parts:
            metadata['content_type'] = 'quote'
            metadata['collection'] = 'quotes'
        else:
            metadata['content_type'] = 'main'
            metadata['collection'] = 'main'
        
        # Extract additional info from filename patterns
        if filename.startswith('email_'):
            parts = filename.split('_', 2)
            if len(parts) >= 2:
                metadata['sequence_number'] = parts[1]
                if len(parts) >= 3:
                    metadata['title'] = parts[2].replace('_', ' ')
        elif filename.startswith('post_'):
            parts = filename.split('_', 2)
            if len(parts) >= 2:
                metadata['sequence_number'] = parts[1]
                if len(parts) >= 3:
                    metadata['title'] = parts[2].replace('_', ' ')
        elif filename.startswith('quote_'):
            parts = filename.split('_', 2)
            if len(parts) >= 2:
                metadata['sequence_number'] = parts[1]
                if len(parts) >= 3:
                    metadata['title'] = parts[2].replace('_', ' ')
        
        return metadata
    
    def load_single_document(self, filepath: str) -> List[Document]:
        """Load a single HTML file using BSHTMLLoader"""
        try:
            # Use BSHTMLLoader for local HTML files
            loader = BSHTMLLoader(filepath)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content loaded from {filepath}")
                return []
            
            # Extract metadata from filename
            file_metadata = self.extract_metadata_from_filename(filepath)
            
            # Enhance documents with metadata
            enhanced_docs = []
            for doc in documents:
                # Strip leading/trailing whitespace from content
                doc.page_content = doc.page_content.strip()
                
                # Split content into lines for metadata extraction
                lines = doc.page_content.split('\n')
                
                # Combine existing metadata with file metadata
                combined_metadata = {**doc.metadata, **file_metadata}
                
                # Add source URL if available in the document
                if 'Source:' in doc.page_content:
                    for line in lines:
                        if line.strip().startswith('Source:'):
                            # Extract URL from "Source: &lt;url&gt;" pattern
                            if 'https://satoshi.nakamotoinstitute.org' in line:
                                start = line.find('https://satoshi.nakamotoinstitute.org')
                                end = line.find(' ', start) if line.find(' ', start) != -1 else len(line)
                                combined_metadata['source_url'] = line[start:end]
                            break
                
                # Extract date if present
                for line in lines:
                    if line.strip().startswith('Date: |'):
                        date_part = line.split('Date: |', 1)[1].strip()
                        combined_metadata['date'] = date_part
                        break
                
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=combined_metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """Load all HTML documents"""
        html_files = self.get_html_files()
        all_documents = []
        
        for filepath in html_files:
            logger.info(f"Loading: {filepath}")
            docs = self.load_single_document(filepath)
            all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} documents")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        # For very short documents (like quotes), don't split
        short_docs = []
        long_docs = []
        
        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                doc.metadata['chunk_index'] = 0
                doc.metadata['total_chunks'] = 1
                short_docs.append(doc)
            else:
                long_docs.append(doc)
        
        # Split long documents
        split_docs = []
        if long_docs:
            for doc in long_docs:
                splits = self.text_splitter.split_documents([doc])
                for i, split in enumerate(splits):
                    split.metadata['chunk_index'] = i
                    split.metadata['total_chunks'] = len(splits)
                split_docs.extend(splits)
        
        # Combine short and split documents
        all_chunks = short_docs + split_docs
        
        logger.info(f"Created {len(all_chunks)} chunks ({len(short_docs)} short, {len(split_docs)} from splitting)")
        return all_chunks
    
    def create_vectorstore(self, documents: List[Document], save_path: str = "satoshi_vectorstore") -> FAISS:
        """Create and save FAISS vectorstore"""
        logger.info(f"Creating vectorstore with {len(documents)} documents...")
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        vectorstore.save_local(save_path)
        logger.info(f"Vectorstore saved to {save_path}")
        
        return vectorstore
    
    def export_raw_documents(self, output_dir: str = "raw_documents") -> None:
        """Export one raw .txt per HTML document and write source URL maps.

        Creates folder structure:
        raw_documents/
          emails/
            <same_base_filename>.txt
            source_map.json
          posts/
            ...
          quotes/
            ...
        """
        html_files = self.get_html_files()
        if not html_files:
            logger.warning("No HTML files found to export.")
            return

        collections = ["emails", "posts", "quotes"]
        # Ensure base and subdirectories exist
        for collection in collections:
            os.makedirs(os.path.join(output_dir, collection), exist_ok=True)

        # Prepare per-collection source maps
        source_maps: Dict[str, Dict[str, str]] = {c: {} for c in collections}

        for filepath in html_files:
            docs = self.load_single_document(filepath)
            if not docs:
                logger.warning(f"Skipping empty document: {filepath}")
                continue

            path = Path(filepath)
            # Determine target collection from first doc metadata
            collection = docs[0].metadata.get("collection", "")
            if collection not in collections:
                # Skip files that are not part of target collections
                logger.info(f"Skipping non-target collection for {filepath}")
                continue

            # Compose single raw text per HTML file
            combined_content = "\n\n".join(doc.page_content for doc in docs if doc.page_content)

            # Determine output filename
            output_filename = f"{path.stem}.txt"
            output_path = os.path.join(output_dir, collection, output_filename)

            # Write raw text
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_content)

            # Capture source URL (prefer the first doc that has it)
            source_url = None
            for doc in docs:
                candidate = doc.metadata.get("source_url")
                if candidate:
                    source_url = candidate
                    break

            # Record mapping; if URL missing, store empty string
            source_maps[collection][output_filename] = source_url or ""

            logger.info(f"Exported raw: {output_path}")

        # Write source_map.json per collection
        for collection, mapping in source_maps.items():
            map_path = os.path.join(output_dir, collection, "source_map.json")
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote source map: {map_path} ({len(mapping)} entries)")

    def export_to_csv(self, chunks: List[Document], csv_path: str):
        import csv
        # Collect all possible metadata keys
        all_keys = set()
        for chunk in chunks:
            all_keys.update(chunk.metadata.keys())
        # Sort keys for consistent order
        metadata_keys = sorted(all_keys)
        # CSV columns: content + metadata_keys
        columns = ['content'] + metadata_keys
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for chunk in chunks:
                row = {'content': chunk.page_content}
                row.update(chunk.metadata)
                writer.writerow(row)
        logger.info(f"Exported {len(chunks)} rows to {csv_path}")

    def run_ingestion(self, save_path: str = "satoshi_vectorstore") -> FAISS:
        """Run the complete ingestion pipeline"""
        logger.info("Starting Satoshi document ingestion...")
        
        # Load documents
        documents = self.load_all_documents()
        
        if not documents:
            raise ValueError("No documents were loaded!")
        
        # Split documents
        chunks = self.split_documents(documents)
        
        # Export to CSV
        self.export_to_csv(chunks, "nakamotoinstitute_files/satoshi.csv")
        
        # Create vectorstore
        vectorstore = self.create_vectorstore(chunks, save_path)
        
        # Print summary
        self.print_ingestion_summary(documents, chunks)
        
        return vectorstore
    
    def print_ingestion_summary(self, documents: List[Document], chunks: List[Document]):
        """Print a summary of the ingestion process"""
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        
        # Count by content type
        content_types = {}
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print(f"Total documents loaded: {len(documents)}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} characters")
        print("\nContent breakdown:")
        for content_type, count in sorted(content_types.items()):
            print(f"  {content_type}: {count} documents")
        
        # Show sample content
        if chunks:
            print(f"\nSample chunk preview:")
            sample_chunk = chunks[0]
            print(f"Content type: {sample_chunk.metadata.get('content_type', 'unknown')}")
            print(f"Title: {sample_chunk.metadata.get('title', 'N/A')}")
            print(f"Content preview: {sample_chunk.page_content[:200]}...")
        
        print("="*60)

def main():
    """Main function to run the ingestion or raw export"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Satoshi documents or export raw text.")
    parser.add_argument("--raw-only", action="store_true", help="Skip chunking/embedding; export raw .txt files and source maps.")
    parser.add_argument("--raw-output", type=str, default="raw_documents", help="Directory to write raw documents to.")
    parser.add_argument("--vectorstore", type=str, default="satoshi_vectorstore", help="Output directory for FAISS vectorstore.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for splitting.")
    parser.add_argument("--embeddings-model", type=str, default="granite-embedding:30m", help="Ollama embeddings model name.")
    args = parser.parse_args()

    # Initialize ingestion
    ingestion = SatoshiDocumentIngestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embeddings_model=args.embeddings_model  
    )
    
    try:
        if args.raw_only:
            ingestion.export_raw_documents(output_dir=args.raw_output)
            print(f"\n✅ Raw export completed successfully!")
            print(f"Raw documents saved to: {args.raw_output}")
        else:
            vectorstore = ingestion.run_ingestion(save_path=args.vectorstore)
            print(f"\n✅ Ingestion completed successfully!")
            print(f"Vectorstore saved to: {args.vectorstore}")
            print(f"You can now use this vectorstore for RAG queries.")
        
    except Exception as e:
        print(f"❌ Operation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()