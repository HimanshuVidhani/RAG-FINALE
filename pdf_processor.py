"""
PDF Processing Module for Research Synthesis Engine.
Handles PDF ingestion, text extraction, chunking, and metadata tagging.
"""

import os
import re
import hashlib
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict]:
    """
    Extract text and metadata from an uploaded PDF file.
    
    Args:
        pdf_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (full_text, metadata_dict)
    """
    reader = PdfReader(pdf_file)
    
    # Extract metadata
    meta = reader.metadata or {}
    metadata = {
        "title": meta.get("/Title", pdf_file.name.replace(".pdf", "")),
        "author": meta.get("/Author", "Unknown"),
        "filename": pdf_file.name,
        "num_pages": len(reader.pages),
        "file_hash": hashlib.md5(pdf_file.getvalue()).hexdigest(),
    }
    
    # Clean up title
    if not metadata["title"] or metadata["title"].strip() == "":
        metadata["title"] = pdf_file.name.replace(".pdf", "").replace("_", " ").title()
    
    # Extract text page by page
    pages_text = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            pages_text.append({
                "text": text,
                "page_num": page_num
            })
    
    full_text = "\n\n".join([p["text"] for p in pages_text])
    
    return full_text, metadata, pages_text


def create_chunks_with_metadata(
    full_text: str,
    pages_text: List[Dict],
    paper_metadata: Dict,
    paper_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Split document text into chunks while preserving paper identity and page numbers.
    
    Args:
        full_text: Complete extracted text
        pages_text: List of dicts with 'text' and 'page_num'
        paper_metadata: Paper-level metadata
        paper_id: Unique identifier for the paper (e.g., "Paper A")
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dicts with text and metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    
    chunks = []
    
    # Process each page separately to maintain page number tracking
    for page_info in pages_text:
        page_text = page_info["text"]
        page_num = page_info["page_num"]
        
        if len(page_text.strip()) < 50:
            continue
        
        page_chunks = text_splitter.split_text(page_text)
        
        for i, chunk_text in enumerate(page_chunks):
            chunk_text = clean_text(chunk_text)
            if len(chunk_text.strip()) < 30:
                continue
                
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "paper_id": paper_id,
                    "paper_title": paper_metadata["title"],
                    "paper_author": paper_metadata["author"],
                    "filename": paper_metadata["filename"],
                    "page_number": page_num,
                    "chunk_index": i,
                    "total_pages": paper_metadata["num_pages"],
                    "source": f"{paper_id}: {paper_metadata['title']}",
                }
            }
            chunks.append(chunk)
    
    return chunks


def clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace and artifacts."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page number artifacts
    text = re.sub(r'\n\d+\n', '\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def detect_sections(text: str) -> List[Dict]:
    """
    Attempt to detect section headers in academic papers.
    
    Args:
        text: Full paper text
        
    Returns:
        List of detected sections with titles and content
    """
    # Common section patterns in academic papers
    section_patterns = [
        r'(?:^|\n)((?:Abstract|Introduction|Background|Related Work|'
        r'Methodology|Methods|Materials and Methods|Experimental Setup|'
        r'Results|Discussion|Conclusion|Conclusions|Future Work|'
        r'References|Acknowledgements|Appendix))\s*\n',
        r'(?:^|\n)(\d+\.?\s+[A-Z][a-zA-Z\s]+)\n',  # Numbered sections
        r'(?:^|\n)([A-Z][A-Z\s]+)\n',  # All-caps headers
    ]
    
    sections = []
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
        for match in matches:
            sections.append({
                "title": match.group(1).strip(),
                "start_pos": match.start(),
            })
    
    # Sort by position and remove duplicates
    sections.sort(key=lambda x: x["start_pos"])
    
    # Add content to each section
    for i, section in enumerate(sections):
        start = section["start_pos"]
        end = sections[i + 1]["start_pos"] if i + 1 < len(sections) else len(text)
        section["content"] = text[start:end].strip()
    
    return sections


def get_paper_label(index: int) -> str:
    """Generate a paper label like 'Paper A', 'Paper B', etc."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(labels):
        return f"Paper {labels[index]}"
    return f"Paper {index + 1}"
