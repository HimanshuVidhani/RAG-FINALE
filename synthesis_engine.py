"""
Synthesis Engine for Research Synthesis & Comparison Engine.
Orchestrates Gemini LLM calls for cross-document analysis, comparison, and synthesis.
"""

from typing import List, Dict, Optional
from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are an expert Research Synthesis & Comparison Assistant. Your role is to analyze, compare, and synthesize findings across multiple research papers.

CRITICAL RULES:
1. **Always cite your sources**: When making any claim, specify which paper it comes from using the format "According to [Paper X: Title]..." or "[Paper X, p.Y]".
2. **Identify agreements and conflicts**: When papers agree, highlight this. When they conflict, clearly state the disagreement with evidence from both sides.
3. **Be precise with data**: Quote specific numbers, percentages, and findings directly from the provided context.
4. **Never hallucinate**: If the provided context doesn't contain enough information to answer, say so explicitly. Do not make up information.
5. **Use structured formatting**: Use headers, bullet points, and tables to organize your analysis.
6. **Distinguish claims**: Clearly differentiate between what each paper claims vs. what is commonly accepted knowledge.
7. **Academic tone**: Maintain a scholarly, objective tone throughout your analysis.

When comparing papers:
- Look for methodological differences
- Note sample size variations
- Identify different datasets or benchmarks used
- Point out temporal differences (when studies were conducted)
- Highlight complementary findings that together paint a fuller picture"""


class SynthesisEngine:
    """
    Core engine that uses Gemini to perform research synthesis tasks:
    - Paper-wise cited Q&A
    - Cross-document comparison
    - Unified summaries
    - Technical methodology synthesis
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    def _generate(self, prompt: str, history: list = None) -> str:
        """Generate content using the Gemini model."""
        contents = []
        if history:
            for turn in history[-6:]:
                contents.append(types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["content"])],
                ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ))
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                max_output_tokens=4096,
            ),
        )
        return response.text
    
    def answer_with_citations(self, query, grouped_context, paper_info, chat_history=None):
        """Answer a question with paper-wise citations."""
        context_text = self._format_grouped_context(grouped_context, paper_info)
        prompt = f"""Based on the following research paper excerpts, answer the user's question with precise citations.

{context_text}

---
USER QUESTION: {query}
---

INSTRUCTIONS:
- Cite each paper explicitly when referencing its content (e.g., "According to Paper A: [Title]...")
- If papers provide conflicting information, highlight the disagreement
- Include page numbers when available
- If the context doesn't contain relevant information, state this clearly
- Structure your response with clear headers if the answer is complex"""

        return self._generate(prompt, chat_history)
    
    def generate_cross_document_summary(self, all_contexts, paper_info):
        """Generate a unified summary covering core themes across all papers."""
        context_parts = []
        for pid, text in all_contexts.items():
            info = next((p for p in paper_info if p["paper_id"] == pid), {})
            title = info.get("title", pid)
            author = info.get("author", "Unknown")
            truncated = text[:6000] if len(text) > 6000 else text
            context_parts.append(f"### {pid}: {title}\n**Author(s):** {author}\n\n{truncated}\n")
        
        all_context = "\n---\n".join(context_parts)
        paper_cols = ' | '.join([f'{p["paper_id"]}' for p in paper_info])
        
        prompt = f"""Analyze the following research papers and create a comprehensive CROSS-DOCUMENT SYNTHESIS SUMMARY.

{all_context}

---

Generate a detailed synthesis that includes:

## 1. Overview
Brief description of each paper and its core contribution.

## 2. Common Themes & Research Questions
What overarching themes connect these papers? What shared research questions do they address?

## 3. Methodological Comparison
| Aspect | {paper_cols} |
Compare the methodologies, datasets, and approaches used.

## 4. Key Findings Comparison
What are the main findings of each paper? Where do they agree? Where do they differ?

## 5. Conflicting Evidence
Identify any contradictory claims or findings between papers, with specific evidence.

## 6. Complementary Insights
How do the papers complement each other? What fuller picture emerges from combining them?

## 7. Research Gaps & Future Directions
Based on all papers, what gaps remain? What future work is suggested?

Be thorough and cite specific papers throughout."""

        return self._generate(prompt)
    
    def synthesize_methodology(self, grouped_context, paper_info, topic):
        """Synthesize and explain complex methodologies across papers."""
        context_text = self._format_grouped_context(grouped_context, paper_info)
        prompt = f"""Analyze and synthesize the technical methodologies described in these research paper excerpts about: {topic}

{context_text}

---

Provide a detailed TECHNICAL METHODOLOGY SYNTHESIS that includes:

## Methodology Overview
Explain what each paper does methodologically, in accessible but technically accurate language.

## Approach Comparison
Create a structured comparison of the different approaches:
- Algorithm/model architecture differences
- Training procedures and hyperparameters
- Evaluation metrics used
- Dataset characteristics

## Technical Deep Dive
Explain the most complex or novel methodology in detail, pulling specifics from the papers.

## Strengths & Limitations
For each methodology, identify key strengths, innovations, potential limitations, and conditions under which each approach works best.

## Unified Understanding
Synthesize a unified understanding of the methodological landscape based on all papers.

Always cite which paper you're referencing."""

        return self._generate(prompt)
    
    def compare_findings(self, grouped_context, paper_info, aspect):
        """Compare specific findings or claims across papers."""
        context_text = self._format_grouped_context(grouped_context, paper_info)
        prompt = f"""Compare the findings across these research papers regarding: {aspect}

{context_text}

---

Provide a structured FINDINGS COMPARISON:

## Summary Table
Create a comparison table showing each paper's position on this topic.

## Detailed Analysis
For each paper, describe what they found/claim, the evidence supporting their claim, and the strength of their evidence.

## Agreement Points
Where do the papers agree? What consensus emerges?

## Disagreement Points  
Where do the papers disagree? What might explain the differences?

## Evidence Assessment
Which paper(s) provide the strongest evidence? Why?

## Synthesized Conclusion
Based on all available evidence, what can we reasonably conclude about "{aspect}"?

Cite specific papers and page numbers throughout."""

        return self._generate(prompt)
    
    def _format_grouped_context(self, grouped_context, paper_info):
        """Format grouped context into a structured prompt section."""
        sections = []
        for paper_id, chunks in grouped_context.items():
            info = next((p for p in paper_info if p["paper_id"] == paper_id), {})
            title = info.get("title", paper_id)
            author = info.get("author", "Unknown")
            header = f"### {paper_id}: {title} (by {author})"
            chunk_texts = []
            for chunk in chunks:
                page = chunk.get("metadata", {}).get("page_number", "?")
                score = chunk.get("relevance_score", 0)
                chunk_texts.append(f"[Page {page}, Relevance: {score:.2f}]\n{chunk['text']}")
            sections.append(f"{header}\n\n" + "\n\n---\n\n".join(chunk_texts))
        return "\n\n" + "=" * 60 + "\n\n".join(sections)
    
    def generate_paper_summary(self, paper_text, paper_info):
        """Generate a detailed summary of a single paper."""
        truncated = paper_text[:8000] if len(paper_text) > 8000 else paper_text
        prompt = f"""Provide a comprehensive summary of this research paper:

**Title:** {paper_info.get('title', 'Unknown')}
**Author(s):** {paper_info.get('author', 'Unknown')}

{truncated}

---

Structure your summary as:

## Key Takeaways (3-5 bullet points)

## Abstract / Overview
What is this paper about? What problem does it address?

## Methodology
What approach/method does the paper use?

## Key Results
What are the main findings and quantitative results?

## Contributions
What are the novel contributions of this paper?

## Limitations
What limitations are mentioned or apparent?

## Future Work
What future directions are suggested?"""

        return self._generate(prompt)
