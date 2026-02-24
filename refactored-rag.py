import sys
import numpy as np
import re
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

class MinimalRAG:
    def __init__(self, documents):
        self.docs = documents
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform([x["txt"] for x in documents])
        
        print("--- Loading LLM (GPT-2) ---")
        # Use a small model for local execution; device=-1 ensures CPU usage
        self.llm = pipeline("text-generation", model="gpt2")
        print("--- Model Loaded ---\n")

    def retrieve(self, query, k=3):
        """Finds documents and returns them with their similarity scores."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # Get top K indices sorted by score descending
        indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for i in indices:
            score = float(similarities[i])
            if score > 0:
                results.append({
                    "id": self.docs[i]["id"],
                    "text": self.docs[i]["txt"],
                    "score": round(score, 4)
                })
        return results

    def generate(self, query, retrieved_docs):
        """Constructs prompt and generates answer."""
        if not retrieved_docs:
            return "I'm sorry, I don't have enough information in my database."

        context = "\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(retrieved_docs)])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question concisely using the context. Cite sources like [1].\n"
            "Answer:"
        )
        
        output = self.llm(prompt, max_new_tokens=50, pad_token_id=50256, truncation=True)
        return output[0]['generated_text'].split("Answer:")[-1].strip()

    def validate(self, answer, retrieved_docs):
        """Calculates a confidence score (0.0 to 1.0) based on response quality."""
        confidence = 1.0
        problems = []
        
        # 1. Relevance Check (average similarity of sources)
        avg_sim = np.mean([d['score'] for d in retrieved_docs]) if retrieved_docs else 0
        if avg_sim < 0.2:
            problems.append("low_source_relevance")
            confidence *= 0.6

        # 2. Citation Check
        if not re.search(r'\[\d+\]', answer):
            problems.append("missing_citations")
            confidence *= 0.7

        # 3. Content Check
        if len(answer) < 15 or "i don't know" in answer.lower():
            problems.append("weak_answer")
            confidence *= 0.5
            
        return {
            "is_valid": confidence > 0.5,
            "confidence_score": round(confidence, 2),
            "problems": problems
        }

def main():
    parser = argparse.ArgumentParser(description="Minimal RAG System with Scoring")
    parser.add_argument("question", type=str, help="The question to ask the RAG system")
    parser.add_argument("-k", type=int, default=2, help="Number of documents to retrieve")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    # Knowledge Base
    documents = [
        {"id": "1", "txt": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace."},
        {"id": "2", "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available."},
        {"id": "3", "txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."},
        {"id": "4", "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations."},
        {"id": "5", "txt": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma."}
    ]

    rag = MinimalRAG(documents)
    
    # Execution
    hits = rag.retrieve(args.question, k=args.k)
    ans = rag.generate(args.question, hits)
    critique = rag.validate(ans, hits)

    # Output Formatting
    if args.json:
        result = {
            "question": args.question,
            "answer": ans,
            "retrieval": hits,
            "validation": critique
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"--- RAG OUTPUT ---")
        print(f"Q: {args.question}")
        print(f"A: {ans}\n")
        
        print(f"--- SOURCES & SCORES ---")
        for i, hit in enumerate(hits):
            print(f"[{i+1}] (Sim: {hit['score']}) {hit['text'][:75]}...")
        
        print(f"\n--- VALIDATION ---")
        status = "PASS" if critique['is_valid'] else "FAIL"
        print(f"Confidence Score: {critique['confidence_score']} {status}")
        if critique['problems']:
            print(f"Issues: {', '.join(critique['problems'])}")

if __name__ == "__main__":
    main()