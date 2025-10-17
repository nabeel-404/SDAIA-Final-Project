from typing import List, Dict
import ollama
from starlette.applications import P


class QueryRewriter:
    """
    Query rewriter that expands and reformulates queries to improve retrieval.
    Uses LLM to generate multiple query variations and synonyms.
    """
    
    def __init__(self, model_name: str = "qwen3:0.6b") -> None:
        self.model_name = model_name
    
    
    def rewrite_query(self, query: str) -> List[str]:
        """
        Generate entity-focused query variations.
        
        Args:
            query: Original query
            
        Returns:
            List of entity-focused query variations
        """
        prompt = f"""You are a query decomposition agent that decompose queries to multiple sub-queries to improve retrieval for multi-hop question-answering.

Question: {query}

Create 2-3 focused queries that target specific entities or concepts mentioned. 

IMPORTANT: Return ONLY the focused queries, one per line"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            raw_queries = response['message']['content'].strip()
            
            queries = []
            for line in raw_queries.split('\n'):
                line = line.strip()
                queries.append(line)
            
            return queries[:3]  # Limit to 3 entity queries
            
        except Exception as e:
            print(f"[QueryRewriter] Entity expansion error: {e}")
            return [query]


if __name__ == "__main__":
    rewriter = QueryRewriter()
    
    test_query = "Were Scott Derrickson and Ed Wood of the same nationality?"
    
    print(f"Original query: {test_query}")
    
    
    print("\nEntity-focused queries:")
    entity_queries = rewriter.rewrite_query(test_query)
    for i, q in enumerate(entity_queries, 1):
        print(f"{i}. {q}")
