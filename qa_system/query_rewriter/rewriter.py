from typing import List, Dict
import ollama


class QueryRewriter:
    """
    Query rewriter that expands and reformulates queries to improve retrieval.
    Uses LLM to generate multiple query variations and synonyms.
    """
    
    def __init__(self, model_name: str = "qwen3:0.6b") -> None:
        self.model_name = model_name
    
    def rewrite_query(self, original_query: str) -> List[str]:
        """
        Rewrite the original query into multiple variations for better retrieval.
        Uses simple rule-based expansion for reliability.
        
        Args:
            original_query: The original user question
            
        Returns:
            List of rewritten query variations
        """
        queries = [original_query]  # Always include original
        
        # Simple rule-based query expansion
        query_lower = original_query.lower()
        
        # Add variations based on common patterns
        if "when was" in query_lower:
            queries.append(original_query.replace("when was", "what year was"))
            queries.append(original_query.replace("when was", "birth date of"))
        
        if "who is" in query_lower:
            queries.append(original_query.replace("who is", "what is the name of"))
            queries.append(original_query.replace("who is", "identity of"))
        
        if "born" in query_lower:
            queries.append(original_query.replace("born", "birth date"))
            queries.append(original_query.replace("born", "birth year"))
        
        # Add entity-focused variations
        if "album" in query_lower:
            queries.append(f"artist who released {original_query.split('album')[1].split('?')[0].strip()} album")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:4]  # Limit to 4 queries max
    
    def expand_entities(self, query: str) -> List[str]:
        """
        Generate entity-focused query variations.
        
        Args:
            query: Original query
            
        Returns:
            List of entity-focused query variations
        """
        prompt = f"""Extract key entities and concepts from this question and create focused queries for each:

Question: {query}

Create 2-3 focused queries that target specific entities or concepts mentioned. 

IMPORTANT: Return ONLY the focused queries, one per line, without any explanations, thinking, or formatting. Do not include <think> tags or reasoning."""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            raw_queries = response['message']['content'].strip()
            
            queries = []
            for line in raw_queries.split('\n'):
                line = line.strip()
                # Filter out thinking, numbering, and formatting
                if (line and 
                    not line.startswith(('1.', '2.', '3.', '-', '*', '<think>', 'Okay', 'Let me', 'First', 'Second', 'Third')) and
                    not line.startswith(('IMPORTANT:', 'Create', 'Return', 'Question:', 'Extract')) and
                    '?' in line):  # Only include lines that are actual questions
                    queries.append(line)
            
            return queries[:3]  # Limit to 3 entity queries
            
        except Exception as e:
            print(f"[QueryRewriter] Entity expansion error: {e}")
            return [query]


if __name__ == "__main__":
    # Test the query rewriter
    rewriter = QueryRewriter()
    
    test_query = "When was the American singer, songwriter, record producer, dancer and actress born who's second studio album is Chapter II?"
    
    print(f"Original query: {test_query}")
    print("\nRewritten queries:")
    rewritten = rewriter.rewrite_query(test_query)
    for i, q in enumerate(rewritten, 1):
        print(f"{i}. {q}")
    
    print("\nEntity-focused queries:")
    entity_queries = rewriter.expand_entities(test_query)
    for i, q in enumerate(entity_queries, 1):
        print(f"{i}. {q}")
