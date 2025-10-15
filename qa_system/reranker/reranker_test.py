#testing the Reranker class from reranker.py

from reranker import Reranker

user_query = "Who invented the light bulb?"
raw_docs = [
    "Thomas Edison invented the electric light bulb in 1879.",
    "Bananas are yellow and grow in tropical regions.",
    "Edison also founded General Electric.",
    "The Wright brothers invented the airplane.",
    "Albert Einstein developed the theory of relativity.",
    "The Great Wall of China is visible from space.",
    "Isaac Newton formulated the laws of motion and universal gravitation.",
    "The capital of France is Paris.",
    "The human body has 206 bones.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "Water boils at 100 degrees Celsius.",
    "Mount Everest is the highest mountain in the world.",
    "Shakespeare wrote many famous plays.",
    "The Amazon rainforest is the largest tropical rainforest.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "The currency of Japan is the yen.",
    "The Eiffel Tower is located in Paris.",
    "The human brain is the most complex organ in the body.",
    "The Great Barrier Reef is the largest coral reef system.",
    "The first manned moon landing was in 1969.",
    "The Statue of Liberty was a gift from France to the United States.",
    "The Nile is the longest river in the world.",
    "The human heart pumps blood throughout the body.",
    "The Taj Mahal is located in India.",
    "The Sahara is the largest hot desert in the world.",
    "The first computer was invented in the 1940s.",
    "The human eye can distinguish about 10 million different colors.",
    "The Colosseum is an ancient amphitheater in Rome.",
    "The Great Depression began in 1929.",
    "The human skeleton provides structure and support to the body.",
    "The Leaning Tower of Pisa is famous for its tilt.",
    "The first successful airplane flight was in 1903.",
    "The human skin is the body's largest organ.",
    "The Golden Gate Bridge is located in San Francisco.",
    "The first telephone was invented by Alexander Graham Bell.",
]

# since `Reranker.rerank()` expects List[Dict] with at least a 'text' field, we convert our raw docs:
retrieved_docs = [{"text": t} for t in raw_docs]

r = Reranker(device = 'cuda', batch_size=16)

top_docs = r.rerank(user_query, retrieved_docs, top_k=10)

print(f"User query: {user_query}")

# Inspect results
for i, d in enumerate(top_docs, 1):
    print(f"{i:02d}. score={d.get('reranker_score'):.6f} | {d['text']}")

# then pass `top_docs` (or just their 'text') to your Ollama LLM
context_chunks = [d["text"] for d in top_docs]


# should print something like the following:
# [Reranker] Loading BAAI/bge-reranker-v2-m3 on cuda...
# [Reranker] Using half precision (fp16)
# User query: Who invented the light bulb?
# 01. score=0.994629 | Thomas Edison invented the electric light bulb in 1879.
# 02. score=0.004433 | Isaac Newton formulated the laws of motion and universal gravitation.
# 03. score=0.004166 | Edison also founded General Electric.
# 04. score=0.003708 | Albert Einstein developed the theory of relativity.
# 05. score=0.002043 | The first telephone was invented by Alexander Graham Bell.
# 06. score=0.000873 | The Mona Lisa was painted by Leonardo da Vinci.
# 07. score=0.000283 | The Wright brothers invented the airplane.
# 08. score=0.000080 | Mount Everest is the highest mountain in the world.
# 09. score=0.000064 | The Eiffel Tower is located in Paris.
# 10. score=0.000041 | Shakespeare wrote many famous plays.