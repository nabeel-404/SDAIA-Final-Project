from typing import Dict, List


class Retriever:
    def __init__(self) -> None:

        # initialize the index 
        # initialize the retriever
        # initialize the document_ids_to_sentence (from the json file)
        pass

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:

        # 1. encoder the query (generate embedding)
        # 2. retrieve top k documents
        # 3. convert the document ids to sentences (from the json file)
        # 4. return the documents (in text)

        return [{'text': 'text'}]

