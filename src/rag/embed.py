import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# OpenAI APIs are costly
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Local Model
_model = SentenceTransformer("all-MiniLM-L6-v2")


# def embed_batch(texts: list[str]) -> list[list[float]]:
#    out: list[list[float]] = []
#    for i in range(0, len(texts), 100):
#        batch = texts[i : i + 100]
#        resp = _client.embeddings.create(
#            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
#            input=batch,
#        )
#        out.extend(d.embedding for d in resp.data)
#    return out

def embed_batch(texts: list[str]) -> list[list[float]]:
    # Generate embeddings locally
    embeddings = _model.encode(texts, batch_size=100, show_progress_bar=False)
    
    # Convert numpy array output back to lists of floats
    return [embedding.tolist() for embedding in embeddings]