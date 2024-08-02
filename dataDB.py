from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from tidb_vector.integrations import TiDBVectorClient
from dotenv import load_dotenv
import uuid

# load the environment variables
load_dotenv()

print("Downloading and loading the embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model_dims = embed_model.get_sentence_embedding_dimension()

def text_to_embedding(text):
    """Generates vector embeddings for the given text."""
    embedding = embed_model.encode(text)
    return embedding.tolist()

vector_store = TiDBVectorClient(
    table_name='movies',
    connection_string=os.environ.get('TIDB_DATABASE_URL'),
    vector_dimension=embed_model_dims,
    drop_existing_table=True,
)

df_movies = pd.read_csv("movies.csv")

documents = []
for _, row in df_movies.iterrows():
    doc = {
        "id": str(uuid.uuid4()),
        "text": row['overview'],
        "embedding": text_to_embedding(row['overview']),
        "metadata": {
            "movie_id": row['id'],
            "title": row['title'],
            "release_date": row['release_date'],
            "genres": row['genres']
        }
    }
    documents.append(doc)

# Insert data into TiDB
vector_store.insert(
    ids=[doc["id"] for doc in documents],
    texts=[doc["text"] for doc in documents],
    embeddings=[doc["embedding"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
)

print(f"Stored {len(documents)} movies in TiDB")