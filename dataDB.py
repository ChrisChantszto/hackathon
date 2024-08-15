from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from tidb_vector.integrations import TiDBVectorClient
from dotenv import load_dotenv
import uuid
from tqdm import tqdm

# load the environment variables
load_dotenv()

print("Downloading and loading the embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model_dims = embed_model.get_sentence_embedding_dimension()

def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text)

def text_to_embedding(text):
    """Generates vector embeddings for the given text."""
    try:
        embedding = embed_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error encoding text: {text}")
        print(f"Error: {e}")
        return [0] * embed_model_dims  # Return a zero vector of the correct dimension

vector_store = TiDBVectorClient(
    table_name='movies',
    connection_string=os.environ.get('TIDB_DATABASE_URL'),
    vector_dimension=embed_model_dims,
    drop_existing_table=True,
)

print("Loading movie data...")
df_movies = pd.read_csv("movies_on.csv")

documents = []
print("Generating embeddings and preparing documents...")
for _, row in tqdm(df_movies.iterrows(), total=len(df_movies), desc="Processing movies"):
    overview = safe_text(row['overview'])
    if not overview:
        continue  # Skip this row if the overview is empty
    doc = {
        "id": str(uuid.uuid4()),
        "text": overview,
        "embedding": text_to_embedding(overview),
        "metadata": {
            "movie_id": row['id'],
            "title": row['title'],
            "genres": row['genres']
        }
    }
    documents.append(doc)

print("Inserting data into TiDB...")
chunk_size = 1000
pbar = tqdm(range(0, len(documents), chunk_size), desc="Inserting chunks")
for i in pbar:
    chunk = documents[i:i + chunk_size]
    vector_store.insert(
        ids=[doc["id"] for doc in chunk],
        texts=[doc["text"] for doc in chunk],
        embeddings=[doc["embedding"] for doc in chunk],
        metadatas=[doc["metadata"] for doc in chunk],
    )
    pbar.set_description(f"Inserted {i+chunk_size} documents")

print(f"Stored {len(documents)} movies in TiDB")