from sentence_transformers import SentenceTransformer
import os
from tidb_vector.integrations import TiDBVectorClient
from dotenv import load_dotenv
import uuid

# load the connection string from the .env file
load_dotenv()

print("Downloading and loading the embedding model...")
embed_model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L12-cos-v5", trust_remote_code=True)
embed_model_dims = embed_model.get_sentence_embedding_dimension()

def text_to_embedding(text):
    """Generates vector embeddings for the given text."""
    embedding = embed_model.encode(text)
    return embedding.tolist()

vector_store = TiDBVectorClient(
   table_name='embedded_documents',
   connection_string=os.environ.get('TIDB_DATABASE_URL'),
   vector_dimension=embed_model_dims,
   drop_existing_table=True,
)

documents = [
    {
        "id": str(uuid.uuid4()),
        "text": "The quick brown fox jumps over the lazy dog",
        "embedding": text_to_embedding("The quick brown fox jumps over the lazy dog"),
        "metadata": {"category": "sentence", "type": "pangram"},
    },
    {
        "id": str(uuid.uuid4()),
        "text": "To be or not to be, that is the question",
        "embedding": text_to_embedding("To be or not to be, that is the question"),
        "metadata": {"category": "quote", "author": "William Shakespeare"},
    },
    {
        "id": str(uuid.uuid4()),
        "text": "The Earth is the third planet from the Sun and the only astronomical object known to harbor life",
        "embedding": text_to_embedding("The Earth is the third planet from the Sun and the only astronomical object known to harbor life"),
        "metadata": {"category": "fact", "subject": "astronomy"},
    },
    {
        "id": str(uuid.uuid4()),
        "text": "Machine learning is a method of data analysis that automates analytical model building",
        "embedding": text_to_embedding("Machine learning is a method of data analysis that automates analytical model building"),
        "metadata": {"category": "definition", "field": "computer science"},
    },
    {
        "id": str(uuid.uuid4()),
        "text": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci",
        "embedding": text_to_embedding("The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci"),
        "metadata": {"category": "art", "artist": "Leonardo da Vinci"},
    },
    {
        "id": str(uuid.uuid4()),
        "text": "Climate change is a long-term shift in global or regional climate patterns",
        "embedding": text_to_embedding("Climate change is a long-term shift in global or regional climate patterns"),
        "metadata": {"category": "environmental science", "topic": "climate"},
    },
]

vector_store.insert(
    ids=[doc["id"] for doc in documents],
    texts=[doc["text"] for doc in documents],
    embeddings=[doc["embedding"] for doc in documents],
    metadatas=[doc["metadata"] for doc in documents],
)

def print_result(query, result):
   print(f"Search result (\"{query}\"):")
   for r in result:
      print(f"- text: \"{r.document}\", distance: {r.distance}")
   print()

# Example queries
queries = [
    "What is the famous Shakespeare quote?",
    "Tell me about our planet",
    "How does AI analyze data?",
    "Famous paintings in art history",
    "Global environmental concerns"
]

for query in queries:
    query_embedding = text_to_embedding(query)
    search_result = vector_store.query(query_embedding, k=2)
    print_result(query, search_result)