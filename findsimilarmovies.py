from sqlalchemy import create_engine, text
import os
import ast
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Create SQLAlchemy engine
engine = create_engine(os.environ.get('TIDB_DATABASE_URL'))

def get_random_movie_id(connection):
    query = text("SELECT id FROM movies ORDER BY RAND() LIMIT 1")
    result = connection.execute(query).fetchone()
    return result[0] if result else None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def string_to_vector(embedding_str):
    return np.array(ast.literal_eval(embedding_str))

def find_similar_movies_db(connection, movie_id):
    # First, get the embedding of the target movie
    query = text("SELECT embedding FROM movies WHERE id = :movie_id")
    result = connection.execute(query, {"movie_id": movie_id}).fetchone()
    
    if result is None:
        return None
    
    target_embedding = string_to_vector(result[0])
    
    # Fetch all other movies
    query = text("SELECT id, document, embedding FROM movies WHERE id != :movie_id")
    results = connection.execute(query, {"movie_id": movie_id}).fetchall()
    
    # Calculate similarities
    similarities = []
    for movie in results:
        movie_embedding = string_to_vector(movie.embedding)
        similarity = cosine_similarity(target_embedding, movie_embedding)
        similarities.append((movie.id, movie.document, similarity))
    
    # Sort by similarity (descending) and return top 5
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:5]

def normalize_embedding(embedding_str):
    # Parse the string representation of the list into an actual list
    embedding_list = ast.literal_eval(embedding_str)
    # Convert the list to a numpy array
    embedding_array = np.array(embedding_list)
    # Normalize the array
    return embedding_array / np.linalg.norm(embedding_array)

def find_similar_movies_python(connection, movie_id):
    query = text("SELECT id, document, embedding FROM movies")
    movies = connection.execute(query).fetchall()
    
    target_embedding = next(normalize_embedding(movie.embedding) for movie in movies if movie.id == movie_id)
    
    similarities = [
        (movie.id, movie.document, cosine_similarity(target_embedding, normalize_embedding(movie.embedding)))
        for movie in movies if movie.id != movie_id
    ]
    
    return sorted(similarities, key=lambda x: x[2], reverse=True)[:5]

def check_embeddings(connection):
    query = text("SELECT id, embedding FROM movies LIMIT 5")
    results = connection.execute(query).fetchall()
    
    for result in results:
        print(f"Movie ID: {result.id}")
        print(f"Embedding type: {type(result.embedding)}")
        print(f"Embedding length: {len(result.embedding)}")
        print(f"First few values: {result.embedding[:5]}")
        print()

def main():
    with engine.connect() as connection:
        movie_id = get_random_movie_id(connection)
        if movie_id is None:
            print("No movies found in the database.")
            return

        print(f"Selected movie ID: {movie_id}")

        # Get details of the selected movie
        query = text("SELECT document, meta FROM movies WHERE id = :movie_id")
        result = connection.execute(query, {"movie_id": movie_id}).fetchone()
        if result:
            print(f"Selected movie overview: {result.document}")
            print(f"Selected movie metadata: {result.meta}")

        print("\nChecking embeddings:")
        check_embeddings(connection)

        print("\nFinding similar movies using database calculation:")
        similar_movies_db = find_similar_movies_db(connection, movie_id)
        
        if similar_movies_db is None:
            print(f"No similar movies found for movie with ID: {movie_id}")
        else:
            for movie_id, document, similarity in similar_movies_db:
                print(f"ID: {movie_id}")
                print(f"Overview: {document}")
                print(f"Similarity: {similarity:.6f}")
                
                meta_query = text("SELECT meta FROM movies WHERE id = :movie_id")
                meta_result = connection.execute(meta_query, {"movie_id": movie_id}).fetchone()
                if meta_result:
                    print(f"Metadata: {meta_result.meta}")
                print()

        print("\nFinding similar movies using Python calculation:")
        similar_movies_python = find_similar_movies_python(connection, movie_id)
        for movie_id, document, similarity in similar_movies_python:
            print(f"ID: {movie_id}")
            print(f"Overview: {document}")
            print(f"Similarity: {similarity:.6f}")
            
            meta_query = text("SELECT meta FROM movies WHERE id = :movie_id")
            meta_result = connection.execute(meta_query, {"movie_id": movie_id}).fetchone()
            if meta_result:
                print(f"Metadata: {meta_result.meta}")
            print()

if __name__ == "__main__":
    main()