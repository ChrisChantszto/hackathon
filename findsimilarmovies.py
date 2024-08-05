from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Create SQLAlchemy engine
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

def find_similar_movies(movie_id):
    query = text("SELECT embedding FROM movies WHERE id = :movie_id")
    
    with engine.connect() as connection:
        result = connection.execute(query, {"movie_id": movie_id}).fetchone()
        
        if result is None:
            return None  # or handle the case where no movie is found
        
        movie_embedding = result[0]
        
        # Now, let's find similar movies
        similar_query = text("""
            SELECT id, document, 
                   1 - (embedding <=> :embedding) AS similarity
            FROM movies
            WHERE id != :movie_id
            ORDER BY similarity DESC
            LIMIT 5
        """)
        
        similar_movies = connection.execute(similar_query, 
                                            {"embedding": movie_embedding, 
                                             "movie_id": movie_id}).fetchall()
        
        return similar_movies

def main():
    # Replace 'some_movie_id' with an actual ID from your database
    movie_id = 'some_movie_id'
    similar_movies = find_similar_movies(movie_id)
    
    if similar_movies is None:
        print(f"No movie found with ID: {movie_id}")
    else:
        print(f"Movies similar to movie with ID {movie_id}:")
        for movie in similar_movies:
            print(f"ID: {movie.id}, Document: {movie.document}, Similarity: {movie.similarity:.4f}")

if __name__ == "__main__":
    main()