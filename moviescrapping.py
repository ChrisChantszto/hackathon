import requests
import pandas as pd
from time import sleep
from tqdm import tqdm

API_KEY = "50f2171bb8fa1065f50ceecb2a3e1669"
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "id": data["id"],
            "title": data["title"],
            "overview": data["overview"],
            "release_date": data["release_date"],
            "genres": [genre["name"] for genre in data["genres"]],
        }
    return None

movies = []

for movie_id in tqdm(range(1, 9999)):
    movie = fetch_movie_details(movie_id)
    if movie:
        movies.append(movie)
    sleep(0.5)

df_movies = pd.DataFrame(movies)
df_movies.to_csv("movies.csv", index=False)
print(f"Fetched {len(movies)} movies")