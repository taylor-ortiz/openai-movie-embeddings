import openai
from dotenv import dotenv_values
import numpy as np
import tenacity
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
import tiktoken
from nomic import atlas
from openai.embeddings_utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances

config = dotenv_values('../.env')
openai.api_key = config["OPENAI_API_KEY"]

df = pd.read_csv('wiki_movie_plots_deduped.csv')
movies = df[df["Origin/Ethnicity"] == 'American'].sort_values("Release Year", ascending=False).head(5000)

movie_plots = movies["Plot"].values

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]

enc = tiktoken.encoding_for_model("text-embedding-ada-002")
total_tokens = sum([len(enc.encode(plot)) for plot in movie_plots])
total_tokens
cost = total_tokens * (.0004 / 1000)
print(f"Estimated cost ${cost:.2f}")


# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

# set path to embedding cache
embedding_cache_path = "movie_embeddings_cache2.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string,
    model="text-embedding-ada-002",
    embedding_cache=embedding_cache
):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

plot_embeddings = [embedding_from_string(plot, model="text-embedding-ada-002") for plot in movie_plots]
print(len(plot_embeddings))

data = movies[["Title", "Genre"]].to_dict("records")

# project = atlas.map_embeddings(
#     embeddings=np.array(plot_embeddings),
#     data=data
# )

def print_recommendations_from_strings(
    strings,
    index_of_source_string,
    k_nearest_neighbors=3,
    model="text-embedding-ada-002"
):
    # Get all of the embeddings that we already have stored
    embeddings = [embedding_from_string(string) for string in strings]
    # Get embedding for our specific query string
    query_embedding = embeddings[index_of_source_string]
    # Get distances between our embedding and all other embeddings
    distances = distances_from_embeddings(query_embedding, embeddings)
    # Get indices of the nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    
    query_string = strings[index_of_source_string]
    match_count = 0
    for i in indices_of_nearest_neighbors:
        if query_string == strings[i]:
            continue
        if match_count >= k_nearest_neighbors:
            break
        match_count += 1
        print(f"Found {match_count} closest match: ")
        print(f"Distance of: {distances[i]}")
        print(strings[i])


print_recommendations_from_strings(movie_plots, 2)