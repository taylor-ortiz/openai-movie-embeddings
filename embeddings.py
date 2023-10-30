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

data = movies[["Title", "Genre"]].to_dict("records")

# project = atlas.map_embeddings(
#     embeddings=np.array(plot_embeddings),
#     data=data
# )

def print_recommendations_from_strings(
    generated_movie_plot_embedding,
    strings,
    k_nearest_neighbors=4,
    model="text-embedding-ada-002"
):
    # Get all of the embeddings that we already have stored
    embeddings = [embedding_from_string(string) for string in strings]
    # Get embedding for our specific query string
    # query_embedding = embeddings[index_of_source_string]
    # Get distances between our embedding and all other embeddings
    distances = distances_from_embeddings(generated_movie_plot_embedding, embeddings)
    # Get indices of the nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    print('what is indices of nearest neighbors? ', indices_of_nearest_neighbors)
    # query_string = strings[index_of_source_string]
    match_count = 0
    for i in indices_of_nearest_neighbors:
        if generated_movie_plot_embedding == strings[i]:
            continue
        if match_count >= k_nearest_neighbors:
            break
        match_count += 1
        print(f"Found {match_count} closest match: ")
        print(f"Distance of: {distances[i]}")
        recommended_movie = movies.iloc[i]  # Using iloc to get the movie by row index
        recommended_title = recommended_movie["Title"]
        condensed_recommended_plot = condense_recommended_plot(strings[i])
        summary = f"""
           Move Title Recommendation: {recommended_title}
           Movie Description: {condensed_recommended_plot}
        """
        print(summary)

def generate_sample_movie_plot(user_input):

    if user_input:

        PROMPT = """
            You are a movie plot building tool.
            A user might provide a summary of varying length to give you context into a movie plot that you should come up with. 
            If no plot is recommended, you should just come up with a random movie plot on your own. 
            We will take the movie plot that you come up with based on user input or at random and send it into a vector database to get nearest neighbors of the plot to provide movie recommendations
            User input is: {user_input}.
            Please only include the text of the plot and nothing else.
            Please only keep the response to one paragraph.
        """

        messages = [
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": f"Write a prompt from the following description {user_input}"
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )

        return response["choices"][0]["message"]["content"]
    else:
        return None
    
def condense_recommended_plot(plot):
    PROMPT = """
            Your job is to take a provided movie plot and summarize the plot so that it can be readable to the user
        """

    messages = [
        {
            "role": "system",
            "content": PROMPT
        },
        {
            "role": "user",
            "content": f"Please summarize the provided plot: {plot}"
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return response["choices"][0]["message"]["content"]
generated_movie_plot = generate_sample_movie_plot('we would like to watch a movie about a spaceship crew that is on a long voyage and ends up somehowe encountering violent alien life and needs to enforce every skill to survive but some of the crew doesnt make it')
# generated_movie_plot = generate_sample_movie_plot('we want to watch a movie about a young boy who grows close with his childhood dog')
if generated_movie_plot != None:
    plot_embedding_from_generated_plot = get_embedding(generated_movie_plot)
    print_recommendations_from_strings(plot_embedding_from_generated_plot, movie_plots, 2)