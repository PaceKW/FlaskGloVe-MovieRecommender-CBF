from flask import Flask, jsonify, request, render_template
import gensim.downloader
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Inisiasi Flask
app = Flask(__name__)

# Load the Glove
glove = gensim.downloader.load('glove-wiki-gigaword-100')

# Load the datasets
df_interaction = pd.read_csv('data/interaction.csv', index_col=0)
df_metadata = pd.read_csv('data/metadata.csv', index_col=0)

# Check Data
df_interaction[df_interaction.duplicated(subset=["user","item"])]
df_interaction[df_interaction.rating.isna()]

df_metadata = df_metadata.dropna(subset = ["original_title","overview"], how="any")

# Split Data
train = df_interaction[df_interaction.split=="train"]
test = df_interaction[df_interaction.split=="test"]

# Text Cleaning
df_metadata["combined"] = df_metadata.original_title + " " + df_metadata.overview
df_metadata["combined"] = df_metadata["combined"].str.lower()
df_metadata["tokenized"] = df_metadata.combined.apply(lambda x: word_tokenize(x))
df_metadata["clean_tokenized"] = df_metadata["tokenized"].apply(lambda tokens: [word for word in tokens if word.isalpha() and word not in stopwords.words("english")])
df_metadata.drop(columns=["combined","tokenized"], inplace=True)

def get_embedding(list_of_tokens):
    embeddings = np.zeros(100)
    for token in list_of_tokens:
        if token in glove:
            embeddings += glove[token]
    return embeddings

df_metadata["embedding"] = df_metadata["clean_tokenized"].apply(lambda x: get_embedding(x))

def recommend(user_id, train, test, df_metadata, top_n=50):
    # Mendapatkan daftar item yang disukai oleh pengguna
    items_of_user = train.query("user==@user_id").item.to_list()

    # Membuat profil pengguna berdasarkan item yang disukai
    embedding_of_movies_of_user = df_metadata.loc[df_metadata.item_id.isin(items_of_user), "embedding"]
    profile_user = np.sum(embedding_of_movies_of_user.values)

    # Menghitung kemiripan cosine antara profil pengguna dan semua film
    df_metadata["cosine"] = df_metadata.embedding.apply(lambda x: cosine_similarity(profile_user.reshape(1,100), x.reshape(1,100))[0][0])

    # Mendapatkan top N rekomendasi
    top_n_items = df_metadata[["item_id","original_title","cosine"]].sort_values("cosine", ascending=False).head(top_n)

    return top_n_items

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    user_id = None
    if request.method == "POST":
        user_id = int(request.form.get("user_id"))  # Konversi ke int
        recommendations = recommend(user_id, test=test, train=train, df_metadata=df_metadata)
        
        # Debugging: Menampilkan user ID dan rekomendasi di terminal
        print(f"User ID received: {user_id}")  
        print("Recommendations for user ID:", recommendations)  # Check recommendations

        # Convert recommendations to a list of dictionaries if it's a DataFrame
        if not recommendations.empty:
            recommendations = recommendations.to_dict(orient='records')
        else:
            recommendations = []
    
    return render_template('index.html', recommendations=recommendations, user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)