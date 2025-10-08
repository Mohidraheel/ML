import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init
init(autoreset=True)


data = pd.read_csv("D:\\Fast\\python\\Intern\\task5\\movies.csv")


print(Fore.CYAN + "\n🎬 Welcome to the Movie Recommender System 🎬\n")
print(Fore.YELLOW + f"Loaded dataset with {data.shape[0]} movies and {data.shape[1]} features.\n")
print(data.isnull().sum())
sns.countplot(y='genres', data=data, order=data['genres'].value_counts().index[:10])
plt.title("Top 10 Genres in Dataset")
plt.show()


selected_features = ['genres', 'director', 'keywords', 'tagline', 'cast']
for feature in selected_features:
    data[feature] = data[feature].fillna('')


combined_features = data['genres'] + ' ' + data['director'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast']


vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(features)

movie_name = input(Fore.GREEN + "Enter your favourite movie name: " + Style.RESET_ALL).strip()

list_of_all_titles = data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if not find_close_match:
    print(Fore.RED + "\n❌ Sorry, no close match found for that movie title. Please try again.\n")
else:
    close_match = find_close_match[0]
    index_of_the_movie = data[data.title == close_match].index[0]

    print(Fore.CYAN + f"\n✅ You searched for: {Fore.YELLOW}{close_match}\n")

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print(Fore.MAGENTA + "🎥 Movies recommended for you:\n" + Style.RESET_ALL)

    for i, movie in enumerate(sorted_similar_movies[1:30], start=1):  
        index = movie[0]
        title_from_index = data.iloc[index]['title']
        print(Fore.WHITE + f"{i:>2}. {Fore.LIGHTBLUE_EX}{title_from_index}")

    print(Fore.CYAN + "\n✨ Enjoy your movie night! ✨\n")
