
import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load product data
df = pd.read_csv("apparel_pricing_data_v1.csv")
df["competitor_prices"] = df["competitor_prices"].apply(ast.literal_eval)
df["hourly_sales"] = df["hourly_sales"].apply(ast.literal_eval)

# TF-IDF matrix for item_name
tfidf = TfidfVectorizer(stop_words="english")
desc_matrix = tfidf.fit_transform(df["item_name"])

# Normalized Black Friday hourly profile
black_friday_profile = np.array([
    2, 3, 5, 6, 8, 10, 9, 7, 6, 5, 4, 4,
    3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1
])
black_friday_profile = black_friday_profile / black_friday_profile.sum()
