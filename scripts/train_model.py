import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Kaggle data
games = pd.read_csv("data/games.csv")

# Drop rows with missing scores (future games)
games = games.dropna(subset=["PTS_home", "PTS_away"])

# Label: 1 if home team wins, else 0
games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)

# Simple features for now (we can upgrade later)
features = games[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]]
labels = games["home_win"]

# Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save the model
with open("models/win_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete!")
print("Accuracy:", model.score(X_test, y_test))
