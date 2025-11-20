import pickle
import numpy as np

class WinProbModel:
    def __init__(self):
        # Try to load trained model
        try:
            with open("models/win_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            self.trained = True
            print("[model] Loaded trained model.")
        except Exception as e:
            print("[model] WARNING: Could not load trained model, using dummy fallback.", e)
            self.model = None
            self.trained = False

    def predict_proba(self, feature_row):
        """
        feature_row is a dict with:
        - home_team_id
        - away_team_id
        """
        if self.trained:
            X = np.array([[feature_row["home_team_id"], feature_row["away_team_id"]]])
            p_home = self.model.predict_proba(X)[0][1]
            return float(p_home)

        # Fallback behavior
        return 0.5


def bootstrap_dummy_training(model):
    """No-op now (kept for backward compatibility)."""
    return
