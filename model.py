import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data_loader import load_news_dataset

MODEL_FILE = "fake_news_pipeline.joblib"

def train_and_save_model():
    """
    Loads data, builds a standard TF-IDF + LogisticRegression pipeline, 
    trains it, and saves the final binary object to disk.
    """
    # 1. Fetch clean dataset
    df = load_news_dataset()
    
    # Optimize dataset size if necessary for memory savings on free tier platforms
    # (Optional: df = df.sample(n=20000, random_state=42) if deployment RAM limits out)
    
    X = df['text']
    y = df['label']
    
    # 2. Split dataset into train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Machine Learning Pipeline...")
    # 3. Design a scikit-learn pipeline for smooth prediction lifecycle
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate operational accuracy
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Training Complete. Test Set Accuracy: {accuracy:.4f}")
    
    # 5. Export binary package
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved locally as '{MODEL_FILE}'")
    return pipeline

def get_prediction_pipeline():
    """
    Helper function to load model if it exists, otherwise trigger training on-the-fly.
    """
    try:
        pipeline = joblib.load(MODEL_FILE)
        print("Existing trained pipeline loaded successfully.")
        return pipeline
    except FileNotFoundError:
        print("Trained model binary not found. Initiating fresh build...")
        return train_and_save_model()

if __name__ == "__main__":
    train_and_save_model()