import os
import glob
import pandas as pd
import kagglehub

def load_news_dataset():
    """
    Downloads the 'real-and-fake-news-dataset' from Kaggle using kagglehub
    and loads the target CSV file into a pandas DataFrame.
    """
    print("Checking/Downloading dataset from Kaggle...")
    # This will download the latest version or return the path if already cached locally
    path = kagglehub.dataset_download("mucahiddemircan/real-and-fake-news-dataset")
    print(f"Path to dataset files: {path}")
    
    # Locate the CSV file inside the downloaded path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the dataset path: {path}")
        
    # Pick the first available CSV file
    target_csv = csv_files[0]
    print(f"Loading data from: {target_csv}")
    
    # Load dataset: contains 'text' and 'label' (1 for Real, 0 for Fake)
    df = pd.read_csv(target_csv)
    
    # Drop missing values to clean text data safely
    df = df.dropna(subset=['text', 'label'])
    return df

if __name__ == "__main__":
    # Test execution
    data = load_news_dataset()
    print(f"Dataset loaded successfully with shape: {data.shape}")
    print(data.head(2))