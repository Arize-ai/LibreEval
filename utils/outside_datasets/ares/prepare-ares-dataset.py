import os
import requests
import pandas as pd

def setup_directories():
    """Create raw and formatted directories."""
    raw_dir = "utils/outside_datasets/ares/raw_datasets"
    formatted_dir = "utils/outside_datasets/ares/formatted_datasets"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(formatted_dir, exist_ok=True)

    return raw_dir, formatted_dir

def download_dataset(file_url, file_path):
    """Download dataset from URL and save to file path."""
    print(f"📥 Downloading from {file_url} ...")
    
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded to {file_path}")
        return True
    else:
        print(f"❌ Failed to download from {file_url}")
        return False

def clean_dataset(df, dataset, required_keys):
    """Clean and format dataset."""
    df["source"] = dataset
    
    # Verify required columns exist
    missing_columns = required_keys - set(df.columns)
    if missing_columns:
        print(f"❌ Missing columns in {dataset}: {missing_columns}")
        return None

    # Keep only required columns and drop nulls
    df = df[list(required_keys)]
    df = df.dropna(subset=["Answer_Faithfulness_Label"])

    # Map labels
    df["Answer_Faithfulness_Label"] = df["Answer_Faithfulness_Label"].map({
        1: "factual",
        0: "hallucinated"
    })

    # Rename columns to standard format
    df = df.rename(columns={
        "Query": "input",
        "Document": "reference", 
        "Answer": "output",
        "Answer_Faithfulness_Label": "label"
    })

    return df

def process_ares_datasets():
    dataset_project_name = "ares"
    raw_dir, formatted_dir = setup_directories()
    
    # GitHub repository details
    GITHUB_REPO = "stanford-futuredata/ARES"
    RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/datasets/eval_datasets"

    # Files to download
    files_to_download = {
        "fever": "fever_ratio_0.5.tsv",
        "hotpotqa": "hotpotqa_ratio_0.5.tsv",
        "nq": "nq_ratio_0.5.tsv",
        "wow": "wow_ratio_0.5.tsv",
    }

    # Required columns
    required_keys = {
        "Document",
        "Query",
        "Answer",
        "Context_Relevance_Label",
        "Answer_Faithfulness_Label",
        "Answer_Relevance_Label",
        "source"
    }

    for dataset, file_name in files_to_download.items():
        # Download dataset
        file_url = f"{RAW_BASE_URL}/{dataset}/{file_name}"
        file_path = os.path.join(raw_dir, f"{dataset_project_name}_{dataset}_{file_name}")
        
        if not download_dataset(file_url, file_path):
            continue

        try:
            # Load and clean data
            df = pd.read_csv(file_path, sep="\t")
            cleaned_df = clean_dataset(df, dataset, required_keys)
            
            if cleaned_df is None:
                continue

            # Save cleaned data
            clean_file_path = os.path.join(formatted_dir, f"{dataset_project_name}_{dataset}_formatted.csv")
            cleaned_df.to_csv(clean_file_path, index=False)
            
            print(cleaned_df.label.value_counts())
            print(f"✅ Cleaned data saved: {clean_file_path}")

        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")

    print("\n✅ All datasets processed successfully.")

if __name__ == "__main__":
    process_ares_datasets()