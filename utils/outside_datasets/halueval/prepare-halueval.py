import pandas as pd
import numpy as np
import os
import requests

def setup_directories():
    """Create raw and formatted directories."""
    raw_dir = "utils/outside_datasets/halueval/raw_datasets"
    formatted_dir = "utils/outside_datasets/halueval/formatted_datasets"

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

def prepare_halu_eval_qa_dataset():
    df = pd.read_json('utils/outside_datasets/halueval/raw_datasets/qa_data.json', lines=True)
    
    # Randomly choose between right_answer and hallucinated_answer
    random_mask = np.random.random(len(df)) < 0.5
    
    # Create new dataframe with selected answers
    result_df = df[['knowledge', 'question']].copy()
    result_df['output'] = np.where(random_mask, 
                                  df['right_answer'],
                                  df['hallucinated_answer'])
    result_df['label'] = np.where(random_mask, 
                                 'factual',
                                 'hallucinated')
    
    # Rename columns to match expected format
    df = result_df.rename(columns={
        'knowledge': 'reference',
        'question': 'input'
    })
    
    df.to_csv('utils/outside_datasets/halueval/formatted_datasets/halu_eval_qa_formatted.csv', index=False)

def prepare_halu_eval_dialogue_dataset():
    df = pd.read_json('utils/outside_datasets/halueval/raw_datasets/dialogue_data.json', lines=True)
    # Create random mask for 50/50 split
    random_mask = np.random.random(len(df)) < 0.5
    
    # Create new dataframe with selected responses
    result_df = df[['knowledge', 'dialogue_history']].copy()
    result_df['output'] = np.where(random_mask,
                                  df['right_response'],
                                  df['hallucinated_response'])
    result_df['label'] = np.where(random_mask,
                                 'factual',
                                 'hallucinated')
    
    # Rename columns to match expected format
    df = result_df.rename(columns={
        'knowledge': 'reference',
        'dialogue_history': 'input'
    })
    
    df.to_csv('utils/outside_datasets/halueval/formatted_datasets/halu_eval_dialogue_formatted.csv', index=False)
    
def prepare_halu_eval_summarization_dataset():
    df = pd.read_json('utils/outside_datasets/halueval/raw_datasets/summarization_data.json', lines=True)
    
    summary_question = 'Summarize the following document:'
    
    # Create random mask for 50/50 split
    random_mask = np.random.random(len(df)) < 0.5
    
    # Create new dataframe with selected summaries
    result_df = df[['document']].copy()
    result_df['output'] = np.where(random_mask,
                                  df['right_summary'],
                                  df['hallucinated_summary'])
    result_df['label'] = np.where(random_mask,
                                 'factual',
                                 'hallucinated')
    
    # Rename columns to match expected format
    df = result_df.rename(columns={
        'document': 'reference'
    })
    
    df['input'] = summary_question

    df.to_csv('utils/outside_datasets/halueval/formatted_datasets/halu_eval_summarization_formatted.csv', index=False)
    
def prepare_halu_eval_general_dataset():
    df = pd.read_json('utils/outside_datasets/halueval/raw_datasets/general_data.json', lines=True)
    
    df = df.rename(columns={
        'user_query': 'input',
        'chatgpt_response': 'output',
    })
    df['label'] = df['hallucination'].apply(lambda x: 'factual' if x == 'no' else 'hallucinated')
    df['reference'] = ''
    
    df.to_csv('utils/outside_datasets/halueval/formatted_datasets/halu_eval_general_formatted.csv', index=False)

if __name__ == '__main__':
    raw_dir, formatted_dir = setup_directories()
    
    # GitHub repository details
    GITHUB_REPO = "RUCAIBox/HaluEval"
    RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/refs/heads/main/data"

    # Files to download
    files_to_download = {
        "dialogue": "dialogue_data.json",
        "general": "general_data.json", 
        "qa": "qa_data.json",
        "summarization": "summarization_data.json"
    }

    for dataset, file_name in files_to_download.items():
        file_url = f"{RAW_BASE_URL}/{file_name}"
        file_path = os.path.join(raw_dir, file_name)
        download_dataset(file_url, file_path)

    prepare_halu_eval_qa_dataset()
    prepare_halu_eval_dialogue_dataset()
    prepare_halu_eval_summarization_dataset()
    prepare_halu_eval_general_dataset()