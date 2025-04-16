import os
import numpy as np
import pandas as pd
from phoenix.evals import HALLUCINATION_PROMPT_TEMPLATE


def read_and_balance_datasets(datasets, output_dir):
    """Read datasets and balance labels within each dataset."""
    os.makedirs(f'combined_datasets_for_tuning/{output_dir}', exist_ok=True)

    # Read and balance each dataset
    balanced_dataframes = []
    for dataset in datasets:
        df = pd.read_csv(dataset)
        print(f"Count of {dataset}: {len(df)}")
        
        min_count = min(df[df['label'] == 'hallucinated'].shape[0], 
                       df[df['label'] == 'factual'].shape[0])
        
        balanced_df = pd.concat([
            df[df['label'] == 'hallucinated'].sample(n=min_count, random_state=42),
            df[df['label'] == 'factual'].sample(n=min_count, random_state=42)
        ])
        print(f"Count after balancing: {len(balanced_df)}")
        balanced_dataframes.append(balanced_df)
    
    # Combine and deduplicate
    balanced_dataset = pd.concat(balanced_dataframes)
    print(f"Count of combined balanced datasets: {len(balanced_dataset)}")
    
    balanced_dataset = balanced_dataset.drop_duplicates(subset=['input', 'reference', 'output'])
    print(f"Count after removing duplicates: {len(balanced_dataset)}")
    
    balanced_dataset.to_csv(f'combined_datasets_for_tuning/{output_dir}/balanced_hallucinations.csv', index=False)
    return balanced_dataset


def create_prompt_completions_dataset(balanced_dataset):
    """Create dataset in prompt-completion format."""
    prompt_completions = []
    for _, row in balanced_dataset.iterrows():
        prompt = HALLUCINATION_PROMPT_TEMPLATE.format(
            variable_values={'input': row['input'], 
                           'reference': row['reference'], 
                           'output': row['output']})
        prompt_completions.append({
            'prompt': prompt,
            'completion': row['label']
        })
    return pd.DataFrame(prompt_completions)


def create_message_format_dataset(dataset):
    """Create dataset in message format."""
    print("Creating message format dataset...")
    
    system_prompt = HALLUCINATION_PROMPT_TEMPLATE.template[:HALLUCINATION_PROMPT_TEMPLATE.template.find("[BEGIN DATA]")]
    user_prompt = HALLUCINATION_PROMPT_TEMPLATE.template[HALLUCINATION_PROMPT_TEMPLATE.template.find("[BEGIN DATA]")-1:]
    
    messages = []
    for _, row in dataset.iterrows():
        messages.append({
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(
                    input=row['input'], 
                    reference=row['reference'], 
                    output=row['output'])},
                {"role": "assistant", "content": row['label']}
            ]
        })
    return pd.DataFrame(messages)

def create_message_format_dataset_nick(dataset):
    """Create dataset in message format."""
    print("Creating message format dataset...")
    
    system_prompt = "You are an 'EVAL assistant' evaluating prompts and responses for hallucinations. Your task is to evaluate the accuracy of AI-generated responses using reference text."
    user_prompt = HALLUCINATION_PROMPT_TEMPLATE.template
    
    messages = []
    for _, row in dataset.iterrows():
        messages.append({
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(
                    input=row['input'], 
                    reference=row['reference'], 
                    output=row['output'])},
                {"role": "assistant", "content": row['label']}
            ]
        })
    return pd.DataFrame(messages)


def split_dataset(dataset, label_column):
    """Split dataset into train/val/test while maintaining label balance."""
    train_size, val_size = 0.7, 0.15
    
    factual = dataset[dataset[label_column] == 'factual']
    hallucinated = dataset[dataset[label_column] == 'hallucinated']
    
    # Split each label group
    train_factual, val_factual, test_factual = np.split(
        factual.sample(frac=1, random_state=42), 
        [int(train_size * len(factual)), int((train_size + val_size) * len(factual))]
    )
    train_hallucinated, val_hallucinated, test_hallucinated = np.split(
        hallucinated.sample(frac=1, random_state=42),
        [int(train_size * len(hallucinated)), int((train_size + val_size) * len(hallucinated))]
    )
    
    return (pd.concat([train_factual, train_hallucinated]),
            pd.concat([val_factual, val_hallucinated]),
            pd.concat([test_factual, test_hallucinated]))


def prompt_completion_format(train_dataset, validation_dataset, test_dataset, output_dir):
    """Save datasets in prompt-completion format."""
    output_path = f'combined_datasets_for_tuning/{output_dir}/prompt_completion_format'
    os.makedirs(output_path, exist_ok=True)

    print("Creating prompt completions datasets...")
    for name, data in [('train', train_dataset), 
                      ('validation', validation_dataset)]:
        df = create_prompt_completions_dataset(data)
        df.to_csv(f'{output_path}/{name}.csv', index=False)
        df.to_json(f'{output_path}/{name}.jsonl', orient='records', lines=True)


def message_format(train_dataset, validation_dataset, test_dataset, output_dir, synthetic=False):
    """Save datasets in message format."""
    output_path = f'combined_datasets_for_tuning/{output_dir}/message_format'
    os.makedirs(output_path, exist_ok=True)

    print("Creating message format datasets...")
    suffix = '_by_synthetic' if synthetic else ''
    
    train_messages = create_message_format_dataset_nick(train_dataset)
    validation_messages = create_message_format_dataset_nick(validation_dataset)
    
    train_messages.to_json(f'{output_path}/train{suffix}_nick.jsonl', orient='records', lines=True)
    validation_messages.to_json(f'{output_path}/validation{suffix}_nick.jsonl', orient='records', lines=True)


def create_full_datasets(datasets, output_dir):
    """Create and save all dataset formats."""
    os.makedirs('combined_datasets_for_tuning', exist_ok=True)

    print("Reading and balancing datasets...")
    balanced_dataset = read_and_balance_datasets(datasets, output_dir)

    print("Splitting dataset...")
    train_dataset, validation_dataset, test_dataset = split_dataset(balanced_dataset, label_column='label')
    test_dataset.to_csv(f'combined_datasets_for_tuning/{output_dir}/test.csv', index=False)
    
    print("Creating prompt completion format...")
    prompt_completion_format(train_dataset, validation_dataset, test_dataset, output_dir)
    
    print("Creating message format...")
    message_format(train_dataset, validation_dataset, test_dataset, output_dir)


def read_and_balance_datasets_by_synthetic(datasets):
    """Read and balance datasets ensuring equal samples across synthetic/non-synthetic."""
    print("Reading datasets...")
    all_dfs = [pd.read_csv(path) for path in datasets]
    
    # Find minimum count across all datasets and labels
    min_count = float('inf')
    for df in all_dfs:
        hallucinated_count = len(df[df['label'] == 'hallucinated'])
        factual_count = len(df[df['label'] == 'factual'])
        min_count = min(min_count, hallucinated_count, factual_count)
    
    print(f"Using {min_count} rows per label per dataset...")
    
    # Balance each dataset
    balanced_dfs = []
    for df in all_dfs:
        balanced_dfs.append(pd.concat([
            df[df['label'] == 'hallucinated'].sample(n=min_count, random_state=42),
            df[df['label'] == 'factual'].sample(n=min_count, random_state=42)
        ]))
    
    return pd.concat(balanced_dfs, ignore_index=True)


def create_datasets_balanced_by_synthetic(datasets, output_dir):
    """Create datasets with synthetic/non-synthetic balance."""
    os.makedirs('combined_datasets_for_tuning', exist_ok=True)

    print("Reading and balancing datasets...")
    balanced_dataset = read_and_balance_datasets_by_synthetic(datasets)

    print("Splitting dataset...")
    train_dataset, validation_dataset, test_dataset = split_dataset(balanced_dataset, label_column='label')
    test_dataset.to_csv(f'combined_datasets_for_tuning/{output_dir}/test_balanced_by_synthetic.csv', index=False)
    
    print("Creating message format...")
    message_format(train_dataset, validation_dataset, output_dir, synthetic=True)


if __name__ == "__main__":
    all_language_datasets = [
        'combined_datasets_for_evals/synthetic_hallucinations_all_languages.csv',
        'combined_datasets_for_evals/non_synthetic_hallucinations_all_languages.csv'
    ]
    
    english_datasets = [
        'combined_datasets_for_evals/synthetic_hallucinations_english.csv',
        'combined_datasets_for_evals/non_synthetic_hallucinations_english.csv'
    ]
    
    create_full_datasets(all_language_datasets, 'all_languages')
    # create_full_datasets(english_datasets, 'english_only')

    # create_datasets_balanced_by_synthetic(all_language_datasets, 'all_languages')
    # create_datasets_balanced_by_synthetic(english_datasets, 'english_only')
    
    