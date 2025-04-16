import os
import pandas as pd

def combine_csvs(directories, international_flag):
    combined_df = pd.DataFrame()
    for directory in directories:
        if directory.endswith('.csv'):
            df = pd.read_csv(directory)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            for root, _, files in os.walk(directory):
                if international_flag and '/en' in root:
                    continue
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def save_combined_csvs(input_directories, output_directory, output_filename, international_flag=False):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    combined_df = combine_csvs(input_directories, international_flag)
    output_file = os.path.join(output_directory, output_filename)
    
    # Remove duplicate rows based on the 'input', 'output', and 'reference' columns
    combined_df = combined_df.drop_duplicates(subset=['input', 'output', 'reference'])
    combined_df.to_csv(output_file, index=False)

def create_csvs_for_evals():
    # Non-synthetic english hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/non_synthetic/en',
                        'labeled_datasets/gpt-4o-hallucinations/non_synthetic/en',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/non_synthetic/en',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/non_synthetic/en'], 
                       'combined_datasets_for_evals', 
                       'non_synthetic_hallucinations_english.csv')
    
    # Non-synthetic international hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/non_synthetic',
                        'labeled_datasets/gpt-4o-hallucinations/non_synthetic',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/non_synthetic',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/non_synthetic'], 
                       'combined_datasets_for_evals', 
                       'non_synthetic_hallucinations_international.csv',
                       international_flag=True)
    
    # Non-synthetic all languages hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/non_synthetic',
                        'labeled_datasets/gpt-4o-hallucinations/non_synthetic',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/non_synthetic',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/non_synthetic'], 
                       'combined_datasets_for_evals', 
                       'non_synthetic_hallucinations_all_languages.csv')
    
    # Synthetic english hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/synthetic/even-split-of-hallucinations-and-factuals/en',
                        'labeled_datasets/gpt-4o-hallucinations/synthetic/even-split-of-hallucinations-and-factuals/en',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/synthetic/even-split-of-hallucinations-and-factuals/en',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/synthetic/even-split-of-hallucinations-and-factuals/en'], 
                       'combined_datasets_for_evals', 
                       'synthetic_hallucinations_english.csv')
    
    # Synthetic international hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/synthetic',
                        'labeled_datasets/gpt-4o-hallucinations/synthetic',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/synthetic',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/synthetic'], 
                       'combined_datasets_for_evals', 
                       'synthetic_hallucinations_international.csv',
                       international_flag=True)
    
    # Synthetic all languages hallucinations
    save_combined_csvs(['labeled_datasets/claude-3-5-sonnet-latest-hallucinations/synthetic/even-split-of-hallucinations-and-factuals',
                        'labeled_datasets/gpt-4o-hallucinations/synthetic/even-split-of-hallucinations-and-factuals',
                        'labeled_datasets/litellm/groq/llama-3.1-8b-instant-hallucinations/synthetic/even-split-of-hallucinations-and-factuals',
                        'labeled_datasets/litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-hallucinations/synthetic/even-split-of-hallucinations-and-factuals'], 
                       'combined_datasets_for_evals', 
                       'synthetic_hallucinations_all_languages.csv')

if __name__ == "__main__":
    create_csvs_for_evals()