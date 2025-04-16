import sys
from main import main
from tqdm import tqdm

def run_evals():
    datasets = [
        'combined_datasets_for_tuning/all_languages/test.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/halueval/formatted_datasets/halu_eval_dialogue_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/halueval/formatted_datasets/halu_eval_qa_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/halueval/formatted_datasets/halu_eval_summarization_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/halueval/formatted_datasets/halu_eval_general_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/ares/formatted_datasets/ares_fever_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/ares/formatted_datasets/ares_hotpotqa_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/ares/formatted_datasets/ares_nq_formatted.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/outside_datasets/ares/formatted_datasets/ares_wow_formatted.csv'
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Advanced logical reasoning.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Default question type.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_english.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Errors, contradictions, or unsolvable questions.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Multimodal content.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_non_english.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_non_synthetic.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Other common hallucinated questions.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_Out-of-scope information.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_synthetic.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Entity-error hallucination.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Incompleteness hallucination.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Outdated information hallucination.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Overclaim hallucination.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Unverifiable information hallucination.csv',
        '/Users/jgilhuly/Documents/dev/GitHub/dataset-generation-research/utils/dataset-analysis/temp/test_hallucination_Relation-error hallucination.csv'
    ]
    
    base_models = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-3.5-turbo',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022',
        'litellm/together_ai/ArizeAI/Qwen/Qwen2-1.5B-Instruct-6897624b'
    ]
    
    fine_tuned_models = [
        'ft:gpt-4o-mini-2024-07-18:arize-ai:jg-3-1:B6F3o8d5',
        'litellm/together_ai/ArizeAI/Qwen2-1.5B-Instruct-1b7ae98e-b6b45d02',
        'litellm/together_ai/ArizeAI/Qwen2-1.5B-Instruct-885cb19a-21805d18'
    ]
    
    def evaluate_model(dataset, model):
        try:
            config = [
                f'--dataset-to-evaluate={dataset}',
                f'--evaluation-models={model}',
                '--provide-explanation=False',
                # '--enable-observability=True',
            ]
            main(config)
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")

    models = base_models + fine_tuned_models
    total_evals = len(datasets) * len(models)

    with tqdm(total=total_evals, desc="Running evaluations") as pbar:
        for dataset in datasets:
            for model in models:
                print(f"Running evaluation for dataset: {dataset}, model: {model}")
                try:
                    evaluate_model(dataset, model)
                except Exception as e:
                    print(f"An error occurred for dataset {dataset} and model {model}: {e}")
                pbar.update(1)


if __name__ == "__main__":
    run_evals()
