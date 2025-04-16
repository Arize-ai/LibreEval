from main import main
from tqdm import tqdm
import logging

logger = logging.getLogger('arize.run_all')

def run_dataset_generation():
    hallucinations_per_config = 500
    
    configurations_databricks = [
        # ENGLISH
        ['--language=en',
        '--website=https://docs.databricks.com/en/index.html',
        '--data-prep-llm={data_prep_llm}',
        '--questions-per-doc=2',
        '--ignore-if-exists=false',
        f'--hallucinations={hallucinations_per_config}',
        '--synthetic=true'],
        
        # # Portugese
        # ['--language=pt',
        # '--website=https://docs.databricks.com/pt/index.html',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        # # Japanese
        # ['--language=ja',
        # '--website=https://docs.databricks.com/ja/index.html',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
    ]

    configurations_mongodb = [
        # ENGLISH
        ['--language=en',
        '--website=https://www.mongodb.com/docs/',
        '--data-prep-llm={data_prep_llm}',
        '--questions-per-doc=2',
        '--ignore-if-exists=false',
        f'--hallucinations={hallucinations_per_config}',
        '--synthetic=true'],
        
        # # Portugese
        # ['--language=pt',
        # '--website=https://www.mongodb.com/pt-br/docs/',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        # #break
        
        # # Japanese
        # ['--language=ja',
        # '--website=https://www.mongodb.com/ja-jp/docs/',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        #  # Korean
        # ['--language=ko',
        # '--website=https://www.mongodb.com/ko-kr/docs/',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        # # Chinese
        # ['--language=zh',
        # '--website=https://www.mongodb.com/zh-cn/docs/',
        # '--data-prep-llm=gpt-4o',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
    ]
    
    configurations_adobe = [
        # ENGLISH
        ['--language=en',
        '--website=https://experienceleague.adobe.com/en/docs',
        '--data-prep-llm={data_prep_llm}',
        '--questions-per-doc=2',
        '--ignore-if-exists=false',
        f'--hallucinations={hallucinations_per_config}',
        '--synthetic=false'],
        
        # # Portugese
        # ['--language=pt',
        # '--website=https://experienceleague.adobe.com/pt/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=false'],
        
        # # Japanese
        # ['--language=ja',
        # '--website=https://experienceleague.adobe.com/ja/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=false'],
        
        #  # Korean
        # ['--language=ko',
        # '--website=https://experienceleague.adobe.com/ko/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=false'],
        
        # # Chinese
        # ['--language=zh',
        # '--website=https://experienceleague.adobe.com/zh/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        # Spanish
        # ['--language=es',
        # '--website=https://experienceleague.adobe.com/es/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true'],
        
        # # French
        # ['--language=fr',
        # '--website=https://experienceleague.adobe.com/fr/docs',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--ignore-if-exists=false',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=false'],
    ]

    configurations = [configurations_adobe, configurations_mongodb, configurations_databricks]
    # configurations = [configurations_databricks]
    data_prep_llms = ['gpt-4o', 'claude-3-5-sonnet-latest', 'litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo']
    
    for config_set in tqdm(configurations, desc="Generating datasets"):
        for config in config_set:
            config_copy = config.copy()
            for data_prep_llm in data_prep_llms:
                config_copy[2] = config[2].replace('{data_prep_llm}', data_prep_llm)
                
                config_copy[6] = "--synthetic=false"
                config_copy.append("--qa-variation=question")
                logging.info("RUNNING CONFIG: ", config_copy)
                main(config_copy)

                # config_copy[6] = "--synthetic=true"
                # config_copy.append("--qa-variation=question")
                # logging.info("RUNNING CONFIG: ", config_copy)
                # main(config_copy)
                
                # config_copy[7] = "--qa-variation=answer"
                # logging.info("RUNNING CONFIG: ", config_copy)
                # main(config_copy)

def run_dataset_generation_other_domains():
    hallucinations_per_config = 500
    
    configurations = [
        # ENGLISH        
        # ['--language=en',
        # '--website={website}',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--use-default-templates=false',
        # '--provide-explanation=true',
        # '--ignore-if-exists=true',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true',
        # '--qa-variation=question'],
        
        # ['--language=en',
        # '--website={website}',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--use-default-templates=false',
        # '--provide-explanation=true',
        # '--ignore-if-exists=true',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=true',
        # '--qa-variation=answer'],
        
        # ['--language=en',
        # '--website={website}',
        # '--data-prep-llm={data_prep_llm}',
        # '--questions-per-doc=2',
        # '--use-default-templates=false',
        # '--provide-explanation=true',
        # '--ignore-if-exists=true',
        # f'--hallucinations={hallucinations_per_config}',
        # '--synthetic=false'],

        ['--language=en',
        '--website={website}',
        '--data-prep-llm={data_prep_llm}',
        '--questions-per-doc=2',
        '--use-default-templates=false',
        '--provide-explanation=true',
        '--ignore-if-exists=true',
        f'--hallucinations={hallucinations_per_config}',
        '--synthetic=false',
        '--qa-variation=question'],
    ]
    
    data_prep_llms = ['gpt-4o', 'claude-3-5-sonnet-latest', 'litellm/together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo']
    
    websites = ['https://www.law.cornell.edu/',
                'https://www.investopedia.com/',
                'https://pmc.ncbi.nlm.nih.gov/',
                'https://www.ncbi.nlm.nih.gov/',
                'https://medlineplus.gov/',
                'https://earthobservatory.nasa.gov/',
                'https://www.noaa.gov/research']
    
    for config in tqdm(configurations, desc="Generating datasets"):
        config_copy = config.copy()
        for website in websites:
            config_copy[1] = config[1].replace('{website}', website)
            for model in data_prep_llms:
                config_copy_copy = config_copy.copy()
                config_copy_copy[2] = config_copy[2].replace('{data_prep_llm}', model)
                try:
                    print(config_copy_copy)
                    main(config_copy_copy)
                except Exception as e:
                    print(f"Error with config: {config_copy_copy}")
                    print(e)

def run_evaluations():
    evaluation_configs = [
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/non_synthetic_hallucinations_all_languages.csv',
        '--evaluation-models={model_name}',
        '--language=all',
        '--synthetic=false'],
        
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/non_synthetic_hallucinations_international.csv',
        '--evaluation-models={model_name}',
        '--language=international',
        '--synthetic=false'],
                
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/non_synthetic_hallucinations_english.csv',
        '--evaluation-models={model_name}',
        '--language=english',
        '--synthetic=false'],
        
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/synthetic_hallucinations_all_languages.csv',
        '--evaluation-models={model_name}',
        '--language=all',
        '--synthetic=true'],
        
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/synthetic_hallucinations_international.csv',
        '--evaluation-models={model_name}',
        '--language=international',
        '--synthetic=true'],
                
        ['--dataset-to-evaluate=combined_datasets_for_evals_rd3/synthetic_hallucinations_english.csv',
        '--evaluation-models={model_name}',
        '--language=english',
        '--synthetic=true'],
    ]
    
    models_to_evaluate = ['gpt-4o-mini', 
                          'claude-3-5-haiku-latest', 
                          'gpt-4o',
                          'claude-3-5-sonnet-latest',
                        #   'litellm/azure_ai/Phi-3-5-mini-instruct-zogdt', 
                        #   'litellm/groq/llama-3.1-8b-instant',
                        ]
    
    for model in models_to_evaluate:
        for config in tqdm(evaluation_configs, desc=f"Evaluating datasets with {model}"):
            config_copy = config.copy()
            config_copy[1] = config[1].replace('{model_name}', model)
            try:
                main(config_copy)
            except Exception as e:
                print(f"Error with config: {config_copy}")
                print(e)

if __name__ == "__main__":
    run_dataset_generation()
    # run_dataset_generation_other_domains()
    # run_evaluations()