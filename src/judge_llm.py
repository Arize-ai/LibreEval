import logging
import os

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from phoenix.evals import (
    HALLUCINATION_PROMPT_TEMPLATE,
    HALLUCINATION_PROMPT_RAILS_MAP,
    MistralAIModel,
    OpenAIModel,
    LiteLLMModel,
    AnthropicModel,
    llm_classify
)

from utils import lookup_hallucination_type, lookup_question_type, classify_hallucination

nest_asyncio.apply()
load_dotenv()

logger = logging.getLogger('arize.judge_llm')


class GenerateLabels:

    def __init__(self, configs):
        self.judge_llm1 = configs.judge_llm1
        self.judge_llm2 = configs.judge_llm2
        self.judge_llm3 = configs.judge_llm3
        self.data_prep_csv_path = configs.data_prep_csv_path
        self.judge_labels_csv_path = configs.judge_labels_csv_path
        self.rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())
        self.models = dict()
        self.configs = configs
        self.provide_explanation = configs.provide_explanation

        # Initialize models based on judge LLM configurations
        for judge_llm in [self.judge_llm1, self.judge_llm2, self.judge_llm3]:
            if not judge_llm:
                continue
                
            llm_code = self.configs.get_llm_code(judge_llm)
            model_key = judge_llm.replace('/', '_')
            
            if llm_code == 'gpt':
                self.models[model_key] = OpenAIModel(model=judge_llm)
            elif llm_code == 'mistral':
                self.models[model_key] = MistralAIModel(model=judge_llm)
            elif llm_code == 'anthropic':
                self.models[model_key] = AnthropicModel(model=judge_llm)
            elif judge_llm.startswith('litellm'):
                self.models[model_key] = LiteLLMModel(
                    model=judge_llm.replace('litellm/', ''),
                    top_p=0.9,
                    max_tokens=249 if judge_llm == self.judge_llm1 else None
                )

        # Set up columns for results
        self.relevant_columns = ["reference",
                                 "input",
                                 "output",
                                 "label",
                                 f"label_{self.judge_llm1}"]
        
        if self.provide_explanation:
            self.relevant_columns.append(f"explanation_{self.judge_llm1}")

        for judge_llm in [self.judge_llm2, self.judge_llm3]:
            if judge_llm:
                self.relevant_columns.append(f"label_{judge_llm}")
                if self.provide_explanation:
                    self.relevant_columns.append(f"explanation_{judge_llm}")

    def generate_labels(self, dataset: pd.DataFrame, model_name):
        print("MODEL NAME", model_name)
        predictions = llm_classify(
            model=self.models.get(model_name.replace('/', '_')),
            template=HALLUCINATION_PROMPT_TEMPLATE,
            system_instruction='You are an assistant that helps determine if answers are factual or hallucinated.',
            rails=self.rails,
            dataframe=dataset,
            provide_explanation=self.provide_explanation,
            concurrency=10
        )
        predictions_columns = ['label', ]
        predictions_columns = predictions_columns if not self.provide_explanation else predictions_columns + [
            'explanation']
        predictions_columns = {col: f"{col}_{model_name}" for col in predictions_columns}
        predictions.rename(columns=predictions_columns, inplace=True)
        dataset = dataset.merge(predictions, left_index=True, right_index=True)
        return dataset

    def generate_labels_and_merge_dataset(self):
        logger.info(f"Loading dataset from {self.data_prep_csv_path}")
        dataset = pd.read_csv(self.data_prep_csv_path)
        
        # Generate labels from all 3 judge LLMs
        logger.info(f"Generating labels using model: {self.judge_llm1}")
        dataset = self.generate_labels(dataset, self.judge_llm1)
        
        logger.info(f"Generating labels using model: {self.judge_llm2}")
        dataset = self.generate_labels(dataset, self.judge_llm2)
        
        logger.info(f"Generating labels using model: {self.judge_llm3}")
        dataset = self.generate_labels(dataset, self.judge_llm3)
        
        logger.info("Merging predictions from all 3 judges")
        dataset['label'] = dataset[[f"label_{self.judge_llm1}", f"label_{self.judge_llm2}", f"label_{self.judge_llm3}"]].mode(axis=1).iloc[:,0]

        dataset['hallucination_type_realized'] = dataset.apply(
            lambda row: classify_hallucination(row['reference'], row['input'], row['output']) if row['label'] == 'hallucinated' 
            else 'n/a',
            axis=1
        )
        
        # Add info labels from the configs
        dataset['rag_model'] = self.configs.data_prep_llm
        dataset['force_even_split'] = self.configs.force_even_split
        dataset['website'] = self.configs.website
        dataset['synthetic'] = str(self.configs.synthetic).strip().lower() == 'true'
        dataset['language'] = self.configs.language
        dataset['question_type'] = dataset['question_type'].apply(lookup_question_type)
        dataset['hallucination_type_encouraged'] = dataset['hallucination_type_encouraged'].apply(lookup_hallucination_type)
        self.relevant_columns += ['rag_model', 
                                  'force_even_split', 
                                  'website', 
                                  'synthetic', 
                                  'language', 
                                  'hallucination_type_realized', 
                                  'question_type', 
                                  'hallucination_type_encouraged']
        dataset = dataset[self.relevant_columns]
        
        # Save the merged dataset to a CSV file
        logger.info(f"Saving merged dataset to {self.judge_labels_csv_path}")
        if os.path.exists(self.judge_labels_csv_path):
            os.rename(
                self.judge_labels_csv_path,
                f"{self.judge_labels_csv_path.split('.')[0]}_old.csv"
            )
        os.makedirs(os.path.dirname(self.judge_labels_csv_path), exist_ok=True)
        dataset.to_csv(self.judge_labels_csv_path, index=False)
        logger.info(f"Merged dataset saved to {self.judge_labels_csv_path}")
        return dataset

    def run(self):
        if os.path.exists(self.judge_labels_csv_path) and self.configs.ignore_if_exists:
            logger.info(f"Labels already generated. Skipping, {self.judge_labels_csv_path}")
            return self.judge_labels_csv_path
        logger.info(f"Generating labels using models :{self.judge_llm1} {self.judge_llm2} {self.judge_llm3}")
        self.generate_labels_and_merge_dataset()
        logger.info(f'Labels generation completed, {self.judge_labels_csv_path}')
        return self.judge_labels_csv_path
