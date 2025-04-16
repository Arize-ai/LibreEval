import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openinference.instrumentation import using_metadata
from phoenix.evals import (
    HALLUCINATION_PROMPT_TEMPLATE,
    HALLUCINATION_PROMPT_RAILS_MAP
)
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import prompts
from configs import Configs
from phoenix.evals import llm_classify
from phoenix.evals import HALLUCINATION_PROMPT_TEMPLATE as PHOENIX_HALLUCINATION_PROMPT_TEMPLATE

import asyncio

logger = logging.getLogger('arize.evaluate_model')


class HallucinationEvaluator:

    def __init__(self, configs: Configs):
        self.configs = configs
        self.models = configs.evaluation_models
        self.dataset = configs.dataset_to_evaluate
        if self.dataset.endswith('.csv'):
            self.dataset = pd.read_csv(self.dataset)
        elif self.dataset.endswith('.jsonl'):
            self.dataset = pd.read_json(self.dataset, lines=True)
        elif self.dataset.endswith('.json'):
            self.dataset = pd.read_json(self.dataset)
            # Rename columns to match expected format
            column_mapping = {
                'context': 'reference',
                'question': 'input', 
                'answer': 'output',
                'is_hallucination': 'label'
            }
            self.dataset = self.dataset.rename(columns=column_mapping)
            self.dataset['label'] = self.dataset['label'].apply(lambda x: 'hallucinated' if x else 'factual')
            logger.info(f"Dataset: {self.dataset.head()}")
        else:
            raise ValueError(f"Unsupported file extension for evaluation dataset: {self.dataset}")
        
        self.rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())

    def evaluate(self) -> Dict[str, Dict[str, Any]]:
        results = asyncio.run(self._evaluate_async())
        return results
    
    async def _evaluate_async(self):
        results = {}
        predictions = {}
        i = 0
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            if 'huggingfacelocal' in model_name:
                model_predictions = await self.generate_predictions_from_huggingface(model_name)
            else:
                model_predictions = await self.generate_predictions_from_base_model(model)
            results[f'model_{i}'] = self.calculate_metrics(model_name, model_predictions)
            predictions[f'model_{i}_{model_name}'] = model_predictions
            i += 1
        self.save_evaluation_results(predictions)
        return results
    
    def printif(self, condition: bool, *args: Any, **kwargs: Any) -> None:
        if condition:
            tqdm.write(*args, **kwargs)
    
    def snap_to_rail(self, raw_string: Optional[str], rails: List[str], verbose: bool = False) -> str:
        """
        Snaps a string to the nearest rail, or returns None if the string cannot be
        snapped to a rail.

        Args:
            raw_string (str): An input to be snapped to a rail.

            rails (List[str]): The target set of strings to snap to.

        Returns:
            str: A string from the rails argument or "UNPARSABLE" if the input
                string could not be snapped.
        """
        if not raw_string:
            return 'NOT_PARSABLE'
        snap_string = raw_string.lower()
        rails = list(set(rail.lower() for rail in rails))
        rails.sort(key=len, reverse=True)
        found_rails = set()
        for rail in rails:
            if rail in snap_string:
                found_rails.add(rail)
                snap_string = snap_string.replace(rail, "")
        if len(found_rails) != 1:
            self.printif(verbose, f"- Cannot snap {repr(raw_string)} to rails")
            return 'NOT_PARSABLE'
        rail = list(found_rails)[0]
        self.printif(verbose, f"- Snapped {repr(raw_string)} to rail: {rail}")
        return rail
    
    async def generate_predictions_from_huggingface(self, model):
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        model = model.replace('huggingfacelocal/', '')
        
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model)

        predictions = []
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc=f"Generating predictions from huggingface model"):
            formatted_query = PHOENIX_HALLUCINATION_PROMPT_TEMPLATE.template.format(
                input=row['input'],
                reference=row['reference'],
                output=row['output']
            )
            generator = pipeline("text-classification", model=model, tokenizer=tokenizer)
            output = generator(formatted_query)
            if output[0]['label'] == 'LABEL_1':
                output_label = 'hallucinated'
            elif output[0]['label'] == 'LABEL_0':
                output_label = 'factual'
            else:
                output_label = output[0]['label']
            predictions.append(output_label)
            
        predictions_df = pd.DataFrame({'label': predictions})
        return predictions_df
    
    async def generate_predictions_from_bert(self, model):
        template = HALLUCINATION_PROMPT_TEMPLATE.template + " The answer is <mask>"
        
        import requests
        import pandas as pd
        from typing import List
        import os
        from tqdm import tqdm
        
        API_URL = os.environ.get("HUGGINGFACE_BERT_API_BASE")
        headers = {
            "Accept": "application/json", 
            "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}",
            "Content-Type": "application/json"
        }

        predictions = []
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc=f"Generating predictions from bert model"):
            # Format template with row data
            formatted_input = template.format(
                input=row['input'],
                reference=row['reference'],
                output=row['output']
            )
            
            # Make API call
            try:
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": formatted_input,
                        "parameters": {}
                    }
                )
                result = response.json()
                # for FT'd roberta:
                # predictions.append(result[0]['label'])
                
                # for base roberta:
                # loop through and find the highest "score" value of either token_str=="hallucinated" or token_str=="factual"
                max_score = -1
                selected_label = "no_response"
                
                for option in result:
                    if option['token_str'].strip() in ['hallucinated', 'factual']:
                        if option['score'] > max_score:
                            max_score = option['score']
                            selected_label = option['token_str'].strip()
                predictions.append(selected_label)
            except Exception as e:
                logger.error(f"Error making API call: {str(e)}")
                predictions.append(None)
                
        # Convert predictions to DataFrame matching expected format
        predictions_df = pd.DataFrame({
            'label': predictions
        })
        
        return predictions_df

    async def generate_predictions_from_base_model(self, model):
        template = HALLUCINATION_PROMPT_TEMPLATE
        if not self.configs.use_default_templates:
            template.explanation_template = prompts.HALLUCINATION_PROMPT_TEMPLATE_WITH_EXPLANATION
            template.template = prompts.HALLUCINATION_PROMPT_BASE_TEMPLATE
            
        with using_metadata({"model": model.model}):
            predictions = llm_classify(
                model=model,
                template=PHOENIX_HALLUCINATION_PROMPT_TEMPLATE,
                rails=self.rails,
                dataframe=self.dataset,
                provide_explanation=self.configs.provide_explanation,
                concurrency=20,
                verbose=False
            )
        return predictions

    def calculate_metrics(self, model_name, predictions: pd.DataFrame) -> Dict[str, Any]:
        merged_df = self.dataset.merge(predictions, left_index=True, right_index=True)
        logger.info(f"Merged DataFrame: {merged_df.head()}")

        total = len(merged_df)
        correct = sum(merged_df['label_x'] == merged_df['label_y'])
        accuracy = correct / total

        true_positives = sum((merged_df['label_x'] == 'hallucinated') & (merged_df['label_y'] == 'hallucinated'))
        false_positives = sum((merged_df['label_x'] == 'factual') & (merged_df['label_y'] == 'hallucinated'))
        false_negatives = sum((merged_df['label_x'] == 'hallucinated') & (merged_df['label_y'] == 'factual'))
        true_negatives = sum((merged_df['label_x'] == 'factual') & (merged_df['label_y'] == 'factual'))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_samples': total,
            'correct_predictions': correct
        }

        # Print the results
        # Log results to console and file
        results_str = f"Evaluation Results for {model_name}:\n"
        for metric, value in results.items():
            metric_str = f"{metric}: {value}\n"
            results_str += metric_str
            logger.info(metric_str.strip())

        if self.configs.generate_confusion_matrix:
            print(merged_df.head(10))
            print(merged_df['label_x'].value_counts())
            print(merged_df['label_y'].value_counts())
            # Create and display confusion matrix
            cm = confusion_matrix(merged_df['label_x'], merged_df['label_y'], labels=['factual', 'hallucinated'])
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['factual', 'hallucinated'],
                        yticklabels=['factual', 'hallucinated'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            confusion_matrix_path = self.configs.confusion_matrix_image_path(model_name)
            plt.savefig(confusion_matrix_path)
            logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
            plt.close()
            
        # Write results to local file
        results_file = self.configs.confusion_matrix_image_path(model_name).replace('.png', '.txt')
        with open(results_file, "w") as f:
            f.write(results_str)
        logger.info(f"Results saved to {results_file}")
        
        return results

    def save_evaluation_results(self, predictions: Dict[str, pd.DataFrame], ):
        merged_df = self.dataset.copy()
        for model_name, model_predictions in predictions.items():
            merged_df[f'{model_name}_prediction'] = model_predictions['label']
            if 'explanation' in model_predictions.columns:
                merged_df[f'{model_name}_explanation'] = model_predictions['explanation']
        merged_df.to_csv(self.configs.evaluation_results_csv_path, index=False)
        logger.info(f"Evaluation results saved to {self.configs.evaluation_results_csv_path}")
