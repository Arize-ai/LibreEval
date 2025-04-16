import argparse
import logging
import os
import sys

from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import MistralAIModel, AnthropicModel, OpenAIModel, LiteLLMModel
from phoenix.otel import register
from dotenv import load_dotenv

logger = logging.getLogger('arize.configs')


class Configs:

    def __init__(self, args=None):
        load_dotenv()
        self.llm_codes = {
            'gpt': 'gpt', 'ft:gpt': 'gpt', 'litellm': 'litellm',
            'mistral': 'mistral', 'claude': 'anthropic', 'huggingfacelocal': 'huggingfacelocal'
        }
        self.path_seperator = '_'
        self.sleep_time = 1
        self._dataset = None
        self.args = self.__parse_args(args)
        self._initialize_configs(self.args)
        self.validate_models()
        self.log_configs()

    def __parse_args(self, args=None):
        """
        :return:
        """
        parser = argparse.ArgumentParser(description="Configuration for dataset generation and evaluation.")
        parser.add_argument('--synthetic', type=str, help='Generate Synthetic Answer', default='True')
        parser.add_argument('--language', type=str, help='Language, Defaults to English', default='en')
        parser.add_argument('--max-depth', type=int, help='Max Depth', default=20)
        parser.add_argument('--no-of-docs', type=int, help='No of Docs', default=0)
        parser.add_argument('--questions-per-doc', type=int, help='No of questions per doc', default=3)
        parser.add_argument('--language-query-param', type=str, help='Language code Query Param, if any url which use language code from query params', default='')
        parser.add_argument('--use-default-templates', type=str, default='true', help='if set false, it will use overwritten templates from prompts.py file, this will be applied for both judge llm and evaluation of models')
        parser.add_argument('--ignore-if-exists', type=str, help='Will not generate qa and labels if the files are existed', default='true')
        parser.add_argument('--generate-confusion-matrix', type=str, help='This flag will enable to generate confusion matrix', default='true')
        parser.add_argument('--provide-explanation', type=str, help='Set true if explanation is needed for labels', default='true')
        parser.add_argument('--hallucinations', type=str, help='number of hallucinations required to generate', default=0)
        parser.add_argument('--website', type=str, help='Website URL to extract data from ex: https://example.com', default='')
        parser.add_argument('--data-prep-llm', type=str, help='Data Prep LLM, Provide full name of the model, ex: gpt-4o-mini', default='gpt-4o-mini')
        parser.add_argument('--judge-llm1', type=str, help='Judge LLM1, Provide full name of the model', default='gpt-4o')
        parser.add_argument('--judge-llm2', type=str, help='Judge LLM2, Provide complete name of the model', default='claude-3-5-sonnet-latest')
        parser.add_argument('--judge-llm3', type=str, help='Judge LLM3, Provide complete name of the model', default='litellm/together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo')
        parser.add_argument('--dataset-to-evaluate', type=str, help="Labeled dataset to evaluate. Accepts a path to a csv file with columns 'input', 'reference', 'output', 'label'", default='')
        parser.add_argument('--evaluation-models', type=str, help='Comma separated list of evaluation models, you can provide full name of the model', default='')
        parser.add_argument('--force-even-split', type=str, help='Force an even split of hallucinations across the dataset. Only applies for synthetic hallucination data.', default='true')
        parser.add_argument('--enable-observability', type=str, help='Enable observability to Phoenix for the project', default='false')
        parser.add_argument('--qa-variation', type=str, help='Whether question, answers, or neither should be forceably varied to fit question and hallucination types', default='neither')
        if args is None:
            args = sys.argv[1:]
        args = parser.parse_args(args)
        self._validate_arguments(args)
        return args

    def _validate_arguments(self, args):
        group1_args = [args.website, args.data_prep_llm, args.judge_llm1]
        group2_args = [args.dataset_to_evaluate]
        if all(group1_args) and any(group2_args):
            raise argparse.ArgumentError(None, "Cannot use both (--website, --data-prep-llm, --judge-llm1) and (--dataset-to-evaluate).")
        elif not (all(group1_args) or all(group2_args)):
            raise argparse.ArgumentError(None, "You must provide either (--website, --data-prep-llm, --judge-llm1) or (--dataset-to-evaluate).")

    def _initialize_configs(self, args):
        self.website = args.website
        self.data_prep_llm = args.data_prep_llm
        self.judge_llm1 = args.judge_llm1
        self.judge_llm2 = args.judge_llm2
        self.judge_llm3 = args.judge_llm3
        self.synthetic = not str(args.synthetic).lower() == 'false'
        self.language = args.language
        self.max_depth = int(args.max_depth)
        self.no_of_docs = int(args.no_of_docs)
        self.hallucinations = int(args.hallucinations)
        self.language_query_param = args.language_query_param
        self.questions_per_doc = int(args.questions_per_doc) or 3
        self.dataset_to_evaluate = args.dataset_to_evaluate
        self.provide_explanation = str(args.provide_explanation).strip().lower() == 'true'
        self.use_default_templates = not str(args.use_default_templates).strip().lower() == 'false'
        self.generate_confusion_matrix = not str(args.generate_confusion_matrix).strip().lower() == 'false'
        self.ignore_if_exists = str(args.ignore_if_exists).strip().lower() == 'true'
        self.force_even_split = str(args.force_even_split).strip().lower() == 'true'
        self.evaluation_model_names = [model.strip() for model in args.evaluation_models.split(',') if model.strip()]
        self.enable_observability = str(args.enable_observability).strip().lower() == 'true'
        if self.hallucinations and self.questions_per_doc:
            self.no_of_docs = (round(self.hallucinations / self.questions_per_doc) + (self.hallucinations % self.questions_per_doc))
        self._evaluation_models = dict()
        self.concurrent_requests = 10
        self.required_instrumentor = [self.judge_llm1, self.judge_llm2, self.judge_llm3, self.data_prep_llm, self.evaluation_model_names]
        self.qa_variation = args.qa_variation
        self._initialize_directories()

    def _initialize_directories(self):
        self._base_website_name = self.website.split('://')[-1].split("/")[0].replace('.', '_')
        self.data_dir = os.path.join("data")
        self.docs_dir = os.path.join(self.data_dir, self._base_website_name, self.language)
        self.output_dir = os.path.join(self.data_dir, "output", self._base_website_name)
        if self.data_prep_llm != "":
            self.labels_dir = os.path.join("labeled_datasets", self.data_prep_llm+'-hallucinations')
        self.evaluations_dir = os.path.join("evaluation_results", self.evaluation_model_names[0])
        for directory in [self.data_dir, self.docs_dir, self.output_dir, self.labels_dir, self.evaluations_dir]:
            os.makedirs(directory, exist_ok=True)

    def log_configs(self):
        if self.no_of_docs != 0 and self.hallucinations != 0:
            logger.info(f"Configs are: No of docs: {self.no_of_docs}, Questions for doc {self.questions_per_doc}, "
                        f"No of hallucinations: {self.hallucinations}, language: {self.language}, "
                        f"Data Prep LLM: {self.data_prep_llm}, Judge LLM1: {self.judge_llm1}, Judge LLM2: {self.judge_llm2}, Judge LLM3: {self.judge_llm3}"
                        f", website: {self.website}")

    @property
    def data_prep_csv_path(self):
        data_type = 'synthetic' if self.synthetic else 'non_synthetic'
        parts = [self.data_prep_llm, data_type, self.no_of_docs, self.questions_per_doc]
        return f'{self.output_dir}/{self.language}_generated_qa_with_contexts_{self.sanitized_path(parts)}.csv'

    @property
    def judge_labels_csv_path(self):
        if not self._dataset:
            path = self.labels_dir
            if self.synthetic:
                path += '/synthetic'
                if self.force_even_split:
                    path += '/even-split-of-hallucinations-and-factuals'
                else:
                    path += '/majority-hallucinations'
            else:
                path += '/non_synthetic'
                
            path += f'/{self.language}'
            data_type = 'synthetic' if self.synthetic else 'non_synthetic'
            variation = '_question' if self.qa_variation == 'question' else '_answer' if self.qa_variation == 'answer' else ''
            parts = [self._base_website_name, self.data_prep_llm, data_type, self.judge_llm1, self.judge_llm2, self.language, variation]
            self._dataset = f'{path}/{self.sanitized_path(parts)}.csv'
            self._dataset = self._dataset.replace("__", '_')
        return self._dataset

    def sanitized_path(self, parts):
        sanitized_models = [str(part).replace(":", "").replace("/", "_") for part in parts if part]
        x = self.path_seperator.join(sanitized_models).replace('-', '_').replace(":", "_")
        return x

    @property
    def evaluation_results_csv_path(self):
        data_type = 'synthetic' if self.synthetic else 'non_synthetic'
        evaluation_models = self.evaluation_models or {}
        even_split = '_even' if self.force_even_split else ''
        if not self.data_prep_llm:
            parts = list(evaluation_models.keys())
            model_path = self.dataset_to_evaluate.split("/")[-1].replace("labels_", 'evaluation_').replace('.csv', '')
            return f'{self.evaluations_dir}/{model_path}_{self.sanitized_path(parts)}{even_split}.csv'.replace('//', '/')
        parts = [data_type, self.judge_llm1, self.judge_llm2, self.judge_llm3 ] + list(evaluation_models.keys()) + [self.language]
        return f'{self.evaluations_dir}/evaluation_{self.sanitized_path(parts)}{even_split}.csv'.replace('//', '/')

    def confusion_matrix_image_path(self, model_name):
        data_type = 'synthetic' if self.synthetic else 'non_synthetic'
        even_split = '_even' if self.force_even_split else ''   
        parts = [self.data_prep_llm, data_type, self.judge_llm1, self.judge_llm2, model_name, self.language]
        return f'{self.evaluations_dir}/confusion_matrix_{self.sanitized_path(parts)}{even_split}.png'.replace('//', '/')

    @property
    def evaluation_models(self):
        if self._evaluation_models:
            return self._evaluation_models
        for model in self.evaluation_model_names:
            if model.startswith('ft:gpt') or model.startswith('gpt'):
                self._evaluation_models[model] = OpenAIModel(model=model, max_tokens=2000)
                self.required_instrumentor.append('gpt')
            elif model.startswith('mistral'):
                self._evaluation_models[model] = MistralAIModel(model=model, max_tokens=2000)
            elif model.startswith("claude"):
                self._evaluation_models[model] = AnthropicModel(model=model, max_tokens=2000)
                self.required_instrumentor.append('anthropic')
            elif model.startswith("litellm/"):
                model = model.replace('litellm/', '')
                self._evaluation_models[model] = LiteLLMModel(model=model, max_tokens=2000)
                self.required_instrumentor.append('litellm')
            elif model.startswith("huggingfacelocal/"):
                self._evaluation_models[model] = model
        return self._evaluation_models

    def initialize_phoenix_instrumentor(self):
        if self.enable_observability:
            if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
                logger.error("PHOENIX_COLLECTOR_ENDPOINT is not set. Please set it to the endpoint of your Phoenix collector.")
                return
            
            project_name = os.environ.get("PHOENIX_PROJECT_NAME") or "default"
            if os.environ.get("PHOENIX_API_KEY"):
                os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.environ.get('PHOENIX_API_KEY')}"
            tracer_provider = register(project_name=project_name)
            
            for instrumentor in self.required_instrumentor:
                if isinstance(instrumentor, list):
                    instrumentor = str(instrumentor)
                    
                if 'gpt' in instrumentor:
                    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
                if "anthropic" in instrumentor or "claude" in instrumentor:
                    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
                if "litellm" in instrumentor:
                    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    def get_llm_code(self, model_name):
        for code in self.llm_codes.keys():
            if model_name.lower().startswith(code):
                return self.llm_codes.get(code)
        raise Exception(f'{model_name} is not supported. Supported models are {", ".join(self.llm_codes.keys())}')

    def validate_models(self):
        if self.data_prep_llm:
            self.get_llm_code(self.data_prep_llm)
        if self.judge_llm1:
            self.get_llm_code(self.judge_llm1)
        if self.judge_llm2:
            self.get_llm_code(self.judge_llm2)
        if self.judge_llm3:
            self.get_llm_code(self.judge_llm3)
        if self.evaluation_model_names:
            [self.get_llm_code(model) for model in self.evaluation_model_names]
