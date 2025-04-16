import asyncio
import logging
import os

import backoff
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import re
from data_prep_llm import OpenAIDataPrepLLM, AnthropicDataPrepLLM, LiteLLMDataPrepLLM

logger = logging.getLogger('arize.generate_questions_on_docs')


@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
async def retryable_task(func, *args, **kwargs):
    """
    This function wraps API calls with backoff retry mechanism.
    The retry uses exponential backoff with full jitter.
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.exception(e)
        raise e


class GenerateQuestionAnswers:
    llm_mappings = {
        'gpt': OpenAIDataPrepLLM,
        'anthropic': AnthropicDataPrepLLM,
        'litellm': LiteLLMDataPrepLLM
    }

    def __init__(self, configs):
        self.output_dir = configs.docs_dir
        data_prep_llm = configs.get_llm_code(configs.data_prep_llm)
        self.llm = self.llm_mappings[data_prep_llm](configs.data_prep_llm, configs)
        self.questions_per_doc = configs.questions_per_doc
        self.hallucinate = configs.synthetic
        self.data_prep_csv_path = configs.data_prep_csv_path
        self.parser = 'html.parser'
        self.configs = configs
        self.semaphore = asyncio.Semaphore(self.configs.concurrent_requests)
        self.force_even_split = configs.force_even_split
        self.qa_variation = configs.qa_variation

    async def extract_paragraphs(self, file_name):
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, self.parser)
        return [p.text.strip() for p in soup.find_all('p') if p.text.strip()]

    async def _generate_question(self, chosen_paragraph):
        question_type = random.randint(1, 5) if self.qa_variation == 'question' else 0 # we limit this to 5 types even though there are 6 types, because models refuse to generate type 6 questions
        question = await retryable_task(self.llm.create_question, chosen_paragraph, question_type)
        return question, question_type

    async def _generate_answer(self, question, chosen_paragraph):
        if not self.hallucinate:
            # non-synthetic, do no encouragement of hallucinations
            answer = await retryable_task(self.llm.generate_non_synthetic_answer, question, chosen_paragraph)
            hallucination_type = 0
            return answer, hallucination_type

        # Handle synthetic hallucination cases
        if self.force_even_split and random.random() >= 0.4:
            # 60% chance of non-synthetic answer when force_even_split is true
            answer = await retryable_task(self.llm.generate_non_synthetic_answer, question, chosen_paragraph)
            hallucination_type = 0
        else:
            # Generate synthetic hallucination
            hallucination_type = random.randint(1, 6) if self.qa_variation == 'answer' else 7
            answer = await retryable_task(self.llm.generate_synthetic_answer, question, chosen_paragraph, hallucination_type)
        
        return answer, hallucination_type

    async def process_document(self, paragraphs):
        questions = list()
        for _ in range(self.questions_per_doc):
            chosen_paragraph = await retryable_task(self.llm.create_informative_paragraph, paragraphs)
            
            question, question_type = await self._generate_question(chosen_paragraph)
            answer, hallucination_type = await self._generate_answer(question, chosen_paragraph)

            questions.append(dict(input=question,
                                reference=chosen_paragraph,
                                output=answer,
                                question_type=question_type,
                                hallucination_type_encouraged=hallucination_type))
        return questions

    async def process_with_semaphore(self, paragraphs):
        async with self.semaphore:  # Limit concurrent tasks with the semaphore
            return await self.process_document(paragraphs)

    async def main_process_documents(self):
        filenames = [f for f in os.listdir(self.configs.docs_dir)[:self.configs.no_of_docs] if f.endswith('.html')]
        # Create tasks for each document to be processed concurrently
        tasks = []
        for filename in filenames:
            paragraphs = await self.extract_paragraphs(filename)  # Assuming `extract_paragraphs` is async
            if not paragraphs:
                continue
            tasks.append(self.process_with_semaphore(paragraphs))
        # Await completion of all document processing tasks
        results = []
        with tqdm(total=len(tasks), desc="Generating Question & Answers....", dynamic_ncols=True) as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                if result:
                    results.extend(result)
                pbar.update(1)  # Update tqdm after each task completion
        x = pd.DataFrame(results)
        return x
    
    # Remove the <|startoftext|> and <|endoftext|> tokens from the 
    # dataframe since these are added by GPTs and can cause issues
    # when fine-tuning models
    def clean_df(self, df):
        pattern = re.compile(r'<\|[^|]+\|>')
        df = df.map(lambda x: pattern.sub('', x) if isinstance(x, str) else x)
        return df
    
    def run(self):
        if os.path.exists(self.data_prep_csv_path) and self.configs.ignore_if_exists:
            logger.info(f"Questions and answers already generated. Skipping, {self.data_prep_csv_path}")
            return self.data_prep_csv_path
        logger.info(f'Generating questions and answers using {self.llm.__class__.__name__}')
        df = asyncio.run(self.main_process_documents())
        logger.info(f'Cleaning CSV...')
        df = self.clean_df(df)
        logger.info(f'Saving to {self.data_prep_csv_path}')
        if os.path.exists(self.data_prep_csv_path):
            os.rename(
                self.data_prep_csv_path,
                f"{self.data_prep_csv_path.split('.')[0]}_old.csv"
            )
        os.makedirs(os.path.dirname(self.data_prep_csv_path), exist_ok=True)
        df.to_csv(self.data_prep_csv_path, index=False)
        logger.info(f"Questions and answers saved to {self.data_prep_csv_path}")
        return self.data_prep_csv_path
