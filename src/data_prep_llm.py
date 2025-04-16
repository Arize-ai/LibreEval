import logging
import os
import time

import tiktoken
from anthropic import AsyncAnthropic
from litellm import acompletion
from openai import AsyncOpenAI
import prompts

logger = logging.getLogger('arize.data_prep_llm')


class DataPrepLLM:
    # def __init__(self, model_name, configs=None):
    def __init__(self, configs=None):
        self.language = configs.language if configs and configs.language else 'English'

    # async def create(self, content, prompt, strip=False):
    #     raise NotImplemented

    # async def extract_answer(self, response):
    #     raise NotImplemented

    async def create_informative_paragraph(self, paragraphs):
        prompt = prompts.SELECT_INFORMATIVE_PARAGRAPH_USER_PROMPT.format(paragraphs=(paragraphs[:15]))
        return await self.create(prompts.SELECT_INFORMATIVE_PARAGRAPH_SYSTEM_PROMPT.format(language=self.language), prompt)

    async def create_question(self, paragraph, question_type):
        if question_type == 1:
            prompt = prompts.GENERATE_QUESTION_TYPE_1_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        elif question_type == 2:
            prompt = prompts.GENERATE_QUESTION_TYPE_2_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        elif question_type == 3:
            prompt = prompts.GENERATE_QUESTION_TYPE_3_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        elif question_type == 4:
            prompt = prompts.GENERATE_QUESTION_TYPE_4_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        elif question_type == 5:
            prompt = prompts.GENERATE_QUESTION_TYPE_5_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        # commented out because models refuse to generate type 6 questions
        # elif question_type == 6:
        #     prompt = prompts.GENERATE_QUESTION_TYPE_6_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        else:
            prompt = prompts.GENERATE_QUESTION_USER_PROMPT.format(paragraph=paragraph, language=self.language)
        system_prompt = prompts.GENERATE_QUESTION_SYSTEM_PROMPT.format(language=self.language)
        return await self.create(system_prompt, prompt)

    async def generate_synthetic_answer(self, question, paragraph, hallucination_type):
        if hallucination_type == 1:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_1_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 2:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_2_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 3:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_3_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 4:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_4_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 5:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_5_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 6:
            prompt = prompts.HALLUCINATION_ANSWER_TYPE_6_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        elif hallucination_type == 7:
            prompt = prompts.HALLUCINATION_ANSWER_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        else:
            prompt = prompts.HALLUCINATION_ANSWER_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        system_prompt = prompts.HALLUCINATION_ANSWER_SYSTEM_PROMPT.format(language=self.language)
        return await self.create(system_prompt, prompt)

    async def generate_non_synthetic_answer(self, question, paragraph):
        prompt = prompts.NON_HALLUCINATION_ANSWER_USER_PROMPT.format(paragraph=paragraph, question=question, language=self.language)
        system_prompt = prompts.NON_HALLUCINATION_ANSWER_SYSTEM_PROMPT.format(language=self.language)
        return await self.create(system_prompt, prompt)

    def count_tokens_and_limit(self, content, max_tokens):
        """
        Count the tokens in the content and truncate it if it exceeds max_tokens.

        Args:
            content (str): The input content to evaluate.
            max_tokens (int): The maximum allowed token count.

        Returns:
            str: The truncated content within the max token limit.
            int: The token count of the truncated content.
        """
        # Initialize tokenizer
        if 'llama' not in self.model:
            return content, 0
        tokenizer = tiktoken.get_encoding("cl100k_base")  # Update encoding for Llama3, if required.
        # Tokenize the content
        tokens = tokenizer.encode(content)
        # Truncate if it exceeds max_tokens
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        # Decode back to text
        truncated_content = tokenizer.decode(tokens)
        return truncated_content, len(tokens)


class OpenAIDataPrepLLM(DataPrepLLM):

    def __init__(self, model_name, configs=None):
        super().__init__(configs)
        self.model = model_name
        self.client = AsyncOpenAI()
        self.model_user_name = "system"
        self.messages_client = self.client.chat.completions

    async def create(self, content, prompt, strip=True):
        response = await self.messages_client.create(
            model=self.model,
            messages=[
                {"role": self.model_user_name, "content": content},
                {"role": "user", "content": prompt}
            ]
        )
        return await self.extract_answer(response, strip)

    async def extract_answer(self, response, strip=True):
        return response.choices[0].message.content.strip()


class AnthropicDataPrepLLM(DataPrepLLM):

    def __init__(self,model_name, configs=None):
        super().__init__(configs)
        self.client = AsyncAnthropic()
        self.model = model_name
        self.model_user_name = "assistant"
        self.max_tokens = 1000
        self.messages_client = self.client.messages

    async def create(self, content, prompt, strip=True):
        response = await self.messages_client.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": self.model_user_name, "content": content},
                {"role": "user", "content": prompt}
            ]
        )
        return await self.extract_answer(response, strip)

    async def extract_answer(self, response, strip=True):
        if not strip:
            return response.content[0].text
        return response.content[0].text.strip('The most informative paragraph is: ').strip()


class LiteLLMDataPrepLLM(DataPrepLLM):

    def __init__(self, model_name, configs=None):
        super().__init__(configs)
        if 'litellm/' in model_name:
            model_name = model_name.split('litellm/')[1]
        self.model = model_name
        self.model_user_name = "system"

    async def create(self, content, prompt, strip=True):
        prompt, _ = self.count_tokens_and_limit(prompt, 1000)
        time.sleep(3) # prevents rate limiting
        params = {}
        if os.getenv("API_BASE_URL"):
            params['api_base'] = os.getenv("API_BASE_URL")
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": self.model_user_name, "content": content},
                {"role": "user", "content": prompt}
            ],
            **params
        )
        return await self.extract_answer(response, strip)

    async def extract_answer(self, response, strip=True):
        return response['choices'][0].message.content
