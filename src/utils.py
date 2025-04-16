from enum import Enum
from openai import OpenAI
from pydantic import BaseModel
import prompts
import logging
import json
from json.decoder import JSONDecodeError

logger = logging.getLogger('arize.utils')

class HallucinationType(Enum):
    RelationError = "Relation-error hallucination"
    Incompleteness = "Incompleteness hallucination"
    OutdatedInformation = "Outdated information hallucination" 
    Overclaim = "Overclaim hallucination"
    UnverifiableInformation = "Unverifiable information hallucination"
    EntityError = "Entity-error hallucination"
    # Other = "Other hallucination"

def lookup_hallucination_type(hallucination_type):
    hallucination_type = str(hallucination_type)
    return {
        '1': "Relation-error hallucination",
        '2': "Incompleteness hallucination",
        '3': "Outdated information hallucination",
        '4': "Overclaim hallucination",
        '5': "Unverifiable information hallucination",
        '6': "Entity-error hallucination",
        '7': "Other hallucination",
        '0': "Non-synthetic. No hallucination encouragement",
    }[hallucination_type]

def lookup_question_type(question_type):
    question_type = str(question_type)
    return {
        '0': "Default question type",
        '1': "Out-of-scope information",
        '2': "Advanced logical reasoning", 
        '3': "Multimodal content",
        '4': "Errors, contradictions, or unsolvable questions",
        '5': "Other common hallucinated questions",
        '6': "Offensive, illegal, or biased responses",
    }[question_type]

class HallucinationTypeResponse(BaseModel):
    hallucination_type: HallucinationType

def classify_hallucination(paragraph, question, answer):
    client = OpenAI()
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            response_format=HallucinationTypeResponse,
        messages=[
            {
                "role": "system",
                "content": prompts.CLASSIFY_HALLUCINATION_SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": prompts.CLASSIFY_HALLUCINATION_USER_PROMPT.format(
                    paragraph=paragraph,
                    question=question,
                    answer=answer
                )
            }
            ],
        )
    except Exception as e:
        logger.error(f"Error classifying hallucination: {e}")
        return "Could not classify hallucination"
    
    ret = response.choices[0].message.content.strip()
    try:
        ret = json.loads(ret)['hallucination_type']
    except JSONDecodeError as e:
        logger.error(f"Error decoding hallucination type json: {e}")
        return ret
    return ret
