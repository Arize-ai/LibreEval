import os
from openai import OpenAI
from typing import Optional

def start_openai_tuning_job(
    training_file: str,
    validation_file: Optional[str] = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: str = "",
) -> str:
    """
    Start a fine-tuning job using the OpenAI API.
    
    Args:
        training_file: Path to training data file
        validation_file: Optional path to validation data file
        model: Base model to fine-tune
        
    Returns:
        job_id: ID of the created fine-tuning job
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Upload training file
    with open(training_file, 'rb') as f:
        training_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    # Upload validation file if provided
    validation_file_id = None
    if validation_file:
        with open(validation_file, 'rb') as f:
            validation_response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
            validation_file_id = validation_response.id
    
    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=training_response.id,
        validation_file=validation_file_id,
        model=model,
        suffix=suffix,
    )
    
    return job.id

if __name__ == "__main__":
    job_id = start_openai_tuning_job(
        training_file="combined_datasets_for_tuning/message_format/train.jsonl",
        validation_file="combined_datasets_for_tuning/message_format/validation.jsonl",
        model="gpt-4o-mini-2024-07-18",
        suffix="jg",
    )
    print(f"Fine-tuning job started with ID: {job_id}")