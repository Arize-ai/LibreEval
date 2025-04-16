import logging.config
import time
import sys

from configs import Configs
from download_docs import DocumentationDownloader
from evaluate_model import HallucinationEvaluator
from generate_questions_on_docs import GenerateQuestionAnswers
from judge_llm import GenerateLabels
from log_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger('arize.main')

def main(args=None):
    # If no arguments are provided, use the default command line arguments
    if args is None:
        args = sys.argv[1:]
    
    # Create a Configs instance with the provided arguments
    configs = Configs(args)
    
    start_time = time.time()
    logger.info("Loading Configs...")
    logger.info(configs.evaluation_model_names)

    configs.initialize_phoenix_instrumentor()

    if configs.website and configs.data_prep_llm and configs.judge_llm1:
        logger.info("Downloading Documents....")
        DocumentationDownloader(configs).download_documentation()
        logger.info(f"Documents Downloaded in {time.time() - start_time} seconds")
        GenerateQuestionAnswers(configs).run()
        logger.info(f"Questions and answers generated in {time.time() - start_time} seconds")
        GenerateLabels(configs).run()
        logger.info(f"Labels generation completed in {time.time() - start_time} seconds")
    if configs.evaluation_models:
        logger.info("Evaluating Models....")
        HallucinationEvaluator(configs).evaluate()
        logger.info(f"Evaluation completed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
