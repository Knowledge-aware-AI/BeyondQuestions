import fire
from .request import Request
from .process_request import ProcessRequest
from .wikidata_utils import *
from .wikipedia_triple_extractor import WikipediaTripleExtractor
import os
import json
from loguru import logger
from datetime import datetime

def setup_logging(results_dir_path: str):
    """
    Configure loguru to log to both console and files.
    Creates a logs directory and sets up rotating file handlers.
    
    Args:
        results_dir_path (str): Base directory for results; logs will be stored in a subdirectory.
    """
    # Create logs directory
    logs_dir = os.path.join(results_dir_path, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (INFO level and above)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler for all logs (DEBUG and above)
    log_file_all = os.path.join(logs_dir, f"eval_all_{timestamp}.log")
    logger.add(
        sink=log_file_all,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="7 days"
    )
    
    # Add file handler for errors only
    log_file_errors = os.path.join(logs_dir, f"eval_errors_{timestamp}.log")
    logger.add(
        sink=log_file_errors,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="500 MB",
        retention="7 days"
    )
    
    logger.info(f"Logging initialized. Logs directory: {logs_dir}")

def main_evaluation(
        entities_file_path:str,
        elicited_triples_dir:str,
        ground_truth_dir_path: str,
        results_dir_path:str = None,
        llm_judge:str = None,
        seed: str = 42,
        sample_size: int = 100,
        evaluate_by_category: bool = False,
        triples_per_category: int = 100,
        evaluate_by_popularity: bool = False,
        triples_per_popularity_bucket: int = 100,
        web_results_count: int = 20,
        web_docs_for_eval: int = 20,
        build_ground_truth_only: bool = False,
        max_workers: int = 5,
        top_k: int = 10,
):  
    """
    Main function to run the evaluation pipeline for the elicited triples.
    
    Evaluation approach:
    - Precision: Uses RAG-based evaluation. For each elicited triple, retrieves top-k passages (configurable via top_k parameter)
      from full ground truth (Wikipedia article + all web documents)
      using dense (embedding-based) retrieval. Single LLM call determines entailment/contradiction/neutral.
    - Recall: Compares ground truth triples against elicited triples to check coverage.
      Uses all ground truth sources (Wikipedia + Wikidata + Web).
    
    Arguments:
    entities_file_path (str): File path storing the wikidata entities
    elicited_triples_dir (str): Dir path where the elicited triples are stored
    ground_truth_dir_path (str): Dir path where the ground truth triples are stored
    results_dir_path (str): Dir path for storing the evaluation results
    llm_judge (str): Name of the model to use as judge in the evaluation
    seed (str): Random seed for sampling in evaluation (default: 42)
    sample_size (int): Number of triples to sample for evaluation (default: 100)
    evaluate_by_category (bool): Whether to evaluate by category (default: False)
    triples_per_category (int): Number of triples to sample per category if evaluate_by_category is True
    evaluate_by_popularity (bool): Whether to evaluate by entity popularity buckets (default: False)
    triples_per_popularity_bucket (int): Number of triples to sample per popularity bucket if evaluate_by_popularity is True
     web_results_count (int): Number of Brave Search results to fetch during ground truth construction (default: 20).
     web_docs_for_eval (int): Number of web documents/results to use during evaluation (default: 20).
     build_ground_truth_only (bool): When True, only builds ground truth and skips evaluation (default: False)
     max_workers (int): Maximum number of parallel workers for ground truth building (default: 10).
     top_k (int): Number of top passages to retrieve in RAG-based precision evaluation (default: 10).
    """
    # Create results directory and setup logging only if results_dir_path is provided
    if results_dir_path is not None:
        if not os.path.exists(results_dir_path):
            os.makedirs(results_dir_path)
        # Setup logging before any other operations
        setup_logging(results_dir_path)
        
        logger.info("="*80)
        logger.info("Evaluation started.")
        logger.info("="*80)
        logger.info(f"Model: {llm_judge}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Sample size: {sample_size}")
        logger.info(f"Results directory: {results_dir_path}")
    else:
        # Remove default handler and add console-only logging for ground truth only mode
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
    
    # Validate parameters (only when not building ground truth only, as we have console-only logging in that case)
    if results_dir_path is not None:
        # Skip evaluation-specific logging when building ground truth only
        if not build_ground_truth_only:
            logger.info(f"Precision evaluation: RAG-based (top-{top_k} passages, single LLM call)")
            logger.info(f"Recall evaluation: LLM verification against all ground truth sources")

        valid_methods = ["web", "wikidata", "wikipedia"]

    with open(entities_file_path, 'r') as file:
        json_content = json.load(file)
    
    if evaluate_by_category == True:
        all_entities = json_content
        all_entities = [item["title"] for category in json_content.values() for item in category]
    else:
        all_entities = json_content
        if isinstance(json_content, dict):
            all_entities = [item["title"] for bucket in json_content.values() for item in bucket]
        else:
            all_entities = [i['title'] for i in all_entities]
    
    """
    Wikipedia-based verification workflow:
    1. Extract Wikipedia articles for each entity
    2. Shorten to max 1000 words
    3. Extract triples from shortened text
    4. Compute precision and recall against elicited triples
    All data is stored in a unified cache file.
    """
    
    # Step 1: Initialize Wikipedia triple extractor
    extractor = WikipediaTripleExtractor(llm_judge=llm_judge, ground_truth_dir_path=ground_truth_dir_path, web_results_count=web_results_count, max_workers=max_workers)
    
    # Step 2: Process entities and extract triples from Wikipedia
    logger.info(f"Processing {len(all_entities)} entities...")
    wikipedia_results = extractor.process_entities_batch(all_entities, max_words=1000)
    #logger.info("Unified Wikipedia cache (articles, shortened versions, triples) has been saved and cached")
    
    # If build_ground_truth_only is True, skip evaluation and return after building ground truth
    if build_ground_truth_only:
        logger.info("Skipping evaluation step (build_ground_truth_only_only=True).")
        return
    
    logger.info("Starting evaluation...")

    # Step 3: Create subdirectory for this model/environment and initialize ProcessRequest
    # Generate environment details for subdirectory naming
    env_details = f"{llm_judge.replace('/', '_')}_rag_eval"
    model_results_dir = os.path.join(results_dir_path, env_details)
    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    logger.info(f"Model results directory: {model_results_dir}")
    
    # Determine RAG cache directory - use ground_truth_dir_path/rag_cache
    rag_cache_dir = os.path.join(ground_truth_dir_path, "rag_cache") if ground_truth_dir_path else None
    if rag_cache_dir and not os.path.exists(rag_cache_dir):
        os.makedirs(rag_cache_dir)
    logger.info(f"RAG cache directory: {rag_cache_dir}")
    
    # Initialize ProcessRequest with the model-specific results directory
    logger.info(f"Processing elicited triples from: {elicited_triples_dir}")
    process_request = ProcessRequest(
        llm_judge, 
        elicited_triples_dir,
        entities_file_path, 
        seed, 
        sample_size,
        model_results_dir,
        ground_truth_dir_path=ground_truth_dir_path,
        web_docs_for_eval=web_docs_for_eval,
        rag_cache_dir=rag_cache_dir,
        top_k=top_k,
    )

    ret_triples = process_request.read_triples_dir()

    # Step 4: Flatten Wikipedia results into dict mapping entity -> result dict
    # (keep full result so callers can choose to compare against the extracted
    # triples OR directly against the shortened_content snippet)
    wikipedia_triples_dict = {}
    for entity_name, result in wikipedia_results.items():
        if result:
            wikipedia_triples_dict[entity_name] = result
    
    logger.info(f"Extracted triples from Wikipedia for {len(wikipedia_triples_dict)} entities")

    # Initialize detailed results file for incremental safe writing
    detailed_output_filename = os.path.join(model_results_dir, f"results_detailed.csv")
    process_request.init_detailed_results_file(detailed_output_filename)

    try:
        # Step 5: Check if category-stratified evaluation is enabled
        if evaluate_by_category:
            logger.info("Category-stratified evaluation enabled (10 triples per category, 110 total)")
            
            # Build category index
            subject_to_category = process_request.build_category_index(entities_file_path)
            
            if not subject_to_category:
                logger.error("Failed to build category index. Falling back to standard evaluation.")
                evaluate_by_category = False
            else:
                logger.info(f"Built category index with {len(subject_to_category)} entities")
                
                # Compute Precision by category
                logger.info("Computing Precision (by category)...")
                process_request.compute_precision_wiki_by_category(ret_triples, wikipedia_triples_dict, subject_to_category, triples_per_category=triples_per_category)
                logger.info("Computing Recall (by category)...")
                process_request.compute_recall_wiki_by_category(ret_triples, wikipedia_triples_dict, subject_to_category, triples_per_category=triples_per_category)
                
                # Save categorical results
                output_filename = os.path.join(model_results_dir, f"results_by_category.csv")
                process_request.write_categorical_results_to_csv(output_filename, process_request.aggregated_data)
                logger.info(f"Category-stratified results saved to {output_filename}")
        
        # Step 6: Check if popularity-stratified evaluation is enabled
        if evaluate_by_popularity:
            logger.info("Popularity-stratified evaluation enabled")
            
            # Extract target popularity bucket from results_dir_path if available
            target_popularity_bucket = None
            if results_dir_path:
                for bucket in ["66-100%", "33-66%", "0-33%"]:
                    if bucket in results_dir_path:
                        target_popularity_bucket = bucket
                        break
            logger.info(f"Target popularity bucket: {target_popularity_bucket}")
            
            # Build popularity index
            subject_to_popularity = process_request.build_popularity_index(entities_file_path)
            
            if not subject_to_popularity:
                logger.error("Failed to build popularity index. Falling back to standard evaluation.")
                evaluate_by_popularity = False
            else:
                logger.info(f"Built popularity index with {len(subject_to_popularity)} entities across {len(set(subject_to_popularity.values()))} buckets")
                
                # Compute Precision by popularity
                logger.info("Computing Precision (by popularity)...")
                process_request.compute_precision_wiki_by_popularity(ret_triples, wikipedia_triples_dict, subject_to_popularity, triples_per_popularity_bucket=triples_per_popularity_bucket, target_bucket=target_popularity_bucket)
                logger.info("Computing Recall (by popularity)...")
                process_request.compute_recall_wiki_by_popularity(ret_triples, wikipedia_triples_dict, subject_to_popularity, triples_per_popularity_bucket=triples_per_popularity_bucket, target_bucket=target_popularity_bucket)
                
                # Save popularity results
                output_filename = os.path.join(model_results_dir, f"results_by_popularity.csv")
                process_request.write_categorical_results_to_csv(output_filename, process_request.aggregated_data)
                logger.info(f"Popularity-stratified results saved to {output_filename}")
    
        # Step 7: If neither category nor popularity stratified is enabled, use standard evaluation
        if not evaluate_by_category and not evaluate_by_popularity:
            # Compute BOTH precision and recall (and keep results together)
            logger.info("Computing Precision...")
            process_request.compute_precision_wiki_dir(ret_triples, wikipedia_triples_dict)
            logger.info("Computing Recall...")
            process_request.compute_recall_wiki_dir(ret_triples, wikipedia_triples_dict)

            # Step 6: Save combined aggregated results
            output_filename = os.path.join(model_results_dir, f"results.csv")
            process_request.write_to_csv(output_filename, process_request.aggregated_data)
            logger.info(f"Aggregated results saved to {output_filename}")
    
    finally:
        # Ensure detailed results file is closed even if an error occurs
        process_request.close_detailed_results_file()
        logger.info(f"Detailed per-triple results saved to {detailed_output_filename}")

    logger.info("="*80)
    logger.info("Triple Verification Completed Successfully")
    logger.info("="*80)


if __name__ == "__main__":
    fire.Fire(main_evaluation)