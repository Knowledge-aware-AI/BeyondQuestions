import os
import time
import sys
import pickle
import random
import fcntl
from collections import defaultdict
from openai import OpenAI
import json
import pandas as pd
import requests
from loguru import logger
from eval.rag_retriever import GroundTruthRAG
import csv
from tqdm import tqdm

from .request import Request
from .wikidata_utils import *
from .wikipedia_utils import search_web
from .rag_retriever import GroundTruthRAG

class ProcessRequest:
    """
    Handles evaluation of elicited RDF triples against ground truth.
    
    Supports two evaluation metrics:
    - Precision: Uses RAG-based evaluation (dense retrieval + single LLM call)
    - Recall: Uses LLM to check if ground truth triples are covered by elicited triples
    
    The RAG-based precision evaluation retrieves top-k passages from combined ground truth
    (Wikipedia + Web) using embedding-based similarity, then makes a single
    LLM call to determine entailment/contradiction/neutral.
    """
    
    def __init__(self, 
                llm_judge, 
                elicited_triples_dir,
                entities_file_path, 
                seed, 
                sample_size,
                results_dir_path,
                ground_truth_dir_path=None,
                web_docs_for_eval: int = 30,
                rag_cache_dir: str = None,
                top_k: int = 10,
        ):
        """
        Initialize the ProcessRequest handler.
        
        Args:
            llm_judge: Name of the LLM model to use as judge
            elicited_triples_dir: Directory path where elicited triples are stored
            entities_file_path: File path storing the entity information
            seed: Random seed for sampling
            sample_size: Number of triples to sample for evaluation (-1 for all)
            results_dir_path: Directory path for storing evaluation results
            ground_truth_dir_path: Directory path for ground truth data (default: None)
            web_docs_for_eval: Number of web documents to use in evaluation (default: 30)
            rag_cache_dir: Directory path for caching RAG indices (default: None)
            top_k: Number of top passages to retrieve in RAG-based precision evaluation (default: 10)
        """


        self.client = OpenAI()

        # directory to store the snippets downloaded from the search query
        self.snippet_dir = os.path.join(os.getcwd(), "snippets")
        self.elicited_triples_dir = elicited_triples_dir
        self.entities_file_path = entities_file_path

        # file location that stores the gold triples from wikidata
        self.gold_triples_file_path = os.path.join(os.getcwd(), "gold.json")
        self.seed = seed
        self.llm_judge = llm_judge
        # sampling: enable only when sample_size is a positive integer
        try:
            sample_size_int = int(sample_size)
        except Exception:
            sample_size_int = -1
        self.sampling = True if sample_size_int > 0 else False
        self.sample_size = sample_size_int
        # deterministic RNG for reproducible sampling using provided seed
        try:
            seed_int = int(seed)
        except Exception:
            # fall back to hashed seed for non-integer seeds
            seed_int = abs(hash(seed)) % (2**32)
        self._rng = random.Random(seed_int)
        self.results_dir_path = results_dir_path
        self.ground_truth_dir_path = ground_truth_dir_path
        self.aggregated_data = []
        # self.detailed_results = []  # No longer used – incremental writes via CSV
        self.detailed_csv_file = None
        self.detailed_csv_writer = None

        self.metric_name = "Precision (RAG)"
        
        # Number of web documents/results to use for evaluation
        self.web_docs_for_eval = web_docs_for_eval
        
        # RAG cache directory for storing pre-built RAG indices
        self.rag_cache_dir = rag_cache_dir
        
        # Number of top passages to retrieve for RAG-based precision evaluation
        self.top_k = top_k
        
    def verify_triples(self, raw_triples):

        """
        Process raw triples, verify from language model, and categorize each response.
        Returns a dict with keys: 'a' (entailment), 'b' (contradiction), 'c' (neutral), and 'noSnippet' (triples without search results).
        """
        request = Request(self.llm_judge)

        # this dict stores results which fall into one of the categories (a, b, c) -- a=entailment, b=contradiction, c=neutral
        results = {"a":[], "b":[], "c":[], "noSnippet": []}
        for each_triple in raw_triples:
            if len(each_triple.get('snippet', [])) > 0:
                each_triple_str = f"({each_triple['subject'].replace('_', ' ')}, {each_triple['predicate'].replace('_', ' ')}, {each_triple['object'].replace('_', ' ')})"

                snippet_str = ""
                for each_snippet in each_triple['snippet']:
                    snippet_str += each_snippet
                    snippet_str += " | "
                
                output = request.verify_triple_language_model(each_triple_str, snippet_str)
                results = self.parse_lm_output(each_triple, results, output)
            else:
                results['noSnippet'].append(each_triple)

        return results
            

    def parse_lm_output(self, current_triple, results, output):
        """
        Parse the LLM output (structured JSON) to extract the answer.
        Output should be a dict with 'answer' field containing a/b/c.
        
        Args:
            current_triple: The triple being verified.
            results: Dictionary accumulating results by category.
            output: Either a dict (from JSON mode) or string (for backward compatibility).
        
        Returns:
            Updated results dictionary.
        """
        try:
            # If output is already a dict (from structured JSON output)
            if isinstance(output, dict):
                answer = output.get("answer", "").lower().strip()
                if answer in ['a', 'b', 'c']:
                    results[answer].append(current_triple)
                    logger.debug(f"Parsed structured JSON answer: {answer}")
                    return results
                else:
                    logger.warning(f"Invalid answer in structured output: {output}")
                    # Treat invalid / unknown answers as neutral
                    results['c'].append(current_triple)
                    return results
            
            # Backward compatibility: if output is a string, try to parse it
            if isinstance(output, str):
                clean_output = output.strip().lower()
                
                # Try parsing as JSON first (in case it's a JSON string)
                try:
                    parsed = json.loads(clean_output)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        answer = parsed.get("answer", "").lower().strip()
                        if answer in ['a', 'b', 'c']:
                            results[answer].append(current_triple)
                            logger.debug(f"Parsed JSON string answer: {answer}")
                            return results
                except json.JSONDecodeError:
                    pass
                
                # Fallback: use regex patterns
                import re
                patterns = [
                    r'(?:the\s+)?best\s+answer\s+is\s+([a-c])',
                    r'(?:the\s+)?answer\s+is\s+([a-c])',
                    r'answer:\s*([a-c])',
                    r'response:\s*([a-c])',
                    r'^([a-c])[.\s)]',
                    r'\n([a-c])[.\s)]',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, clean_output)
                    if match:
                        answer_letter = match.group(1)
                        results[answer_letter].append(current_triple)
                        logger.debug(f"Matched pattern '{pattern}' with answer: {answer_letter}")
                        return results
                
                # Last resort: search for standalone letters from end
                lines = clean_output.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    match = re.search(r'^\s*([a-c])\s*[.\)]*\s*$', line)
                    if match:
                        answer_letter = match.group(1)
                        results[answer_letter].append(current_triple)
                        logger.debug(f"Found standalone letter at end: {answer_letter}")
                        return results
                
                logger.warning(f'LLM response did not contain a clear answer (a/b/c). Response:\n{output}')
                # Treat ambiguous / unparsable responses as neutral
                results['c'].append(current_triple)
                return results
        
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            # On parse errors, classify as neutral
            results['c'].append(current_triple)
            return results
    
    def read_triples_dir(self):
        """
        Read all CSV files from the elicited triples directory.
        Handles the case where CSV files are nested inside a single subfolder.
        
        Returns:
            dict: Dictionary with filename (without extension) as key and list of triples as value.
        """
        raw_triples = {}

        # Ensure the directory exists
        if not os.path.isdir(self.elicited_triples_dir):
            raise ValueError(f"{self.elicited_triples_dir} is not a valid directory.")

        # First, check if there are any CSV files directly in the directory
        csv_files_direct = [f for f in os.listdir(self.elicited_triples_dir) if f.endswith(".csv")]
        
        # If no CSV files found directly, check if there's exactly one subdirectory
        # (common pattern for some models where CSV is inside a nested folder)
        search_dir = self.elicited_triples_dir
        if not csv_files_direct:
            subdirs = [d for d in os.listdir(self.elicited_triples_dir) 
                      if os.path.isdir(os.path.join(self.elicited_triples_dir, d))]
            
            if len(subdirs) == 1:
                # Exactly one subfolder - look for CSV inside it
                nested_dir = os.path.join(self.elicited_triples_dir, subdirs[0])
                csv_files_nested = [f for f in os.listdir(nested_dir) if f.endswith(".csv")]
                
                if csv_files_nested:
                    logger.info(f"No CSV found directly in {self.elicited_triples_dir}, found CSV in nested folder: {subdirs[0]}")
                    search_dir = nested_dir
                else:
                    logger.warning(f"No CSV found in {self.elicited_triples_dir} or its subfolder {subdirs[0]}")
            elif len(subdirs) > 1:
                logger.warning(f"Multiple subdirectories found in {self.elicited_triples_dir}, no CSV files directly available")

        # Iterate over all files in the (potentially nested) search directory
        for file in os.listdir(search_dir):
            if file.endswith(".csv"):  
                file_path = os.path.join(search_dir, file)
                # get the filename without the extension
                file_key = os.path.splitext(file)[0]  

                with open(file_path, mode='r', newline='', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    raw_triples[file_key] = [row for row in csv_reader]  
        
        # return type is a dict with key as filename and value as rows read from that file
        return raw_triples



    def get_triples_statistics(self, data_triples):
        """
        Log basic statistics from the wikidata triples file.
        Prints the unique subjects found in the triples.
        """
        unique_subjects = set(entry["subject"] for entry in data_triples)
        logger.debug(f"Unique Subjects: {unique_subjects}")

    def get_brave_results(self, query: str) -> str:
        """
        Search the web using Brave Search API and return combined snippets.
        
        Args:
            query: The search query string.
        
        Returns:
            A string containing combined snippets from search results, or empty string if no results.
        """
        results = search_web(query, num_results=10)
        if not results:
            logger.warning(f"No web search results found for query: {query}")
            return ""
        
        # Combine all snippets into one string
        combined_snippets = " | ".join([r.get("snippet", "") for r in results if r.get("snippet")])
        return combined_snippets

    def query_snippets(self, data_triples):
        """
        Process raw triples, make the query to the search engine and get the snippets
        Wait for 2 seconds before making a new query (using free subscription for now that it requires min 1 sec wait time)
        """
        for each_triple in data_triples:
            returned_snippet = self.get_brave_results(each_triple['subject'].replace("_", " ") + " " + each_triple['object'].replace("_", " "))
            # get the snippet returned by api search call and add a new key to the triple dict
            each_triple['snippet'] = returned_snippet
            time.sleep(2)

        if not os.path.exists(self.snippet_dir):
            os.makedirs(self.snippet_dir)
            logger.debug(f"Directory created: {self.snippet_dir}")
        else:
            logger.debug(f"Directory already exists: {self.snippet_dir}")

        file_path = os.path.join(self.snippet_dir, f"{self.seed}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(data_triples, f)
        
        return data_triples


    def subject_based_lookup(self, current_subject, data_triples):
        """
        Select all triples for a given subject from the list of raw triples.
        
        Args:
            current_subject: The subject to filter triples by.
            data_triples: List of triple dictionaries.
        
        Returns:
            list: List of triple strings for the given subject.
        """
        subject_list_triple = []
        for each_triple in data_triples:
            if each_triple['subject'] == current_subject:
                each_triple_str = f"({current_subject}, {each_triple['predicate']}, {each_triple['object']})"
                subject_list_triple.append(each_triple_str)

        
        return subject_list_triple

    def stratified_sample_from_grouped(self, subj_to_triples: dict, m: int):
        """
        Stratified sampling from a dict mapping subject -> list of triples.
        Sampling procedure (your specification): repeatedly pick a subject uniformly
        at random, then pick a random triple from that subject. Continue until
        we have m unique triples or run out of available triples.

        Returns a list of sampled triples (<= m).
        """
        # make a shallow copy of available triples per subject
        available = {s: list(v) for s, v in subj_to_triples.items() if v}
        subjects = list(available.keys())
        sampled = []

        # Safety: if no subjects, return empty
        if not subjects:
            return sampled

        attempts = 0
        # Continue until we have m samples or no subjects with available triples
        while len(sampled) < m and subjects:
            subj = self._rng.choice(subjects)
            trio_list = available.get(subj)
            if not trio_list:
                # remove exhausted subject
                subjects.remove(subj)
                continue
            triple = self._rng.choice(trio_list)

            # Prefer unique triples in the sampled list
            if triple not in sampled:
                sampled.append(triple)
                # remove chosen triple from availability
                trio_list.remove(triple)
                if not trio_list:
                    subjects.remove(subj)
            else:
                # duplicate: remove from availability to avoid repeated collisions
                try:
                    trio_list.remove(triple)
                except ValueError:
                    pass
                if not trio_list:
                    subjects.remove(subj)

            attempts += 1
            # safety break to avoid pathological infinite loops
            if attempts > m * 50:
                break

        return sampled

    def stratified_sample_total(self, triples_list: list, m: int):
        """
        Convenience wrapper: group triples by subject and call
        stratified_sample_from_grouped.
        """
        subj_to_triples = {}
        for t in triples_list:
            subj = t.get('subject')
            subj_to_triples.setdefault(subj, []).append(t)

        return self.stratified_sample_from_grouped(subj_to_triples, m)

    def init_detailed_results_file(self, filename):
        """Initialize the detailed results CSV file for incremental writing."""
        headers = [
            'subject', 'predicate', 'object', 'result', 'result_category',
            'reasoning', 'metric', 'source_file', 'retrieved_passages'
        ]
        self.detailed_csv_file = open(filename, mode='w', newline='', encoding='utf-8')
        self.detailed_csv_writer = csv.DictWriter(
            self.detailed_csv_file, fieldnames=headers, quoting=csv.QUOTE_MINIMAL, escapechar='\\'
        )
        self.detailed_csv_writer.writeheader()
        self.detailed_csv_file.flush()
        logger.info(f"Initialized detailed results file for incremental writing: {filename}")

    def append_detailed_result(self, row):
        """Append a single result row to the detailed CSV and flush to disk."""
        if self.detailed_csv_writer is None:
            logger.error("Detailed CSV writer not initialized – cannot append row.")
            return
        result_categories = {'a': 'Entailment', 'b': 'Contradiction', 'c': 'Neutral', 'error': 'Error'}
        result = row.get('result', 'c')
        result_category = result_categories.get(result, 'Unknown')
        output_row = {
            'subject': row.get('subject', ''),
            'predicate': row.get('predicate', ''),
            'object': row.get('object', ''),
            'result': result,
            'result_category': result_category,
            'reasoning': row.get('reasoning', ''),
            'metric': row.get('metric', ''),
            'source_file': row.get('source_file', ''),
            'retrieved_passages': row.get('retrieved_passages', '')
        }
        try:
            self.detailed_csv_writer.writerow(output_row)
            self.detailed_csv_file.flush()
        except Exception as e:
            logger.error(f"Failed to write detailed result row: {e}", exc_info=True)

    def close_detailed_results_file(self):
        """Close the detailed results CSV file."""
        if self.detailed_csv_file is not None:
            self.detailed_csv_file.close()
            self.detailed_csv_file = None
            self.detailed_csv_writer = None
            logger.info("Detailed results file closed.")

    def entity_based_stats(self, raw_triples, results):

        """
        Log entity-wise statistics based on verification results.
        Counts entailment, contradiction, and neutral results per subject and logs summary information.
        """
        # dict for recording subject to count of a,b,c (entailment, contradiction, neutral)
        subject_to_key_counts = {}

        for key, triples in results.items():
            for triple in triples:
                triple_split = triple.strip("()").split(", ")
                if len(triple_split) != 3:
                    logger.warning(f"Skipping invalid triple: {triple}")
                    continue

                subject, b, c = triple_split

                # Initialize nested dictionary for the subject if it doesn't exist
                if subject not in subject_to_key_counts:
                    subject_to_key_counts[subject] = {}

                # Increment the count for the current key
                if key not in subject_to_key_counts[subject]:
                    subject_to_key_counts[subject][key] = 0
                subject_to_key_counts[subject][key] += 1

        logger.debug(subject_to_key_counts)
        subjects_entailment_or_contradiction = []
        subjects_neutral = []

        for subject, freq_count in subject_to_key_counts.items():
            if "a" not in freq_count and "b" not in freq_count:
                if "c" in freq_count:
                    subjects_neutral.append(subject)

            if "a" in freq_count or "b" in freq_count:
                if "c" not in freq_count:
                    subjects_entailment_or_contradiction.append(subject)

        logger.debug(f"Subjects with entailment or contradiction: {subjects_entailment_or_contradiction}")
        logger.debug(f"Count subjects with entailment or contradiction: {len(subjects_entailment_or_contradiction)}")
        logger.debug(f"Subjects neutral: {subjects_neutral}")
        logger.debug(f"Subjects neutral count: {len(subjects_neutral)}")


    def write_to_csv(self, filename, data):
        """
        Write aggregated evaluation results to a CSV file.
        
        Args:
            filename: Path to the output CSV file.
            data: List of dictionaries containing evaluation metrics.
        """
        headers = ['Entailment', 'Contradiction', 'Neutral', 'Entailment_ratio', 'Contradiction_ratio', 'Neutral_ratio', 'Total #Triples', 'Error_count', 'Error_ratio', "Metric", "Source Elicited File", "Source Prompt File"]

        with open(filename, mode = 'w', newline = "") as file:
            writer = csv.DictWriter(file, fieldnames = headers)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    
    def write_detailed_results_to_csv(self, filename, detailed_results):
        """
        Write detailed results (each triple with its classification and reasoning) to CSV.
        
        Args:
            filename (str): Path to the output CSV file.
            detailed_results (list): List of dicts with keys: subject, predicate, object, result, reasoning, metric, source_file, retrieved_passages.
        """
        headers = ['subject', 'predicate', 'object', 'result', 'result_category', 'reasoning', 'metric', 'source_file', 'retrieved_passages']
        
        # Map result letters to categories (human-readable)
        result_categories = {
            'a': 'Entailment',
            'b': 'Contradiction',
            'c': 'Neutral',
            'error': 'Error'
        }

        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                
                for row in detailed_results:
                    # Add result_category as human-readable name
                    result = row.get('result', 'c')
                    result_category = result_categories.get(result, 'Unknown')
                    output_row = {
                        'subject': row.get('subject', ''),
                        'predicate': row.get('predicate', ''),
                        'object': row.get('object', ''),
                        'result': result,
                        'result_category': result_category,
                        'reasoning': row.get('reasoning', ''),
                        'metric': row.get('metric', ''),
                        'source_file': row.get('source_file', ''),
                        'retrieved_passages': row.get('retrieved_passages', '')
                    }
                    writer.writerow(output_row)
            
            logger.info(f"Detailed results written to {filename} ({len(detailed_results)} triples)")
        except Exception as e:
            logger.error(f"Error writing detailed results to CSV: {e}", exc_info=True)
        
    def read_parse_jinja_file(self, elicited_csv_file):
        """
        Read jinja mapping file and return source prompt file name
        """

        jinja_file_mapping = os.path.abspath(os.path.join(os.getcwd(), "..", "jinja_index_mapping.txt"))
        mapping_dict = dict()
        with open(jinja_file_mapping, "r") as file:
            for line in file:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:  
                    source_file, mapped_value = parts
                    #mapping_dict[source_file] = mapped_value
                    mapping_dict[mapped_value] = source_file
        
        return mapping_dict[elicited_csv_file]


    def compute_precision_wiki_dir(self, elicited_triples, wikipedia_triples_dict):
        """
        Compute precision using RAG-based evaluation.
        For each elicited triple, retrieves top-k passages (configurable via top_k parameter) from full ground truth
        (Wikipedia article + all web documents) using
        dense (embedding-based) retrieval, then makes a single LLM call to determine
        entailment/contradiction/neutral.
        Stores both aggregated metrics and detailed per-triple results.
        
        OPTIMIZATION: Groups triples by entity and builds RAG index once per entity
        instead of once per triple, significantly reducing API calls.
        
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to full ground truth data.
        
        Returns:
            Updated self.aggregated_data with precision metrics.
        """
        request = Request(self.llm_judge)
        
        for filename, triples_list in elicited_triples.items():
            logger.debug(f"Computing Precision ({self.metric_name}) for file: {filename}")
            results = {"a": [], "b": [], "c": []}
            # Detailed results written incrementally via self.append_detailed_result()

            # Apply stratified sampling (entity-then-triple) to obtain a total of m samples
            if self.sampling:
                triples_list = self.stratified_sample_total(triples_list, self.sample_size)
            
            # Group triples by entity (subject_name) to build RAG index once per entity
            triples_by_entity = defaultdict(list)
            for each_triple in triples_list:
                subject_name = each_triple.get('subject_name', '')
                if subject_name:
                    triples_by_entity[subject_name].append(each_triple)
            
            # Process each entity once, building RAG index once
            for subject_name, entity_triples in tqdm(triples_by_entity.items(), desc=f"Computing Precision ({self.metric_name}) for {filename}"):
                # Get Wikipedia data for this subject
                wiki_data = wikipedia_triples_dict.get(subject_name, None)
                
                # If no wiki data at all, mark all triples for this entity as neutral
                if not wiki_data:
                    logger.warning(f"No Wikipedia/Web data found for subject: {subject_name}. Marking as Neutral.")
                    for each_triple in entity_triples:
                        results['c'].append(each_triple)
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "result": "c",
                            "reasoning": "No Wikipedia/Web data found for subject",
                            "metric": f"Precision ({self.metric_name})",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    continue
                
                # Build RAG index ONCE for this entity (with caching and locking)
                try:
                    rag = None
                    cache_path = GroundTruthRAG.get_cache_path_for_entity(self.rag_cache_dir, subject_name) if self.rag_cache_dir else None
                    
                    # Build new index if needed (use lock to prevent race conditions)
                    if cache_path:
                        lock_path = cache_path + ".lock"
                        
                        # Ensure lock directory exists before acquiring lock
                        lock_dir = os.path.dirname(lock_path)
                        if lock_dir and not os.path.exists(lock_dir):
                            os.makedirs(lock_dir, exist_ok=True)
                        
                        # Use 'a' mode to avoid truncating existing lock file, create if needed
                        lock_file_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
                        try:
                            fcntl.flock(lock_file_fd, fcntl.LOCK_EX)
                            try:
                                # Check if cache already exists (another worker may have built it while waiting)
                                rag = GroundTruthRAG.load_from_disk(cache_path)
                                
                                if rag is None:
                                    # Build the RAG index
                                    rag = GroundTruthRAG()
                                    rag.build_index(wiki_data)
                                    rag.cache_embeddings()
                                    try:
                                        rag.save_to_disk(cache_path)
                                        logger.debug(f"RAG index built and saved for {subject_name}")
                                    except Exception as e:
                                        logger.warning(f"Failed to save RAG cache for {subject_name}: {e}")
                            finally:
                                fcntl.flock(lock_file_fd, fcntl.LOCK_UN)
                        finally:
                            os.close(lock_file_fd)
                    else:
                        # No cache path, build without locking
                        rag = GroundTruthRAG()
                        rag.build_index(wiki_data)
                        rag.cache_embeddings()
                except Exception as e:
                    logger.error(f"Error building RAG index for {subject_name}: {e}")
                    # Record errors for all triples of this entity; do not count in a/b/c
                    for each_triple in entity_triples:
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "result": "error",
                            "reasoning": f"RAG index build error: {str(e)}",
                            "metric": f"Precision ({self.metric_name})",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    continue
                
                # Process each triple for this entity (reusing the same RAG index)
                for each_triple in entity_triples:
                    subject = each_triple.get('subject', '')
                    predicate = each_triple.get('predicate', '')
                    obj = each_triple.get('object', '')
                    elicited_triple_str = f"({subject}, {predicate}, {obj})"
                    
                    try:
                        # Retrieve top-k relevant passages using pre-built index
                        retrieved_passages = rag.retrieve_top_k(elicited_triple_str, k=self.top_k)
                        
                        if not retrieved_passages:
                            final_category = "c"
                            final_reasoning = "No relevant passages found in ground truth"
                            passages_str = ""
                        else:
                            # Format passages for LLM (truncated to 500 chars per passage)
                            passages_text = "\n\n".join([
                                f"[{p['source'].upper()}] {p['content'][:500]}" 
                                for p in retrieved_passages
                            ])
                            
                            # Call LLM judge with triple and retrieved passages
                            output = request.verify_triple_with_rag(
                                elicited_triple_str,
                                passages_text,
                                [p['source'] for p in retrieved_passages]
                            )
                            final_category = output.get("answer", "c") if isinstance(output, dict) else "c"
                            final_reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                            final_reasoning = f"RAG retrieved: {', '.join(set([p['source'] for p in retrieved_passages]))} | {final_reasoning}"
                        
                        results[final_category].append(each_triple)
                        
                        passages_str = "\n\n---\n\n".join([
                            f"[{p['source'].upper()}] {p['content'][:2000]}"
                            for p in retrieved_passages
                        ]) if retrieved_passages else ""
                        
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "result": final_category,
                            "reasoning": final_reasoning,
                            "metric": f"Precision ({self.metric_name})",
                            "source_file": filename,
                            "retrieved_passages": passages_str
                        })
                        
                    except Exception as e:
                        logger.error(f"Error verifying triple {elicited_triple_str}: {e}")
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "result": "error",
                            "reasoning": f"Verification error after retries: {str(e)}",
                            "metric": f"Precision ({self.metric_name})",
                            "source_file": filename,
                            "retrieved_passages": passages_str if 'passages_str' in locals() else ""
                        })
            
            # Compute metrics
            total_triples = len(triples_list)
            if total_triples > 0:
                error_count = total_triples - (len(results['a']) + len(results['b']) + len(results['c']))
                results_dict = {
                    "Entailment": len(results['a']),
                    "Contradiction": len(results['b']),
                    "Neutral": len(results['c']),
                    "Entailment_ratio": len(results['a']) / total_triples,
                    "Contradiction_ratio": len(results['b']) / total_triples,
                    "Neutral_ratio": len(results['c']) / total_triples,
                    "Total #Triples": total_triples,
                    "Error_count": error_count,
                    "Error_ratio": error_count / total_triples if total_triples > 0 else 0,
                    "Metric": f"Precision ({self.metric_name})",
                    "Source Elicited File": str(filename),
                }
                self.aggregated_data.append(results_dict)
                logger.info(f"Precision (Wiki+Web) for {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}, Errors={error_count}")

    def compute_recall_wiki_dir(self, elicited_triples, wikipedia_triples_dict):
        """
        Compute recall: check if Wikipedia + Web triples are entailed/plausible by elicited triples.
        Uses OR logic: a GT triple is covered if ANY source (Wikipedia OR Web) supports it.
        Stores both aggregated metrics and detailed per-triple results.
        
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples (now includes Web).
        
        Returns:
            Updated self.aggregated_data with recall metrics.
        """
        request = Request(self.llm_judge)
        
        for filename, elicited_list in elicited_triples.items():
            logger.debug(f"Computing Recall (Wiki + Web) for file: {filename}")
            results = {"a": [], "b": [], "c": []}
            # Detailed results written incrementally via self.append_detailed_result()

            # Collect ALL ground truth triples by subject (Wikipedia + Web)
            # Use all sources for recall computation
            wiki_facts_per_subject = {}
            for subject, wiki_value in wikipedia_triples_dict.items():
                all_facts_for_subject = []
                
                if isinstance(wiki_value, dict):
                    # Use ground truth sources (Wikipedia + Web)
                    wiki_triples = wiki_value.get('triples', [])
                    if wiki_triples:
                        all_facts_for_subject.extend(wiki_triples)
                    
                    # Use all Web triples
                    web_triples = wiki_value.get('web_triples', [])
                    if web_triples:
                        all_facts_for_subject.extend(web_triples)
                else:
                    # Legacy format - just a list
                    all_facts_for_subject.extend(wiki_value)
                
                if all_facts_for_subject:
                    wiki_facts_per_subject[subject] = all_facts_for_subject
            
            # Apply stratified sampling (entity-then-triple) to obtain a total of m samples
            if self.sampling:
                all_wiki_facts = self.stratified_sample_from_grouped(wiki_facts_per_subject, self.sample_size)
            else:
                all_wiki_facts = [item for value_list in wiki_facts_per_subject.values() for item in value_list]
            
            # For each ground truth fact, check if it's entailed by elicited triples
            for wiki_fact in tqdm(all_wiki_facts, desc=f"Computing Recall (Wiki+Web) for {filename}"):
                subject = wiki_fact.get('subject', '')
                predicate = wiki_fact.get('predicate', '')
                obj = wiki_fact.get('object', '')
                
                # Get elicited triples for this subject
                elicited_triples_for_subject = self.subject_based_lookup(subject, elicited_list)
                
                if not elicited_triples_for_subject:
                    logger.warning(f"No elicited triples found for subject: {subject}. Marking as Neutral.")
                    results['c'].append(wiki_fact)
                    # Store detailed result
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "result": "c",
                        "reasoning": "No elicited triples found for subject",
                        "metric": "Recall (Wikipedia+Web)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                    continue
                
                # Convert elicited triples to string format
                elicited_triples_str = ", ".join(elicited_triples_for_subject)
                
                # Convert Wikipedia triple to string
                wiki_fact_str = f"({subject}, {predicate}, {obj})"
                
                # Ask LLM to judge
                try:
                    output = request.verify_triple_lm_wikidata(wiki_fact_str, elicited_triples_str)
                    
                    # Extract category and reasoning
                    category = output.get("answer", "c") if isinstance(output, dict) else "c"
                    reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                    
                    results[category].append(wiki_fact)
                    
                    # Store detailed result
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "result": category,
                        "reasoning": reasoning,
                        "metric": "Recall (Wikipedia+Web)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                except Exception as e:
                    logger.error(f"Error verifying triple {wiki_fact_str}: {e}")
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "result": "error",
                        "reasoning": f"Verification error after retries: {str(e)}",
                        "metric": "Recall (Wikipedia+Web)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
            
            # Compute metrics
            total_facts = len(all_wiki_facts)
            if total_facts > 0:
                error_count = total_facts - (len(results['a']) + len(results['b']) + len(results['c']))
                results_dict = {
                    "Entailment": len(results['a']),
                    "Contradiction": len(results['b']),
                    "Neutral": len(results['c']),
                    "Entailment_ratio": len(results['a']) / total_facts,
                    "Contradiction_ratio": len(results['b']) / total_facts,
                    "Neutral_ratio": len(results['c']) / total_facts,
                    "Total #Triples": total_facts,
                    "Error_count": error_count,
                    "Error_ratio": error_count / total_facts if total_facts > 0 else 0,
                    "Metric": "Recall (Wikipedia+Web)",
                    "Source Elicited File": str(filename),
                }
                self.aggregated_data.append(results_dict)
                logger.info(f"Recall (Wiki+Web) for {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}, Errors={error_count}")

    def _lookup_category_helper(self, name, subject_to_category):
        """
        Helper method to lookup category with multiple normalization attempts.
        
        Args:
            name (str): The subject name to look up
            subject_to_category (dict): Mapping of subject_name -> category
        
        Returns:
            str: Category name or 'unknown' if not found
        """
        if not name:
            return 'unknown'
        
        candidates = [
            name,
            name.strip(),
            name.replace('_', ' '),
            name.replace(' ', '_'),
            name.lower(),
            name.lower().replace('_', ' '),
            name.lower().replace(' ', '_')
        ]
        
        for cand in candidates:
            cat = subject_to_category.get(cand)
            if cat:
                return cat
        
        return 'unknown'


    def build_category_index(self, entities_file_path):
        """
        Build a lookup index from subject_name (title) to category.
        
        Args:
            entities_file_path (str): Path to the JSON file containing categorized entities.
        
        Returns:
            dict: Mapping of subject_name -> category
        """
        subject_to_category = {}
        
        try:
            with open(entities_file_path, 'r', encoding='utf-8') as f:
                entities_by_category = json.load(f)

            # entities_by_category structure: {"category": [{"title": ..., "wikibase_item": ..., "length": ...}, ...]}
            categories = list(entities_by_category.items())

            # Iterate all categories (do not skip any)
            for category, entities_list in categories:
                for entity in entities_list:
                    title = entity.get('title', '')
                    if not title:
                        continue

                    # Add multiple normalized variants to improve matching robustness
                    variants = set()
                    v = title.strip()
                    variants.add(v)
                    variants.add(v.replace('_', ' '))
                    variants.add(v.replace(' ', '_'))
                    variants.add(v.lower())
                    variants.add(v.lower().replace('_', ' '))
                    variants.add(v.lower().replace(' ', '_'))

                    for variant in variants:
                        if variant:
                            subject_to_category.setdefault(variant, category)

            logger.info(f"Built category index with {len(subject_to_category)} title-variants across {len(categories)} categories")
            return subject_to_category
        except Exception as e:
            logger.error(f"Error building category index: {e}", exc_info=True)
            return {}


    def build_popularity_index(self, entities_file_path):
        """
        Build a lookup index from subject_name (title) to popularity bucket.
        
        Args:
            entities_file_path (str): Path to the JSON file containing entities with popularity buckets.
        
        Returns:
            dict: Mapping of subject_name -> popularity_bucket
        """
        subject_to_popularity = {}
        
        try:
            with open(entities_file_path, 'r', encoding='utf-8') as f:
                entities_by_popularity = json.load(f)

            # entities_by_popularity structure: {"66-100%": [{"title": ..., "length": ..., "wikidata_id": ..., "popularity_bucket": ...}, ...], ...}
            buckets = list(entities_by_popularity.items())

            # Iterate all popularity buckets (sorted from high to low popularity)
            for bucket, entities_list in buckets:
                for entity in entities_list:
                    title = entity.get('title', '')
                    if not title:
                        continue

                    # Add multiple normalized variants to improve matching robustness
                    variants = set()
                    v = title.strip()
                    variants.add(v)
                    variants.add(v.replace('_', ' '))
                    variants.add(v.replace(' ', '_'))
                    variants.add(v.lower())
                    variants.add(v.lower().replace('_', ' '))
                    variants.add(v.lower().replace(' ', '_'))

                    for variant in variants:
                        if variant:
                            subject_to_popularity.setdefault(variant, bucket)

            logger.info(f"Built popularity index with {len(subject_to_popularity)} title-variants across {len(buckets)} buckets")
            return subject_to_popularity
        except Exception as e:
            logger.error(f"Error building popularity index: {e}", exc_info=True)
            return {}


    def stratified_sample_by_popularity(self, triples_list, subject_to_popularity, triples_per_bucket=100):
        """Two-level stratified sampling across popularity buckets:
        1. Sample triples from each subject within each bucket (subject-level balance)
        2. Cap total samples per bucket at triples_per_bucket (bucket-level balance)

        Args:
            triples_list (list): List of triple dicts.
            subject_to_popularity (dict): Mapping of subject_name -> popularity_bucket.
            triples_per_bucket (int): Maximum number of triples to sample per popularity bucket.

        Returns:
            list: Sampled triples (max triples_per_bucket per bucket).
        """
        # Group triples by bucket AND subject (two-level grouping)
        triples_by_bucket_and_subject = {}
        unmatched_triples = []

        for triple in triples_list:
            subject_name = triple.get('subject_name', '')
            # Try lookup on subject_name, then on subject
            bucket = self._lookup_category_helper(subject_name, subject_to_popularity)
            if bucket == 'unknown':
                bucket = self._lookup_category_helper(triple.get('subject', ''), subject_to_popularity)

            if bucket != 'unknown':
                if bucket not in triples_by_bucket_and_subject:
                    triples_by_bucket_and_subject[bucket] = {}

                # Group by subject within bucket
                if subject_name not in triples_by_bucket_and_subject[bucket]:
                    triples_by_bucket_and_subject[bucket][subject_name] = []

                triples_by_bucket_and_subject[bucket][subject_name].append(triple)
            else:
                # Subject not found in popularity index
                unmatched_triples.append(triple)

        if unmatched_triples:
            sample_subjects = [t.get('subject_name') or t.get('subject') for t in unmatched_triples[:10]]
            logger.warning(f"Found {len(unmatched_triples)} triples with subjects not in popularity index; examples: {sample_subjects}")

        # Two-level sampling: sample from each subject first, then cap per bucket
        sampled_triples = []
        for bucket, subjects_dict in triples_by_bucket_and_subject.items():
            bucket_samples = []
            total_subjects = len(subjects_dict)

            # Calculate the number of triples to sample from each subject
            triples_per_subject = min(triples_per_bucket // total_subjects, 2)

            # Level 1: Sample from each subject within this bucket
            for subject_name, subject_triples in subjects_dict.items():
                sampled = self._rng.sample(subject_triples, k=min(triples_per_subject, len(subject_triples)))
                bucket_samples.extend(sampled)

            # If we still need more samples, fill the rest with random triples from all subjects
            while len(bucket_samples) < triples_per_bucket:
                subject_name = self._rng.choice(list(subjects_dict.keys()))
                subject_triples = subjects_dict[subject_name]
                if subject_triples:
                    triple = self._rng.choice(subject_triples)
                    if triple not in bucket_samples:
                        bucket_samples.append(triple)

            sampled_triples.extend(bucket_samples)
            logger.info(f"Sampled {len(bucket_samples)} triples from bucket '{bucket}' across {len(subjects_dict)} subjects (available: {sum(len(v) for v in subjects_dict.values())})")

        logger.info(f"Total sampled triples across all popularity buckets: {len(sampled_triples)}")
        return sampled_triples


    def stratified_sample_by_category(self, triples_list, subject_to_category, triples_per_category=100):
        """Two-level stratified sampling across categories:
        1. Sample triples from each subject within each category (subject-level balance)
        2. Cap total samples per category at triples_per_category (category-level balance)

        Args:
            triples_list (list): List of triple dicts.
            subject_to_category (dict): Mapping of subject_name -> category.
            triples_per_category (int): Maximum number of triples to sample per category.

        Returns:
            list: Sampled triples (max triples_per_category per category).
        """
        # Group triples by category AND subject (two-level grouping)
        triples_by_category_and_subject = {}
        unmatched_triples = []

        for triple in triples_list:
            subject_name = triple.get('subject_name', '')
            # try lookup on subject_name, then on subject
            category = self._lookup_category_helper(subject_name, subject_to_category)
            if category == 'unknown':
                category = self._lookup_category_helper(triple.get('subject', ''), subject_to_category)

            if category != 'unknown':
                if category not in triples_by_category_and_subject:
                    triples_by_category_and_subject[category] = {}

                # Group by subject within category
                if subject_name not in triples_by_category_and_subject[category]:
                    triples_by_category_and_subject[category][subject_name] = []

                triples_by_category_and_subject[category][subject_name].append(triple)
            else:
                # Subject not found in category index
                unmatched_triples.append(triple)

        if unmatched_triples:
            # Log count and a small sample to help debugging mismatches
            sample_subjects = [t.get('subject_name') or t.get('subject') for t in unmatched_triples[:10]]
            logger.warning(f"Found {len(unmatched_triples)} triples with subjects not in category index; examples: {sample_subjects}")

        # Two-level sampling: sample from each subject first, then cap per category
        sampled_triples = []
        for category, subjects_dict in triples_by_category_and_subject.items():
            category_samples = []
            total_subjects = len(subjects_dict)

            # Calculate the number of triples to sample from each subject
            triples_per_subject = min(triples_per_category // total_subjects, 2)

            # Level 1: Sample from each subject within this category
            for subject_name, subject_triples in subjects_dict.items():
                sampled = self._rng.sample(subject_triples, k=min(triples_per_subject, len(subject_triples)))
                category_samples.extend(sampled)

            # If we still need more samples, fill the rest with random triples from all subjects
            while len(category_samples) < triples_per_category:
                subject_name = self._rng.choice(list(subjects_dict.keys()))
                subject_triples = subjects_dict[subject_name]
                if subject_triples:
                    triple = self._rng.choice(subject_triples)
                    if triple not in category_samples:
                        category_samples.append(triple)

            sampled_triples.extend(category_samples)
            logger.info(f"Sampled {len(category_samples)} triples from category '{category}' across {len(subjects_dict)} subjects (available: {sum(len(v) for v in subjects_dict.values())})")

        logger.info(f"Total sampled triples across all categories: {len(sampled_triples)}")
        return sampled_triples

    def build_subject_to_title_mapping(self, wikipedia_triples_dict):
        """
        Build a mapping from Wikipedia triple subjects back to entity titles.
        This is needed because Wikipedia triples use normalized subjects (e.g., "King I")
        while we need to look up the original entity title for category matching.
        
        Args:
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples.
        
        Returns:
            dict: Mapping of triple subject -> entity title
        """
        subject_to_title = {}
        
        for entity_title, wiki_value in wikipedia_triples_dict.items():
            # Extract triples from wiki_value
            wiki_triples = wiki_value if isinstance(wiki_value, list) else wiki_value.get('triples', []) if isinstance(wiki_value, dict) else []
            
            # Map each triple's subject back to the entity title
            for triple in wiki_triples:
                triple_subject = triple.get('subject', '')
                if triple_subject:
                    # Store the entity title for this subject
                    subject_to_title[triple_subject] = entity_title
        
        logger.debug(f"Built subject-to-title mapping with {len(subject_to_title)} entries")
        
        return subject_to_title

    def compute_recall_wiki_by_category(self, elicited_triples, wikipedia_triples_dict, subject_to_category, triples_per_category=100):
        """
        Compute recall by category: two-level stratified sampling (100 Wikipedia triples per category) then LLM verification.
        Stores both aggregated metrics (per-category) and detailed per-triple results.
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples.
            subject_to_category (dict): Mapping of subject_name (title) -> category.
        Returns:
            Updated self.aggregated_data with per-category recall metrics.
        """
        request = Request(self.llm_judge)
        for filename, elicited_list in elicited_triples.items():
            logger.debug(f"Computing Recall (Wiki) by category for file: {filename}")
            # Build mapping from Wikipedia triple subjects to entity titles
            subject_to_title = self.build_subject_to_title_mapping(wikipedia_triples_dict)
            # Collect Wikipedia triples by category AND subject (for balanced stratified sampling)
            wiki_facts_by_category_and_subject = {}
            category_debug_counts = {}  # Debug: track how many triples per category
            for subject_name, wiki_value in wikipedia_triples_dict.items():
                # Look up category directly from subject_to_category (built from entities file)
                category = self._lookup_category_helper(subject_name, subject_to_category)
                if category == 'unknown':
                    logger.debug(f"Could not determine category for Wikipedia subject: {subject_name}")
                    continue  # Skip uncategorized Wikipedia facts
                # Extract triples from wiki_value (which could be a dict or list)
                wiki_triples = wiki_value if isinstance(wiki_value, list) else wiki_value.get('triples', []) if isinstance(wiki_value, dict) else []
                if wiki_triples:
                    if category not in wiki_facts_by_category_and_subject:
                        wiki_facts_by_category_and_subject[category] = {}
                        category_debug_counts[category] = 0
                    # Add subject_name to each triple for later category lookup
                    for triple in wiki_triples:
                        triple['subject_name'] = subject_name
                    wiki_facts_by_category_and_subject[category][subject_name] = wiki_triples
                    category_debug_counts[category] += len(wiki_triples)
            # Log category distribution
            logger.info(f"Wikipedia triples distribution by category: {category_debug_counts}")
            # Two-level stratified sampling: up to triples_per_category per category
            sampled_wiki_facts = []
            for category, subjects_dict in wiki_facts_by_category_and_subject.items():
                category_samples = []
                
                # Only sample if triples_per_category is greater than 0
                if triples_per_category > 0:
                    # First pass: sample at most one fact from each subject (only if we need at least 1 triple total)
                    for subject_name, facts in subjects_dict.items():
                        if facts and len(category_samples) < triples_per_category:
                            sampled = self._rng.choice(facts)
                            category_samples.append(sampled)
                    
                    # Additional passes: sample more facts from subjects until we reach triples_per_category
                    while len(category_samples) < triples_per_category and subjects_dict:
                        added = False
                        for subject_name, facts in list(subjects_dict.items()):
                            if facts and len(category_samples) < triples_per_category:
                                sampled = self._rng.choice(facts)
                                if sampled not in category_samples:
                                    category_samples.append(sampled)
                                    added = True
                                    if len(category_samples) >= triples_per_category:
                                        break
                        if len(category_samples) >= triples_per_category:
                            break
                        if not added:
                            break
                
                sampled_wiki_facts.extend(category_samples)
                logger.info(f"Sampled {len(category_samples)} Wikipedia facts from category '{category}' across {len(subjects_dict)} subjects")
            if not sampled_wiki_facts:
                logger.warning(f"No sampled Wikipedia facts for {filename}")
                continue
            # Group results by category
            results_by_category = {}
            # For each sampled Wikipedia fact, check if it's entailed by elicited triples
            for wiki_fact in tqdm(sampled_wiki_facts, desc=f"Computing Recall (Wiki) by category for {filename}"):
                subject = wiki_fact.get('subject', '')
                predicate = wiki_fact.get('predicate', '')
                obj = wiki_fact.get('object', '')
                subject_name = wiki_fact.get('subject_name', '')  # This was added in the sampling loop above
                # Look up category using the subject_name we added earlier
                category = self._lookup_category_helper(subject_name, subject_to_category)
                if category == 'unknown':
                    # This shouldn't happen since we filtered during collection, but log it
                    logger.error(f"UNEXPECTED: Could not determine category for subject_name: {subject_name}, subject: {subject}")
                    continue
                # Initialize category results if needed
                if category not in results_by_category:
                    results_by_category[category] = {"a": [], "b": [], "c": []}
                # Get elicited triples for this subject
                elicited_triples_for_subject = self.subject_based_lookup(subject, elicited_list)
                if not elicited_triples_for_subject:
                    logger.warning(f"No elicited triples found for subject: {subject} (category: {category})")
                    results_by_category[category]['c'].append(wiki_fact)
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": category,
                        "result": "c",
                        "reasoning": "No elicited triples found for subject",
                        "metric": "Recall (Wikipedia-Category)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                    continue
                elicited_triples_str = ", ".join(elicited_triples_for_subject)
                wiki_fact_str = f"({subject}, {predicate}, {obj})"
                try:
                    output = request.verify_triple_lm_wikidata(wiki_fact_str, elicited_triples_str)
                    category_result = output.get("answer", "c") if isinstance(output, dict) else "c"
                    reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                    results_by_category[category][category_result].append(wiki_fact)
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": category,
                        "result": category_result,
                        "reasoning": reasoning,
                        "metric": "Recall (Wikipedia-Category)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                except Exception as e:
                    logger.error(f"Error verifying triple {wiki_fact_str}: {e}")
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": category,
                        "result": "error",
                        "reasoning": f"Verification error after retries: {str(e)}",
                        "metric": "Recall (Wikipedia-Category)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
            # Compute per-category metrics
            for category, results in results_by_category.items():
                total_facts = len(results['a']) + len(results['b']) + len(results['c'])
                if total_facts > 0:
                    results_dict = {
                        "Category": category,
                        "Entailment": len(results['a']),
                        "Contradiction": len(results['b']),
                        "Neutral": len(results['c']),
                        "Entailment_ratio": len(results['a']) / total_facts,
                        "Contradiction_ratio": len(results['b']) / total_facts,
                        "Neutral_ratio": len(results['c']) / total_facts,
                        "Total #Triples": total_facts,
                        "Metric": "Recall (Wikipedia-Category)",
                        "Source Elicited File": str(filename),
                    }
                    self.aggregated_data.append(results_dict)
                    logger.info(f"Recall (Wiki-Category) for {category} in {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}")

    def compute_precision_wiki_by_category(self, elicited_triples, wikipedia_triples_dict, subject_to_category, triples_per_category=100):
        """
        Compute precision by category: two-level stratified sampling (100 triples per category) then LLM verification.
        Stores both aggregated metrics (per-category) and detailed per-triple results.
        
        OPTIMIZATION: Groups triples by entity and builds RAG index once per entity instead of once per triple,
        significantly reducing API calls and enabling caching to disk.
        
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples.
            subject_to_category (dict): Mapping of subject_name -> category.
        Returns:
            Updated self.aggregated_data with per-category precision metrics.
        """
        request = Request(self.llm_judge)
        for filename, triples_list in elicited_triples.items():
            logger.debug(f"Computing Precision (Wiki) by category for file: {filename}")
            sampled_triples = []
            triples_by_category_and_subject = {}
            for triple in triples_list:
                subject_name = triple.get('subject_name', '')
                category = self._lookup_category_helper(subject_name, subject_to_category)
                if category == 'unknown':
                    category = self._lookup_category_helper(triple.get('subject', ''), subject_to_category)
                if category != 'unknown':
                    if category not in triples_by_category_and_subject:
                        triples_by_category_and_subject[category] = {}
                    if subject_name not in triples_by_category_and_subject[category]:
                        triples_by_category_and_subject[category][subject_name] = []
                    triples_by_category_and_subject[category][subject_name].append(triple)
            for category, subjects_dict in triples_by_category_and_subject.items():
                category_samples = []
                if triples_per_category > 0:
                    initial_per_subject = min(1, triples_per_category)
                    if initial_per_subject > 0:
                        for subject_name, subject_triples in subjects_dict.items():
                            if subject_triples and len(category_samples) < triples_per_category:
                                sampled = self._rng.choice(subject_triples)
                                category_samples.append(sampled)
                    while len(category_samples) < triples_per_category and subjects_dict:
                        added = False
                        for subject_name, subject_triples in list(subjects_dict.items()):
                            if subject_triples and len(category_samples) < triples_per_category:
                                sampled = self._rng.choice(subject_triples)
                                if sampled not in category_samples:
                                    category_samples.append(sampled)
                                    added = True
                                    if len(category_samples) >= triples_per_category:
                                        break
                        if len(category_samples) >= triples_per_category:
                            break
                        if not added:
                            break
                sampled_triples.extend(category_samples)
                logger.info(f"Sampled {len(category_samples)} triples from category '{category}' across {len(subjects_dict)} subjects")
            if not sampled_triples:
                logger.warning(f"No sampled triples for {filename}")
                continue
            
            triples_by_entity = defaultdict(list)
            for triple in sampled_triples:
                subject_name = triple.get('subject_name', '')
                if subject_name:
                    triples_by_entity[subject_name].append(triple)
            
            total_triples_count = sum(len(v) for v in triples_by_entity.values())
            triples_progress_bar = tqdm(total=total_triples_count, desc=f"Computing Precision by category", unit="triple")
            
            results_by_category = {}
            
            for subject_name, entity_triples in triples_by_entity.items():
                category = self._lookup_category_helper(subject_name, subject_to_category)
                if category == 'unknown':
                    category = self._lookup_category_helper(entity_triples[0].get('subject', ''), subject_to_category)
                if category == 'unknown':
                    logger.warning(f"Could not determine category for subject: {subject_name}")
                
                if category not in results_by_category:
                    results_by_category[category] = {"a": [], "b": [], "c": []}
                
                wiki_data = wikipedia_triples_dict.get(subject_name, None)
                if not wiki_data:
                    logger.warning(f"No Wikipedia data found for subject: {subject_name}")
                    for each_triple in entity_triples:
                        results_by_category[category]['c'].append(each_triple)
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "category": category,
                            "result": "c",
                            "reasoning": "No Wikipedia data found for subject",
                            "metric": "Precision (Wikipedia-Category)",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    triples_progress_bar.update(len(entity_triples))
                    continue
                
                try:
                    rag = None
                    cache_path = GroundTruthRAG.get_cache_path_for_entity(self.rag_cache_dir, subject_name) if self.rag_cache_dir else None
                    
                    # Try to load from cache first (outside lock - fast path)
                    if cache_path and os.path.exists(cache_path):
                        try:
                            rag = GroundTruthRAG.load_from_disk(cache_path)
                        except Exception as e:
                            logger.warning(f"Failed to load RAG cache for {subject_name}, rebuilding: {e}")
                    
                    # Build new index if not loaded from cache (use lock to prevent race conditions)
                    if rag is None:
                        lock_path = cache_path + ".lock" if cache_path else None
                        
                        if lock_path:
                            lock_dir = os.path.dirname(lock_path)
                            if lock_dir and not os.path.exists(lock_dir):
                                os.makedirs(lock_dir, exist_ok=True)
                            
                            with open(lock_path, 'w') as lock_file:
                                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                                try:
                                    # Double-check after acquiring lock
                                    if os.path.exists(cache_path):
                                        try:
                                            rag = GroundTruthRAG.load_from_disk(cache_path)
                                        except Exception as e:
                                            logger.warning(f"Failed to load RAG cache after lock for {subject_name}, rebuilding: {e}")
                                    else:
                                        # Build the RAG index
                                        rag = GroundTruthRAG()
                                        rag.build_index(wiki_data)
                                        rag.cache_embeddings()
                                        if cache_path:
                                            try:
                                                rag.save_to_disk(cache_path)
                                            except Exception as e:
                                                logger.warning(f"Failed to save RAG cache for {subject_name}: {e}")
                                finally:
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        else:
                            # No cache path, build without locking
                            rag = GroundTruthRAG()
                            rag.build_index(wiki_data)
                            rag.cache_embeddings()
                            if cache_path:
                                rag.save_to_disk(cache_path)
                except Exception as e:
                    logger.error(f"Error building RAG index for {subject_name}: {e}")
                    for each_triple in entity_triples:
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "category": category,
                            "result": "error",
                            "reasoning": f"RAG index build error: {str(e)}",
                            "metric": "Precision (Wikipedia-Category)",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    triples_progress_bar.update(len(entity_triples))
                    continue
                
                for each_triple in entity_triples:
                    subject = each_triple.get('subject', '')
                    predicate = each_triple.get('predicate', '')
                    obj = each_triple.get('object', '')
                    elicited_triple_str = f"({subject}, {predicate}, {obj})"
                    
                    try:
                        retrieved_passages = rag.retrieve_top_k(elicited_triple_str, k=self.top_k)
                        
                        if not retrieved_passages:
                            category_result = "c"
                            reasoning = "No relevant passages found"
                            passages_str = ""
                        else:
                            passages_text = "\n\n".join([
                                f"[{p['source'].upper()}] {p['content'][:500]}" 
                                for p in retrieved_passages
                            ])
                            output = request.verify_triple_with_rag(
                                elicited_triple_str,
                                passages_text,
                                [p['source'] for p in retrieved_passages]
                            )
                            category_result = output.get("answer", "c") if isinstance(output, dict) else "c"
                            reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                            reasoning = f"RAG: {', '.join(set([p['source'] for p in retrieved_passages]))} | {reasoning}"
                    
                        results_by_category[category][category_result].append(each_triple)
                        passages_str = "\n\n---\n\n".join([
                            f"[{p['source'].upper()}] {p['content'][:2000]}"
                            for p in retrieved_passages
                        ]) if retrieved_passages else ""
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "category": category,
                            "result": category_result,
                            "reasoning": reasoning,
                            "metric": "Precision (Wikipedia-Category)",
                            "source_file": filename,
                            "retrieved_passages": passages_str
                        })
                        
                    except Exception as e:
                        logger.error(f"Error verifying triple {elicited_triple_str}: {e}")
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "category": category,
                            "result": "error",
                            "reasoning": f"Verification error after retries: {str(e)}",
                            "metric": "Precision (Wikipedia-Category)",
                            "source_file": filename,
                            "retrieved_passages": passages_str if 'passages_str' in locals() else ""
                        })
                    
                    triples_progress_bar.update(1)
            
            triples_progress_bar.close()
            
            for category, results in results_by_category.items():
                total_triples = len(results['a']) + len(results['b']) + len(results['c'])
                if total_triples > 0:
                    results_dict = {
                        "Category": category,
                        "Entailment": len(results['a']),
                        "Contradiction": len(results['b']),
                        "Neutral": len(results['c']),
                        "Entailment_ratio": len(results['a']) / total_triples,
                        "Contradiction_ratio": len(results['b']) / total_triples,
                        "Neutral_ratio": len(results['c']) / total_triples,
                        "Total #Triples": total_triples,
                        "Metric": "Precision (Wikipedia-Category)",
                        "Source Elicited File": str(filename),
                    }
                    self.aggregated_data.append(results_dict)
                    logger.info(f"Precision (Wiki-Category) for {category} in {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}")

    def compute_precision_wiki_by_popularity(self, elicited_triples, wikipedia_triples_dict, subject_to_popularity, triples_per_popularity_bucket=100, target_bucket=None):
        """
        Compute precision by popularity bucket: two-level stratified sampling (100 triples per bucket) then LLM verification with RAG.
        Stores both aggregated metrics (per-bucket) and detailed per-triple results.
        
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples.
            subject_to_popularity (dict): Mapping of subject_name -> popularity_bucket.
            triples_per_popularity_bucket (int): Number of triples to sample per popularity bucket.
            target_bucket (str): If set, only compute metrics for this bucket (for single-bucket evaluation).
        
        Returns:
            Updated self.aggregated_data with per-bucket precision metrics.
        """
        request = Request(self.llm_judge)
        for filename, triples_list in elicited_triples.items():
            logger.debug(f"Computing Precision (Wiki) by popularity for file: {filename}")
            # Two-level stratified sampling: 100 triples per popularity bucket
            sampled_triples = []
            # Group triples by bucket AND subject (two-level grouping)
            triples_by_bucket_and_subject = {}
            for triple in triples_list:
                subject_name = triple.get('subject_name', '')
                bucket = self._lookup_category_helper(subject_name, subject_to_popularity)
                if bucket == 'unknown':
                    bucket = self._lookup_category_helper(triple.get('subject', ''), subject_to_popularity)
                if bucket != 'unknown':
                    if bucket not in triples_by_bucket_and_subject:
                        triples_by_bucket_and_subject[bucket] = {}
                    # Group by subject within bucket
                    if subject_name not in triples_by_bucket_and_subject[bucket]:
                        triples_by_bucket_and_subject[bucket][subject_name] = []
                    triples_by_bucket_and_subject[bucket][subject_name].append(triple)
            # Two-level sampling: sample from each subject first, then cap per bucket
            for bucket, subjects_dict in triples_by_bucket_and_subject.items():
                # Skip non-target buckets if target_bucket is specified
                if target_bucket and bucket != target_bucket:
                    logger.info(f"Skipping bucket '{bucket}' (target is '{target_bucket}')")
                    continue
                bucket_samples = []
                
                # Only sample if triples_per_popularity_bucket is greater than 0
                if triples_per_popularity_bucket > 0:
                    # First pass: sample at most one triple from each subject (only if we need at least 1 triple total)
                    for subject_name, subject_triples in subjects_dict.items():
                        if subject_triples and len(bucket_samples) < triples_per_popularity_bucket:
                            sampled = self._rng.choice(subject_triples)
                            bucket_samples.append(sampled)
                    
                    # Additional passes: sample more triples from subjects until we reach triples_per_popularity_bucket
                    while len(bucket_samples) < triples_per_popularity_bucket and subjects_dict:
                        added = False
                        for subject_name, subject_triples in list(subjects_dict.items()):
                            if subject_triples and len(bucket_samples) < triples_per_popularity_bucket:
                                sampled = self._rng.choice(subject_triples)
                                if sampled not in bucket_samples:
                                    bucket_samples.append(sampled)
                                    added = True
                                    if len(bucket_samples) >= triples_per_popularity_bucket:
                                        break
                        if len(bucket_samples) >= triples_per_popularity_bucket:
                            break
                        if not added:
                            break
                
                sampled_triples.extend(bucket_samples)
                logger.info(f"Sampled {len(bucket_samples)} triples from bucket '{bucket}' across {len(subjects_dict)} subjects")
            if not sampled_triples:
                logger.warning(f"No sampled triples for {filename}")
                continue
            # Group results by bucket for per-bucket metrics
            results_by_bucket = {}
            triples_by_entity = defaultdict(list)
            for triple in sampled_triples:
                subject_name = triple.get('subject_name', '')
                if subject_name:
                    triples_by_entity[subject_name].append(triple)
            
            total_triples_count = sum(len(v) for v in triples_by_entity.values())
            triples_progress_bar = tqdm(total=total_triples_count, desc=f"Computing Precision by popularity", unit="triple")
            
            for subject_name, entity_triples in triples_by_entity.items():
                bucket = self._lookup_category_helper(subject_name, subject_to_popularity)
                if bucket == 'unknown':
                    bucket = self._lookup_category_helper(entity_triples[0].get('subject', ''), subject_to_popularity)
                if bucket == 'unknown':
                    logger.warning(f"Could not determine popularity bucket for subject: {subject_name}")
                
                if bucket not in results_by_bucket:
                    results_by_bucket[bucket] = {"a": [], "b": [], "c": []}
                
                wiki_data = wikipedia_triples_dict.get(subject_name, None)
                if not wiki_data:
                    logger.warning(f"No Wikipedia data found for subject: {subject_name}")
                    for each_triple in entity_triples:
                        results_by_bucket[bucket]['c'].append(each_triple)
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "category": bucket,
                            "result": "c",
                            "reasoning": "No Wikipedia data found for subject",
                            "metric": "Precision (Wikipedia-Popularity)",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    triples_progress_bar.update(len(entity_triples))
                    continue
                
                try:
                    rag = None
                    cache_path = GroundTruthRAG.get_cache_path_for_entity(self.rag_cache_dir, subject_name) if self.rag_cache_dir else None
                    
                    # Try to load from cache first (outside lock - fast path)
                    if cache_path and os.path.exists(cache_path):
                        try:
                            rag = GroundTruthRAG.load_from_disk(cache_path)
                        except Exception as e:
                            logger.warning(f"Failed to load RAG cache for {subject_name}, rebuilding: {e}")
                    
                    # Build new index if not loaded from cache (use lock to prevent race conditions)
                    if rag is None:
                        lock_path = cache_path + ".lock" if cache_path else None
                        
                        if lock_path:
                            lock_dir = os.path.dirname(lock_path)
                            if lock_dir and not os.path.exists(lock_dir):
                                os.makedirs(lock_dir, exist_ok=True)
                            
                            with open(lock_path, 'w') as lock_file:
                                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                                try:
                                    # Double-check after acquiring lock
                                    if os.path.exists(cache_path):
                                        try:
                                            rag = GroundTruthRAG.load_from_disk(cache_path)
                                        except Exception as e:
                                            logger.warning(f"Failed to load RAG cache after lock for {subject_name}, rebuilding: {e}")
                                    else:
                                        # Build the RAG index
                                        rag = GroundTruthRAG()
                                        rag.build_index(wiki_data)
                                        rag.cache_embeddings()
                                        if cache_path:
                                            try:
                                                rag.save_to_disk(cache_path)
                                            except Exception as e:
                                                logger.warning(f"Failed to save RAG cache for {subject_name}: {e}")
                                finally:
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        else:
                            # No cache path, build without locking
                            rag = GroundTruthRAG()
                            rag.build_index(wiki_data)
                            rag.cache_embeddings()
                            if cache_path:
                                rag.save_to_disk(cache_path)
                except Exception as e:
                    logger.error(f"Error building RAG index for {subject_name}: {e}")
                    for each_triple in entity_triples:
                        self.append_detailed_result({
                            "subject": each_triple.get('subject', ''),
                            "predicate": each_triple.get('predicate', ''),
                            "object": each_triple.get('object', ''),
                            "category": bucket,
                            "result": "error",
                            "reasoning": f"RAG index build error: {str(e)}",
                            "metric": "Precision (Wikipedia-Popularity)",
                            "source_file": filename,
                            "retrieved_passages": ""
                        })
                    triples_progress_bar.update(len(entity_triples))
                    continue
                
                for each_triple in entity_triples:
                    subject = each_triple.get('subject', '')
                    predicate = each_triple.get('predicate', '')
                    obj = each_triple.get('object', '')
                    elicited_triple_str = f"({subject}, {predicate}, {obj})"
                    
                    try:
                        retrieved_passages = rag.retrieve_top_k(elicited_triple_str, k=self.top_k)
                        
                        if not retrieved_passages:
                            bucket_result = "c"
                            reasoning = "No relevant passages found"
                            passages_str = ""
                        else:
                            passages_text = "\n\n".join([
                                f"[{p['source'].upper()}] {p['content'][:500]}" 
                                for p in retrieved_passages
                            ])
                            output = request.verify_triple_with_rag(
                                elicited_triple_str,
                                passages_text,
                                [p['source'] for p in retrieved_passages]
                            )
                            bucket_result = output.get("answer", "c") if isinstance(output, dict) else "c"
                            reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                            reasoning = f"RAG: {', '.join(set([p['source'] for p in retrieved_passages]))} | {reasoning}"
                    
                        results_by_bucket[bucket][bucket_result].append(each_triple)
                        passages_str = "\n\n---\n\n".join([
                            f"[{p['source'].upper()}] {p['content'][:2000]}"
                            for p in retrieved_passages
                        ]) if retrieved_passages else ""
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "category": bucket,
                            "result": bucket_result,
                            "reasoning": reasoning,
                            "metric": "Precision (Wikipedia-Popularity)",
                            "source_file": filename,
                            "retrieved_passages": passages_str
                        })
                        
                    except Exception as e:
                        logger.error(f"Error verifying triple {elicited_triple_str}: {e}")
                        self.append_detailed_result({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "category": bucket,
                            "result": "error",
                            "reasoning": f"Verification error after retries: {str(e)}",
                            "metric": "Precision (Wikipedia-Popularity)",
                            "source_file": filename,
                            "retrieved_passages": passages_str if 'passages_str' in locals() else ""
                        })
                    
                    triples_progress_bar.update(1)
            
            triples_progress_bar.close()
            # Compute per-bucket metrics
            for bucket, results in results_by_bucket.items():
                total_triples = len(results['a']) + len(results['b']) + len(results['c'])
                if total_triples > 0:
                    results_dict = {
                        "Category": bucket,
                        "Entailment": len(results['a']),
                        "Contradiction": len(results['b']),
                        "Neutral": len(results['c']),
                        "Entailment_ratio": len(results['a']) / total_triples,
                        "Contradiction_ratio": len(results['b']) / total_triples,
                        "Neutral_ratio": len(results['c']) / total_triples,
                        "Total #Triples": total_triples,
                        "Metric": "Precision (Wikipedia-Popularity)",
                        "Source Elicited File": str(filename),
                    }
                    self.aggregated_data.append(results_dict)
                    logger.info(f"Precision (Wiki-Popularity) for {bucket} in {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}")

    def compute_recall_wiki_by_popularity(self, elicited_triples, wikipedia_triples_dict, subject_to_popularity, triples_per_popularity_bucket=100, target_bucket=None):
        """
        Compute recall by popularity bucket: two-level stratified sampling (100 Wikipedia triples per bucket) then LLM verification.
        Stores both aggregated metrics (per-bucket) and detailed per-triple results.
        
        Parameters:
            elicited_triples (dict): Dictionary where keys are filenames and values are lists of elicited triples.
            wikipedia_triples_dict (dict): Dictionary mapping entity names to Wikipedia-extracted triples.
            subject_to_popularity (dict): Mapping of subject_name (title) -> popularity_bucket.
            triples_per_popularity_bucket (int): Number of triples to sample per popularity bucket.
            target_bucket (str): If set, only compute metrics for this bucket (for single-bucket evaluation).
        
        Returns:
            Updated self.aggregated_data with per-bucket recall metrics.
        """
        request = Request(self.llm_judge)
        for filename, elicited_list in elicited_triples.items():
            logger.debug(f"Computing Recall (Wiki) by popularity for file: {filename}")
            # Build mapping from Wikipedia triple subjects to entity titles
            subject_to_title = self.build_subject_to_title_mapping(wikipedia_triples_dict)
            # Collect Wikipedia triples by bucket AND subject (for balanced stratified sampling)
            wiki_facts_by_bucket_and_subject = {}
            bucket_debug_counts = {}
            for subject_name, wiki_value in wikipedia_triples_dict.items():
                # Look up bucket directly from subject_to_popularity (built from entities file)
                bucket = self._lookup_category_helper(subject_name, subject_to_popularity)
                if bucket == 'unknown':
                    logger.debug(f"Could not determine popularity bucket for Wikipedia subject: {subject_name}")
                    continue
                # Extract triples from wiki_value (which could be a dict or list)
                wiki_triples = wiki_value if isinstance(wiki_value, list) else wiki_value.get('triples', []) if isinstance(wiki_value, dict) else []
                if wiki_triples:
                    if bucket not in wiki_facts_by_bucket_and_subject:
                        wiki_facts_by_bucket_and_subject[bucket] = {}
                        bucket_debug_counts[bucket] = 0
                    # Add subject_name to each triple for later bucket lookup
                    for triple in wiki_triples:
                        triple['subject_name'] = subject_name
                    wiki_facts_by_bucket_and_subject[bucket][subject_name] = wiki_triples
                    bucket_debug_counts[bucket] += len(wiki_triples)
            # Log bucket distribution
            logger.info(f"Wikipedia triples distribution by popularity bucket: {bucket_debug_counts}")
            # Two-level stratified sampling: up to triples_per_popularity_bucket per bucket
            sampled_wiki_facts = []
            for bucket, subjects_dict in wiki_facts_by_bucket_and_subject.items():
                # Skip non-target buckets if target_bucket is specified
                if target_bucket and bucket != target_bucket:
                    logger.info(f"Skipping bucket '{bucket}' (target is '{target_bucket}')")
                    continue
                bucket_samples = []
                
                # Only sample if triples_per_popularity_bucket is greater than 0
                if triples_per_popularity_bucket > 0:
                    # First pass: sample at most one fact from each subject (only if we need at least 1 triple total)
                    for subject_name, facts in subjects_dict.items():
                        if facts and len(bucket_samples) < triples_per_popularity_bucket:
                            sampled = self._rng.choice(facts)
                            bucket_samples.append(sampled)
                    
                    # Additional passes: sample more facts from subjects until we reach triples_per_popularity_bucket
                    while len(bucket_samples) < triples_per_popularity_bucket and subjects_dict:
                        added = False
                        for subject_name, facts in list(subjects_dict.items()):
                            if facts and len(bucket_samples) < triples_per_popularity_bucket:
                                sampled = self._rng.choice(facts)
                                if sampled not in bucket_samples:
                                    bucket_samples.append(sampled)
                                    added = True
                                    if len(bucket_samples) >= triples_per_popularity_bucket:
                                        break
                        if len(bucket_samples) >= triples_per_popularity_bucket:
                            break
                        if not added:
                            break
                
                sampled_wiki_facts.extend(bucket_samples)
                logger.info(f"Sampled {len(bucket_samples)} Wikipedia facts from bucket '{bucket}' across {len(subjects_dict)} subjects")
            if not sampled_wiki_facts:
                logger.warning(f"No sampled Wikipedia facts for {filename}")
                continue
            # Group results by bucket
            results_by_bucket = {}
            # For each sampled Wikipedia fact, check if it's entailed by elicited triples
            for wiki_fact in tqdm(sampled_wiki_facts, desc=f"Computing Recall (Wiki) by popularity for {filename}"):
                subject = wiki_fact.get('subject', '')
                predicate = wiki_fact.get('predicate', '')
                obj = wiki_fact.get('object', '')
                subject_name = wiki_fact.get('subject_name', '')
                # Look up bucket using the subject_name we added earlier
                bucket = self._lookup_category_helper(subject_name, subject_to_popularity)
                if bucket == 'unknown':
                    logger.error(f"UNEXPECTED: Could not determine bucket for subject_name: {subject_name}, subject: {subject}")
                    continue
                # Initialize bucket results if needed
                if bucket not in results_by_bucket:
                    results_by_bucket[bucket] = {"a": [], "b": [], "c": []}
                # Get elicited triples for this subject
                elicited_triples_for_subject = self.subject_based_lookup(subject, elicited_list)
                if not elicited_triples_for_subject:
                    logger.warning(f"No elicited triples found for subject: {subject} (bucket: {bucket})")
                    results_by_bucket[bucket]['c'].append(wiki_fact)
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": bucket,
                        "result": "c",
                        "reasoning": "No elicited triples found for subject",
                        "metric": "Recall (Wikipedia-Popularity)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                    continue
                elicited_triples_str = ", ".join(elicited_triples_for_subject)
                wiki_fact_str = f"({subject}, {predicate}, {obj})"
                try:
                    output = request.verify_triple_lm_wikidata(wiki_fact_str, elicited_triples_str)
                    bucket_result = output.get("answer", "c") if isinstance(output, dict) else "c"
                    reasoning = output.get("reasoning", "") if isinstance(output, dict) else ""
                    results_by_bucket[bucket][bucket_result].append(wiki_fact)
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": bucket,
                        "result": bucket_result,
                        "reasoning": reasoning,
                        "metric": "Recall (Wikipedia-Popularity)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
                except Exception as e:
                    logger.error(f"Error verifying triple {wiki_fact_str}: {e}")
                    self.append_detailed_result({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "category": bucket,
                        "result": "error",
                        "reasoning": f"Verification error after retries: {str(e)}",
                        "metric": "Recall (Wikipedia-Popularity)",
                        "source_file": filename,
                        "retrieved_passages": ""
                    })
            # Compute per-bucket metrics
            for bucket, results in results_by_bucket.items():
                total_facts = len(results['a']) + len(results['b']) + len(results['c'])
                if total_facts > 0:
                    results_dict = {
                        "Category": bucket,
                        "Entailment": len(results['a']),
                        "Contradiction": len(results['b']),
                        "Neutral": len(results['c']),
                        "Entailment_ratio": len(results['a']) / total_facts,
                        "Contradiction_ratio": len(results['b']) / total_facts,
                        "Neutral_ratio": len(results['c']) / total_facts,
                        "Total #Triples": total_facts,
                        "Metric": "Recall (Wikipedia-Popularity)",
                        "Source Elicited File": str(filename),
                    }
                    self.aggregated_data.append(results_dict)
                    logger.info(f"Recall (Wiki-Popularity) for {bucket} in {filename}: Entailment={len(results['a'])}, Contradiction={len(results['b'])}, Neutral={len(results['c'])}")
    
    def write_categorical_results_to_csv(self, filename, data):
        """
        Write per-category results to CSV with category column.
        
        Args:
            filename (str): Path to output CSV file.
            data (list): List of result dicts (with 'Category' column).
        """
        headers = ['Category', 'Entailment', 'Contradiction', 'Neutral', 'Entailment_ratio', 'Contradiction_ratio', 'Neutral_ratio', 'Total #Triples', 'Metric', 'Source Elicited File']
        
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            logger.info(f"Categorical results written to {filename}")
        except Exception as e:
            logger.error(f"Error writing categorical results to CSV: {e}", exc_info=True)