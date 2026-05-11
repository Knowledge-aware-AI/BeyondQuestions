import os
import json
import pickle
import threading
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from .request import Request
from .wikipedia_utils import (
    get_wikipedia_article,
    save_unified_wikipedia_cache,
    load_unified_wikipedia_cache,
    word_count,
    search_web,
    fetch_full_document
)

"""
Module for extracting triples from Wikipedia articles using LLM.
Coordinates Wikipedia retrieval, text shortening, and triple extraction.
Unified cache stores original articles, shortened versions, and extracted triples together.
"""

class WikipediaTripleExtractor:
    def __init__(self, ground_truth_dir_path, llm_judge: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct", web_results_count: int = 30, max_workers: int = 4):
        """
        Initialize the Wikipedia triple extractor.
        
        Args:
            llm_judge (str): Name of the LLM model to use.
            ground_truth_dir_path (str): Directory to store unified Wikipedia cache. If None, uses cwd.
            web_results_count (int): Number of Brave Search results to fetch per entity (default: 30).
            max_workers (int): Maximum number of parallel workers for entity processing (default: 4).
        """
        self.request = Request(llm_judge)
        self.llm_judge = llm_judge
        self.ground_truth_dir_path = ground_truth_dir_path or os.getcwd()
        self.web_results_count = web_results_count
        self.max_workers = max_workers
        # Single unified cache file containing all Wikipedia data
        self.unified_cache_file = os.path.join(self.ground_truth_dir_path, "GT.json")
        
        # Thread lock for synchronized cache access
        self._cache_lock = threading.Lock()
        
        # Load existing unified cache
        self.unified_cache = load_unified_wikipedia_cache(self.unified_cache_file)
        #logger.info(f"WikipediaTripleExtractor initialized")
        logger.info(f"Ground truth file: {self.unified_cache_file}")
        logger.info(f"Loaded {len(self.unified_cache)} entities from cache")
        logger.info(f"Max workers for parallel processing: {self.max_workers}")
    
    def get_wikipedia_article_for_entity(self, entity_name: str) -> Optional[Dict]:
        """
        Get Wikipedia article for an entity, using unified cache if available.
        
        Args:
            entity_name (str): The entity name to search for.
        
        Returns:
            dict: Article data with 'title', 'content', 'url' or None if not found.
        """
        # Check if entity exists in unified cache with complete data
        if entity_name in self.unified_cache and "original_content" in self.unified_cache[entity_name]:
            logger.debug(f"Using cached Wikipedia data for: {entity_name}")
            cached_data = self.unified_cache[entity_name]
            return {
                "title": cached_data.get("title", entity_name),
                "content": cached_data["original_content"],
                "url": cached_data.get("wiki_url", ""),
            }
        
        # Fetch from Wikipedia
        logger.debug(f"Fetching Wikipedia article for: {entity_name}")
        article = get_wikipedia_article(entity_name)
        
        #if article:
        #    logger.info(f"Successfully retrieved Wikipedia article for: {entity_name} ({len(article['content'])} chars)")
        #else:
        #    logger.warning(f"Failed to retrieve Wikipedia article for: {entity_name}")
        
        if not article:
            logger.warning(f"Failed to retrieve Wikipedia article for: {entity_name}")

        return article
    
    def process_entity(self, entity_name: str, max_words: int = 1000) -> Optional[Dict]:
        """
        Process a single entity: retrieve Wikipedia article, shorten, and extract triples.
        All data (original, shortened, triples) is stored in unified cache.
        Uses cached data at each stage to avoid redundant API/LLM calls.
        
        Args:
            entity_name (str): The entity to process.
            max_words (int): Maximum words for shortened text.
        
        Returns:
            dict: Contains 'entity', 'original_content', 'triples', 'wiki_url'
                  or None if Wikipedia article not found.
        """
        # Check if entity is FULLY cached (has all required fields)
        if entity_name in self.unified_cache:
            cached_entry = self.unified_cache[entity_name]
            if all(key in cached_entry for key in ["original_content", "triples"]):
                # Validate that triples list is not empty
                if cached_entry.get("triples") and len(cached_entry["triples"]) > 0:
                    # Also validate that web_search_results is not empty (to handle cases where Brave Search API failed with 402)
                    web_results = cached_entry.get("web_search_results")
                    if web_results and len(web_results) > 0:
                        #logger.info(f"Entity {entity_name} is fully cached, skipping all API and LLM calls")
                        return cached_entry
                    else:
                        logger.warning(f"Entity {entity_name} has empty web_search_results in cache (likely from 402 error), will re-fetch web data")
                        # Fall through to re-fetch web search results
                else:
                    logger.warning(f"Entity {entity_name} has empty triples in cache, will attempt re-extraction")
                    # Fall through to re-extract triples
        
        # Step 1: Retrieve Wikipedia article (check cache first)
        article = self.get_wikipedia_article_for_entity(entity_name)
        if not article:
            logger.warning(f"Could not retrieve Wikipedia article for: {entity_name}")
            return None
        
        original_content = article["content"]
        logger.info(f"Retrieved Wikipedia article for {entity_name} ({word_count(original_content)} words)")
        
        # Step 2: Use full Wikipedia content (no shortening)
        original_word_count = word_count(original_content)
        shortened_content = original_content
        logger.info(f"Using full Wikipedia content ({original_word_count} words)")
        
        # Step 3: Extract triples from full Wikipedia content (check cache first)
        # Only use cached triples if they exist AND are non-empty (to avoid stuck empty lists)
        if entity_name in self.unified_cache and "triples" in self.unified_cache[entity_name]:
            cached_triples = self.unified_cache[entity_name]["triples"]
            if cached_triples:  # Only use if list is not empty
                triples = cached_triples
                logger.debug(f"Using cached triples for {entity_name} ({len(triples)} triples)")
            else:
                logger.warning(f"Found empty cached triples for {entity_name}, re-extracting...")
                logger.debug(f"Extracting triples from {entity_name}...")
                triples = self.request.extract_triples_from_text(entity_name, shortened_content)
                logger.debug(f"Re-extracted {len(triples)} triples from {entity_name}")
        else:
            logger.info(f"Extracting triples from Wikipedia article of entity: {entity_name}")
            triples = self.request.extract_triples_from_text(entity_name, shortened_content)
            # extract_triples_from_text now returns a structured list of dicts
            logger.info(f"Extracted {len(triples)} triples from Wikipedia article of entity: {entity_name}")
        
        result = {
            "entity": entity_name,
            "title": article.get("title", entity_name),
            "original_content": original_content,
            "triples": triples,
            "wiki_url": article["url"],
            "original_word_count": original_word_count,
            "extracted_triple_count": len(triples),
        }
        
        # Step 5: Fetch web search results
        if entity_name in self.unified_cache and "web_search_results" in self.unified_cache[entity_name]:
            cached_web_results = self.unified_cache[entity_name]["web_search_results"]
            if cached_web_results and len(cached_web_results) > 0:
                result["web_search_results"] = cached_web_results
                result["web_search_count"] = len(cached_web_results)
                logger.debug(f"Using cached web search results for {entity_name} ({len(cached_web_results)} results)")
                
                # Check if full documents were cached
                if "web_full_documents" in self.unified_cache[entity_name]:
                    result["web_full_documents"] = self.unified_cache[entity_name]["web_full_documents"]
                    result["web_full_documents_count"] = len(result["web_full_documents"])
                else:
                    # Fetch full documents (no shortening)
                    logger.info(f"Fetching full documents for cached web results {entity_name}...")
                    full_documents = []
                    for item in cached_web_results:
                        url = item.get("url", "")
                        if url:
                            full_content = fetch_full_document(url)
                            if full_content:
                                full_documents.append({
                                    "url": url,
                                    "title": item.get("title", ""),
                                    "content": full_content,
                                    "word_count": len(full_content.split())
                                })
                    result["web_full_documents"] = full_documents
                    result["web_full_documents_count"] = len(full_documents)
                
                # Check if triples were extracted from web results
                if "web_triples" in self.unified_cache[entity_name]:
                    result["web_triples"] = self.unified_cache[entity_name]["web_triples"]
                    result["web_triple_count"] = len(result["web_triples"])
                else:
                    # Extract triples from web snippets
                    logger.info(f"Extracting triples from web search results for {entity_name}...")
                    web_triples = self._extract_triples_from_web_results(cached_web_results, entity_name)
                    result["web_triples"] = web_triples
                    result["web_triple_count"] = len(web_triples)
            else:
                # Empty cache - re-fetch
                logger.info(f"Fetching web search results for {entity_name}...")
                web_results = search_web(entity_name, num_results=self.web_results_count)
                result["web_search_results"] = web_results
                result["web_search_count"] = len(web_results)
                
                # Fetch full document content (no shortening)
                logger.info(f"Fetching full documents for {entity_name}...")
                full_documents = []
                for item in web_results:
                    url = item.get("url", "")
                    if url:
                        full_content = fetch_full_document(url)
                        if full_content:
                            full_documents.append({
                                "url": url,
                                "title": item.get("title", ""),
                                "content": full_content,
                                "word_count": len(full_content.split())
                            })
                result["web_full_documents"] = full_documents
                result["web_full_documents_count"] = len(full_documents)
                
                # Extract triples from web snippets
                logger.info(f"Extracting triples from web search results for {entity_name}...")
                web_triples = self._extract_triples_from_web_results(web_results, entity_name)
                result["web_triples"] = web_triples
                result["web_triple_count"] = len(web_triples)
                logger.info(f"Fetched {len(web_results)} web search results, {len(full_documents)} full documents, and extracted {len(web_triples)} triples for {entity_name}")
        else:
            # No cache entry - re-fetch
            logger.info(f"Fetching web search results for {entity_name}...")
            web_results = search_web(entity_name, num_results=self.web_results_count)
            result["web_search_results"] = web_results
            result["web_search_count"] = len(web_results)
            
            # Fetch full document content (no shortening)
            logger.info(f"Fetching full documents for {entity_name}...")
            full_documents = []
            for item in web_results:
                url = item.get("url", "")
                if url:
                    full_content = fetch_full_document(url)
                    if full_content:
                        full_documents.append({
                            "url": url,
                            "title": item.get("title", ""),
                            "content": full_content,
                            "word_count": len(full_content.split())
                        })
            result["web_full_documents"] = full_documents
            result["web_full_documents_count"] = len(full_documents)
            
            # Extract triples from web snippets
            logger.info(f"Extracting triples from web search results for {entity_name}...")
            web_triples = self._extract_triples_from_web_results(web_results, entity_name)
            result["web_triples"] = web_triples
            result["web_triple_count"] = len(web_triples)
            logger.info(f"Fetched {len(web_results)} web search results, {len(full_documents)} full documents, and extracted {len(web_triples)} triples for {entity_name}")
        
        # Store in unified cache (thread-safe)
        with self._cache_lock:
            self.unified_cache[entity_name] = result
            save_unified_wikipedia_cache(self.unified_cache, self.unified_cache_file)
        #logger.debug(f"Cached complete Wikipedia data for {entity_name}")
        #logger.info(f"Entity {entity_name} processing complete: {len(triples)} triples, {original_word_count} original words, {word_count(shortened_content)} shortened words")
        
        return result
    
    def _extract_triples_from_web_results(self, web_results: list, entity_name: str) -> list:
        """
        Extract RDF triples from web search result snippets.
        
        Args:
            web_results (list): List of dicts with 'title', 'url', 'snippet' keys.
            entity_name (str): The entity name to use as subject.
        
        Returns:
            list: List of extracted triples as dicts.
        """
        all_triples = []
        
        for result in web_results:
            snippet = result.get('snippet', '')
            title = result.get('title', '')
            
            if not snippet:
                continue
            
            try:
                # Extract triples from this snippet
                triples = self.request.extract_triples_from_text(entity_name, snippet)
                
                # Add source info to each triple
                for triple in triples:
                    triple['_web_source_title'] = title
                    triple['_web_source_url'] = result.get('url', '')
                
                all_triples.extend(triples)
                
            except Exception as e:
                logger.warning(f"Failed to extract triples from web result '{title}': {e}")
        
        logger.debug(f"Extracted {len(all_triples)} triples from {len(web_results)} web results")
        return all_triples
    
    def process_entities_batch(self, entity_names: List[str], max_words: int = 1000) -> Dict:
        """
        Process multiple entities in parallel and store all data in unified cache.
        
        Args:
            entity_names (list): List of entity names to process.
            max_words (int): Maximum words for shortened text.
        
        Returns:
            dict: Mapping entity name -> processed result (also stored in unified cache).
        """
        results = {}
        successful = 0
        failed = 0
        
        logger.info(f"Starting parallel batch processing of {len(entity_names)} entities with {self.max_workers} workers")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_entity = {
                executor.submit(self.process_entity, entity_name, max_words): entity_name
                for entity_name in entity_names
            }
            
            # Process results as they complete with tqdm progress bar
            for future in tqdm(as_completed(future_to_entity), total=len(future_to_entity), 
                            desc="Processing Wikipedia articles"):
                entity_name = future_to_entity[future]
                try:
                    result = future.result()
                    if result:
                        results[entity_name] = result
                        successful += 1
                    else:
                        results[entity_name] = None
                        failed += 1
                        logger.debug(f"No result for entity: {entity_name}")
                except Exception as e:
                    logger.error(f"Error processing {entity_name}: {e}", exc_info=True)
                    results[entity_name] = None
                    failed += 1
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed out of {len(entity_names)} total")
        return results
    
    def get_cached_results(self, entity_names: List[str] = None) -> Dict:
        """
        Retrieve all cached Wikipedia data for entities.
        
        Args:
            entity_names (list, optional): List of entity names to retrieve. If None, returns all cached entities.
        
        Returns:
            dict: Mapping entity name -> complete cached Wikipedia data.
        """
        if entity_names is None:
            return self.unified_cache
        
        results = {}
        for entity_name in entity_names:
            if entity_name in self.unified_cache:
                results[entity_name] = self.unified_cache[entity_name]
        
        return results
    
    def export_unified_cache(self, output_filename: str = "GT.json"):
        """
        Export the unified cache to a JSON file for reference or backup.
        
        Args:
            output_filename (str): Filename for the export.
        """
        output_path = os.path.join(self.ground_truth_dir_path, output_filename)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.unified_cache, f, indent=4)
            logger.info(f"Unified cache exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export unified cache: {e}")
    
    def repair_empty_triples(self, entity_names: List[str] = None) -> Dict:
        """
        Repair entities with empty or missing triple lists by re-extracting triples.
        This is useful if extraction failed silently in previous runs.
        Handles both entities with empty triples list and entities missing the 'triples' key entirely.
        
        Args:
            entity_names (list, optional): Specific entities to repair. If None, scans all cached entities.
        
        Returns:
            dict: Mapping of entity_name -> number of triples recovered.
        """
        repair_results = {}
        cache_modified = False
        
        entities_to_check = entity_names if entity_names else list(self.unified_cache.keys())
        
        logger.info(f"Scanning {len(entities_to_check)} entities for empty or missing triple lists...")
        
        for entity_name in entities_to_check:
            if entity_name not in self.unified_cache:
                logger.debug(f"Entity {entity_name} not in cache, skipping")
                continue
            
            cached_entry = self.unified_cache[entity_name]
            
            # Check if triples are missing OR empty - either condition requires repair
            has_triples = "triples" in cached_entry
            is_empty = has_triples and len(cached_entry["triples"]) == 0
            is_missing = not has_triples
            
            if is_empty or is_missing:
                if is_missing:
                    logger.warning(f"Found MISSING triples key for {entity_name}, attempting extraction...")
                else:
                    logger.warning(f"Found EMPTY triples for {entity_name}, attempting repair...")
                
                # Get content for extraction (use original_content)
                if "original_content" in cached_entry:
                    content = cached_entry["original_content"]
                else:
                    logger.error(f"No content available for {entity_name}, cannot repair")
                    repair_results[entity_name] = -1
                    continue
                
                # Re-extract triples
                try:
                    logger.debug(f"Re-extracting triples for {entity_name}...")
                    triples = self.request.extract_triples_from_text(entity_name, content)
                    
                    if triples:
                        # Update cache (thread-safe)
                        with self._cache_lock:
                            self.unified_cache[entity_name]["triples"] = triples
                            self.unified_cache[entity_name]["extracted_triple_count"] = len(triples)
                        repair_results[entity_name] = len(triples)
                        cache_modified = True
                        logger.info(f"Successfully repaired {entity_name}: extracted {len(triples)} triples")
                    else:
                        repair_results[entity_name] = 0
                        logger.warning(f"Re-extraction for {entity_name} still returned empty list")
                except Exception as e:
                    repair_results[entity_name] = -1
                    logger.error(f"Error re-extracting triples for {entity_name}: {e}", exc_info=True)
            else:
                repair_results[entity_name] = len(cached_entry.get("triples", []))
        
        # Save cache once after all repairs are done (ensures consistency)
        if cache_modified:
            logger.info("Saving cache after repairs...")
            with self._cache_lock:
                save_unified_wikipedia_cache(self.unified_cache, self.unified_cache_file)
            logger.debug("Cache saved successfully")
        
        # Summary
        empty_count = sum(1 for v in repair_results.values() if v == 0)
        repaired_count = sum(1 for v in repair_results.values() if v > 0)
        error_count = sum(1 for v in repair_results.values() if v == -1)
        
        logger.info(f"Repair complete: {repaired_count} repaired, {empty_count} still empty, {error_count} errors")
        
        return repair_results
