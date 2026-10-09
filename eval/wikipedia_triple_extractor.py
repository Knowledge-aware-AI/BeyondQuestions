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
    def __init__(self, ground_truth_dir_path, llm_judge: str = "google/gemma-4-26B-A4B-it", web_results_count: int = 30, disable_web_results_topup: bool = False, max_workers: int = 4, judge_api_url: str = None, judge_api_key: str = None, max_brave_calls: int = 1000):
        """
        Initialize the Wikipedia triple extractor.

        Args:
            llm_judge (str): Name of the LLM model to use.
            ground_truth_dir_path (str): Directory to store unified Wikipedia cache. If None, uses cwd.
            web_results_count (int): Number of Brave Search results to fetch per entity (default: 30).
            disable_web_results_topup (bool): When True, an entity with any cached web_search_results
                is treated as fully cached and never re-fetched to reach web_results_count (default: False).
            max_workers (int): Maximum number of parallel workers for entity processing (default: 4).
            judge_api_url (str): Base URL for the judge API (overrides default selection).
            judge_api_key (str): API key for the judge API (overrides default selection).
            max_brave_calls (int): Maximum total Brave Search API calls across the run (default: 1000).
        """
        self.request = Request(llm_judge, judge_api_url=judge_api_url, judge_api_key=judge_api_key)
        self.llm_judge = llm_judge
        self.ground_truth_dir_path = ground_truth_dir_path or os.getcwd()
        self.web_results_count = web_results_count
        self.disable_web_results_topup = disable_web_results_topup
        self.max_workers = max_workers
        self._brave_calls = 0
        self._max_brave_calls = max_brave_calls
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

    def _search_web_guarded(self, entity_name: str, num_results: int, offset: int = 0) -> list:
        """Call search_web only if the global Brave API call budget has not been exhausted."""
        with self._cache_lock:
            if self._brave_calls >= self._max_brave_calls:
                logger.warning(
                    f"Brave Search limit ({self._max_brave_calls}) reached — skipping web search for '{entity_name}'"
                )
                return []
            self._brave_calls += 1
        return search_web(entity_name, num_results=num_results, offset=offset)

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
        # Check if entity is FULLY cached (has all required fields) AND already has
        # at least as many web results as currently requested (self.web_results_count).
        # If the cache has fewer web results than requested (e.g. corpus previously
        # built with 20 and now rebuilding with a higher target), fall through so the
        # web-fetching step below can extend the cache instead of skipping it.
        if entity_name in self.unified_cache:
            cached_entry = self.unified_cache[entity_name]
            if all(key in cached_entry for key in ["original_content", "triples"]):
                # Validate that triples list is not empty
                if cached_entry.get("triples") and len(cached_entry["triples"]) > 0:
                    # Also validate that web_search_results is not empty (to handle cases where Brave Search API failed with 402)
                    web_results = cached_entry.get("web_search_results")
                    if web_results and len(web_results) > 0:
                        if self.disable_web_results_topup or len(web_results) >= self.web_results_count:
                            #logger.info(f"Entity {entity_name} is fully cached, skipping all API and LLM calls")
                            return cached_entry
                        else:
                            logger.info(
                                f"Entity {entity_name} cached with {len(web_results)} web results, "
                                f"extending to target {self.web_results_count}"
                            )
                            # Fall through to fetch additional web results
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
        
        # Step 5: Fetch web search results page-by-page, extending any cached results up to
        # self.web_results_count, and checkpoint raw vs. LLM-deduped-novel triple counts after
        # every page so we can plot triples-vs-#docs curves (does raw growth reflect genuinely
        # new information, or just redundant restatements from lower-relevance results?).
        cached_entry = self.unified_cache.get(entity_name, {})
        cached_web_results = cached_entry.get("web_search_results") or []
        cached_full_documents = cached_entry.get("web_full_documents") or []
        cached_web_triples = cached_entry.get("web_triples") or []
        cached_checkpoints = cached_entry.get("web_checkpoints") or []

        page_size = 20
        web_results = list(cached_web_results)
        full_documents = list(cached_full_documents)
        web_triples = list(cached_web_triples)
        checkpoints = list(cached_checkpoints)

        # "Known" baseline for novelty judging: Wikipedia triples plus every raw web
        # triple already accounted for by prior checkpoints (whether it was itself
        # judged novel or a duplicate, it's still "already available" information).
        known_triples = list(triples)
        if checkpoints:
            # Resume from the last checkpoint: all web triples accounted for so far are "known"
            known_triples.extend(web_triples)
            novel_cumulative = checkpoints[-1]["novel_triples_cumulative"]
        elif cached_web_triples:
            # Old cache predates checkpointing: treat the cached batch as checkpoint 0
            logger.info(f"Backfilling checkpoint 0 for {entity_name} ({len(cached_web_results)} cached docs, no prior checkpoint metadata)")
            novel_flags = self.request.dedup_triples_llm(known_triples, cached_web_triples)
            novel_count = sum(novel_flags)
            known_triples.extend(cached_web_triples)
            checkpoints.append({
                "docs_cumulative": len(cached_web_results),
                "raw_triples_cumulative": len(cached_web_triples),
                "novel_triples_cumulative": novel_count,
                "new_docs": len(cached_web_results),
                "new_raw_triples": len(cached_web_triples),
                "new_novel_triples": novel_count,
            })
            novel_cumulative = novel_count
        else:
            novel_cumulative = 0

        already_have = len(web_results)
        seen_urls = {item.get("url", "") for item in web_results}
        start_page = already_have // page_size
        num_pages_needed = -(-self.web_results_count // page_size)  # ceil

        for page in range(start_page, num_pages_needed):
            if len(web_results) >= self.web_results_count:
                break
            page_results = self._search_web_guarded(entity_name, num_results=page_size, offset=page)
            if not page_results:
                logger.info(f"No more web results available for {entity_name} at page {page}, stopping pagination early")
                break

            batch = [item for item in page_results if item.get("url", "") not in seen_urls]
            for item in batch:
                seen_urls.add(item.get("url", ""))
            if not batch:
                if len(page_results) < page_size:
                    break
                continue

            logger.info(f"Fetching {len(batch)} new full documents (page {page}) for {entity_name}...")
            batch_full_documents = []
            for item in batch:
                url = item.get("url", "")
                if url:
                    full_content = fetch_full_document(url)
                    if full_content:
                        batch_full_documents.append({
                            "url": url,
                            "title": item.get("title", ""),
                            "content": full_content,
                            "word_count": len(full_content.split())
                        })

            batch_triples = self._extract_triples_from_web_results(batch, entity_name)
            novel_flags = self.request.dedup_triples_llm(known_triples, batch_triples) if batch_triples else []
            batch_novel_count = sum(novel_flags)

            web_results.extend(batch)
            full_documents.extend(batch_full_documents)
            web_triples.extend(batch_triples)
            known_triples.extend(batch_triples)
            novel_cumulative += batch_novel_count

            checkpoints.append({
                "docs_cumulative": len(web_results),
                "raw_triples_cumulative": len(web_triples),
                "novel_triples_cumulative": novel_cumulative,
                "new_docs": len(batch),
                "new_raw_triples": len(batch_triples),
                "new_novel_triples": batch_novel_count,
            })

            logger.info(
                f"Checkpoint for {entity_name}: {len(web_results)} docs -> "
                f"{len(web_triples)} raw triples, {novel_cumulative} novel triples cumulative "
                f"(+{len(batch_triples)} raw / +{batch_novel_count} novel this page)"
            )

            if len(page_results) < page_size:
                logger.info(f"Brave returned fewer than {page_size} results for {entity_name} at page {page}, stopping pagination early")
                break

        result["web_search_results"] = web_results
        result["web_search_count"] = len(web_results)
        result["web_full_documents"] = full_documents
        result["web_full_documents_count"] = len(full_documents)
        result["web_triples"] = web_triples
        result["web_triple_count"] = len(web_triples)
        result["web_checkpoints"] = checkpoints
        result["web_novel_triple_count"] = novel_cumulative

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
