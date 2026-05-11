import os
import requests
import json
from loguru import logger
from typing import Optional, Dict
from .network_utils import network_retry

"""
Helper methods for retrieving and processing Wikipedia articles for entities.
"""

@network_retry(max_retries=5, initial_delay=1.0)
def get_wikipedia_article(entity_name: str, language: str = "en") -> Optional[Dict]:
    """
    Retrieve Wikipedia article content for a given entity name.
    
    Args:
        entity_name (str): The name of the entity to search for.
        language (str): The language code for Wikipedia (default is "en").
    
    Returns:
        dict: Contains 'title', 'content', and 'url' if found, None otherwise.
    """
    
    url = f"https://{language}.wikipedia.org/w/api.php"
    
    headers = {
        "User-Agent": "MyWikipediaClient/1.0 (anonymous@example.com)"
    }
    
    # First, search for the article
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": entity_name,
        "srnamespace": 0,
    }
    
    try:
        logger.info(f"Searching Wikipedia for entity: {entity_name}")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        search_result = response.json()
        
        if not search_result.get("query", {}).get("search"):
            logger.warning(f"No Wikipedia article found for entity: {entity_name}")
            return None
        
        # Get the first search result
        article_title = search_result["query"]["search"][0]["title"]
        #logger.info(f"Found article: {article_title}")
        
        # Now fetch the full article content
        params = {
            "action": "query",
            "format": "json",
            "titles": article_title,
            "prop": "extracts",
            "explaintext": True,
        }
        
        logger.debug(f"Fetching full article for: {article_title}")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        page = pages[page_id]
        
        if "extract" in page:
            article_url = f"https://{language}.wikipedia.org/wiki/{article_title.replace(' ', '_')}"
            content_length = len(page["extract"])
            word_count = len(page["extract"].split())
            #logger.info(f"Successfully retrieved Wikipedia article: {article_title} ({content_length} chars, {word_count} words)")
            return {
                "title": article_title,
                "content": page["extract"],
                "url": article_url,
            }
        else:
            logger.warning(f"Could not extract content from Wikipedia for: {entity_name}")
            return None
            
    except requests.RequestException as e:
        logger.error(f"Network error fetching Wikipedia article for {entity_name}: {e}", exc_info=True)
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Parse error processing Wikipedia response for {entity_name}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Wikipedia article for {entity_name}: {e}", exc_info=True)
        return None


def save_unified_wikipedia_cache(cache_dict: Dict, cache_file_path: str):
    """
    Save unified Wikipedia cache containing articles, shortened versions, and extracted triples.
    
    Args:
        cache_dict (dict): Dictionary mapping entity names to complete Wikipedia data.
        cache_file_path (str): Path to save the unified cache file.
    """
    try:
        # Create directory if it doesn't exist
        cache_dir = os.path.dirname(cache_file_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.debug(f"Created cache directory: {cache_dir}")
        
        with open(cache_file_path, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=4)
        cache_size = len(json.dumps(cache_dict)) / (1024 * 1024)  # Size in MB
        logger.info(f"Unified ground truth saved: {cache_file_path} ({len(cache_dict)} entities, {cache_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save unified ground truth to {cache_file_path}: {e}", exc_info=True)


def load_unified_wikipedia_cache(cache_file_path: str) -> Dict:
    """
    Load unified Wikipedia cache from file.
    Unified cache contains original articles, shortened versions, and extracted triples.
    
    Args:
        cache_file_path (str): Path to the cache file.
    
    Returns:
        dict: Cached Wikipedia data, or empty dict if file doesn't exist.
    """
    try:
        with open(cache_file_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Loaded unified Wikipedia cache: {cache_file_path} ({len(cache)} entities)")
        return cache
    except FileNotFoundError:
        logger.debug(f"Unified Wikipedia cache file not found: {cache_file_path} (this is normal on first run)")
        return {}
    except Exception as e:
        logger.error(f"Failed to load unified Wikipedia cache from {cache_file_path}: {e}", exc_info=True)
        return {}


def word_count(text: str) -> int:
    """
    Calculate the word count of a text.
    
    Args:
        text (str): The text to count words in.
    
    Returns:
        int: The number of words.
    """
    return len(text.split())


# Brave Search API configuration
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "YOUR_BRAVE_API_KEY_HERE")


@network_retry(max_retries=5, initial_delay=1.0)
def search_web(entity_name: str, num_results: int = 10) -> list:
    """
    Search the web for an entity using Brave Search API.
    
    Args:
        entity_name (str): The name of the entity to search for.
        num_results (int): Number of search results to return (default: 10). Maximum is 20 for Brave Search API.
    
    Returns:
        list: List of dicts with 'title', 'url', 'snippet' keys, or empty list if failed.
    """
    # Cap num_results to maximum supported by Brave Search API
    MAX_BRAVE_RESULTS = 20
    if num_results > MAX_BRAVE_RESULTS:
        logger.warning(f"Brave Search API supports max {MAX_BRAVE_RESULTS} results, capping from {num_results}")
        num_results = MAX_BRAVE_RESULTS
    
    if BRAVE_API_KEY == "YOUR_BRAVE_API_KEY_HERE":
        logger.warning("Brave API key not configured. Set BRAVE_API_KEY environment variable.")
        return []
    
    url = "https://api.search.brave.com/res/v1/web/search"
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    params = {
        "q": entity_name,
        "count": num_results,
    }
    
    try:
        logger.info(f"Searching web for: {entity_name}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        logger.info(f"Brave Search API response status: {response.status_code} for {entity_name}")
        response.raise_for_status()
        
        # Handle empty responses
        if not response.text:
            logger.warning(f"Brave Search API returned empty response for {entity_name}")
            return []
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            # Check if response is HTML (likely API key/authentication issue)
            if response.text.strip().startswith('<'):
                logger.error(f"Brave Search API returned HTML (possible invalid API key or authentication issue) for {entity_name}")
                logger.error(f"Response content (first 500 chars): {response.text[:500]}")
            else:
                logger.error(f"Brave Search API returned invalid JSON for {entity_name}: {e}")
                logger.error(f"Response content (first 500 chars): {response.text[:500] if response.text else 'empty'}")
            return []
        
        results = []
        web_results = data.get("web", {}).get("results", [])
        
        for item in web_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", "")
            })
        
        logger.info(f"Found {len(results)} web search results for: {entity_name}")
        return results
        
    except requests.RequestException as e:
        logger.error(f"Brave Search API error for {entity_name}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error in web search for {entity_name}: {e}", exc_info=True)
        return []


@network_retry(max_retries=5, initial_delay=1.0)
def fetch_full_document(url: str, max_length: int = 50000) -> str:
    """
    Fetch the full content of a web page and extract text.
    
    Args:
        url (str): The URL of the web page to fetch.
        max_length (int): Maximum number of characters to return (default: 50000).
    
    Returns:
        str: The extracted text content, or empty string if fetch fails.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Try to extract text content
        content = response.text
        
        # Simple HTML stripping - remove scripts, styles, and tags
        import re
        # Remove script and style elements
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        # Decode HTML entities
        import html
        content = html.unescape(content)
        
        # Truncate if too long
        if len(content) > max_length:
            content = content[:max_length]
        
        return content.strip()
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch document from {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error processing document from {url}: {e}")
        return ""
