import requests
import json
import time
import openai
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get base directory (parent of ENTITIES/DOMAINS directory)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_BASE_DIR, "execution.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

DOMAINS = [
    "person",
    "organization",
    "location",
    "event",
    "work_of_art",
    "artifact",
    "scientific_concept",
    "cultural_concept",
    "animal",
    "plant"
]

PER_DOMAIN = 100 #30
TOTAL_TITLES = PER_DOMAIN * len(DOMAINS)
BATCH_SIZE = 10
MAX_WORKERS = 5  # Parallel threads for API calls

OUTPUT_FILE = os.path.join(_BASE_DIR, "wikipedia_entities_by_domain_300.json")
LOG_FILE = os.path.join(_BASE_DIR, "problematic_entities_domains.log")
CHECKPOINT_FILE = os.path.join(_BASE_DIR, "checkpoint.json")
GOOD_ENTITIES_FILE = os.path.join(_BASE_DIR, "good_entities_partial.json")

MIN_LENGTH = 2000
MIN_STATEMENTS = 10

# LLM client (keeps original initialization)
client = openai.OpenAI(base_url="", api_key='')

# State
fetched_titles = set()
# domain -> list of candidate dicts
domain_candidates = {d: [] for d in DOMAINS}

HEADERS = {"User-Agent": "MyWikipediaClient/1.0"}


def domain_quota_met():
    return all(len(domain_candidates[d]) >= PER_DOMAIN for d in DOMAINS)


def save_checkpoint():
    """Save current progress to checkpoint file"""
    checkpoint = {
        'fetched_titles': list(fetched_titles),
        'domain_candidates': domain_candidates
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    logger.info(f"Checkpoint saved to {CHECKPOINT_FILE}")


def load_checkpoint():
    """Load progress from checkpoint file if it exists"""
    global fetched_titles, domain_candidates
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            fetched_titles = set(checkpoint.get('fetched_titles', []))
            domain_candidates = checkpoint.get('domain_candidates', {d: [] for d in DOMAINS})
            # Ensure all domains exist
            for d in DOMAINS:
                if d not in domain_candidates:
                    domain_candidates[d] = []
            logger.info(f"Checkpoint loaded. Current progress: {dict((d, len(domain_candidates[d])) for d in DOMAINS)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return False


def save_good_entities():
    """Save currently collected good entities to a separate file"""
    partial = {}
    for d in DOMAINS:
        partial[d] = domain_candidates.get(d, [])
    os.makedirs(os.path.dirname(GOOD_ENTITIES_FILE), exist_ok=True)
    with open(GOOD_ENTITIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)
    logger.info(f"Partial good entities saved to {GOOD_ENTITIES_FILE}")



def collect_random_batch(batch_size=BATCH_SIZE):
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": batch_size,
        "rnnamespace": 0,
    }
    logger.debug(f"Collecting random batch of size {batch_size}")
    r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    batch = data.get('query', {}).get('random', [])
    logger.debug(f"Received batch with {len(batch)} titles")
    return batch


def get_pages_info(pageids):
    params = {
        "action": "query",
        "format": "json",
        "pageids": "|".join(pageids),
        "prop": "info|categories|pageprops|templates",
        "cllimit": "max",
        "tllimit": "max",
    }
    r = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get('query', {}).get('pages', {})


def process_page_parallel(page_dict):
    """Process a single page in parallel: get wikidata statements"""
    pageid, page = page_dict
    try:
        wikibase_item = page.get('pageprops', {}).get('wikibase_item')
        if wikibase_item:
            statements = count_wikidata_statements(wikibase_item)
            return pageid, page, statements
    except Exception as e:
        logger.debug(f"Error getting statements for page {pageid}: {e}")
    return pageid, page, 0


def count_wikidata_statements(wikibase_item):
    try:
        params = {
            "action": "wbgetentities",
            "ids": wikibase_item,
            "props": "claims",
            "format": "json",
        }
        r = requests.get(WIKIDATA_API_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        ent = data.get('entities', {}).get(wikibase_item)
        if ent:
            return len(ent.get('claims', {}))
    except Exception:
        return 0
    return 0


def evaluate_with_llm(title):
    prompt = f"""Evaluate the Wikipedia entity '{title}' according to these criteria on a scale of 1-5:
- Informativeness: How informative is this entity/topic (1 means not informative, 5 means very informative)?
- Ambiguity: How ambiguous is the entity name (1 means very ambiguous, 5 means not ambiguous)?
- Suitability as factual ground truth for LLM evaluation benchmark: How suitable is this as a factual reference for testing LLMs (1 means not suitable, 5 means very suitable)?

Also assign this entity to exactly one of these domains (respond with the domain string): {DOMAINS}

If any score is less than 3, mark as problematic. Provide a brief reason.

Respond ONLY with valid JSON in this exact format, no extra text:
{{
    "informativeness": int,
    "ambiguity": int,
    "suitability": int,
    "problematic": bool,
    "reason": "string",
    "domain": "one of the domain strings from the list"
}}"""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip("```json").strip("```")
        if not content:
            return None
        return json.loads(content)
    except Exception:
        return None


def finalize_selection():
    logger.info("Finalizing entity selection")
    final = {}
    for d in DOMAINS:
        candidates = domain_candidates.get(d, [])
        # unique by title using dict to preserve order
        seen = {}
        for c in candidates:
            if c['title'] not in seen:
                seen[c['title']] = c
        uniq = list(seen.values())
        # take top 10 entities
        final[d] = uniq[:PER_DOMAIN]
        logger.info(f"Domain '{d}': selected {len(final[d])}/{PER_DOMAIN} entities")
    return final


# def generate_nonexistent_entities(count=10):
#     logger.info(f"Generating {count} non-existent entities")
#     prompt = f"""Generate {count} fictional, non-existent Wikipedia-style entity names (short, plausible proper-noun style titles) and a one-sentence description for each.
# Return ONLY valid JSON in this exact format (no extra text):
# [
#   {{"title": "string", "description": "string"}},
#   ...
# ]
# 
# The entities must be obviously fictional (not real people, places, companies, works, etc.)."""
#     try:
#         response = client.chat.completions.create(
#             model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=800,
#         )
#         content = response.choices[0].message.content.strip()
#         # try to extract JSON
#         try:
#             data = json.loads(content)
#             if isinstance(data, list):
#                 logger.info(f"Successfully generated {len(data[:count])} non-existent entities")
#                 return data[:count]
#         except Exception:
#             # try to find JSON substring
#             import re
#             m = re.search(r"\[\s*\{.*?\}\s*\]", content, re.S)
#             if m:
#                 try:
#                     data = json.loads(m.group(0))
#                     if isinstance(data, list):
#                         logger.info(f"Successfully generated {len(data[:count])} non-existent entities")
#                         return data[:count]
#                 except Exception:
#                     pass
#     except Exception as e:
#         logger.warning(f"Failed to generate non-existent entities: {e}")
# 
#     # Fallback: generate synthetic names
#     logger.info("Using fallback synthetic entity names")
#     fallback = []
#     for i in range(1, count + 1):
#         fallback.append({
#             "title": f"Fictitia-{i}",
#             "description": "Fictional entity created for evaluation purposes."
#         })
#     return fallback


def main(max_iterations=3000):
    logger.info("="*60)
    logger.info("Starting entity collection and evaluation")
    logger.info(f"Domains: {DOMAINS}")
    logger.info(f"Target: {PER_DOMAIN} entities per domain")
    logger.info("="*60)
    
    # Load checkpoint to resume from previous run
    load_checkpoint()
    
    iterations = 0
    with tqdm(total=TOTAL_TITLES, desc="Collecting and evaluating entities") as pbar:
        # Update progress bar to reflect loaded progress
        for d in DOMAINS:
            pbar.update(len(domain_candidates[d]))
        
        while not domain_quota_met() and iterations < max_iterations:
            # Check again in case quota was just met
            if domain_quota_met():
                break
            iterations += 1
            logger.info(f"Iteration {iterations} - Current candidates: {dict((d, len(domain_candidates[d])) for d in DOMAINS)}")
            batch = collect_random_batch(BATCH_SIZE)
            pageids = [str(p['id']) for p in batch]
            pages = get_pages_info(pageids)
            
            # Process pages in parallel for wikidata statements
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_page_parallel, item): item[0] for item in pages.items()}
                processed_pages = {}
                for future in as_completed(futures):
                    # Check if quota met after each batch
                    if domain_quota_met():
                        break
                    pageid, page, statements = future.result()
                    processed_pages[pageid] = (page, statements)
            
            for pageid, (page, statements) in processed_pages.items():
                if domain_quota_met():
                    break
                try:
                    title = page['title']
                    if title in fetched_titles:
                        logger.debug(f"Skipping {title}: already fetched")
                        continue
                    fetched_titles.add(title)
                    length = page.get('length', 0)
                    templates = [t.get('title', '') for t in page.get('templates', [])]
                    categories = [c.get('title', '') for c in page.get('categories', [])]
                    wikibase_item = page.get('pageprops', {}).get('wikibase_item')

                    # disambiguation check
                    is_disambig = any('disambiguation' in (t or '').lower() for t in templates) or any('disambig' in (t or '').lower() for t in templates)
                    if is_disambig:
                        logger.debug(f"Skipping {title}: disambiguation page")
                        continue
                    if length < MIN_LENGTH:
                        logger.debug(f"Skipping {title}: too short ({length} chars)")
                        continue
                    if not wikibase_item:
                        logger.debug(f"Skipping {title}: no wikibase item")
                        continue

                    if statements < MIN_STATEMENTS:
                        logger.debug(f"Skipping {title}: insufficient wikidata statements ({statements})")
                        continue

                    # LLM evaluation
                    logger.info(f"Evaluating {title} with LLM")
                    result = evaluate_with_llm(title)
                    if not result:
                        logger.warning(f"Failed to evaluate {title}")
                        continue
                    if result.get('problematic'):
                        logger.info(f"Marked as problematic: {title} - {result.get('reason')}")
                        with open(LOG_FILE, 'a', encoding='utf-8') as log:
                            log.write(json.dumps({
                                'title': title,
                                'reason': result.get('reason'),
                                'informativeness': result.get('informativeness'),
                                'ambiguity': result.get('ambiguity'),
                                'suitability': result.get('suitability')
                            }, ensure_ascii=False) + "\n")
                        continue

                    domain = result.get('domain')
                    if domain not in DOMAINS:
                        logger.warning(f"Invalid domain for {title}: {domain}")
                        continue

                    candidate = {
                        'title': title,
                        'length': length,
                        'wikibase_item': wikibase_item
                    }

                    # add candidate if domain quota not met
                    if len(domain_candidates[domain]) < PER_DOMAIN:
                        domain_candidates[domain].append(candidate)
                        logger.info(f"Added {title} to domain '{domain}' ({len(domain_candidates[domain])}/{PER_DOMAIN})")
                        pbar.update(1)
                except Exception as e:
                    # Log error but continue processing
                    logger.error(f"Error processing entity: {str(e)}", exc_info=True)
                    with open(LOG_FILE, 'a', encoding='utf-8') as log:
                        log.write(json.dumps({
                            'title': title if 'title' in locals() else 'unknown',
                            'error': str(e),
                            'type': 'processing_error'
                        }, ensure_ascii=False) + "\n")
                    continue
            
            # Save checkpoint and partial good entities after each iteration
            save_checkpoint()
            save_good_entities()

            time.sleep(0.01)

    final = finalize_selection()
    # # Add generated non-existent entities
    # nonexist = generate_nonexistent_entities(count=10)
    # final['non-existant'] = nonexist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved domain-organized entities to {OUTPUT_FILE}")
    # Save final good entities file
    save_good_entities()
    logger.info("Execution completed successfully")


if __name__ == '__main__':
    main()
    