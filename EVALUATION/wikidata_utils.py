import requests
import re
from tqdm import tqdm 
import torch
from sentence_transformers import SentenceTransformer, util
from loguru import logger
import json

"""

py file containing helper methods for fetching data from wikidata for entities
"""

def get_wikidata_entity_id_from_wikipedia_title(wikipedia_title: str, language: str = 'en') -> str:
    """
    Get Wikidata entity ID from a Wikipedia article title using Wikipedia's API.
    This is more reliable than name search when there are disambiguation differences.
    
    Args:
        wikipedia_title (str): The Wikipedia article title.
        language (str): The language code for Wikipedia (default is "en").
    
    Returns:
        str: The Wikidata entity ID (e.g., "Q42"), or None if not found.
    """
    url = f"https://{language}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "MyWikidataClient/1.0 (anonymous@example.com)"
    }
    
    params = {
        "action": "query",
        "titles": wikipedia_title,
        "prop": "pageprops",
        "format": "json",
        "redirects": 1  # Follow redirects to get final page
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.debug(f"Failed to fetch Wikipedia page props. HTTP Status: {response.status_code}")
            return None
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":  # Page not found
                logger.debug(f"Wikipedia page not found: {wikipedia_title}")
                return None
            
            # Get wikibase_item property which contains the Wikidata ID
            wikibase_item = page_data.get("pageprops", {}).get("wikibase_item")
            if wikibase_item:
                #logger.info(f"Found Wikidata ID {wikibase_item} for entity: {wikipedia_title}")
                return wikibase_item
        
        logger.debug(f"No Wikidata item found for Wikipedia title: {wikipedia_title}")
        return None
    
    except Exception as e:
        logger.error(f"Error getting Wikidata ID from Wikipedia title {wikipedia_title}: {e}")
        return None


def extract_base_name(entity_name: str) -> str:
    """
    Extract base name from entity name by removing disambiguation suffixes.
    E.g., "Splendour (apple)" -> "Splendour"
    
    Args:
        entity_name (str): The entity name potentially with disambiguation.
    
    Returns:
        str: The base name without disambiguation.
    """
    # Remove content in parentheses like (apple), (film), etc.
    base_name = re.sub(r'\s*\([^)]*\)', '', entity_name).strip()
    return base_name


def get_wikidata_entity_id(entity_name, language='en'):
    """
    Get Wikidata entity ID for an entity name, with fallback strategies for disambiguation.
    
    Tries multiple approaches:
    1. Wikipedia interlinking (most reliable when Wikipedia page exists)
    2. Exact name search on Wikidata
    3. Base name search (removing disambiguation like " (apple)")
    
    Args:
        entity_name (str): The entity name to search for.
        language (str): The language code (default is 'en').
    
    Returns:
        str: The Wikidata entity ID, or None if not found.
    """
    
    # Strategy 1: Try Wikipedia interlinking first (most reliable)
    # This works even when Wikipedia and Wikidata have different names
    wikidata_id = get_wikidata_entity_id_from_wikipedia_title(entity_name, language)
    if wikidata_id:
        logger.info(f"Found Wikidata ID {wikidata_id} via Wikipedia interlink for: {entity_name}")
        return wikidata_id
    
    # If entity_name has disambiguation, also try with base name via Wikipedia
    base_name = extract_base_name(entity_name)
    if base_name and base_name != entity_name:
        wikidata_id = get_wikidata_entity_id_from_wikipedia_title(base_name, language)
        if wikidata_id:
            logger.info(f"Found Wikidata ID {wikidata_id} via Wikipedia interlink using base name '{base_name}' for: {entity_name}")
            return wikidata_id
    
    # Strategy 2: Try exact name search on Wikidata
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": language,
        "format": "json"
    }

    headers = {
        "User-Agent": "MyWikidataClient/1.0 (anonymous@example.com)"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        results = response.json().get("search", [])
        if results:
            logger.info(f"Found Wikidata ID {results[0].get('id')} for exact match: {entity_name}")
            return results[0].get("id")
    else:
        logger.warning(
            f"Failed to fetch data. HTTP Status Code: {response.status_code}, Response: {response.text}"
        )
    
    # Strategy 3: Try with base name (remove disambiguation like " (apple)")
    if base_name and base_name != entity_name:
        logger.info(f"Trying base name '{base_name}' for entity: {entity_name}")
        params["search"] = base_name
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            results = response.json().get("search", [])
            if results:
                logger.info(f"Found Wikidata ID {results[0].get('id')} using base name '{base_name}' for: {entity_name}")
                return results[0].get("id")
    
    logger.warning(f"No Wikidata ID found for entity: {entity_name}")
    return None



def sanity_check_entity(entity_name, language = 'en'):
    """
    Given an entity string check if it exists on wikidata
    """

    if get_wikidata_entity_id(entity_name)!= None:
        return True
    else:
        return False


def fetch_wikidata_claims(entity_id):
    """
    Fetches properties for claims given a Wikidata entity using its ID.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    headers = {
        "User-Agent": "MyWikidataClient/1.0 (anonymous@example.com)"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch entity data for {entity_id}")
    data = response.json()
    claims = data['entities'][entity_id]['claims']
    properties = {}
    for prop, values in claims.items():
        properties[prop] = [v['mainsnak']['datavalue']['value'] for v in values if 'datavalue' in v['mainsnak']]

    return properties


def get_wikidata_entity_name(entity_id, language="en"):
    """
    Fetches the name (label) of a Wikidata entity given its ID.
    
    Args:
        entity_id (str): The Wikidata entity ID (e.g., "Q42").
        language (str): The language code for the label (default is "en").
    
    Returns:
        str: The name (label) of the entity or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    headers = {
        "User-Agent": "MyWikidataClient/1.0 (anonymous@example.com)"
    }

    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "languages": language,
        "props": "labels",
        "format": "json"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        entities = data.get("entities", {})
        if entity_id in entities:
            labels = entities[entity_id].get("labels", {})
            if language in labels:
                return labels[language].get("value")
        logger.debug(f"No label found for entity ID: {entity_id}")
        return None
    else:
        logger.info(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return None



def get_wikidata_property_label(property_id, language="en"):
    """
    Fetches the label of a Wikidata property given its ID.
    
    Args:
        property_id (str): The Wikidata property ID (e.g., "P31").
        language (str): The language code for the label (default is "en").
    
    Returns:
        str: The label of the property or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    headers = {
        "User-Agent": "MyWikidataClient/1.0 (anonymous@example.com)"
    }
    params = {
        "action": "wbgetentities",
        "ids": property_id,
        "languages": language,
        "props": "labels",
        "format": "json"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        entities = data.get("entities", {})
        if property_id in entities:
            labels = entities[property_id].get("labels", {})
            if language in labels:
                return labels[language].get("value")
        logger.debug(f"No label found for property ID: {property_id}")
        return None
    else:
        logger.debug(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return None


def convert_wikidata_claims_to_triples(wikidata_claims, current_subject, exp_output_type = 'str'):
    all_triple_wikidata_claims = []
    for prop, values in tqdm(wikidata_claims.items(), desc = "Matching triples"):
        for value in values:
            if type(value) == dict:
                prop_label = get_wikidata_property_label(prop)
                if 'id' in value:
                    referred_entity = get_wikidata_entity_name(value['id'])
                    if prop_label!=None and referred_entity!=None:
                        #curr_wikidata_claim_triple = [current_subject, prop_label, referred_entity]
                        curr_wikidata_claim_triple = {'subject': current_subject, 'predicate': prop_label, 'object': referred_entity}
                        curr_wikidata_claim_string = f"({current_subject}, {prop_label}, {referred_entity})"
                        if exp_output_type == 'str':
                            all_triple_wikidata_claims.append(curr_wikidata_claim_string)
                        else:
                            all_triple_wikidata_claims.append(curr_wikidata_claim_triple)

    
    return all_triple_wikidata_claims


def get_wikidata_triples_for_entity(entity_name: str, language: str = 'en') -> dict:
    """
    Get Wikidata triples for a single entity.
    
    Args:
        entity_name (str): The name of the entity to look up.
        language (str): The language code for labels (default: 'en').
    
    Returns:
        dict: Contains 'wikidata_id', 'wikidata_triples' (list of dicts), 
              'raw_claims', 'wikidata_url', or None if entity not found.
    """
    # Get Wikidata ID
    entity_id = get_wikidata_entity_id(entity_name, language)
    if not entity_id:
        logger.warning(f"No Wikidata ID found for entity: {entity_name}")
        return None
    
    # Fetch claims
    try:
        claims = fetch_wikidata_claims(entity_id)
    except Exception as e:
        logger.error(f"Failed to fetch Wikidata claims for {entity_name} (ID: {entity_id}): {e}")
        return None
    
    # Convert to readable triples
    triples = convert_wikidata_claims_to_triples(claims, entity_name, 'dict')
    
    # Build Wikidata URL
    wikidata_url = f"https://www.wikidata.org/wiki/{entity_id}"
    
    return {
        'wikidata_id': entity_id,
        'wikidata_triples': triples,
        'raw_claims': claims,
        'wikidata_url': wikidata_url,
        'triple_count': len(triples)
    }



def soft_match_triples_with_claims(current_triple, wikidata_claims, threshold_score = 0.8):
    """
    Matches a list of triples with Wikidata claims using semantic similarity.
    If the asked triple matches if any of the wikidata claims then it can be considered plausible
    return True else False

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    triple_texts = [f"{current_triple['subject']} {current_triple['predicate']} {current_triple['object']}"]
    triple_embeddings = model.encode(triple_texts, convert_to_tensor=True, device = device)

    # variable for storing all wikidata claims converted to triple format
    all_triple_wikidata_claims = []

    for prop, values in tqdm(wikidata_claims.items(), desc = "Matching triples"):
        for value in values:
            if type(value) == dict:
                prop_label = get_wikidata_property_label(prop)
                if 'id' in value:
                    referred_entity = get_wikidata_entity_name(value['id'])
                    if prop_label!=None and referred_entity!=None:
                        curr_wikidata_claim_triple = [current_triple['subject'], prop_label, referred_entity]
                        curr_wikidata_claim_string = f"{current_triple['subject']} {prop_label} {referred_entity}"
                        all_triple_wikidata_claims.append(curr_wikidata_claim_string)
    

    claim_embedding = model.encode(all_triple_wikidata_claims, convert_to_tensor=True, device = device)
    similarities = util.pytorch_cos_sim(triple_embeddings, claim_embedding)

    # record indices for matching triples semantically
    indices = (similarities > threshold_score).nonzero(as_tuple=True)
    values = similarities[indices]

    if len(values) > 0:
        return True
    
    return False


def soft_match_utils(raw_triples):
    """
    Perform soft matching for triples generated with elicitation prompt on claims scraped from wikidata
    """ 

    # list recording plausible claims 
    plausible_triples = []

    for each_triple in raw_triples:
        subject_entity_id = get_wikidata_entity_id(each_triple['subject'])
        wikidata_collection_claims = fetch_wikidata_claims(subject_entity_id)
        if soft_match_triples_with_claims(each_triple, wikidata_collection_claims) == True:
            plausible_triples.append(each_triple)
    

    logger.info(f"Collection of plausible triples: {plausible_triples}")
    return plausible_triples


def create_gold_triples_file(data_triples, gold_file_path):

    """
    Create gold triples file so that every time eval framework is used, web api lookup can be prevented
    """
    
    gold_triples = dict()
    for each_triple in data_triples:
        if each_triple['subject'] not in gold_triples:
            subject_entity_id = get_wikidata_entity_id(each_triple['subject'])
            if subject_entity_id is not None:
                wikidata_claims = fetch_wikidata_claims(subject_entity_id)
                all_triples_wikidata = convert_wikidata_claims_to_triples(wikidata_claims, each_triple['subject'], 'dict')
                gold_triples[each_triple['subject']] = all_triples_wikidata
    
    with open(gold_file_path, "w") as json_file:
        json.dump(gold_triples, json_file, indent=4)

    logger.info(f"Gold triples written to {gold_file_path}")

