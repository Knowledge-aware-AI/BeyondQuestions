import requests
import json
import time
import openai
import os
import signal
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Get base directory (ENTITIES directory)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants and configuration
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
TOTAL_TITLES = 10000 #100
BATCH_SIZE = 10  # Wikipedia allows up to 10 random pages per request for non-bots
OUTPUT_FILE = os.path.join(_BASE_DIR, "10000_random_entities_wikipedia.json")
LOG_FILE = os.path.join(_BASE_DIR, "problematic_entities.log")
CHECKPOINT_FILE = os.path.join(_BASE_DIR, "checkpoint.json")

# Thresholds
MIN_LENGTH = 2000  # minimum article length in characters
MIN_STATEMENTS = 10  # minimum number of statements on Wikidata

# --- state management ------------------------------------------------------

def save_state():
    """Persist the current progress into the checkpoint file."""
    state = {
        "good_entities": good_entities,
        "fetched_titles": list(fetched_titles),
    }
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as cp:
            json.dump(state, cp, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


def load_state():
    """Load progress from the checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as cp:
                state = json.load(cp)
            return state.get("good_entities", []), set(state.get("fetched_titles", []))
        except Exception as e:
            print(f"Failed to load checkpoint, starting fresh: {e}")
    return [], set()


def save_and_exit(signum, frame):
    """Signal handler to save state before exiting."""
    print(f"Received signal {signum}, saving state and exiting...")
    save_state()
    sys.exit(0)

# register signal handlers so that ctrl-c or termination saves progress
signal.signal(signal.SIGINT, save_and_exit)
signal.signal(signal.SIGTERM, save_and_exit)

# Initialize
good_entities, fetched_titles = load_state()
if good_entities:
    print(f"Resuming from checkpoint: {len(good_entities)} entities already collected.")


# LLM client
client = openai.OpenAI(base_url="", api_key='')

with tqdm(total=TOTAL_TITLES, desc="Collecting and evaluating entities", initial=len(good_entities)) as pbar:
    while len(good_entities) < TOTAL_TITLES:
        batch = []
        
        # Collect a batch of entities that pass hard constraints
        while len(batch) < BATCH_SIZE:
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnlimit": BATCH_SIZE,
                "rnnamespace": 0  # main/article namespace only
            }

            headers = {
                "User-Agent": "MyWikipediaClient/1.0"
            }

            response = requests.get(WIKI_API_URL, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            random_pages = data["query"]["random"]
            pageids = [str(page["id"]) for page in random_pages]

            # Get details for these pages
            params2 = {
                "action": "query",
                "format": "json",
                "pageids": "|".join(pageids),
                "prop": "info|categories|pageprops|templates",
                "cllimit": "max",
                "tllimit": "max"
            }

            response2 = requests.get(WIKI_API_URL, params=params2, headers=headers)
            response2.raise_for_status()

            data2 = response2.json()
            pages = data2["query"]["pages"]

            for pageid, page in pages.items():
                title = page["title"]
                if title in fetched_titles:
                    continue  # already processed
                length = page.get("length", 0)
                categories = [cat["title"] for cat in page.get("categories", [])]
                templates = [t["title"] for t in page.get("templates", [])]
                wikibase_item = page.get("pageprops", {}).get("wikibase_item")

                # Check if disambiguation page
                is_disambig = (
                    any("disambiguation" in t.lower() for t in templates) or
                    any("disambig" in t.lower() for t in templates) or
                    any("dab" in t.lower() for t in templates) or
                    "Category:Disambiguation pages" in categories or
                    "Category:All disambiguation pages" in categories
                )

                if is_disambig:
                    print(f"Skipping disambiguation page: {title}")
                    fetched_titles.add(title)
                    save_state()
                    continue

                if length < MIN_LENGTH:
                    print(f"Skipping short article: {title} (length: {length})")
                    fetched_titles.add(title)
                    save_state()
                    continue

                statements = 0
                if wikibase_item:
                    # Query Wikidata for number of statements
                    wd_params = {
                        "action": "wbgetentities",
                        "ids": wikibase_item,
                        "props": "claims",
                        "format": "json"
                    }
                    wd_response = requests.get(WIKIDATA_API_URL, params=wd_params, headers=headers)
                    wd_response.raise_for_status()
                    wd_data = wd_response.json()
                    entity = wd_data["entities"].get(wikibase_item)
                    if entity:
                        statements = len(entity.get("claims", {}))
                        print(f"Article: {title}, Wikidata item: {wikibase_item}, Statements: {statements}")
                else:
                    print(f"Article: {title} has no Wikidata item.")
                    fetched_titles.add(title)
                    save_state()
                    continue  # No Wikidata item, skip

                if statements < MIN_STATEMENTS:
                    print(f"Skipping article with few statements: {title} (statements: {statements})")
                    fetched_titles.add(title)
                    save_state()
                    continue

                # Passed hard constraints
                batch.append({"title": title, "length": length})
                fetched_titles.add(title)
                save_state()
                if len(batch) >= BATCH_SIZE:
                    break

            # Be polite to the APIs
            time.sleep(0.1)
            if len(batch) >= BATCH_SIZE:
                break

        # Now evaluate the batch
        for entity in batch:
            title = entity["title"]
            prompt = f"""Evaluate the Wikipedia entity '{title}' according to these criteria on a scale of 1-5:
- Informativeness: How informative is this entity/topic (1 means not informative, 5 means very informative)?
- Ambiguity: How ambiguous is the entity name (1 means very ambiguous, 5 means not ambiguous)?
- Suitability as factual ground truth for LLM evaluation benchmark: How suitable is this as a factual reference for testing LLMs (1 means not suitable, 5 means very suitable)?

If any score is less than 3, mark as problematic. Provide a brief reason.

Respond ONLY with valid JSON in this exact format, no extra text:
{{
    "informativeness": int,
    "ambiguity": int,
    "suitability": int,
    "problematic": bool,
    "reason": "string"
}}"""

            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip("```json").strip("```")
                if not content:
                    print(f"Empty response for {title}")
                    continue
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as je:
                    print(f"Invalid JSON for {title}: {je}. Content: {content[:200]}...")
                    continue
                if result.get('problematic', False):
                    print(f"Marking problematic entity: {title} Reason: {result.get('reason')}")
                    # Log problematic
                    with open(LOG_FILE, "a", encoding="utf-8") as log:
                        log_entry = {
                            "title": title,
                            "informativeness": result.get('informativeness'),
                            "ambiguity": result.get('ambiguity'),
                            "suitability": result.get('suitability'),
                            "reason": result.get('reason')
                        }
                        log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                else:
                    good_entities.append(entity)
                    # save progress immediately after a successful addition
                    save_state()
                    print(f"Kept entity: {title}, length: {entity['length']}")
                    pbar.update(1)
                    if len(good_entities) >= TOTAL_TITLES:
                        break
            except Exception as e:
                print(f"Error evaluating {title}: {e}")
                continue

# Save to JSON
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(good_entities, f, ensure_ascii=False, indent=2)

print(f"Saved {len(good_entities)} Wikipedia article titles to {OUTPUT_FILE}")

# final checkpoint cleanup – we don't need the interim file any more
try:
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Removed checkpoint file {CHECKPOINT_FILE}")
except Exception as e:
    print(f"Could not remove checkpoint file: {e}")

# Plot the article lengths
lengths = [e["length"] for e in good_entities]
plt.figure(figsize=(10,6))
plt.bar(range(len(lengths)), lengths)
plt.xlabel('Entity Index')
plt.ylabel('Article Length (characters)')
plt.title('Article Lengths of Kept Entities')
plt.savefig(os.path.join(_BASE_DIR, 'article_lengths_plot.png'))
print("Plot saved to", os.path.join(_BASE_DIR, 'article_lengths_plot.png'))

# Plot the article lengths sorted from shortest to longest
plt.figure(figsize=(10,6))
sorted_lengths = sorted(lengths)
plt.bar(range(len(sorted_lengths)), sorted_lengths)
plt.xlabel('Entity Index (sorted)')
plt.ylabel('Article Length (characters)')
plt.title('Article Lengths of Kept Entities (Sorted from Shortest to Longest)')
plt.savefig(os.path.join(_BASE_DIR, 'article_lengths_plot_sorted.png'))
print("Sorted plot saved to", os.path.join(_BASE_DIR, 'article_lengths_plot_sorted.png'))