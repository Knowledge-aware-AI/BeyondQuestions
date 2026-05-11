#!/usr/bin/env python3
"""
Script to retrieve RAG passages from cached ground truth data for a given entity.
Usage: python retrieve_passages.py <entity_name> [ground_truth_dir] [k]

Example: python retrieve_passages.py "Notgrove railway station" "ground_truth/random/200" 10
"""

import os
import sys
import pickle
import argparse
import json

# Get base directory (parent of OKBENCH directory)
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_gt = os.path.join(_base_dir, "ground_truth", "random", "200")

def get_cache_path(ground_truth_dir, entity_name):
    """Get the cache path for an entity."""
    entity_sanitized = entity_name.replace(" ", "_").replace("/", "_")
    return os.path.join(ground_truth_dir, "rag_cache", entity_sanitized)

def load_passages(cache_path):
    """Load passages from the cache."""
    passages_file = os.path.join(cache_path, "passages.pkl")
    if not os.path.exists(passages_file):
        return None
    
    with open(passages_file, 'rb') as f:
        return pickle.load(f)

def retrieve_passages_for_entity(entity_name, ground_truth_dir, k=5):
    """
    Retrieve top-k passages for a given entity from the cached RAG index.
    
    Args:
        entity_name: Name of the entity to retrieve passages for
        ground_truth_dir: Path to the ground truth directory containing rag_cache
        k: Number of top passages to retrieve (default: 5)
    
    Returns:
        List of top-k passage dicts or None if not found
    """
    cache_path = get_cache_path(ground_truth_dir, entity_name)
    
    if not os.path.exists(cache_path):
        print(f"Cache not found for entity: {entity_name}")
        print(f"Expected path: {cache_path}")
        return None
    
    passages = load_passages(cache_path)
    if passages is None:
        print(f"Could not load passages for: {entity_name}")
        return None
    
    print(f"Found {len(passages)} total passages for: {entity_name}")
    print(f"Showing top {k} passages:\n")
    
    for i, p in enumerate(passages[:k], 1):
        print(f"--- Passage {i} [{p.get('source', 'unknown').upper()}] ---")
        content = p.get('content', '')
        # Truncate long passages for display
        if len(content) > 1000:
            content = content[:1000] + "..."
        print(content)
        print()
    
    return passages[:k]

def main():
    parser = argparse.ArgumentParser(description="Retrieve RAG passages from cached ground truth")
    parser.add_argument("entity", help="Entity name to retrieve passages for")
    parser.add_argument("ground_truth_dir", nargs="?", default=_default_gt,
                        help=f"Path to ground truth directory (default: {_default_gt})")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of top passages to retrieve (default: 10)")
    parser.add_argument("-a", "--all", action="store_true", help="Show all passages instead of just top-k")
    
    args = parser.parse_args()
    
    passages = retrieve_passages_for_entity(args.entity, args.ground_truth_dir, args.top_k if not args.all else 1000)

if __name__ == "__main__":
    main()