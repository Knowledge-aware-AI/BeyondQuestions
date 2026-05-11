from openai import OpenAI
import json
import os
from loguru import logger
from typing import Dict, List
from .network_utils import network_retry

class Request:
    """
    Handles LLM-based verification of RDF triples.
    
    Provides methods to verify triples against different sources:
    - Single snippet (used for recall)
    - Wikidata triples (used for recall)
    - Retrieved RAG passages (used for precision with new RAG-based approach)
    """
    
    def __init__(self, llm_judge, max_tokens = 3000):
        """
        Initialize the Request handler.
        
        Args:
            llm_judge: Name of the LLM model to use for verification
            max_tokens: Maximum tokens in LLM response (default: 3000)
        """
        self.llm_judge = llm_judge
        self.max_tokens = max_tokens

        if llm_judge.startswith("gpt-") and 'oss' not in llm_judge:
            self.client = OpenAI()
        else:
            self.client = OpenAI(base_url=os.getenv("SCADSAI_BASE_URL"), api_key=os.getenv("SCADSAI_API_KEY"))

    @network_retry(max_retries=6, initial_delay=1.0)
    def verify_triple_lm_snippet(self, triple, snippet):
        """
        Verify an RDF triple against a text snippet.
        
        Used for recall computation (checking if GT triples are covered by elicited triples).
        
        Args:
            triple: RDF triple string in format "(subject, predicate, object)"
            snippet: Text snippet to verify against
            
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
                  a=entailment, b=contradiction, c=neutral
        """

        triple_prompt_str = f"Statement to verify: {triple}."
        snippet_prompt_str = f"Snippet to verify from: {snippet}"
        
        # Define the JSON schema for structured output
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against a snippet",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given snippet? \
                            a) The snippet entails the RDF triple.\
                            b) The snippet contradicts the RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the snippet alone. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": snippet_prompt_str},
        ]
        
        logger.debug("=== LLM CALL: verify_triple_lm_snippet ===")
        logger.debug(f"Triple: {triple}")
        logger.debug(f"Snippet length: {len(snippet)} characters")
        logger.debug(f"Model: {self.llm_judge}")
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format=response_schema
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")
        
        logger.debug(f"Snippet verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result
    

    @network_retry(max_retries=6, initial_delay=1.0)
    def verify_triple_lm_wikidata(self, triple, gold_triples):
        """
        Verify an RDF triple against a list of Wikidata triples.
        
        Used for recall computation (checking if GT triples are covered by elicited triples).
        
        Args:
            triple: RDF triple string in format "(subject, predicate, object)"
            gold_triples: List of gold RDF triples to verify against
            
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
                  a=entailment, b=contradiction, c=neutral
        """

        triple_prompt_str = f"Statement to verify: {triple}."
        gold_prompt_str = f"List of triples to verify from: {str(gold_triples)}"
        
        # Define the JSON schema for structured output
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against a list of triples",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given list of triples? \
                            a) The list of triples entails the given RDF triple.\
                            b) The list of triples contradicts the given RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the list of triples alone. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": gold_prompt_str},
        ]

        logger.debug("=== LLM CALL: verify_triple_lm_wikidata ===")
        logger.debug(f"Triple to verify: {triple}")
        logger.debug(f"Gold triples length: {len(str(gold_triples))} characters")
        logger.debug(f"Model: {self.llm_judge}")
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format=response_schema
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")
        
        logger.debug(f"Wikidata verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result

    @network_retry(max_retries=5, initial_delay=1.0)
    def shorten_wikipedia_text(self, text: str, max_words: int = 1000) -> str:
        """
        Shorten a Wikipedia article text to approximately max_words using an LLM.
        
        Args:
            text (str): The full Wikipedia text to shorten.
            max_words (int): Target word count for the shortened text.
        
        Returns:
            str: The shortened text.
        """
        messages = [
            {"role": "user", "content": f"Please summarize the following text to approximately {max_words} words. Keep the most important information and maintain factual accuracy:\n\n{text}"}
        ]
        
        input_word_count = len(text.split())
        logger.debug("=== LLM CALL: shorten_wikipedia_text ===")
        logger.debug(f"Target words: {max_words}")
        logger.debug(f"Input text length: {len(text)} characters, {input_word_count} words")
        logger.debug(f"Model: {self.llm_judge}")
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.llm_judge,
                max_tokens=max_words + 200,  # Allow some buffer
                temperature=0.3,
            )
            
            result = response.choices[0].message.content
            output_word_count = len(result.split())
            # logger.debug(f"Output text length: {len(result)} characters, {output_word_count} words")
            logger.debug(f"Wikipedia text shortened: {input_word_count} => {output_word_count} words")
            
            return result
        except Exception as e:
            logger.error(f"Error in shorten_wikipedia_text: {e}", exc_info=True)
            return text
    
    @network_retry(max_retries=5, initial_delay=1.0)
    def shorten_web_document_text(self, text: str, max_words: int = 1000) -> str:
        """
        Shorten a web document text to approximately max_words using an LLM.
        This is similar to shorten_wikipedia_text but with different prompt.
        
        Args:
            text (str): The full web document text to shorten.
            max_words (int): Target word count for the shortened text.
        
        Returns:
            str: The shortened text.
        """
        messages = [
            {"role": "user", "content": f"Please summarize the following web document content to approximately {max_words} words. Keep the most important factual information and maintain accuracy. Preserve key details about people, places, events, dates, and numbers:\n\n{text}"}
        ]
        
        input_word_count = len(text.split())
        logger.debug("=== LLM CALL: shorten_web_document_text ===")
        logger.debug(f"Target words: {max_words}")
        logger.debug(f"Input text length: {len(text)} characters, {input_word_count} words")
        logger.debug(f"Model: {self.llm_judge}")
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.llm_judge,
                max_tokens=max_words + 200,  # Allow some buffer
                temperature=0.3,
            )
            
            result = response.choices[0].message.content
            output_word_count = len(result.split())
            logger.debug(f"Web document text shortened: {input_word_count} => {output_word_count} words")
            
            return result
        except Exception as e:
            logger.error(f"Error in shorten_web_document_text: {e}", exc_info=True)
            return text

    @network_retry(max_retries=5, initial_delay=1.0)
    def extract_triples_from_text(self, entity_name: str, text: str) -> List[Dict]:
        """
        Extract RDF triples from a given text for a specific entity.
        Returns structured JSON output with guaranteed list of triples.
        
        Args:
            entity_name (str): The subject entity for the triples.
            text (str): The text to extract triples from.
        
        Returns:
            list: List of dicts with 'subject', 'predicate', 'object' keys.
        """
        # Define the JSON schema for structured output
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleExtraction",
                "description": "Extract RDF triples from text",
                "schema": {
                    "type": "object",
                    "properties": {
                        "triples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {"type": "string"},
                                    "predicate": {"type": "string"},
                                    "object": {"type": "string"}
                                },
                                "required": ["subject", "predicate", "object"]
                            },
                            "description": "List of extracted triples"
                        }
                    },
                    "required": ["triples"]
                }
            }
        }
        
        messages = [
            {"role": "user", "content": f"""Extract RDF triples from the following text about '{entity_name}'. 
The subject of all triples should be '{entity_name}'.
Return the triples as a JSON array with objects containing 'subject', 'predicate', 'object' fields.

Text:
{text}"""}
        ]
        
        logger.debug("=== LLM CALL: extract_triples_from_text ===")
        logger.debug(f"Entity: {entity_name}")
        logger.debug(f"Input text length: {len(text)} characters, {len(text.split())} words")
        logger.debug(f"Model: {self.llm_judge}")
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.llm_judge,
                max_tokens=2000,
                temperature=0.0,
                response_format=response_schema
            )
            
            response_text = response.choices[0].message.content
            result = json.loads(response_text)
            triples = result.get("triples", [])
            
            # Validation: warn if no triples were extracted
            if not triples or len(triples) == 0:
                logger.warning(f"No triples extracted for entity '{entity_name}'. Response was: {response_text[:200]}")
            else:
                logger.debug(f"Extracted {len(triples)} triples")
            
            #logger.info(f"Triple extraction for '{entity_name}': {len(triples)} triples extracted")
            
            return triples
        except Exception as e:
            logger.error(f"Error in extract_triples_from_text for {entity_name}: {e}", exc_info=True)
            return []
    
    @network_retry(max_retries=5, initial_delay=1.0)
    def verify_triple_all_sources(self, triple: str, wikipedia_snippet: str, wikidata_triples: list, web_snippets: list) -> dict:
        """
        Verify a triple against all ground truth sources in a single LLM call.
        
        NOTE: This method is no longer used for precision evaluation (now uses RAG-based
        approach via verify_triple_with_rag). It is kept for potential future use.
        
        Args:
            triple (str): The RDF triple to verify.
            wikipedia_snippet (str): Wikipedia article text.
            wikidata_triples (list): List of Wikidata triples.
            web_snippets (list): List of web search result dicts.
        
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
        """
        triple_prompt_str = f"Statement to verify: {triple}."
        
        # Build context from all sources
        context_parts = []
        
        if wikipedia_snippet:
            context_parts.append(f"Wikipedia article excerpt: {wikipedia_snippet}")
        
        if wikidata_triples and len(wikidata_triples) > 0:
            wikidata_str = str([
                f"({t.get('subject', '')}, {t.get('predicate', '')}, {t.get('object', '')})"
                for t in wikidata_triples
            ])
            context_parts.append(f"Wikidata triples: {wikidata_str}")
        
        if web_snippets and len(web_snippets) > 0:
            web_str = " | ".join([
                f"{r.get('title', '')}: {r.get('snippet', '')}"
                for r in web_snippets
            ])
            context_parts.append(f"Web search results: {web_str}")
        
        context_str = "\n\n".join(context_parts)
        context_prompt_str = f"Ground truth context:\n{context_str}"
        
        # Define JSON schema for structured output
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against all ground truth sources",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given ground truth context? \
                            a) The ground truth entails the RDF triple.\
                            b) The ground truth contradicts the RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the ground truth alone. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": context_prompt_str},
        ]
        
        logger.debug("=== LLM CALL: verify_triple_all_sources ===")
        logger.debug(f"Triple: {triple}")
        logger.debug(f"Wikipedia: {len(wikipedia_snippet)} chars, Wikidata: {len(wikidata_triples)} triples, Web: {len(web_snippets)} results")
        logger.debug(f"Model: {self.llm_judge}")
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format=response_schema
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")
        
        logger.debug(f"All-sources verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result
    
    @network_retry(max_retries=6, initial_delay=1.0)
    def verify_triple_with_rag(self, triple: str, rag_context: str, sources: list) -> dict:
        """
        Verify an RDF triple against retrieved RAG passages in a single LLM call.
        
        Used for precision evaluation with the RAG-based approach. The LLM receives
        the triple and top-k retrieved passages, then determines if the triple is
        entailed, contradicted, or neutral based on the passages.
        
        Args:
            triple: RDF triple string in format "(subject, predicate, object)"
            rag_context: Combined text of all retrieved passages
            sources: List of source names for retrieved passages (e.g., ['wikipedia', 'wikidata'])
            
        Returns:
            dict: {"answer": "a" | "b" | "c", "reasoning": "..."}
                  a=entailment, b=contradiction, c=neutral
        """
        triple_prompt_str = f"Statement to verify: {triple}."
        context_prompt_str = f"Retrieved ground truth passages:\n{rag_context}"
        
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "TripleVerification",
                "description": "Verification of an RDF triple against retrieved passages",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["a", "b", "c"],
                            "description": "a) entailment, b) contradiction, c) neutral"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reasoning for the choice"
                        }
                    },
                    "required": ["answer", "reasoning"]
                }
            }
        }
        
        sources_str = ", ".join(sources) if sources else "unknown"
        
        messages = [
            {"role": "user",
                "content": "Can the given RDF triple be inferred from the given retrieved passages? \
                            a) The passages entail the RDF triple.\
                            b) The passages contradict the RDF triple. \
                            c) The truth of the given RDF triple cannot be determined from the passages alone. \
                            Respond with JSON containing 'answer' (one of a/b/c) and 'reasoning'."},
            {"role": "user", "content": triple_prompt_str},
            {"role": "user", "content": context_prompt_str},
        ]
        
        logger.debug("=== LLM CALL: verify_triple_with_rag ===")
        logger.debug(f"Triple: {triple}")
        logger.debug(f"Sources: {sources_str}")
        logger.debug(f"Model: {self.llm_judge}")
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_judge,
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format=response_schema
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        # Validate structured output
        if not isinstance(result, dict) or result.get("answer") not in ['a', 'b', 'c']:
            raise ValueError(f"Invalid LLM response for triple '{triple}': {result}")
        
        logger.debug(f"RAG verification completed: {triple[:50]}... => {result.get('answer')}")
        
        return result