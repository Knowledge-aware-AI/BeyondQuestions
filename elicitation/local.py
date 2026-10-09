import openai as oa
from openai import OpenAI, api_key
import json
import queue
import time
import threading
import logging
import csv
import os
import re
from pathlib import Path

import requests
from eval.network_utils import network_retry
from .template_utils import resolve_templates, extract_system_prompt_from_jinja, TwoStepTemplate
from .prompter_parser import PromptTwoStep


def main_elicitation_other(
        entities_file_path:str,
        model_elicitation:str,
        api_url_elicitation:str,
        api_key_elicitation:str,
        elicited_triples_dir:str,
        evaluate_by_category:bool,
        reasoning_effort_elicitation:str=None,
        different_elicitation_triple_ranges:bool=False,
        evaluate_by_popularity:bool=False,
        prompt_template_dir_elicitation:str=None,
        use_all_prompts:bool=False,
        prompt_templates:str=None,
        nthreads:int=10,
        multi_turn_elicitation:bool=False,
        multi_turn_max_turns:int=6,
        multi_turn_min_new_triples:int=2,
    ):

    """
    Main function to run the entity-based triple extraction pipeline using local LLM API.
    Arguments:
        entities_file_path (str): File path storing the wikidata entities
        model_elicitation (str): Name of the model to use for elicitation
        elicited_triples_dir (str): Dir path for storing the elicited triples
        evaluate_by_category (bool): Whether to evaluate by category (default: False)
        reasoning_effort_elicitation (str, optional): Reasoning effort level for elicitation prompts.
        different_elicitation_triple_ranges (bool): Whether to run separate elicitation for different triple count ranges.
        evaluate_by_popularity (bool): Whether to evaluate by popularity buckets.
        prompt_template_dir_elicitation (str, optional): Directory containing .jinja prompt templates.
            When provided, the system prompt is extracted from each template and used for elicitation.
            When None, a default built-in prompt is used.
        use_all_prompts (bool): When True, run elicitation for every .jinja file in the template directory.
        prompt_templates (str, optional): Comma-separated list of specific .jinja filenames to use.
        nthreads (int): Number of entities to query concurrently per batch (default: 10).
            Lower this if you hit per-key rate limits (e.g. scads.ai token/min limits).
        multi_turn_elicitation (bool): When True, after the initial (standard) elicitation turn,
            keep the same chat going with bias-free follow-up nudges (e.g. "Anything else?"),
            accumulating unique triples across turns. Not supported together with two-step
            templates (those are skipped when this is enabled). Default: False.
        multi_turn_max_turns (int): Hard cap on the number of turns per entity, including the
            initial one, when multi_turn_elicitation is True (default: 6).
        multi_turn_min_new_triples (int): Stop the conversation once a turn yields fewer than this
            many new unique triples (default: 2). A turn returning zero triples (explicit "nothing
            more" via an empty JSON array) always stops the conversation.
    """
    # === CONFIGURATION ===
    verbose = True

    # === LOGGING SETUP ===
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler("kbc_entity_extraction.log"),
            logging.StreamHandler()
        ]
    )

    # === LLM WRAPPER ===
    @network_retry(max_retries=5, initial_delay=1.0)
    def promptLLMLocal(message):
        """message may be a plain string (single user turn) or a list of
        chat-message dicts (for multi-turn conversations)."""
        client = OpenAI(base_url=api_url_elicitation, api_key=api_key_elicitation)
        messages = message if isinstance(message, list) else [{"role": "user", "content": message}]

        if reasoning_effort_elicitation:
            logging.info(f"Prompting LLM with reasoning effort: {reasoning_effort_elicitation}")
            r = client.chat.completions.create(
                messages=messages,
                model=model_elicitation,
                temperature=0,
                extra_body={"reasoning": {"enabled": True, "effort": reasoning_effort_elicitation}},
            )
            message_obj = r.choices[0].message
            reasoning = getattr(message_obj, "reasoning", None) or getattr(message_obj, "reasoning_content", None)
            return message_obj.content, reasoning
        else:
            logging.info(f"Prompting LLM without reasoning effort")
            r = client.chat.completions.create(
                messages=messages,
                model=model_elicitation,
                temperature=0
            )
            return r.choices[0].message.content, None

    # === FILE UTILS ===
    def append_to_jsonl_file(file_path, new_data):
        dirpath = os.path.dirname(file_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(new_data) + '\n')
    '''
    def remove_json_delimiters(input_string):
        """Strip common markdown/codeblock wrappers and surrounding text so the
        remaining string is hopefully pure JSON. This uses regex to remove leading
        and trailing fences like ```json or ``` and also strips incidental
        backticks and whitespace.
        """
        if not isinstance(input_string, str):
            return input_string

        s = input_string.strip()

        # Remove <think>...</think> blocks
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)

        # Remove leading/trailing triple-backtick blocks (with optional json tag)
        s = re.sub(r"^(```\s*json\s*\n?|```\s*)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n?```\s*$", "", s)

        # If the model wrapped the JSON in a text block like "```json\n{...}\n```",
        # this should remove the fences. Also remove any leading "json" markers.
        s = s.strip()

        return s
    '''
    '''
    def remove_json_delimiters(input_string):
        if not isinstance(input_string, str):
            return input_string

        s = input_string.strip()

        # Remove <think>...</think> blocks
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)

        # Remove leading/trailing triple-backtick blocks
        s = re.sub(r"^(```\s*json\s*\n?|```\s*)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n?```\s*$", "", s)

        # Remove accidental leading text before JSON array
        json_start = s.find("[")
        if json_start != -1:
            s = s[json_start:]

        return s.strip()
    
    
    def extract_json_array(text):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return match.group(0)
        return None
    '''


    def extract_json_array(text: str) -> list | None:
        if not isinstance(text, str):
            return None

        s = text.strip()

        # 1. Strip <think>...</think> blocks
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

        # 2. Strip markdown code fences
        s = re.sub(r"^```+\s*(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n?```+\s*$", "", s).strip()

        # 3. Fast path: whole string is valid JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass

        # 4. Tuple extraction — handles all observed variants:
        #      ("s", "p", "o")                   — quoted
        #      - ("s", "p", "o")                 — bulleted + quoted
        #      (s, p, o)                          — unquoted
        #      - (s, p, o)                        — bulleted + unquoted
        #      (s, p, o)  # comment              — with trailing comment
        #
        # Strategy: find every (...) group on its own line, then split on commas.
        # Using a lenient pattern that matches both quoted and unquoted values.

        VALUE = r'"[^"]*"|[^,()#\n]+'  # quoted string OR unquoted run of chars

        tuple_matches = re.findall(
            rf'\(\s*({VALUE})\s*,\s*({VALUE})\s*,\s*({VALUE})\s*\)',
            s
        )

        def clean_value(v: str) -> str:
            v = v.strip()
            # Strip surrounding quotes if present
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            # Strip trailing comments (e.g. "# first mention")
            v = re.sub(r'\s*#.*$', '', v).strip()
            return v

        if tuple_matches:
            return [
                {"subject": clean_value(subj), "predicate": clean_value(pred), "object": clean_value(obj)}
                for subj, pred, obj in tuple_matches
                if clean_value(subj) and clean_value(pred) and clean_value(obj)
            ]

        # 5. Find outermost '[' ... ']' respecting nesting and strings
        start = s.find("[")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        end = -1

        for i, ch in enumerate(s[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        candidate = s[start:end + 1]

        # 6. Attempt direct JSON parse
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # 7. Truncation recovery: trim to last complete object element
        last_complete = candidate.rfind("},")
        if last_complete != -1:
            truncated = candidate[:last_complete + 1] + "]"
            try:
                parsed = json.loads(truncated)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None


    # === RANGES ===
    popular_ranges = ["5-10", "10-20", "25-50", "33-66", "50-100", "75-150", "100-200"]
    longtail_ranges = ["1", "1-2", "3-5", "3-7", "5-10", "8-15", "10-20"]

    # === PATHS (will be set per run) ===
    entities_input_path = entities_file_path

    # === TEMPLATE RESOLUTION ===
    if prompt_template_dir_elicitation:
        selected_templates = resolve_templates(
            prompt_template_dir=prompt_template_dir_elicitation,
            use_all_prompts=use_all_prompts,
            prompt_templates=prompt_templates,
        )
        logging.info(f"Selected {len(selected_templates)} template(s): {[t.name if isinstance(t, TwoStepTemplate) else os.path.basename(t) for t in selected_templates]}")

    multi_prompt = len(selected_templates) > 1

    def _prompt_output_dir(base_dir, template_path):
        """Return per-template subdirectory when multi-prompt mode is active."""
        if template_path is None:
            result = base_dir
        # Always use subdirectory for TwoStepTemplate
        elif isinstance(template_path, TwoStepTemplate):
            result = os.path.join(base_dir, template_path.name)
        # For single-step templates, only use subdirectory if multi_prompt is True
        elif not multi_prompt:
            result = base_dir
        else:
            stem = Path(template_path).stem
            result = os.path.join(base_dir, stem)
        # Keep multi-turn output separate from standard single-turn output
        # for the same template so results don't collide.
        if multi_turn_elicitation and not isinstance(template_path, TwoStepTemplate):
            result = os.path.join(result, "multi_turn")
        return result

    def _get_two_step_system_prompt(two_step_template, step):
        """Extract system prompt from two-step template."""
        from jinja2 import Environment, FileSystemLoader
        if step == "predicates":
            path = Path(two_step_template.predicate_template)
        else:
            path = Path(two_step_template.object_template)
        env = Environment(loader=FileSystemLoader(str(path.parent)))
        tmpl = env.get_template(path.name)
        tokens = {"low": 4096, "medium": 10000, "high": 15000}
        params = {"max_tokens": 4096, "top_p": 0, "temperature": 0, "frequency_penalty": 0, "presence_penalty": 0}
        kwargs = {"subject_name": "__PLACEHOLDER__", "model": model_elicitation, "reasoning_effort": reasoning_effort_elicitation, "max_completion_tokens": tokens, **params}
        if step == "objects":
            kwargs["predicate"] = "__PRED_PLACEHOLDER__"
            kwargs["custom_id"] = "__PLACEHOLDER__"
        rendered = tmpl.render(**kwargs)
        payload = json.loads(rendered)
        for msg in payload["body"]["messages"]:
            if msg.get("role") == "system":
                return msg["content"]
        raise ValueError(f"No system message in {path}")

    def _get_system_prompt_from_template(tmpl_path, pop_range=None, lt_range=None):
        """Extract the system message content from a jinja template.
        Optionally substitute dynamic range values (for different_elicitation_triple_ranges).
        """
        sys_prompt = extract_system_prompt_from_jinja(
            template_path=tmpl_path,
            model_elicitation=model_elicitation,
            reasoning_effort=reasoning_effort_elicitation,
        )
        if pop_range and lt_range:
            sys_prompt, _ = re.subn(
                r"(between )([0-9]+\s*to\s*[0-9]+|[0-9]+–[0-9]+|[0-9]+ or more|[0-9]+\s*–\s*[0-9]+)",
                lambda m: m.group(1) + pop_range,
                sys_prompt,
                count=1,
            )
            sys_prompt, _ = re.subn(
                r"(very low, like )([0-9]+|[0-9]+-[0-9]+|[0-9]+–[0-9]+|[0-9]+ to [0-9]+)",
                lambda m: m.group(1) + lt_range,
                sys_prompt,
                count=1,
            )
        return sys_prompt

    # === TRIPLE EXTRACTION ===
    def getTriples(entity, result_queue, errorQueue, pop_range, lt_range, parse_errors_path, empty_results_path, system_prompt=None, reasoning_traces_path=None):
        # Build the prompt.
        # When a system_prompt is provided (from a jinja template), use it directly
        # and append the entity.  Otherwise fall back to the built-in hardcoded prompt.
        if system_prompt is not None:
            prompt = f"{system_prompt}\n\nSubject: \"{entity}\""
        else:
            prompt = (
                f"You are a knowledge base construction expert. Given a subject entity, "
                f"return all facts that you know for the subject as a JSON array of "
                f"triples. Each triple must be a JSON object with exactly these keys: "
                f'"subject" (string), "predicate" (string), "object" (string). '
                f"If there are multiple objects for the same predicate, return them as "
                f"separate triple objects (one object per triple). The number of facts may be very high, between {pop_range} for very popular subjects. For less popular subjects, the number of facts can be very low, like {lt_range}.\n\n"
                f"Important rules (must follow exactly):\n"
                f"- Output MUST be valid JSON and nothing else (no explanatory text).\n"
                f"- The top-level JSON value MUST be an array (e.g. []).\n"
                f"- If you don't know the subject or it is not a named entity, return [] (empty array).\n"
                f"- If the subject is a named entity include at least one triple with predicate \"instanceOf\".\n"
                f"- Keep properties concise; do not include nested objects or arrays inside triple fields.\n\n"
                f"Subject: \"{entity}\""
            )
        
        logging.info(f"Querying entity: {entity}")
        try:
            output_string, reasoning = promptLLMLocal(prompt)
            if reasoning and reasoning_traces_path:
                append_to_jsonl_file(reasoning_traces_path, {"entity": entity, "reasoning": reasoning})
            try:
                #clean = remove_json_delimiters(output_string)
                #json_str = extract_json_array(clean)

                #if json_str is None:
                #    raise ValueError("No JSON array found in response")

                #linetriples = json.loads(json_str)
                linetriples = extract_json_array(output_string)
                if linetriples is None:
                    raise ValueError("Failed to extract JSON array from response")

                # Normalize list-of-lists [["s","p","o"], ...] → list-of-dicts
                if linetriples and isinstance(linetriples[0], list):
                    keys = ("subject", "predicate", "object")
                    normalized = []
                    for item in linetriples:
                        if isinstance(item, list) and len(item) == 3:
                            normalized.append(dict(zip(keys, [str(v) for v in item])))
                    if normalized:
                        linetriples = normalized

                # Validate structure and try to repair if invalid
                valid, reason = validate_triples(linetriples)
                if not valid:
                    #logging.warning(f"   Initial validation failed for {entity}: {reason}. Trying repair prompt...")
                    #repaired = try_repair_output(output_string, entity)
                    #if repaired is not None:
                    #    try:
                    #        linetriples = json.loads(repaired)
                    #        valid, reason = validate_triples(linetriples)
                    logging.warning(f"Initial validation failed for {entity}: {reason}.")
                    #    except Exception as e:
                    #        valid = False
                    #        reason = f"JSON parse after repair failed: {e}"

                if not valid:
                    append_to_jsonl_file(parse_errors_path, {"error": reason, "entity": entity, "response": str(output_string)[:500]})
                    result_queue.put((entity, []))
                    logging.warning(f"   Failed to parse/validate JSON for: {entity} -- {reason}")
                else:
                    if not linetriples:
                        append_to_jsonl_file(empty_results_path, {"entity": entity, "message": "Empty result"})
                        logging.info(f"   Received 0 triples for: {entity}")
                    else:
                        result_queue.put((entity, linetriples))
                        logging.info(f"   Received {len(linetriples)} triples for: {entity}")
                    
            except Exception as e:
                append_to_jsonl_file(parse_errors_path, {"error": str(e), "entity": entity, "response": str(output_string)[:500]})
                result_queue.put((entity, []))
                logging.warning(f"   Failed to parse JSON for: {entity}")
                
        except Exception as e:
            errorQueue.put(e)
            logging.exception(f"Exception during prompt for entity: {entity}")


    # === MULTI-TURN ELICITATION ===
    # Bias-free follow-up nudges: no mention of the entity or any category
    # (e.g. education, career) so they don't steer the model toward a
    # particular kind of fact and don't need per-entity adaptation.
    FOLLOWUP_PROMPTS = [
        "Anything else?",
        "Do you know more?",
        "Please continue. Is there more?",
        "Any additional facts?",
        "What else can you tell me?",
    ]

    def getTriplesMultiTurn(entity, result_queue, errorQueue, pop_range, lt_range, parse_errors_path,
                             empty_results_path, max_turns, min_new_triples, system_prompt=None,
                             reasoning_traces_path=None):
        if system_prompt is not None:
            initial_prompt = f"{system_prompt}\n\nSubject: \"{entity}\""
        else:
            initial_prompt = (
                f"You are a knowledge base construction expert. Given a subject entity, "
                f"return all facts that you know for the subject as a JSON array of "
                f"triples. Each triple must be a JSON object with exactly these keys: "
                f'"subject" (string), "predicate" (string), "object" (string). '
                f"If there are multiple objects for the same predicate, return them as "
                f"separate triple objects (one object per triple). The number of facts may be very high, between {pop_range} for very popular subjects. For less popular subjects, the number of facts can be very low, like {lt_range}.\n\n"
                f"Important rules (must follow exactly):\n"
                f"- Output MUST be valid JSON and nothing else (no explanatory text).\n"
                f"- The top-level JSON value MUST be an array (e.g. []).\n"
                f"- If you don't know the subject or it is not a named entity, return [] (empty array).\n"
                f"- If the subject is a named entity include at least one triple with predicate \"instanceOf\".\n"
                f"- Keep properties concise; do not include nested objects or arrays inside triple fields.\n\n"
                f"Subject: \"{entity}\""
            )

        logging.info(f"Querying entity (multi-turn): {entity}")
        messages = [{"role": "user", "content": initial_prompt}]
        seen_keys = set()
        accumulated_triples = []

        try:
            for turn_idx in range(max_turns):
                output_string, reasoning = promptLLMLocal(messages)
                if reasoning and reasoning_traces_path:
                    append_to_jsonl_file(reasoning_traces_path, {"entity": entity, "turn": turn_idx, "reasoning": reasoning})
                messages.append({"role": "assistant", "content": output_string})

                linetriples = extract_json_array(output_string)
                new_count = 0

                if linetriples:
                    if isinstance(linetriples[0], list):
                        keys = ("subject", "predicate", "object")
                        normalized = []
                        for item in linetriples:
                            if isinstance(item, list) and len(item) == 3:
                                normalized.append(dict(zip(keys, [str(v) for v in item])))
                        if normalized:
                            linetriples = normalized

                    valid, reason = validate_triples(linetriples)
                    if valid:
                        for t in linetriples:
                            key = (str(t.get('subject', '')).strip(), str(t.get('predicate', '')).strip(), str(t.get('object', '')).strip())
                            if key not in seen_keys:
                                seen_keys.add(key)
                                accumulated_triples.append(t)
                                new_count += 1
                    else:
                        append_to_jsonl_file(parse_errors_path, {"error": reason, "entity": entity, "turn": turn_idx, "response": str(output_string)[:500]})
                        logging.warning(f"   [{entity}] turn {turn_idx + 1}: validation failed -- {reason}")
                else:
                    append_to_jsonl_file(empty_results_path, {"entity": entity, "turn": turn_idx, "message": "No triples parsed (explicit stop or parse failure)"})

                logging.info(f"   [{entity}] turn {turn_idx + 1}: {new_count} new unique triples (total {len(accumulated_triples)})")

                if new_count < min_new_triples:
                    logging.info(f"   [{entity}] Stopping after turn {turn_idx + 1}: new triples ({new_count}) below threshold ({min_new_triples})")
                    break

                if turn_idx == max_turns - 1:
                    logging.info(f"   [{entity}] Reached max turns ({max_turns})")
                    break

                followup = FOLLOWUP_PROMPTS[turn_idx % len(FOLLOWUP_PROMPTS)]
                messages.append({"role": "user", "content": followup})

            result_queue.put((entity, accumulated_triples))
            logging.info(f"   [{entity}] Multi-turn elicitation complete: {len(accumulated_triples)} total unique triples")

        except Exception as e:
            errorQueue.put(e)
            logging.exception(f"Exception during multi-turn prompt for entity: {entity}")

    def validate_triples(triples):
        """Return (True, '') if triples is a valid list of triple dicts.
        Otherwise return (False, reason string).
        """
        if not isinstance(triples, list):
            return False, f"Top-level JSON is not a list (got {type(triples).__name__})"

        for i, t in enumerate(triples):
            if not isinstance(t, dict):
                return False, f"Element {i} is not an object (got {type(t).__name__})"
            # required keys
            for k in ("subject", "predicate", "object"):
                if k not in t:
                    return False, f"Element {i} missing required key '{k}'"
                # values must be scalars (we coerce to string later)
                if isinstance(t[k], (list, dict)):
                    return False, f"Element {i} key '{k}' must be a string (got {type(t[k]).__name__})"
        return True, ""

    '''
    def try_repair_output(original_output, entity):
        """Ask the model to reformat its previous output into strict JSON array
        following the triple schema. Returns the repaired JSON string or None.
        """
        try:
            repair_prompt = (
                "The previous response did not follow the required JSON schema. "
                "Please reformat only the content (no explanation) as a valid JSON array "
                "of objects with keys 'subject', 'predicate', and 'object' (all strings). "
                "If you do not know the subject, return an empty array.\n\n"
                f"Subject: \"{entity}\"\n\n"
                + "Previous response:\n```\n"
                + str(original_output)
                + "\n```"
            )

            repaired = promptLLMLocal(repair_prompt)
            repaired = remove_json_delimiters(repaired)
            return repaired
        except Exception as e:
            logging.exception(f"Repair attempt failed for entity {entity}: {e}")
            return None
    '''

    # === DEDUPLICATION ===
    def deduplicate_triples(triples):
        seen = set()
        unique_triples = []
        for t in triples:
            try:
                key = (t.get('subject', '').strip(), t.get('predicate', '').strip(), t.get('object', '').strip())
                if key not in seen:
                    seen.add(key)
                    unique_triples.append(t)
            except Exception as e:
                logging.error(f"Error processing triple {t}: {e}")
                continue
        return unique_triples

    def storeTriplesInCSV(entity, triples, triples_output_path):
        unique_triples = deduplicate_triples(triples)
        if not unique_triples:
            return 0
        logging.info(f"   Deduplicated {len(triples) - len(unique_triples)} triples (from {len(triples)} to {len(unique_triples)})")
        # Ensure output directory exists
        csv_dir = os.path.dirname(triples_output_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(triples_output_path)
        has_confidence = any('confidence' in triple for triple in unique_triples)
        with open(triples_output_path, 'a', newline='', encoding='utf-8') as csvfile:
            # Match the schema used by gpt_kbc.py: include subject_name column
            fieldnames = ['subject', 'predicate', 'object', 'subject_name']
            if has_confidence:
                fieldnames.append('confidence')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for triple in unique_triples:
                row = {
                    'subject': triple.get('subject', ''),
                    'predicate': triple.get('predicate', ''),
                    'object': triple.get('object', ''),
                    'subject_name': entity
                }
                if has_confidence:
                    row['confidence'] = triple.get('confidence', '')
                writer.writerow(row)
        return len(unique_triples)

    # === LOAD ENTITIES ===
    def loadEntities():
            with open(entities_input_path, 'r') as file:
                json_content = json.load(file)
            
            if evaluate_by_category == True:
                all_values = json_content
                all_values = [item["title"] for category in json_content.values() for item in category]
                return all_values
            else:
                all_values = json_content
                #all_values = [item for key, values in json_content.items() for item in values]
                all_values = [i['title'] for i in all_values]
                return all_values


    # === MAIN EXECUTION ===

    # Helper: run threaded pool for a list of entities, writing to given paths.
    # Rolling pool: nthreads entities are always in flight, so one slow entity never holds up
    # the others. Each entity's triples are written as soon as it finishes. An entity whose
    # request fails (after retries) is logged and left unrecorded, so a later run can fill it in;
    # it never discards the results of other entities.
    def _run_batch_loop(entities, triples_output_path, parse_errors_path, empty_results_path,
                        pop_range, lt_range, system_prompt=None, reasoning_traces_path=None):
        from concurrent.futures import ThreadPoolExecutor
        totals = {"triples": 0, "processed": 0, "failed": 0}
        write_lock = threading.Lock()

        def process_entity(entity):
            if isinstance(entity, dict):
                entity_name = entity.get('label') or entity.get('name') or entity.get('id') or str(entity)
            else:
                entity_name = str(entity)
            result_queue = queue.Queue()
            errorQueue = queue.Queue()
            getTriples(entity_name, result_queue, errorQueue, pop_range, lt_range,
                       parse_errors_path, empty_results_path, system_prompt, reasoning_traces_path)
            if not errorQueue.empty():
                error_msg = str(errorQueue.get())[:100]
                logging.error(f"Error in thread processing ({entity_name}, left for a later run): {error_msg}")
                with write_lock:
                    totals["failed"] += 1
                time.sleep(60)  # same back-off as before, now only for this slot
                return
            with write_lock:
                while not result_queue.empty():
                    name, triples = result_queue.get()
                    totals["triples"] += storeTriplesInCSV(name, triples, triples_output_path)
                    totals["processed"] += 1
                done = totals["processed"] + totals["failed"]
                if done % nthreads == 0:
                    logging.info(f"Progress: processed {totals['processed']}, failed {totals['failed']}, "
                                 f"of {len(entities)}; total triples: {totals['triples']}  {time.strftime('%X %x %Z')}")

        logging.info(f"Processing {len(entities)} entities, {nthreads} at a time (rolling)")
        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            for future in [pool.submit(process_entity, e) for e in entities]:
                try:
                    future.result()
                except Exception as e:
                    logging.exception(f"Unexpected error in entity worker: {e}")
        logging.info(f"All entities done. Processed: {totals['processed']}/{len(entities)}, "
                     f"failed (left for a later run): {totals['failed']}, Total triples: {totals['triples']}")
        return totals["triples"], totals["processed"]

    def _run_multi_turn_batch_loop(entities, triples_output_path, parse_errors_path, empty_results_path,
                                    pop_range, lt_range, max_turns, min_new_triples, system_prompt=None,
                                    reasoning_traces_path=None):
        total_triples = 0
        processed_count = 0
        for batch_start in range(0, len(entities), nthreads):
            batch_end = min(batch_start + nthreads, len(entities))
            batch = entities[batch_start:batch_end]
            threads = []
            result_queue = queue.Queue()
            errorQueue = queue.Queue()
            logging.info(f"Processing multi-turn batch {batch_start // nthreads + 1} (entities {batch_start + 1} to {batch_end})")
            for entity in batch:
                if isinstance(entity, dict):
                    entity_name = entity.get('label') or entity.get('name') or entity.get('id') or str(entity)
                else:
                    entity_name = str(entity)
                thread = threading.Thread(
                    target=getTriplesMultiTurn,
                    args=(entity_name, result_queue, errorQueue, pop_range, lt_range,
                          parse_errors_path, empty_results_path, max_turns, min_new_triples,
                          system_prompt, reasoning_traces_path)
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            if errorQueue.empty():
                while not result_queue.empty():
                    entity_name, triples = result_queue.get()
                    triples_count = storeTriplesInCSV(entity_name, triples, triples_output_path)
                    total_triples += triples_count
                    processed_count += 1
            else:
                error_msg = str(errorQueue.get())[:100]
                logging.error(f"Error in thread processing: {error_msg}")
                time.sleep(60)
            logging.info(f"Batch completed. Processed: {processed_count}/{len(entities)}, Total triples: {total_triples}")
            logging.info(f"{time.strftime('%X %x %Z')}\n")
            time.sleep(1)
        return total_triples, processed_count

    def _run_two_step_loop(entities, two_step_template, output_dir):
        """Run two-step elicitation locally (predicates -> objects).

        Incremental and resumable: each entity is processed end to end (predicates, then all its
        objects) and its triples are appended to elicited_triples.csv as soon as it finishes.
        Entities with no predicates go to empty_entity_results.jsonl, unparseable predicate
        responses to entity_extraction_errors.jsonl. An entity hit by a request failure (after
        retries) is not recorded at all, so a later run over the missing entities redoes it whole.
        Entities run in parallel (nthreads), and each entity's object calls run 4 at a time."""
        import requests
        from concurrent.futures import ThreadPoolExecutor

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Starting two-step: {two_step_template.name}")

        pred_system = _get_two_step_system_prompt(two_step_template, "predicates")
        obj_system = _get_two_step_system_prompt(two_step_template, "objects")

        csv_path = os.path.join(output_dir, "elicited_triples.csv")
        parse_errors_path = os.path.join(output_dir, "entity_extraction_errors.jsonl")
        empty_results_path = os.path.join(output_dir, "empty_entity_results.jsonl")
        write_lock = threading.Lock()
        OBJECT_WORKERS = 4

        def post(system_prompt, user_content):
            """One chat call. Raises on any request failure so @network_retry retries it."""
            chat_url = api_url_elicitation.rstrip('/') + "/chat/completions"
            r = requests.post(
                chat_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key_elicitation}" if api_key_elicitation else ""
                },
                json={
                    "model": model_elicitation,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0
                },
                timeout=120
            )
            if r.status_code != 200:
                raise RuntimeError(f"Non-200 response: status={r.status_code}, body={r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"]

        def strip_wrappers(content):
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            cleaned = re.sub(r"^```+\s*(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
            return re.sub(r"\n?```+\s*$", "", cleaned).strip()

        @network_retry(max_retries=5, initial_delay=1.0)
        def get_predicates(entity):
            """Returns (predicates, None) or (None, raw_content) when the response can't be parsed."""
            content = post(pred_system, entity)
            logging.debug(f"Raw predicate response for {entity}: {content[:200]}...")
            result = extract_json_array(content)
            if result is not None:
                if all(isinstance(p, str) for p in result):
                    logging.info(f"Extracted {len(result)} predicates for {entity}")
                    return result, None
                if all(isinstance(p, dict) for p in result):
                    predicates = [p.get("predicate") or p.get("name") or p.get("label") or str(p)
                                  for p in result if p]
                    predicates = [p for p in predicates if p]
                    logging.info(f"Extracted {len(predicates)} predicates (from dicts) for {entity}")
                    return predicates, None
            try:
                obj = json.loads(strip_wrappers(content))
                if isinstance(obj, dict) and "predicates" in obj:
                    predicates = [str(p) for p in obj["predicates"] if p]
                    logging.info(f"Extracted {len(predicates)} predicates (dict wrapper) for {entity}")
                    return predicates, None
            except json.JSONDecodeError:
                pass
            logging.warning(f"Could not extract predicates for {entity}: {content[:300]}")
            return None, content

        @network_retry(max_retries=5, initial_delay=1.0)
        def get_objects(entity, predicate):
            """Returns the triples for one (entity, predicate); [] when the response can't be parsed."""
            content = post(obj_system, f"Subject: {entity}\nPredicate: {predicate}")
            logging.debug(f"Raw objects response for {entity}/{predicate}: {content[:200]}...")

            def make_triples(objects):
                return [
                    {"subject": entity, "predicate": predicate, "object": str(o), "subject_name": entity}
                    for o in objects if o
                ]

            result = extract_json_array(content)
            if result is not None:
                if all(isinstance(o, str) for o in result):
                    triples = make_triples(result)
                    logging.info(f"Extracted {len(triples)} objects for {entity}/{predicate}")
                    return triples
                if all(isinstance(o, dict) for o in result):
                    objects = [o.get("object") or o.get("value") or o.get("name") or str(o)
                               for o in result if o]
                    objects = [o for o in objects if o]
                    triples = make_triples(objects)
                    logging.info(f"Extracted {len(triples)} objects (from dicts) for {entity}/{predicate}")
                    return triples
            try:
                obj = json.loads(strip_wrappers(content))
                if isinstance(obj, dict) and "objects" in obj:
                    triples = make_triples(obj["objects"])
                    logging.info(f"Extracted {len(triples)} objects (dict wrapper) for {entity}/{predicate}")
                    return triples
            except json.JSONDecodeError:
                pass
            logging.warning(f"Could not extract objects for {entity}/{predicate}: {content[:300]}")
            return []

        def write_entity(triples):
            with write_lock:
                new_file = not os.path.isfile(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "subject_name"])
                    if new_file:
                        writer.writeheader()
                    writer.writerows(triples)

        def record(path, data):
            with write_lock:
                append_to_jsonl_file(path, data)

        names = [e.get('label') or e.get('name') or e.get('id') or str(e) if isinstance(e, dict) else str(e)
                 for e in entities]
        done = {'entities': 0, 'triples': 0}

        def process_entity(name):
            try:
                predicates, raw = get_predicates(name)
                if predicates is None:
                    record(parse_errors_path, {"error": "Failed to extract predicates", "entity": name, "response": str(raw)[:500]})
                    return
                if not predicates:
                    record(empty_results_path, {"entity": name, "message": "Empty result (no predicates)"})
                    return
                with ThreadPoolExecutor(max_workers=OBJECT_WORKERS) as pool:
                    per_predicate = list(pool.map(lambda p: get_objects(name, p), predicates))
                triples = [t for ts in per_predicate for t in ts]
                if triples:
                    write_entity(triples)
                else:
                    record(empty_results_path, {"entity": name, "message": f"Empty result ({len(predicates)} predicates, no objects)"})
                with write_lock:
                    done['entities'] += 1
                    done['triples'] += len(triples)
                    logging.info(f"Two-step [{two_step_template.name}] {name}: {len(predicates)} predicates, "
                                 f"{len(triples)} triples ({done['entities']}/{len(names)} entities done)")
            except Exception as e:
                logging.error(f"Two-step [{two_step_template.name}] {name}: not recorded after retries, "
                              f"will be redone by a run over the missing entities: {e}")

        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            list(pool.map(process_entity, names))

        logging.info(f"Two-step [{two_step_template.name}] finished: {done['entities']}/{len(names)} entities, "
                     f"{done['triples']} triples appended to {csv_path}")
        return done['triples'], len(names)
    
    if different_elicitation_triple_ranges:
        assert len(popular_ranges) == len(longtail_ranges), "Popular and long-tail ranges must have the same length."
        for tmpl_path in selected_templates:
            # Handle TwoStepTemplate vs regular template path
            if isinstance(tmpl_path, TwoStepTemplate):
                tmpl_label = tmpl_path.name
                # Skip two-step in different_elicitation_triple_ranges mode (not supported)
                logging.warning(f"Two-step template '{tmpl_label}' not supported in different_elicitation_triple_ranges mode. Skipping.")
                continue
            else:
                tmpl_label = Path(tmpl_path).stem if tmpl_path else "default"
            tmpl_base_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)

            for idx, (pop_range, lt_range) in enumerate(zip(popular_ranges, longtail_ranges), 1):
                range_tag = f"pop_{pop_range.replace('-', '_')}_lt_{lt_range.replace('-', '_')}"
                logging.info(f"[Template {tmpl_label}] [Range {idx}/{len(popular_ranges)}] popular: {pop_range}, long-tail: {lt_range}")

                system_prompt = (
                    _get_system_prompt_from_template(tmpl_path, pop_range, lt_range)
                    if tmpl_path else None
                )

                triples_output_path = os.path.join(tmpl_base_dir, range_tag, "elicited_triples.csv")
                parse_errors_path = os.path.join(tmpl_base_dir, range_tag, "entity_extraction_errors.jsonl")
                empty_results_path = os.path.join(tmpl_base_dir, range_tag, "empty_entity_results.jsonl")
                reasoning_traces_path = os.path.join(tmpl_base_dir, range_tag, "reasoning_traces.jsonl")

                entities = loadEntities()
                if not entities:
                    logging.error("No entities loaded. Skipping this range.")
                    continue
                logging.info(f"Loaded {len(entities)} entities for processing")

                if multi_turn_elicitation:
                    total_triples, processed_count = _run_multi_turn_batch_loop(
                        entities, triples_output_path, parse_errors_path, empty_results_path,
                        pop_range, lt_range, multi_turn_max_turns, multi_turn_min_new_triples,
                        system_prompt, reasoning_traces_path
                    )
                else:
                    total_triples, processed_count = _run_batch_loop(
                        entities, triples_output_path, parse_errors_path, empty_results_path,
                        pop_range, lt_range, system_prompt, reasoning_traces_path
                    )
                logging.info(f"==== Finished [{tmpl_label}] range {range_tag} ====")
                logging.info(f"Total entities processed: {processed_count}")
                logging.info(f"Total triples extracted: {total_triples}")
                logging.info(f"Output CSV: {triples_output_path}")

    elif evaluate_by_popularity:
        logging.info("==== Entity-based Triple Extraction Pipeline Started (by Popularity) ====")

        with open(entities_input_path, 'r') as file:
            entities_by_bucket = json.load(file)

        buckets = ["high", "mid", "low"]
        total_triples_grand = 0
        processed_count_grand = 0

        for tmpl_path in selected_templates:
            # Handle TwoStepTemplate vs regular template path
            if isinstance(tmpl_path, TwoStepTemplate):
                tmpl_label = tmpl_path.name
                # Skip two-step in evaluate_by_popularity mode (not supported)
                logging.warning(f"Two-step template '{tmpl_label}' not supported in evaluate_by_popularity mode. Skipping.")
                continue
            else:
                tmpl_label = Path(tmpl_path).stem if tmpl_path else "default"
            tmpl_base_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)
            system_prompt = (
                _get_system_prompt_from_template(tmpl_path) if tmpl_path else None
            )

            for bucket in buckets:
                if bucket not in entities_by_bucket:
                    logging.warning(f"Bucket {bucket} not found in entities file. Skipping.")
                    continue

                logging.info(f"[Template {tmpl_label}] [Bucket {bucket}] Processing popularity bucket")
                bucket_entities = entities_by_bucket[bucket]
                entities = [entity['title'] for entity in bucket_entities]

                if not entities:
                    logging.warning(f"No entities found in bucket {bucket}. Skipping.")
                    continue

                logging.info(f"Loaded {len(entities)} entities for bucket {bucket}")

                triples_output_path = os.path.join(tmpl_base_dir, bucket, "elicited_triples.csv")
                parse_errors_path = os.path.join(tmpl_base_dir, bucket, "entity_extraction_errors.jsonl")
                empty_results_path = os.path.join(tmpl_base_dir, bucket, "empty_entity_results.jsonl")
                reasoning_traces_path = os.path.join(tmpl_base_dir, bucket, "reasoning_traces.jsonl")

                if multi_turn_elicitation:
                    total_triples, processed_count = _run_multi_turn_batch_loop(
                        entities, triples_output_path, parse_errors_path, empty_results_path,
                        popular_ranges[4], longtail_ranges[4], multi_turn_max_turns, multi_turn_min_new_triples,
                        system_prompt, reasoning_traces_path
                    )
                else:
                    total_triples, processed_count = _run_batch_loop(
                        entities, triples_output_path, parse_errors_path, empty_results_path,
                        popular_ranges[4], longtail_ranges[4], system_prompt, reasoning_traces_path
                    )
                total_triples_grand += total_triples
                processed_count_grand += processed_count
                logging.info(f"==== Finished [{tmpl_label}] bucket {bucket} ====")
                logging.info(f"Entities processed: {processed_count}")
                logging.info(f"Triples extracted: {total_triples}")
                logging.info(f"Output CSV: {triples_output_path}\n")

        logging.info("==== Entity-based Triple Extraction Pipeline Finished (all buckets) ====")
        logging.info(f"Total entities processed: {processed_count_grand}")
        logging.info(f"Total triples extracted: {total_triples_grand}")

    else:
        logging.info("==== Entity-based Triple Extraction Pipeline Started ====")
        entities = loadEntities()
        if not entities:
            logging.error("No entities loaded. Exiting.")
            return
        logging.info(f"Loaded {len(entities)} entities for processing")

        for tmpl_path in selected_templates:
            # Check if this is a two-step template
            if isinstance(tmpl_path, TwoStepTemplate):
                logging.info(f"==== Running two-step elicitation: {tmpl_path.name} ====")
                total_triples, processed_count = _run_two_step_loop(
                    entities, tmpl_path, _prompt_output_dir(elicited_triples_dir, tmpl_path)
                )
                logging.info(f"==== Two-Step Complete [{tmpl_path.name}] ====")
                logging.info(f"Entities: {processed_count}, Triples: {total_triples}")
                continue

            tmpl_label = Path(tmpl_path).stem if tmpl_path else "default"
            tmpl_base_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)
            system_prompt = (
                _get_system_prompt_from_template(tmpl_path) if tmpl_path else None
            )

            logging.info(f"==== Running elicitation with template: {tmpl_label} ====")

            triples_output_path = os.path.join(tmpl_base_dir, "elicited_triples.csv")
            parse_errors_path = os.path.join(tmpl_base_dir, "entity_extraction_errors.jsonl")
            empty_results_path = os.path.join(tmpl_base_dir, "empty_entity_results.jsonl")
            reasoning_traces_path = os.path.join(tmpl_base_dir, "reasoning_traces.jsonl")

            if multi_turn_elicitation:
                total_triples, processed_count = _run_multi_turn_batch_loop(
                    entities, triples_output_path, parse_errors_path, empty_results_path,
                    popular_ranges[4], longtail_ranges[4], multi_turn_max_turns, multi_turn_min_new_triples,
                    system_prompt, reasoning_traces_path
                )
            else:
                total_triples, processed_count = _run_batch_loop(
                    entities, triples_output_path, parse_errors_path, empty_results_path,
                    popular_ranges[4], longtail_ranges[4], system_prompt, reasoning_traces_path
                )
            logging.info(f"==== Entity-based Triple Extraction Pipeline Finished [{tmpl_label}] ====")
            logging.info(f"Total entities processed: {processed_count}")
            logging.info(f"Total triples extracted: {total_triples}")
            logging.info(f"Output CSV: {triples_output_path}")
