import fire
from .gpt_kbc import GPTKBCRunner
from .prompter_parser import PromptJSONSchema, PromptTwoStep
from .template_utils import resolve_templates, TwoStepTemplate
from eval.network_utils import network_retry
import os
import json
import time
import csv
from pathlib import Path
from typing import Literal


def main_elicitation_openai_batched(
        model_elicitation:str,
        prompt_template_dir_elicitation:str,              
        entities_file_path:str,
        elicited_triples_dir:str,           
        poll_interval:int = 30,  # seconds between batch status checks
        evaluate_by_category:bool = False,
        reasoning_effort_elicitation:Literal["low", "medium", "high"] = None,
        different_elicitation_triple_ranges:bool = False,
        evaluate_by_popularity:bool = False,
        run_id: str = None,
        use_all_prompts: bool = False,
        prompt_templates: str = None,
        openai_api_key: str = None,
):
    
    """
    Main arguments to perform elicitation with possible different prompts

    Arguments:
        model_elicitation (str): Name of the model to use for elicitation
        prompt_template_dir_elicitation (str): Dir which stores multiple jinja files
        entities_file_path (str): File path storing the wikidata entities
        elicited_triples_dir (str): Dir path for storing the elicited triples
        poll_interval (int): Seconds between status checks (default: 300)
        evaluate_by_category (bool): Whether to evaluate by category (default: False)
        reasoning_effort_elicitation (Literal["low", "medium", "high"]): Reasoning effort level
        use_all_prompts (bool): When True, run elicitation for every .jinja file in the directory.
            Each template gets its own output subdirectory.
        prompt_templates (str): Comma-separated list of specific .jinja filenames to use.
            When set, overrides use_all_prompts.
    """

    print(f"Starting elicitation with reasoning effort: {reasoning_effort_elicitation}")

    # Resolve template selection using shared utility
    selected_templates = resolve_templates(
        prompt_template_dir=prompt_template_dir_elicitation,
        use_all_prompts=use_all_prompts,
        prompt_templates=prompt_templates,
    )
    print(f"Selected {len(selected_templates)} template(s): {[os.path.basename(t) if isinstance(t, str) else t.name for t in selected_templates]}")

    # Determine whether we are in multi-prompt mode
    multi_prompt = len(selected_templates) > 1

    # Helper: return output dir for a given template path
    def _prompt_output_dir(base_dir: str, template_path: str) -> str:
        """If multi-prompt, nest results under a subdirectory named after the template stem."""
        if not multi_prompt:
            return base_dir
        stem = Path(template_path).stem  # e.g. "prompt_elicitation.json"
        return os.path.join(base_dir, stem)

    # For single-template mode keep the same file_name variable used further below
    selected_template = selected_templates[0]
    file_name = os.path.basename(selected_template) if isinstance(selected_template, str) else selected_template.name

    # Ranges for popular and long-tail entities
    popular_ranges = ["5-10", "10-20", "25-50", "33-66", "75-150", "100-200"]
    longtail_ranges = ["1", "1-2", "3-5", "3-7", "8-15", "10-20"]

    if different_elicitation_triple_ranges:
        import logging
        import re as _re
        logging.basicConfig(level=logging.INFO)
        assert len(popular_ranges) == len(longtail_ranges), "Popular and long-tail ranges must have the same length."

        gpt_runners = []  # Keep track of all runners for polling

        for tmpl in selected_templates:
            # Dispatch based on template type
            if isinstance(tmpl, TwoStepTemplate):
                # Two-step templates are not supported for different_elicitation_triple_ranges yet
                logging.warning(f"Skipping two-step template '{tmpl.name}' in different_elicitation_triple_ranges mode. Use regular mode.")
                continue

            tmpl_path = tmpl
            tmpl_stem = Path(tmpl_path).stem
            # Base dir for this template (only nested when multi-prompt)
            tmpl_base_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)

            for idx, (pop_range, lt_range) in enumerate(zip(popular_ranges, longtail_ranges), 1):
                logging.info(f"[Template {tmpl_stem}] [Range {idx}/{len(popular_ranges)}] popular: {pop_range}, long-tail: {lt_range}")
                with open(tmpl_path, 'r') as f:
                    template_content = f.read()

                # Replace both the popular and long-tail range in the system message
                new_content = template_content
                new_content, n1 = _re.subn(
                    r"(between )([0-9]+\s*to\s*[0-9]+|[0-9]+–[0-9]+|[0-9]+ or more|[0-9]+\s*–\s*[0-9]+)",
                    lambda m: m.group(1) + pop_range,
                    new_content,
                    count=1
                )
                new_content, n2 = _re.subn(
                    r"(very low, like )([0-9]+|[0-9]+-[0-9]+|[0-9]+–[0-9]+|[0-9]+ to [0-9]+)",
                    lambda m: m.group(1) + lt_range,
                    new_content,
                    count=1
                )

                # Add range info to template and output files
                range_tag = f"pop_{pop_range.replace('-', '_')}_lt_{lt_range.replace('-', '_')}"
                tmp_template_path = f"/tmp/tmp_prompt_{tmpl_stem}_{range_tag}.jinja"
                with open(tmp_template_path, 'w') as f:
                    f.write(new_content)

                logging.info(f"Using template: {tmp_template_path}")
                prompter_parser_module = PromptJSONSchema(
                    template_path_elicitation=tmp_template_path,
                    model_elicitation=model_elicitation,
                    reasoning_effort=reasoning_effort_elicitation,
                )

                # Use range_tag in source_file_name and job_description for traceability
                gpt_runner = GPTKBCRunner(
                    source_file_name=f"{os.path.basename(tmp_template_path)}_{range_tag}",
                    curr_index=1,
                    entities_file_path=entities_file_path,
                    elicited_triples_dir=os.path.join(tmpl_base_dir, range_tag),
                    prompter_parser_module=prompter_parser_module,
                    evaluate_by_category=evaluate_by_category,
                    job_description=f"Knowledge Elicitation ({tmpl_stem} / {range_tag})",
                    run_id=run_id,
                    openai_api_key=openai_api_key,
                )

                list_of_subjects = gpt_runner.get_list_of_subjects()
                logging.info(f"Submitting batch for popular {pop_range}, long-tail {lt_range} with {len(list_of_subjects)} subjects...")
                gpt_runner.loop(subjects_to_expand=list_of_subjects)
                logging.info(f"Batch for popular {pop_range}, long-tail {lt_range} submitted.")

                gpt_runners.append(gpt_runner)  # Track this runner for polling

        logging.info("All triple ranges processed. Polling for completion...")
        _poll_until_complete(
            gpt_runners=gpt_runners,
            poll_interval=poll_interval
        )
    elif evaluate_by_popularity:
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Load entities grouped by popularity bucket
        with open(entities_file_path, 'r') as f:
            entities_by_bucket = json.load(f)
        
        # Popularity buckets in order
        buckets = ["66-100%", "33-66%", "0-33%"]
        gpt_runners = []

        for tmpl in selected_templates:
            # Dispatch based on template type
            if isinstance(tmpl, TwoStepTemplate):
                logging.warning(f"Skipping two-step template '{tmpl.name}' in evaluate_by_popularity mode. Use regular mode.")
                continue

            tmpl_path = tmpl
            tmpl_stem = Path(tmpl_path).stem
            tmpl_base_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)

            for bucket in buckets:
                if bucket not in entities_by_bucket:
                    logging.warning(f"Bucket {bucket} not found in entities file. Skipping.")
                    continue

                bucket_entities = entities_by_bucket[bucket]
                logging.info(f"[Template {tmpl_stem}] [Bucket {bucket}] Processing {len(bucket_entities)} entities")

                prompter_parser_module = PromptJSONSchema(
                    template_path_elicitation=tmpl_path,
                    model_elicitation=model_elicitation,
                    reasoning_effort=reasoning_effort_elicitation,
                )

                # Use bucket tag in source_file_name and output directory
                gpt_runner = GPTKBCRunner(
                    source_file_name=f"{tmpl_stem}_{bucket.replace('%', 'pct').replace('-', '_')}",
                    curr_index=1,
                    entities_file_path=entities_file_path,
                    elicited_triples_dir=os.path.join(tmpl_base_dir, bucket),
                    prompter_parser_module=prompter_parser_module,
                    evaluate_by_category=evaluate_by_category,
                    job_description=f"Knowledge Elicitation ({tmpl_stem} / Popularity {bucket})",
                    run_id=run_id,
                    openai_api_key=openai_api_key,
                )

                # Extract titles from bucket entities
                list_of_subjects = [entity['title'] for entity in bucket_entities]
                logging.info(f"Submitting batch for bucket {bucket} with {len(list_of_subjects)} subjects...")
                gpt_runner.loop(subjects_to_expand=list_of_subjects)
                logging.info(f"Batch for bucket {bucket} submitted.")

                gpt_runners.append(gpt_runner)

        logging.info("All popularity buckets processed. Polling for completion...")
        _poll_until_complete(
            gpt_runners=gpt_runners,
            poll_interval=poll_interval
        )
    else:
        # ---- Multi-template loop (or single-template when len == 1) ----
        gpt_runners = []
        for tmpl in selected_templates:
            # Dispatch based on template type
            if isinstance(tmpl, TwoStepTemplate):
                # Run two-step elicitation
                two_step_runner = _run_two_step_batched(
                    two_step_template=tmpl,
                    entities_file_path=entities_file_path,
                    model_elicitation=model_elicitation,
                    elicited_triples_dir=elicited_triples_dir,
                    reasoning_effort=reasoning_effort_elicitation,
                    evaluate_by_category=evaluate_by_category,
                    run_id=run_id,
                    poll_interval=poll_interval,
                    openai_api_key=openai_api_key,
                )
                if two_step_runner:
                    gpt_runners.append(two_step_runner)
                continue

            # Single-prompt template (existing logic)
            tmpl_path = tmpl
            tmpl_name = os.path.basename(tmpl_path)
            out_dir = _prompt_output_dir(elicited_triples_dir, tmpl_path)

            prompter_parser_module = PromptJSONSchema(
                template_path_elicitation=tmpl_path,
                model_elicitation=model_elicitation,
                reasoning_effort=reasoning_effort_elicitation,
            )

            gpt_runner = GPTKBCRunner(
                source_file_name=tmpl_name,
                curr_index=1,
                entities_file_path=entities_file_path,
                elicited_triples_dir=out_dir,
                prompter_parser_module=prompter_parser_module,
                evaluate_by_category=evaluate_by_category,
                run_id=run_id,
                openai_api_key=openai_api_key,
            )

            list_of_subjects = gpt_runner.get_list_of_subjects()
            gpt_runner.loop(subjects_to_expand=list_of_subjects)
            gpt_runners.append(gpt_runner)

        # After all batches submitted, poll until all are done
        if gpt_runners:
            _poll_until_complete(
                gpt_runners=gpt_runners,
                poll_interval=poll_interval
            )


def _run_two_step_batched(
    two_step_template: TwoStepTemplate,
    entities_file_path: str,
    model_elicitation: str,
    elicited_triples_dir: str,
    reasoning_effort: str,
    evaluate_by_category: bool,
    run_id: str,
    poll_interval: int,
    openai_api_key: str = None,
):
    """
    Run two-step elicitation via OpenAI Batch API.

    Step 1: For each subject, elicit predicates using predicates.jinja
    Step 2: For each (subject, predicate) pair, elicit objects using objects.jinja
    Then assemble triples and write to CSV.
    """
    import logging
    from loguru import logger
    from openai import OpenAI
    from .prompter_parser.exceptions import ParsingException

    logging.basicConfig(level=logging.INFO)
    logger.info(f"=== Starting Two-Step Elicitation: {two_step_template.name} ===")

    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        openai_client = OpenAI()
    prompter = PromptTwoStep(
        predicate_template_path=two_step_template.predicate_template,
        object_template_path=two_step_template.object_template,
        model_elicitation=model_elicitation,
        reasoning_effort=reasoning_effort,
    )

    # Output directory for this strategy
    out_dir = os.path.join(elicited_triples_dir, two_step_template.name)
    os.makedirs(out_dir, exist_ok=True)

    # Load entities
    with open(entities_file_path, 'r') as f:
        json_content = json.load(f)

    if evaluate_by_category:
        entities = [item["title"] for category in json_content.values() for item in category]
    else:
        entities = [i['title'] for i in json_content]

    logger.info(f"Loaded {len(entities)} entities for processing")

    # === STEP 1: Predicate Elicitation ===
    logger.info("=== Step 1: Predicate Elicitation ===")

    # Create batch requests for all entities
    batch_requests = []
    for entity in entities:
        req = prompter.get_predicate_prompt(entity)
        batch_requests.append(req)

    # Write to temp file and upload
    temp_batch_file = f"/tmp/two_step_predicates_{int(time.time())}.jsonl"
    with open(temp_batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    # Upload to OpenAI
    with open(temp_batch_file, "rb") as f:
        input_file = openai_client.files.create(file=f, purpose="batch")

    # Create batch for Step 1
    batch_step1 = openai_client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Two-Step Elicitation - Step 1: Predicates ({two_step_template.name})"}
    )
    logger.info(f"Step 1 batch submitted: {batch_step1.id}")

    # Poll for completion
    while True:
        batch_status = openai_client.batches.retrieve(batch_step1.id)
        if batch_status.status == "completed":
            break
        elif batch_status.status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Step 1 batch failed with status: {batch_status.status}")
        logger.info(f"Step 1 batch status: {batch_status.status}. Waiting {poll_interval}s...")
        time.sleep(poll_interval)

    # Download Step 1 results
    result_content = openai_client.files.content(batch_status.output_file_id).content
    step1_results_path = f"/tmp/step1_results_{batch_step1.id}.json"
    with open(step1_results_path, "wb") as f:
        f.write(result_content)

    # Parse Step 1 results to get predicates per entity
    entity_to_predicates: dict[str, list[str]] = {}
    parse_errors = []

    with open(step1_results_path, "r") as f:
        for line in f:
            try:
                # DEBUG: Log the raw response
                logger.debug(f"Step 1 raw response: {line[:200]}...")
                subject_name, predicates = prompter.parse_predicate_response(line)
                entity_to_predicates[subject_name] = predicates
            except Exception as e:
                parse_errors.append({"error": str(e), "line": line[:500]})
                logger.warning(f"Failed to parse Step 1 response: {e}")

    # Save any parse errors
    if parse_errors:
        errors_path = os.path.join(out_dir, "step1_parse_errors.jsonl")
        with open(errors_path, "w") as f:
            for err in parse_errors:
                f.write(json.dumps(err) + "\n")
        logger.warning(f"Saved {len(parse_errors)} Step 1 parse errors to {errors_path}")

    logger.info(f"Step 1 complete. Entities with predicates: {len(entity_to_predicates)}")

    # === STEP 2: Object Elicitation ===
    logger.info("=== Step 2: Object Elicitation ===")

    # Create batch requests for all (entity, predicate) pairs
    batch_requests_step2 = []
    for entity, predicates in entity_to_predicates.items():
        for pred in predicates:
            req = prompter.get_object_prompt(entity, pred)
            batch_requests_step2.append(req)

    logger.info(f"Submitting {len(batch_requests_step2)} requests for Step 2")

    if not batch_requests_step2:
        logger.warning("No Step 2 requests to submit (no predicates collected). Writing empty CSV.")
        csv_path = os.path.join(out_dir, "elicited_triples.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "subject_name"])
            writer.writeheader()
        return None

    # Write to temp file and upload
    temp_batch_file2 = f"/tmp/two_step_objects_{int(time.time())}.jsonl"
    with open(temp_batch_file2, "w") as f:
        for req in batch_requests_step2:
            f.write(json.dumps(req) + "\n")

    # Upload to OpenAI
    with open(temp_batch_file2, "rb") as f:
        input_file2 = openai_client.files.create(file=f, purpose="batch")

    # Create batch for Step 2
    batch_step2 = openai_client.batches.create(
        input_file_id=input_file2.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Two-Step Elicitation - Step 2: Objects ({two_step_template.name})"}
    )
    logger.info(f"Step 2 batch submitted: {batch_step2.id}")

    # Poll for completion
    while True:
        batch_status2 = openai_client.batches.retrieve(batch_step2.id)
        if batch_status2.status == "completed":
            break
        elif batch_status2.status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Step 2 batch failed with status: {batch_status2.status}")
        logger.info(f"Step 2 batch status: {batch_status2.status}. Waiting {poll_interval}s...")
        time.sleep(poll_interval)

    # Download Step 2 results
    result_content2 = openai_client.files.content(batch_status2.output_file_id).content
    step2_results_path = f"/tmp/step2_results_{batch_step2.id}.json"
    with open(step2_results_path, "wb") as f:
        f.write(result_content2)

    # Parse Step 2 results and assemble triples
    all_triples = []
    parse_errors_step2 = []

    with open(step2_results_path, "r") as f:
        for line in f:
            try:
                # DEBUG: Log the raw response
                logger.debug(f"Step 2 raw response: {line[:200]}...")
                triples = prompter.parse_object_response(line)
                all_triples.extend(triples)
            except Exception as e:
                parse_errors_step2.append({"error": str(e), "line": line[:500]})
                logger.warning(f"Failed to parse Step 2 response: {e}")

    # Save any parse errors
    if parse_errors_step2:
        errors_path2 = os.path.join(out_dir, "step2_parse_errors.jsonl")
        with open(errors_path2, "w") as f:
            for err in parse_errors_step2:
                f.write(json.dumps(err) + "\n")
        logger.warning(f"Saved {len(parse_errors_step2)} Step 2 parse errors to {errors_path2}")

    logger.info(f"Step 2 complete. Total triples collected: {len(all_triples)}")

    # === Write CSV ===
    csv_path = os.path.join(out_dir, "elicited_triples.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "subject_name"])
        writer.writeheader()
        writer.writerows(all_triples)

    logger.info(f"Triples written to {csv_path}")
    logger.info(f"=== Two-Step Elicitation Complete: {two_step_template.name} ===")

    # Return None - no runner to poll since we handled everything directly
    return None


def _poll_until_complete(gpt_runners=None, gpt_runner=None, poll_interval=None):
    """
    Poll until all batches are completed.
    Accepts either a list of runners (gpt_runners) or a single runner (gpt_runner) for backward compatibility.
    """

    import time
    from loguru import logger
    import os

    # Handle both single runner and list of runners
    if gpt_runners is None:
        gpt_runners = [gpt_runner]
    elif not isinstance(gpt_runners, list):
        gpt_runners = [gpt_runners]

    logger.info(f"Starting auto-verification loop. Checking every {poll_interval}s...")

    start_time = time.time()
    max_wait = 60 * 60 * 2  # 2 hours max

    while True:

        if time.time() - start_time > max_wait:
            logger.error("Timeout reached. Exiting polling loop.")
            break

        # Check status updates for all runners
        for runner in gpt_runners:
            runner.check_batch_status_dir()
            runner.check_and_process_all()

        # Check if any in-progress files remain across all runners
        total_in_progress = 0
        for runner in gpt_runners:
            in_progress_files = [
                f for f in os.listdir(runner.in_progress_dir_path)
                if f.startswith("in_progress_") and f.endswith(".json")
            ] if os.path.exists(runner.in_progress_dir_path) else []
            total_in_progress += len(in_progress_files)

        if total_in_progress == 0:
            logger.info("All batches completed and processed. Exiting.")
            break

        logger.info(
            f"{total_in_progress} batch(es) still running. Sleeping {poll_interval}s..."
        )

        time.sleep(poll_interval)

    # Cleanup directories after all polling is complete
    for runner in gpt_runners:
        runner.cleanup_dirs()
    logger.info("Cleanup complete.")
