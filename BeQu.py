from typing import Literal
import fire
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
from concurrent.futures import ThreadPoolExecutor, as_completed
from elicitation import main_elicitation_openai_batched, main_elicitation_other
from elicitation.gpt_kbc import GPTKBCRunner
from elicitation.prompter_parser import PromptJSONSchema
from eval import main_evaluation
from experiment_tracker import ExperimentTracker
import re

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)

def compute_f1(precision_val, recall_val):
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)


def combine_all_results(results_dir_path: str, results_summary: list) -> str:
    """
    Combines all individual model results into a single summary CSV file.
    For each (Model, Setting) pair, compute F1 score from Precision and Recall rows.
    The file is saved in /results directory (outside of any model subdir).
    
    Args:
        results_dir_path: Base results directory (e.g., ./OKBENCH/results)
        results_summary: List of tuples (model_name, results_path)
    
    Returns:
        Path to the combined summary file
    """
    import csv
    
    base_results_dir = os.path.join(os.path.dirname(results_dir_path), "combined_results")
    combined_path = os.path.join(base_results_dir, "combined_results_automatic.csv")
    
    os.makedirs(base_results_dir, exist_ok=True)
    
    all_rows = []
    
    for model_name, results_path in results_summary:
        if not results_path or not os.path.exists(results_path):
            continue
        
        for root, dirs, files in os.walk(results_path):
            for f in files:
                if f in ["results.csv", "results_by_category.csv", "results_by_popularity.csv"]:
                    csv_path = os.path.join(root, f)
                    try:
                        with open(csv_path, 'r', newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                row_copy = dict(row)
                                row_copy['Model'] = model_name
                                
                                subdir = os.path.relpath(root, results_path)
                                if subdir.endswith("_rag_eval"):
                                    subdir = os.path.dirname(subdir)
                                row_copy['Setting'] = subdir
                                
                                row_copy['Result_File'] = f
                                
                                all_rows.append(row_copy)
                    except Exception as e:
                        print(f"Warning: Could not read {csv_path}: {e}")
    
    if not all_rows:
        print("Warning: No results found to combine")
        return combined_path
    
    grouped = {}
    for row in all_rows:
        model = row.get('Model', '')
        setting = row.get('Setting', '')
        domain = row.get('Category', '') if setting == 'domains' else ''
        key = (model, setting, domain)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    
    final_rows = []
    for (model, setting, domain), rows in grouped.items():
        precision_row = None
        recall_row = None
        for row in rows:
            metric = row.get('Metric', '')
            if 'Precision' in metric:
                precision_row = row
            elif 'Recall' in metric:
                recall_row = row
        
        combined = {
            'Model': model,
            'Setting': setting,
            'Domain': domain if domain else '',
            'Total #Triples': rows[0].get('Total #Triples', '500') if rows else '500',
        }
        
        for col in ['Entailment', 'Contradiction', 'Neutral', 'Entailment_ratio', 'Contradiction_ratio', 'Neutral_ratio', 'Error_count', 'Error_ratio']:
            prec_val = float(precision_row.get(col, 0)) if precision_row else 0
            rec_val = float(recall_row.get(col, 0)) if recall_row else 0
            combined[f'{col}_Precision'] = prec_val
            combined[f'{col}_Recall'] = rec_val
        
        if precision_row and recall_row:
            prec_ent = float(precision_row.get('Entailment_ratio', 0))
            rec_ent = float(recall_row.get('Entailment_ratio', 0))
            combined['Entailment_F1'] = compute_f1(prec_ent, rec_ent)
        else:
            combined['Entailment_F1'] = 0.0
        
        combined['Result_File'] = rows[0].get('Result_File', '') if rows else ''
        
        final_rows.append(combined)
    
    fieldnames = ['Model', 'Setting', 'Domain', 'Result_File', 'Total #Triples',
                  'Entailment_Precision', 'Entailment_Recall', 'Entailment_F1',
                  'Contradiction_Precision', 'Contradiction_Recall',
                  'Neutral_Precision', 'Neutral_Recall',
                  'Entailment_ratio_Precision', 'Entailment_ratio_Recall',
                  'Contradiction_ratio_Precision', 'Contradiction_ratio_Recall',
                  'Neutral_ratio_Precision', 'Neutral_ratio_Recall',
                  'Error_count_Precision', 'Error_count_Recall',
                  'Error_ratio_Precision', 'Error_ratio_Recall']
    
    with open(combined_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)
    
    return combined_path

def BeQu(
        entities_file_path:str = None,
        api:Literal["openai_batched", "custom", "openrouter"] = None,
        model_elicitation = None,
        elicited_triples_dir:str = None,
        ground_truth_dir_path:str = None,
        results_dir_path:str = None,
        prompt_template_dir_elicitation:str = None,
        reasoning_effort_elicitation:Literal["low", "medium", "high"] = None,
        llm_judge:str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        llm_judge_api:Literal["custom", "openrouter"] = "custom",
        seed:int = 42,
        sample_size:int = 500,
        evaluate_by_category:bool = False,
        triples_per_category:int = 500,
        non_existing_entities:bool = False,
        different_elicitation_triple_ranges:bool = False,
        evaluate_by_popularity:bool = False,
        skip_if_exists:bool = True,
        skip_elicitation:bool = False,
        skip_evaluation:bool = False,
        build_ground_truth_only:bool = False,
        run_id: str = None,
        use_all_prompts: bool = False,
        prompt_templates: str = None,
        web_results_count: int = 20,
        web_docs_for_eval: int = 20,
        evaluate_all_models: bool = False,
        top_k: int = 10,
        recover_errors: bool = False,
        recover_errors_dir: str = None,
        ):

        """
        Main function to run the entire BeQu pipeline, from elicitation to evaluation.
        Arguments:
                entities_file_path (str): File path storing the entities (random, by domain, by popularity)
                api (Literal["openai_batched", "custom", "openrouter"]): API to use for elicitation
                model_elicitation (str): Name of the model to use for elicitation
                prompt_template_dir_elicitation (str): Dir which stores multiple jinja files for elicitation
                reasoning_effort_elicitation:Literal["low", "medium", "high"] = None: Reasoning effort level for elicitation prompts. Works only for models that support this parameter
                elicited_triples_dir (str): Dir path for storing the elicited triples
                ground_truth_dir_path (str): Dir path where the ground truth triples are stored
                results_dir_path (str): Dir path for storing the evaluation results
                llm_judge (str): Name of the model to use as judge in the evaluation
                llm_judge_api (Literal["custom", "openrouter"]): API backend for the LLM judge. Use "openrouter" to route judge calls through OpenRouter instead of custom (default: "custom")
                seed (int): Random seed for sampling in evaluation (default: 42)
                sample_size (int): Number of triples to sample for evaluation (default: 100)
                evaluate_by_category (bool): Whether to evaluate by category (default: False)
                triples_per_category (int): Number of triples to sample per category if evaluate_by_category is True (default: 500)
                skip_if_exists (bool): Skip experiment if same config was already run (default: True)
                different_elicitation_triple_ranges (bool): Whether to use different triple ranges for popular vs long-tail entities during elicitation (default: False)
                evaluate_by_popularity (bool): Whether to evaluate results by entity popularity (default: False)
                non_existing_entities (bool): Whether to include non-existing entities in the elicitation (default: False)
                skip_elicitation (bool): Whether to skip the elicitation step and only run evaluation (default: False)
                skip_evaluation (bool): Whether to skip the evaluation step and only run elicitation (default: False)
                build_ground_truth_only (bool): When True, only builds ground truth and skips both elicitation and evaluation (default: False)
                run_id (str): Optional identifier used to isolate intermediate files when running
                    multiple BeQu processes simultaneously. If omitted a unique id is
                    generated automatically.
                use_all_prompts (bool): When True, run elicitation for every .jinja file in the template directory, storing results in per-template subdirectories (default: False).
                prompt_templates (str): Comma-separated list of specific .jinja filenames (e.g. "prompt_a.jinja,prompt_b.jinja") to use for elicitation. Overrides use_all_prompts when set (default: None).
                 web_results_count (int): Number of Brave Search results to fetch during ground truth construction (default: 20). Maximum is 20 due to Brave API limits.
                 web_docs_for_eval (int): Number of web documents/results to use during evaluation (default: 20). Must be <= web_results_count.
                 evaluate_all_models (bool): When True, automatically discover all models and subdirectories in elicited_triples_dir that contain elicited_triples.csv and run evaluation for each. Results directories will mirror the elicited_triples structure (default: False). Uses skip_if_exists to avoid re-running evaluations.
                  top_k (int): Number of top passages to retrieve in RAG-based precision evaluation (default: 10).
         
        Evaluation approach:
                Precision is computed using RAG-based evaluation: for each elicited triple, the top-k passages (configurable via top_k parameter)
                are retrieved from the full ground truth (Wikipedia article + all web documents)
                using dense (embedding-based) retrieval. A single LLM call determines entailment/contradiction/neutral.
                Recall compares ground truth triples against elicited triples to check coverage.
        """
        # Compute base directory (directory containing BeQu.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Error recovery mode: re-process failed triples
        if recover_errors and recover_errors_dir:
            from eval.recover_errors import recover_errors as do_recover_errors
            if ground_truth_dir_path is None:
                ground_truth_dir_path = os.path.join(base_dir, "ground_truth", "random", "200")
            logger.info(f"Running error recovery mode")
            do_recover_errors(
                    root_results_dir=recover_errors_dir,
                    llm_judge=llm_judge,
                    ground_truth_dir_path=ground_truth_dir_path,
                    web_docs_for_eval=web_docs_for_eval,
                    top_k=top_k,
                )
            return
        
        # Set default paths if not provided
        if elicited_triples_dir is None:
            elicited_triples_dir = os.path.join(base_dir, "elicited_triples")
        if results_dir_path is None:
            results_dir_path = os.path.join(base_dir, "results")
        if prompt_template_dir_elicitation is None:
            prompt_template_dir_elicitation = os.path.join(base_dir, "elicitation", "templates", "prompts")
        
        # Set judge API URL and key based on llm_judge_api
        if llm_judge_api == "openrouter":
                judge_api_url = "https://openrouter.ai/api/v1"
                judge_api_key = OPENROUTER_API_KEY
        else:  # default: custom
                judge_api_url = os.getenv("CUSTOM_BASE_URL")
                judge_api_key = CUSTOM_API_KEY

        # Set API URL and key based on selected API
        api_url_elicitation = None  # Default to None; will be set based on api if needed
        api_key_elicitation = None
        if api == "custom":
                api_url_elicitation = os.getenv("CUSTOM_BASE_URL")
                api_key_elicitation = CUSTOM_API_KEY
        elif api == "openrouter":
                api_url_elicitation = "https://openrouter.ai/api/v1"
                api_key_elicitation = OPENROUTER_API_KEY
        # For openai_batched, use provided or default values (likely OpenAI's API)
        elif api == "openai_batched":
                api_key_elicitation = OPENAI_API_KEY

        # Initialize experiment tracker
        tracker = ExperimentTracker(".experiment_tracking.json")

        def discover_all_model_configs():
                all_configs = []
                if not os.path.exists(elicited_triples_dir):
                        print(f"Warning: elicited_triples_dir does not exist: {elicited_triples_dir}")
                        return all_configs
                
                def get_entities_file_path(eval_by_pop, eval_by_cat, non_exist):
                        default_entities = os.path.join(base_dir, "ENTITIES", "EXPERIMENTS_200_random_entities_wikipedia.json")
                        if eval_by_cat:
                                entities_base = os.path.join(os.path.dirname(entities_file_path) if entities_file_path else os.path.join(base_dir, "ENTITIES"), "DOMAINS")
                                if os.path.isdir(entities_base):
                                        domain_file = os.path.join(entities_base, "wikipedia_entities_by_domain_1000.json")
                                        if os.path.exists(domain_file):
                                                return domain_file
                        if eval_by_pop:
                                entities_base = os.path.join(os.path.dirname(entities_file_path) if entities_file_path else os.path.join(base_dir, "ENTITIES"), "POPULARITY")
                                if os.path.isdir(entities_base):
                                        pop_file = os.path.join(entities_base, "EXPERIMENTS_200_random_entities_wikipedia_popularity.json")
                                        if os.path.exists(pop_file):
                                                return pop_file
                        return entities_file_path if entities_file_path else default_entities
                
                def get_ground_truth_dir_path(eval_by_cat, eval_by_pop):
                        default_gt = os.path.join(base_dir, "ground_truth", "random", "200")
                        if eval_by_cat:
                                gt_base = os.path.join(os.path.dirname(ground_truth_dir_path) if ground_truth_dir_path else os.path.join(base_dir, "ground_truth"), "domains")
                                if os.path.isdir(gt_base):
                                        return gt_base
                        if eval_by_pop:
                                gt_base = os.path.join(os.path.dirname(ground_truth_dir_path) if ground_truth_dir_path else os.path.join(base_dir, "ground_truth"), "popularity")
                                if os.path.isdir(gt_base):
                                        return gt_base
                        return ground_truth_dir_path if ground_truth_dir_path else default_gt
                
                for model_name in os.listdir(elicited_triples_dir):
                        model_path = os.path.join(elicited_triples_dir, model_name)
                        if not os.path.isdir(model_path):
                                continue
                        
                        model_sanitized = sanitize_filename(model_name)
                        
                        for root, dirs, files in os.walk(model_path):
                                if "elicited_triples.csv" in files or any(f.startswith("elicited_triples") and f.endswith(".csv") for f in files):
                                        rel_path = os.path.relpath(root, model_path)
                                        
                                        if rel_path == ".":
                                                subdir = "random"
                                        else:
                                                subdir = rel_path
                                        
                                        eval_by_pop = False
                                        eval_by_cat = False
                                        non_exist = False
                                        diff_ranges = False
                                        reason_effort = None
                                        use_prompts = False
                                        
                                        parts = subdir.split(os.sep)
                                        first_part = parts[0]
                                        
                                        if first_part == "popularity":
                                                eval_by_pop = True
                                        elif first_part == "domains":
                                                eval_by_cat = True
                                                if len(parts) > 1 and parts[1] == "non-existent":
                                                        continue
                                        elif first_part == "ranges":
                                                diff_ranges = True
                                        elif first_part == "random":
                                                if len(parts) > 1:
                                                        reason_effort = parts[1]
                                        elif first_part == "prompts":
                                                use_prompts = True
                                        
                                        results_subdir = subdir
                                        
                                        config_entities_file_path = get_entities_file_path(eval_by_pop, eval_by_cat, non_exist)
                                        config_ground_truth_dir_path = get_ground_truth_dir_path(eval_by_cat, eval_by_pop)
                                        
                                        config = {
                                                "entities_file_path": config_entities_file_path,
                                                "api": api,
                                                "model_elicitation": model_name,
                                                "prompt_template_dir_elicitation": prompt_template_dir_elicitation,
                                                "reasoning_effort_elicitation": reason_effort,
                                                "non_existing_entities": non_exist,
                                                "elicited_triples_dir": root,
                                                "ground_truth_dir_path": config_ground_truth_dir_path,
                                                "results_dir_path": os.path.join(results_dir_path, model_sanitized, results_subdir) if results_dir_path else None,
                                                "api_url_elicitation": api_url_elicitation,
                                                "api_key_elicitation": api_key_elicitation,
                                                "llm_judge": llm_judge,
                                                "seed": seed,
                                                "sample_size": sample_size,
                                                "evaluate_by_category": eval_by_cat,
                                                "triples_per_category": triples_per_category,
                                                "different_elicitation_triple_ranges": diff_ranges,
                                                "evaluate_by_popularity": eval_by_pop,
                                                "use_all_prompts": use_prompts,
                                                "prompt_templates": None,
                                        }
                                        
                                        all_configs.append((model_name, config))
                
                return all_configs

        if evaluate_all_models:
                print(f"\n{'='*70}")
                print("DISCOVERING ALL MODELS AND SETTINGS IN elicited_triples...")
                print(f"{'='*70}\n")
                
                all_configs = discover_all_model_configs()
                total = len(all_configs)
                print(f"Found {total} model/subdirectory configurations to evaluate")
                
                if total == 0:
                        print("No configurations found. Exiting.")
                        return []
                
                results_summary = []

                # Pre-build ground truth once before parallel evaluation to avoid
                # multiple workers racing to fetch the same Brave Search results.
                print(f"\n{'='*70}")
                print("Pre-building ground truth before parallel evaluation...")
                print(f"{'='*70}\n")
                from eval import main_evaluation as _main_eval
                _sample_config = all_configs[0][1]
                _main_eval(
                        entities_file_path=_sample_config["entities_file_path"],
                        elicited_triples_dir=_sample_config["elicited_triples_dir"],
                        ground_truth_dir_path=_sample_config["ground_truth_dir_path"],
                        results_dir_path=None,
                        llm_judge=llm_judge,
                        seed=seed,
                        sample_size=sample_size,
                        evaluate_by_category=evaluate_by_category,
                        triples_per_category=triples_per_category,
                        evaluate_by_popularity=evaluate_by_popularity,
                        triples_per_popularity_bucket=triples_per_category,
                        web_results_count=web_results_count,
                        web_docs_for_eval=web_docs_for_eval,
                        build_ground_truth_only=True,
                        top_k=top_k,
                        judge_api_url=judge_api_url,
                        judge_api_key=judge_api_key,
                )
                print(f"Ground truth ready. Starting parallel evaluation...")

                def evaluate_single_config(args):
                        i, (model, config), total = args
                        model_name = config["model_elicitation"]
                        results_path = config["results_dir_path"]
                        
                        print(f"\n{'='*70}")
                        print(f"Model {i}/{total}: {model_name}")
                        print(f"Subdir: {os.path.relpath(config['elicited_triples_dir'], os.path.join(elicited_triples_dir, model_name))}")
                        print(f"Entities: {config['entities_file_path']}")
                        print(f"Ground Truth: {config['ground_truth_dir_path']}")
                        print(f"Results: {results_path}")
                        print(f"{'='*70}")
                        
                        if skip_if_exists:
                                previous_run = tracker.check_experiment(config)
                                if previous_run:
                                        print(f"EXPERIMENT ALREADY RUN. Skipping.")
                                        print(f"Previous run: {previous_run}")
                                        return None
                        
                        if config.get("non_existing_entities"):
                                print(f"Skipping evaluation for non-existing entities. No ground truth available.")
                                return None
                        
                        from eval import main_evaluation
                        main_evaluation(
                                entities_file_path=config["entities_file_path"],
                                elicited_triples_dir=config["elicited_triples_dir"],
                                ground_truth_dir_path=config["ground_truth_dir_path"],
                                results_dir_path=config["results_dir_path"],
                                llm_judge=config["llm_judge"],
                                seed=config["seed"],
                                sample_size=config["sample_size"],
                                evaluate_by_category=config["evaluate_by_category"],
                                triples_per_category=config["triples_per_category"],
                                evaluate_by_popularity=config["evaluate_by_popularity"],
                                triples_per_popularity_bucket=triples_per_category,
                                web_results_count=web_results_count,
                                web_docs_for_eval=web_docs_for_eval,
                                build_ground_truth_only=False,
                                top_k=top_k,
                                judge_api_url=judge_api_url,
                                judge_api_key=judge_api_key,
                        )
                        
                        tracker.register_experiment(config, results_path=config["results_dir_path"])
                        print(f"Completed evaluation {i}/{total}")
                        return (model_name, config["results_dir_path"])
                
                configs_with_index = list(enumerate(all_configs, 1))
                
                completed_count = 0
                failed_count = 0
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_config = {executor.submit(evaluate_single_config, (i, mc, total)): (i, mc) 
                                            for i, mc in configs_with_index}
                        for future in as_completed(future_to_config):
                                i, (model, config) = future_to_config[future]
                                try:
                                        result = future.result()
                                        if result is not None:
                                                results_summary.append(result)
                                                completed_count += 1
                                        else:
                                                failed_count += 1
                                except Exception as exc:
                                        failed_count += 1
                                        print(f"Model {config.get('model_elicitation', 'unknown')} generated an exception: {exc}")
                
                print(f"\n{'='*70}")
                print(f"ALL EVALUATIONS COMPLETED: {completed_count}/{total} successful, {failed_count} skipped/failed")
                print(f"{'='*70}\n")
                
                if results_summary:
                        combined_summary_path = combine_all_results(results_dir_path, results_summary)
                        print(f"Combined results saved to: {combined_summary_path}")
                
                return results_summary
        
        if isinstance(model_elicitation, list):
                models = model_elicitation
        elif isinstance(model_elicitation, str):
                stripped = model_elicitation.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                        models = [m.strip().strip("'\"") for m in stripped[1:-1].split(",") if m.strip()]
                else:
                        models = [model_elicitation]
        else:
                models = []

        # Auto-discover models from elicited_triples_dir when skipping elicitation.
        # Only includes folders where the expected subpath (matching the same flags used
        # during evaluation) contains at least one CSV — this naturally excludes nested
        # experiment directories like gpt_evolution_experiment.
        #
        # With --reasoning_effort_elicitation auto, discovers all effort variants
        # (bare random/ and any random/<effort>/ subdirs) per model and builds a flat
        # list of (model, effort) jobs.
        auto_effort_jobs = []  # list of (model_folder_name, effort_or_None)

        if skip_elicitation and not models and os.path.isdir(elicited_triples_dir):
                def _has_csvs(path):
                        return os.path.isdir(path) and any(f.endswith(".csv") for f in os.listdir(path))

                def _expected_subpath(model_folder):
                        if evaluate_by_popularity:
                                return os.path.join(model_folder, "popularity")
                        elif evaluate_by_category:
                                return os.path.join(model_folder, "domains")
                        elif non_existing_entities:
                                return os.path.join(model_folder, "domains", "non-existent")
                        elif different_elicitation_triple_ranges:
                                return os.path.join(model_folder, "ranges")
                        elif reasoning_effort_elicitation and isinstance(reasoning_effort_elicitation, str) and reasoning_effort_elicitation not in ("auto",) and "," not in reasoning_effort_elicitation:
                                return os.path.join(model_folder, "random", reasoning_effort_elicitation)
                        elif use_all_prompts or prompt_templates is not None:
                                return os.path.join(model_folder, "prompts")
                        else:
                                return os.path.join(model_folder, "random")

                # Parse effort list. Fire may pass "none,medium" as a tuple ('none','medium').
                # Normalise to a list or the sentinel "auto".
                effort_list = None
                if isinstance(reasoning_effort_elicitation, (list, tuple)):
                        effort_list = [None if e.strip().lower() == "none" else e.strip() for e in reasoning_effort_elicitation]
                elif reasoning_effort_elicitation == "auto":
                        effort_list = "auto"
                elif reasoning_effort_elicitation and "," in reasoning_effort_elicitation:
                        effort_list = [
                                None if e.strip().lower() == "none" else e.strip()
                                for e in reasoning_effort_elicitation.split(",")
                        ]

                if effort_list is not None:
                        # Discover models, then build (model, effort) jobs for each requested effort
                        for d in sorted(os.listdir(elicited_triples_dir)):
                                model_dir = os.path.join(elicited_triples_dir, d)
                                if not os.path.isdir(model_dir):
                                        continue
                                random_dir = os.path.join(model_dir, "random")
                                if not os.path.isdir(random_dir):
                                        continue
                                if effort_list == "auto":
                                        # Include bare random/ and all effort subdirs that have CSVs
                                        if _has_csvs(random_dir):
                                                auto_effort_jobs.append((d, None))
                                        for sub in sorted(os.listdir(random_dir)):
                                                sub_path = os.path.join(random_dir, sub)
                                                if os.path.isdir(sub_path) and _has_csvs(sub_path):
                                                        auto_effort_jobs.append((d, sub))
                                else:
                                        # Include only the explicitly requested efforts that exist
                                        for effort in effort_list:
                                                path = random_dir if effort is None else os.path.join(random_dir, effort)
                                                if _has_csvs(path):
                                                        auto_effort_jobs.append((d, effort))
                        models = list(dict.fromkeys(m for m, _ in auto_effort_jobs))
                        if auto_effort_jobs:
                                print(f"Discovered {len(auto_effort_jobs)} (model, effort) jobs to evaluate:")
                                for m, e in auto_effort_jobs:
                                        print(f"  {m}  effort={e or 'none'}")
                else:
                        models = [
                                d for d in sorted(os.listdir(elicited_triples_dir))
                                if os.path.isdir(os.path.join(elicited_triples_dir, d))
                                and _has_csvs(_expected_subpath(os.path.join(elicited_triples_dir, d)))
                        ]
                        if models:
                                print(f"Auto-discovered {len(models)} models to evaluate: {models}")

        results_summary = []

        def run_single_model(model, effort_override=None):
                model_sanitized = sanitize_filename(model)
                is_multi_effort = reasoning_effort_elicitation and (
                        isinstance(reasoning_effort_elicitation, (list, tuple))
                        or reasoning_effort_elicitation == "auto"
                        or (isinstance(reasoning_effort_elicitation, str) and "," in reasoning_effort_elicitation)
                )
                effective_effort = effort_override if effort_override is not None else (
                        None if is_multi_effort else reasoning_effort_elicitation
                )
                elicited_triples_dir_value = (
                os.path.join(elicited_triples_dir, model_sanitized, "popularity") if evaluate_by_popularity
                else os.path.join(elicited_triples_dir, model_sanitized, "domains") if evaluate_by_category
                else os.path.join(elicited_triples_dir, model_sanitized, "domains", "non-existent") if non_existing_entities
                else os.path.join(elicited_triples_dir, model_sanitized, "ranges") if different_elicitation_triple_ranges
                else os.path.join(elicited_triples_dir, model_sanitized, "random", effective_effort) if effective_effort
                else os.path.join(elicited_triples_dir, model_sanitized, "prompts") if use_all_prompts or prompt_templates is not None
                else os.path.join(elicited_triples_dir, model_sanitized, "random")
                )
                elicited_triples_base = os.path.join(elicited_triples_dir, model_sanitized)
                results_subdir = os.path.relpath(elicited_triples_dir_value, elicited_triples_base)
                config = {
                        "entities_file_path": entities_file_path,
                        "api": api,
                        "model_elicitation": model,
                        "prompt_template_dir_elicitation": prompt_template_dir_elicitation,
                        "reasoning_effort_elicitation": effective_effort,
                        "non_existing_entities": non_existing_entities,
                        "elicited_triples_dir": elicited_triples_dir_value,
                        "ground_truth_dir_path": ground_truth_dir_path,
                        "results_dir_path": os.path.join(results_dir_path, model_sanitized, results_subdir) if skip_evaluation==False else None,
                        "api_url_elicitation": api_url_elicitation,
                        "api_key_elicitation": api_key_elicitation,
                        "llm_judge": llm_judge,
                        "seed": seed,
                        "sample_size": sample_size,
                        "evaluate_by_category": evaluate_by_category,
                        "triples_per_category": triples_per_category,
                        "different_elicitation_triple_ranges": different_elicitation_triple_ranges,
                        "evaluate_by_popularity": evaluate_by_popularity,
                        "use_all_prompts": use_all_prompts,
                        "prompt_templates": prompt_templates,
                }

                # Check if experiment already exists
                if skip_if_exists:
                        previous_run = tracker.check_experiment(config)
                        if previous_run:
                                print(f"\n{'='*70}")
                                print(f"EXPERIMENT ALREADY RUN for model: {model}")
                                print(f"Configuration matches a previous run from: {previous_run}")
                                print(f"Skipping execution to avoid redundant computation.")
                                print(f"Results stored in: {config['results_dir_path']}")
                                print(f"{'='*70}\n")
                                return (model, config['results_dir_path'])

                # -------------------------
                # ELICITATION
                # -------------------------
                # Skip elicitation if skip_elicitation is True or if building ground truth only
                if not skip_elicitation and not build_ground_truth_only:
                        if api == "openai_batched":
                                print(f"Starting elicitation with OpenAI Batch API for model: {model} ...")
                                main_elicitation_openai_batched(
                                        entities_file_path=entities_file_path,
                                        model_elicitation=model,
                                        prompt_template_dir_elicitation=prompt_template_dir_elicitation,
                                        reasoning_effort_elicitation=effective_effort,
                                        elicited_triples_dir=config['elicited_triples_dir'],
                                        evaluate_by_category=evaluate_by_category,
                                        different_elicitation_triple_ranges=different_elicitation_triple_ranges,
                                        evaluate_by_popularity=evaluate_by_popularity,
                                        run_id=run_id,
                                        use_all_prompts=use_all_prompts,
                                        prompt_templates=prompt_templates,
                                        openai_api_key=api_key_elicitation,
                                )
                        elif api in ["custom", "openrouter"]:
                                print(f"Starting elicitation with {api} API for model: {model} ...")
                                main_elicitation_other(
                                        entities_file_path=entities_file_path,
                                        model_elicitation=model,
                                        api_url_elicitation=api_url_elicitation,
                                        api_key_elicitation=api_key_elicitation,
                                        reasoning_effort_elicitation=effective_effort,
                                        elicited_triples_dir=config['elicited_triples_dir'],
                                        evaluate_by_category=evaluate_by_category,
                                        different_elicitation_triple_ranges=different_elicitation_triple_ranges,
                                        evaluate_by_popularity=evaluate_by_popularity,
                                        prompt_template_dir_elicitation=prompt_template_dir_elicitation,
                                        use_all_prompts=use_all_prompts,
                                        prompt_templates=prompt_templates,
                                )
                        else:
                                print(f"Unsupported API choice: {api}. Please choose 'openai_batched', 'custom', or 'openrouter'.")
                                return None
                else:
                        if build_ground_truth_only:
                                print("Skipping elicitation step (build_ground_truth_only=True).")
                        else:
                                print("Skipping elicitation step as requested.")

                # -------------------------
                # EVALUATION
                # -------------------------
                if non_existing_entities:
                        print("Skipping evaluation since non-existing entities were used for elicitation. No ground truth available for evaluation.")
                        return None

                if not skip_evaluation:
                        prompts_subdirs = []
                        if use_all_prompts or prompt_templates is not None:
                                prompts_base = config['elicited_triples_dir']
                                if os.path.isdir(prompts_base):
                                        for item in os.listdir(prompts_base):
                                                item_path = os.path.join(prompts_base, item)
                                                if os.path.isdir(item_path):
                                                        csv_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
                                                        if csv_files:
                                                                prompts_subdirs.append(item)
                                
                                if not prompts_subdirs:
                                        csv_files_base = [f for f in os.listdir(prompts_base) if f.endswith(".csv")] if os.path.isdir(prompts_base) else []
                                        if csv_files_base:
                                                prompts_subdirs = [None]
                        
                        if prompts_subdirs:
                                print(f"Found {len(prompts_subdirs)} prompt template subdirectories to evaluate")
                                
                                def evaluate_prompt_subdir(subdir_name):
                                        if subdir_name is None:
                                                subdir_elicited = config['elicited_triples_dir']
                                                subdir_results = config['results_dir_path']
                                        else:
                                                subdir_elicited = os.path.join(config['elicited_triples_dir'], subdir_name)
                                                base_results_dir = os.path.join(results_dir_path, model_sanitized)
                                                subdir_results = os.path.join(base_results_dir, "prompts", subdir_name)
                                        
                                        print(f"\n{'='*70}")
                                        print(f"Evaluating prompt template: {subdir_name or 'root'}")
                                        print(f"{'='*70}")
                                        
                                        subdir_config = dict(config)
                                        subdir_config['elicited_triples_dir'] = subdir_elicited
                                        subdir_config['results_dir_path'] = subdir_results
                                        
                                        if skip_if_exists:
                                                previous_run = tracker.check_experiment(subdir_config)
                                                if previous_run:
                                                        print(f"EXPERIMENT ALREADY RUN. Skipping.")
                                                        return None
                                        
                                        main_evaluation(
                                                entities_file_path=entities_file_path,
                                                elicited_triples_dir=subdir_elicited,
                                                ground_truth_dir_path=ground_truth_dir_path,
                                                results_dir_path=subdir_results,
                                                llm_judge=llm_judge,
                                                seed=seed,
                                                sample_size=sample_size,
                                                evaluate_by_category=evaluate_by_category,
                                                triples_per_category=triples_per_category,
                                                evaluate_by_popularity=evaluate_by_popularity,
                                                triples_per_popularity_bucket=triples_per_category,
                                                web_results_count=web_results_count,
                                                web_docs_for_eval=web_docs_for_eval,
                                                build_ground_truth_only=build_ground_truth_only,
                                                top_k=top_k,
                                                judge_api_url=judge_api_url,
                                                judge_api_key=judge_api_key,
                                        )
                                        
                                        if not build_ground_truth_only:
                                                tracker.register_experiment(subdir_config, results_path=subdir_results)
                                                print(f"Completed evaluation for prompt template: {subdir_name}")
                                        
                                        return subdir_name
                                
                                # Pre-build ground truth once, sequentially, before spawning parallel
                                # prompt-subdir evaluations — prevents N threads from racing to fetch
                                # the same Brave Search results simultaneously.
                                print(f"\n{'='*70}")
                                print("Pre-building ground truth before parallel prompt evaluation...")
                                print(f"{'='*70}\n")
                                main_evaluation(
                                        entities_file_path=entities_file_path,
                                        elicited_triples_dir=config['elicited_triples_dir'],
                                        ground_truth_dir_path=ground_truth_dir_path,
                                        results_dir_path=None,
                                        llm_judge=llm_judge,
                                        seed=seed,
                                        sample_size=sample_size,
                                        evaluate_by_category=evaluate_by_category,
                                        triples_per_category=triples_per_category,
                                        evaluate_by_popularity=evaluate_by_popularity,
                                        triples_per_popularity_bucket=triples_per_category,
                                        web_results_count=web_results_count,
                                        web_docs_for_eval=web_docs_for_eval,
                                        build_ground_truth_only=True,
                                        top_k=top_k,
                                        judge_api_url=judge_api_url,
                                        judge_api_key=judge_api_key,
                                )
                                print(f"Ground truth ready. Running {len(prompts_subdirs)} prompt evaluations in parallel...")

                                max_workers_prompts = min(len(prompts_subdirs), 5)
                                with ThreadPoolExecutor(max_workers=max_workers_prompts) as executor:
                                        futures = {executor.submit(evaluate_prompt_subdir, subdir): subdir for subdir in prompts_subdirs}
                                        completed = 0
                                        for future in as_completed(futures):
                                                try:
                                                        result = future.result()
                                                        if result is not None:
                                                                completed += 1
                                                except Exception as exc:
                                                        print(f"Prompt template evaluation failed: {exc}")
                                
                                print(f"\n{'='*70}")
                                print(f"ALL EVALUATIONS COMPLETED for model: {model} ({completed}/{len(prompts_subdirs)})")
                                print(f"{'='*70}\n")
                                
                                return (model, config['results_dir_path'])
                        else:
                                print(f"Starting evaluation of elicited triples for model: {model} ...")
                                main_evaluation(
                                        entities_file_path=entities_file_path,
                                        elicited_triples_dir=config['elicited_triples_dir'],
                                        ground_truth_dir_path=ground_truth_dir_path,
                                        results_dir_path=config['results_dir_path'],
                                        llm_judge=llm_judge,
                                        seed=seed,
                                        sample_size=sample_size,
                                        evaluate_by_category=evaluate_by_category,
                                        triples_per_category=triples_per_category,
                                        evaluate_by_popularity=evaluate_by_popularity,
                                        triples_per_popularity_bucket=triples_per_category,
                                        web_results_count=web_results_count,
                                        web_docs_for_eval=web_docs_for_eval,
                                        build_ground_truth_only=build_ground_truth_only,
                                        top_k=top_k,
                                        judge_api_url=judge_api_url,
                                        judge_api_key=judge_api_key,
                                )

                                if not build_ground_truth_only:
                                        tracker.register_experiment(config, results_path=config['results_dir_path'])

                                        print(f"\n{'='*70}")
                                        print(f"EXPERIMENT COMPLETED AND REGISTERED for model: {model}")
                                        print(f"Configuration saved to tracking database.")
                                        print(f"{'='*70}\n")

                                        return (model, config['results_dir_path'])
                                else:
                                        print(f"\n{'='*70}")
                                        print(f"Ground truth saved to: {ground_truth_dir_path}")
                                        print(f"{'='*70}\n")
                                        return (model, ground_truth_dir_path)

                        return None

        # Handle case: build ground truth only without any model
        if build_ground_truth_only and len(models) == 0:
                print(f"\n{'='*70}")
                print("Building ground truth only...")
                print(f"{'='*70}\n")
                
                # Call main_evaluation directly to build ground truth (no results_dir_path needed)
                main_evaluation(
                        entities_file_path=entities_file_path,
                        elicited_triples_dir=elicited_triples_dir,
                        ground_truth_dir_path=ground_truth_dir_path,
                        results_dir_path=None,
                        llm_judge=llm_judge,
                        seed=seed,
                        sample_size=sample_size,
                        evaluate_by_category=evaluate_by_category,
                        triples_per_category=triples_per_category,
                        evaluate_by_popularity=evaluate_by_popularity,
                        triples_per_popularity_bucket=triples_per_category,
                        web_results_count=web_results_count,
                        web_docs_for_eval=web_docs_for_eval,
                        build_ground_truth_only=True,
                        top_k=top_k,
                        judge_api_url=judge_api_url,
                        judge_api_key=judge_api_key,
                )
                
                print(f"\n{'='*70}")
                print(f"Ground truth built and saved to: {ground_truth_dir_path}")
                print(f"{'='*70}\n")
                return
        
        # Build the flat job list: list of (model, effort_or_None)
        if auto_effort_jobs:
                jobs = auto_effort_jobs
        else:
                jobs = [(m, None) for m in models]

        if not jobs:
                print(f"ERROR: No elicited triples found in {elicited_triples_dir} for the given flags. Pass --model_elicitation explicitly or check the directory.")
                return

        if len(jobs) == 1:
                result = run_single_model(jobs[0][0], effort_override=jobs[0][1])
                if result is not None:
                        results_summary.append(result)
        else:
                # Multiple jobs: pre-build ground truth once before parallel execution
                # to avoid all workers racing to fetch the same Brave Search results.
                print(f"\n{'='*70}")
                print("Pre-building ground truth before parallel elicitation/evaluation...")
                print(f"{'='*70}\n")
                main_evaluation(
                        entities_file_path=entities_file_path,
                        elicited_triples_dir=elicited_triples_dir,
                        ground_truth_dir_path=ground_truth_dir_path,
                        results_dir_path=None,
                        llm_judge=llm_judge,
                        seed=seed,
                        sample_size=sample_size,
                        evaluate_by_category=evaluate_by_category,
                        triples_per_category=triples_per_category,
                        evaluate_by_popularity=evaluate_by_popularity,
                        triples_per_popularity_bucket=triples_per_category,
                        web_results_count=web_results_count,
                        web_docs_for_eval=web_docs_for_eval,
                        build_ground_truth_only=True,
                        top_k=top_k,
                        judge_api_url=judge_api_url,
                        judge_api_key=judge_api_key,
                )
                print(f"Ground truth ready. Running {len(jobs)} evaluation jobs in parallel...")
                with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                        future_to_job = {executor.submit(run_single_model, m, e): (m, e) for m, e in jobs}
                        for future in as_completed(future_to_job):
                                model, _ = future_to_job[future]
                                try:
                                        result = future.result()
                                        if result is not None:
                                                results_summary.append(result)
                                except Exception as exc:
                                        print(f"Model {model} generated an exception: {exc}")

if __name__ == "__main__":
        fire.Fire(BeQu)
