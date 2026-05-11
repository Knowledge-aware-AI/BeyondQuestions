import json
import time
from pathlib import Path
import os
import random
import openai
import csv 
import shutil
from loguru import logger
from openai import OpenAI
from openai.types import Batch as OpenAIBatch
from tqdm import tqdm
from .prompter_parser import AbstractPrompterParser
from .prompter_parser.exceptions import ParsingException
import re
from eval.network_utils import network_retry

class GPTKBCRunner:
    def __init__(
            self,
            source_file_name: str,
            curr_index: int,                # current index needed while submitting the batch request 
            entities_file_path:str,
            elicited_triples_dir:str,
            job_description: str = "Knowledge Elicitation", 
            prompter_parser_module: AbstractPrompterParser = None,
            evaluate_by_category: bool = False,
            run_id: str = None,  # optional override for workspace isolation
            openai_api_key: str = None,  # optional API key for OpenAI
        ):

        """
        Main arguments for GPTKBC runner class

        Arguments:
            source_file_name (str): Name of the jinja file based on which the elicitation is done and batch request is submitted
            curr_index (int): Index used for identifying the current prompt file
            entities_file_path (str): File path for handpicked wikidata entities
            elicited_triples_dir (str): Dir path for storing the elicited triples
            job_description (str): String based description of the current job
            prompter_parser_module: Abstract prompter parser class object      
            evaluate_by_category (bool): Whether to evaluate by category (default: False)
            #scaling_factor_elicitation, not implemented yet
            #reasoning_effort_elicitation:Literal["low", "medium", "high"] = None, not implemented yet  
        """

        logger.info('Initialize the GPTKC Runner ..')
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = OpenAI()
        self.job_description = job_description
        self.prompter_parser_module = prompter_parser_module
        self.curr_index = curr_index
        self.source_file_name = source_file_name

        self.tmp_folder = os.getcwd()
        self.entities_file_path = entities_file_path
        self.csv_dir_path = elicited_triples_dir

        # Generate a unique run identifier early so each runner gets its own workspace
        # this lets two processes share the same top-level elicited_triples_dir but
        # keep their in_progress/completed files separate. The caller may override via `run_id`.
        import random
        # fallback id: timestamp plus process id plus a small random number to avoid collisions
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = f"{int(time.time())}_{os.getpid()}_{random.randint(0,9999)}"
        # timestamp used for batch filenames can just be numeric
        self.timestamp = str(int(time.time()))

        # Create model-specific working directories to avoid conflicts when running multiple models
        # and different runs within the same directory. Each run gets a subfolder under .work
        work_dir_base = os.path.join(elicited_triples_dir, '.work', self.run_id)
        self.in_progress_dir_path = os.path.join(work_dir_base, 'progress_dir')
        self.completed_dir_path = os.path.join(work_dir_base, 'completed_dir')
        self.batch_results_dir = os.path.join(work_dir_base, 'results_dir')
        self.batch_request_dir = os.path.join(work_dir_base, 'batch_request')

        self.jinja_file_mapping = os.path.abspath(os.path.join(os.getcwd(), "..", "jinja_index_mapping.txt"))
        #self.jinja_file_mapping = os.getcwd() + '/jinja_index_mapping.txt'

        # file path used for storing temporary batch json objects until they are submitted
        self.batch_record_file_path = os.getcwd() + f"/batch_records_{self.timestamp}.jsonl"
        self.evaluate_by_category = evaluate_by_category
        #self.scaling_factor_elicitation = scaling_factor_elicitation
    

    def get_list_of_subjects(self) -> list[str]:
        """
        Read the wikidata entities file and get a list of subjects
        """
        with open(self.entities_file_path, 'r') as file:
            json_content = json.load(file)

        if self.evaluate_by_category == True:
            all_values = json_content
            all_values = [item["title"] for category in json_content.values() for item in category]
            return all_values
        else:
            all_values = json_content
            #all_values = [item for key, values in json_content.items() for item in values]
            all_values = [i['title'] for i in all_values]
            return all_values
    
    def create_dir(self):
        directory_list = [
            self.completed_dir_path,
            self.batch_results_dir,
            self.batch_request_dir,
            self.csv_dir_path,
        ]

        for directory in directory_list:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def cleanup_dirs(self):
        """
        Remove all files in in_progress, completed, batch_results, and batch_request directories after processing results.
        """
        dirs_to_clean = [
            self.in_progress_dir_path,
            self.completed_dir_path,
            self.batch_results_dir,
            self.batch_request_dir,
        ]
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.error(f'Failed to delete {file_path}. Reason: {e}')
                            
    def check_and_process_all(self):
        """Check all in-progress batches and process any that have completed."""
        self.create_dir()
        # Move any newly completed batches from in-progress to completed
        self.check_batch_status_dir()

        # Ensure we process all completed files and generate CSVs if missing
        for file_name in os.listdir(self.completed_dir_path):
            if not file_name.startswith("completed_") or not file_name.endswith(".json"):
                continue
            completed_file_path = os.path.join(self.completed_dir_path, file_name)
            with open(completed_file_path, 'r') as f:
                data = json.load(f)
                batch_file_id = data.get('batch_id')
            match = re.search(r'completed_(\d+)\.json', file_name)
            if match:
                file_index = str(match.group(1))
                result_file_path = os.path.join(self.batch_results_dir, f"batch_results_{file_index}.json")
                csv_file_path = os.path.join(self.csv_dir_path, f"elicited_triples_{file_index}.csv")
                # Process the completed batch if the batch results or the CSV is missing
                if (not os.path.exists(result_file_path)) or (not os.path.exists(csv_file_path)):
                    self.process_completed_batch_dir(batch_file_id, file_index)

    def loop(self, subjects_to_expand):
        """Main loop."""
        logger.info("Starting the main loop ...")
        self.create_dir()

        logger.info("Submitting a new batch.")
        self.create_batch_dir(subjects_to_expand)
        logger.info("Processing completed batches...")
        self.check_and_process_all()


    def process_completed_batch_dir(self, batch_id, csv_file_index):
        """
        Input: batch_id that has been completed.
        Processes the completed batch by downloading results, writing them to the batch_results_dir,
        and saving the parsed triples to a CSV file.
        """
        logger.info(f"Processing a newly completed batch: `{batch_id}`.")

        # Ensure the batch results directory exists
        os.makedirs(self.batch_results_dir, exist_ok=True)

        # Path where batch results should be stored
        result_file_path = os.path.join(self.batch_results_dir, f"batch_results_{csv_file_index}.json")

        # If the batch results file doesn't already exist, download it from OpenAI
        if not os.path.exists(result_file_path):
            logger.info("Downloading results from OpenAI.")
            openai_batch = self.openai_client.batches.retrieve(batch_id)
            input_file_id = openai_batch.input_file_id
            output_file_id = openai_batch.output_file_id
            batch_result = self.openai_client.files.content(output_file_id).content

            # Write batch results to the batch_results_dir
            with open(result_file_path, "wb") as f:
                f.write(batch_result)
            logger.info(f"Batch results written to `{result_file_path}`.")
        else:
            logger.info(f"Using existing batch results at `{result_file_path}`.")

        # Read the results and process triples
        raw_triples = []
        with open(result_file_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    # Parse elicitation responses
                    raw_triples_from_line = self.parse_elicitation_response(line)

                    raw_triples.extend(raw_triples_from_line)
                except json.JSONDecodeError:
                    logger.error(f"JSONDecodeError at line {line_number}: {line.strip()}")
                except Exception as e:
                    logger.error(f"Unexpected error while parsing line {line_number}: {line.strip()} | Error: {e}")

        logger.info(f"Found {len(raw_triples):,} raw triples in the batch results.")

        # Ensure the subfolder for this run exists
        os.makedirs(self.csv_dir_path, exist_ok=True)

        # Define the CSV filename in the subfolder
        csv_filename = os.path.join(self.csv_dir_path, f"elicited_triples_{csv_file_index}.csv")

        # Write the triples to a CSV file in the subfolder
        self.write_triples_to_csv(raw_triples, csv_filename)
        logger.info(f"Raw triples written to `{csv_filename}`.")

    # This method has been modified from the one in Prompter Parser Class
    def parse_elicitation_response(self, response: str) -> list[dict]:
        response_object = json.loads(response.strip())

        subject_name = response_object["custom_id"]
        choice = response_object["response"]["body"]["choices"][0]

        # check if the request was stopped correctly
        finish_reason = choice["finish_reason"]

        if finish_reason != "stop":
            raise ParsingException(f"finish_reason={finish_reason}")

        message = choice["message"]
        # check if the request was refused
        refusal = message["refusal"]
        if refusal:
            raise ParsingException(f"refusal={refusal}")

        output_string = message["content"]
        generated_json_object = json.loads(output_string)

        # check if the response object contains the key "facts"
        key = "facts"
        if (type(generated_json_object) != dict
                or key not in generated_json_object):
            raise ParsingException(f"Key '{key}' not found in response")

        # get the triples from the response object
        # ignore if the triple is not in the correct format (no error)
        raw_triples = []
        for line_triple in generated_json_object[key]:
            if ("subject" in line_triple
                    and "predicate" in line_triple
                    and "object" in line_triple):
                line_triple["subject_name"] = subject_name
                raw_triples.append(line_triple)
            else:
                logger.warning(
                    f"Subject: {subject_name}. "
                    f"Invalid triple format: {line_triple}")

        return raw_triples

    
    def write_triples_to_csv(self, raw_triples, input_file_path):
        """
        Write the raw triples to the csv file for further processing
        """
        try:
            with open(input_file_path, mode = 'w', newline='', encoding = 'utf-8') as csv_file:
                fieldnames = ["subject", "predicate", "object", "subject_name"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(raw_triples)
                logger.info("Data written to CSV succesfully ...")

        except Exception as e:
            print("An error occured: {e}")

    @network_retry(max_retries=5, initial_delay=2.0)
    def create_batch_dir(self, subjects_to_expand: list[str], max_tries: int = 5):
        """
        Push the data to a batch request for OpenAI.
        Write the in-progress batch ID to a JSON file for recording in `self.in_progress_dir_path`
        with a filename format `in_progress_<self.curr_index>.json`.
        """
        batch_request = []
        
        # Generate batch request data
        for subject in subjects_to_expand:
            req = self.prompter_parser_module.get_elicitation_prompt(subject_name=subject)
            batch_request.append(req)

        # Write batch request data to a temporary file
        with open(self.batch_record_file_path, "w") as f:
            for obj in batch_request:
                f.write(json.dumps(obj) + "\n")

        # Upload the batch request file to OpenAI
        try:
            batch_input_file = self.openai_client.files.create(
                file=open(self.batch_record_file_path, "rb"),
                purpose="batch"
            )
        finally:
            if os.path.exists(self.batch_record_file_path):
                os.remove(self.batch_record_file_path)

        batch_input_file_id = batch_input_file.id

        openai_batch = None
        for num_tries in range(max_tries):
            try:
                # Create batch
                openai_batch = self.openai_client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": self.job_description
                    }
                )
                logger.info(f"Batch file created successfully. Batch ID: `{openai_batch.id}`.")
                break
            except openai.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                logger.info("Waiting for 60 seconds before retrying.")
                time.sleep(60)
                continue

        if openai_batch is None:
            raise Exception(f"Failed to create batch file after {max_tries} attempts.")

        # Prepare data for writing to the in-progress directory
        data = {"batch_id": openai_batch.id}

        # Create the in-progress directory if it doesn't exist
        os.makedirs(self.in_progress_dir_path, exist_ok=True)

        # Determine the filename for the current batch
        in_progress_file_path = os.path.join(
            self.in_progress_dir_path,
            f"in_progress_{self.timestamp}.json"
        )

        # Write batch ID to the in-progress file
        with open(in_progress_file_path, "w") as f:
            json.dump(data, f)
        
        with open(self.jinja_file_mapping, 'a') as f:
            f.write(f"{self.source_file_name} elicited_triples_{self.timestamp}\n")
        
        logger.info(f"Data processed from jinja file name ... {self.source_file_name}")

        logger.info(f"Batch ID recorded at {in_progress_file_path}")

    '''
    def check_batch_status_dir(self):
        """
        Check the current batch status for jobs submitted to OpenAI.
        
        Moves completed files from the in-progress directory to the completed directory while preserving
        the numbered index of the filenames. Returns True if at least one batch has been completed, 
        otherwise False.
        """
        status_in_progress = ["created", "validating", "in_progress", "finalizing", "parsing"]
        any_completed = False

        # Ensure the in-progress directory exists
        if os.path.exists(self.in_progress_dir_path):

            # Iterate over all files in the in-progress directory
            for file_name in os.listdir(self.in_progress_dir_path):
                if file_name.endswith(".json") and file_name.startswith("in_progress_"):
                    file_path = os.path.join(self.in_progress_dir_path, file_name)
                    
                    # Read the JSON file
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    batch_file_id = data.get("batch_id", None)
                    openai_batch = self.openai_client.batches.retrieve(batch_file_id)
                    current_status = openai_batch.status
                    
                    logger.info(f"Current status of batch {batch_file_id} in {file_name}: {current_status}")
                    
                    # Check if the status is completed
                    if current_status in ["validating", "in_progress", "finalizing"]:
                        logger.info(f"Batch {batch_file_id} still running with status {current_status}")
                        return  # keep it in progress
                    
                    if current_status == "completed":
                        # Determine the new filename for the completed directory
                        completed_file_name = file_name.replace("in_progress_", "completed_")
                        completed_file_path = os.path.join(self.completed_dir_path, completed_file_name)

                        # Move the file to the completed directory
                        shutil.move(file_path, completed_file_path)
                        logger.info(f"Moved {file_name} to {completed_file_path}")
                        any_completed = True

                    elif current_status in ["failed", "expired", "cancelled"]:
                        logger.error(f"Batch {batch_file_id} failed with status {current_status}")
                        shutil.move(file_path, os.path.join(self.completed_dir_path, file_name))

                    else:
                        logger.warning(f"Unexpected status {current_status} for batch {batch_file_id} in {file_name}.")

        return any_completed
    '''

    def check_batch_status_dir(self):
        """
        Check the current batch status for jobs submitted to OpenAI.

        Moves completed files from the in-progress directory to the completed directory
        while preserving the numbered index of the filenames.
        Returns True if at least one batch has been completed, otherwise False.
        """

        any_completed = False

        if os.path.exists(self.in_progress_dir_path):

            for file_name in os.listdir(self.in_progress_dir_path):
                if file_name.endswith(".json") and file_name.startswith("in_progress_"):

                    file_path = os.path.join(self.in_progress_dir_path, file_name)

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    batch_file_id = data.get("batch_id", None)
                    openai_batch = self.openai_client.batches.retrieve(batch_file_id)
                    current_status = openai_batch.status

                    logger.info(
                        f"Current status of batch {batch_file_id} in {file_name}: {current_status}"
                    )

                    # --------------------------------------------------
                    # ACTIVE STATES → DO NOTHING (KEEP FILE IN PLACE)
                    # --------------------------------------------------
                    if current_status in [
                        "created",
                        "validating",
                        "in_progress",
                        "finalizing",
                        "parsing"
                    ]:
                        logger.info(
                            f"Batch {batch_file_id} still running with status {current_status}"
                        )
                        continue  # ✅ DO NOT RETURN

                    # --------------------------------------------------
                    # COMPLETED
                    # --------------------------------------------------
                    elif current_status == "completed":

                        completed_file_name = file_name.replace(
                            "in_progress_", "completed_"
                        )
                        completed_file_path = os.path.join(
                            self.completed_dir_path, completed_file_name
                        )

                        shutil.move(file_path, completed_file_path)

                        logger.info(
                            f"Moved {file_name} to {completed_file_path}"
                        )

                        any_completed = True

                    # --------------------------------------------------
                    # FAILED STATES
                    # --------------------------------------------------
                    elif current_status in ["failed", "expired", "cancelled"]:

                        logger.error(
                            f"Batch {batch_file_id} failed with status {current_status}"
                        )

                        failed_path = os.path.join(
                            self.completed_dir_path, file_name.replace(
                                "in_progress_", "failed_"
                            )
                        )

                        shutil.move(file_path, failed_path)

                    # --------------------------------------------------
                    # UNKNOWN STATUS
                    # --------------------------------------------------
                    else:
                        logger.warning(
                            f"Unexpected status {current_status} for batch {batch_file_id}"
                        )
                        # ⚠️ Do NOT move or delete the file

        return any_completed