import json
import hashlib
import os
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentTracker:
    """
    Manages tracking of experiment configurations to avoid re-running identical experiments.
    Stores experiment metadata in a JSON file and checks for duplicates before execution.
    """
    
    def __init__(self, tracking_file: str = ".experiment_tracking.json"):
        """
        Initialize the experiment tracker.
        
        Args:
            tracking_file (str): Path to the JSON file where experiment history is stored.
                                 Default: .experiment_tracking.json in the current directory.
        """
        self.tracking_file = Path(tracking_file)
        self._ensure_tracking_file_exists()
    
    def _ensure_tracking_file_exists(self) -> None:
        """Create the tracking file if it doesn't exist."""
        if not self.tracking_file.exists():
            self.tracking_file.write_text(json.dumps([], indent=2))
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute a hash of the configuration dictionary.
        
        Args:
            config (Dict[str, Any]): Configuration parameters to hash.
        
        Returns:
            str: SHA-256 hash of the sorted config.
        """
        # Sort the config to ensure consistent hashing
        sorted_config = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(sorted_config.encode()).hexdigest()
    
    def _load_experiments(self) -> list:
        """Load all tracked experiments from file."""
        try:
            content = self.tracking_file.read_text()
        except FileNotFoundError:
            return []
        if not content.strip():
            return []
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # never treat a damaged tracker as empty: that would redo (and overwrite) every
            # finished experiment and replace the tracker with a single entry
            raise RuntimeError(f"{self.tracking_file} is not valid JSON ({e}); restore it from a backup") from e
    
    def _save_experiments(self, experiments: list) -> None:
        """Save experiments to file. Atomic: written to a temp file, then renamed over the old one,
        so a process killed mid-write can never leave a truncated tracker behind."""
        tmp = self.tracking_file.with_name(f"{self.tracking_file.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(experiments, indent=2))
        os.replace(tmp, self.tracking_file)

    @contextmanager
    def _locked(self):
        """Exclusive lock (across threads and processes) around read-modify-write of the tracker,
        so registrations finishing at the same moment cannot overwrite each other."""
        fd = os.open(str(self.tracking_file) + ".lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    
    def check_experiment(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Check if an experiment with the given configuration has already been run.
        
        Args:
            config (Dict[str, Any]): Configuration parameters for the experiment.
        
        Returns:
            Optional[str]: Timestamp of previous run if found, None otherwise.
        """
        config_hash = self._compute_config_hash(config)
        experiments = self._load_experiments()
        
        for exp in experiments:
            if exp.get("config_hash") == config_hash:
                return exp.get("timestamp")
        
        return None
    
    def register_experiment(self, config: Dict[str, Any], results_path: Optional[str] = None) -> None:
        """
        Register a new experiment configuration as completed.
        
        Args:
            config (Dict[str, Any]): Configuration parameters for the experiment.
            results_path (Optional[str]): Path where results are stored (optional).
        """
        config_hash = self._compute_config_hash(config)
        timestamp = datetime.now().isoformat()
        
        experiment_record = {
            "config_hash": config_hash,
            "timestamp": timestamp,
            "config": config,
        }
        
        if results_path:
            experiment_record["results_path"] = str(results_path)
        
        with self._locked():
            experiments = self._load_experiments()
            experiments.append(experiment_record)
            self._save_experiments(experiments)
    
    def get_all_experiments(self) -> list:
        """
        Retrieve all tracked experiments.
        
        Returns:
            list: List of all experiment records.
        """
        return self._load_experiments()
    
    def clear_tracking(self) -> None:
        """Clear all tracked experiments (use with caution!)."""
        self.tracking_file.write_text(json.dumps([], indent=2))
        print(f"Cleared all experiments from {self.tracking_file}")
    
    def remove_experiment(self, config: Dict[str, Any]) -> bool:
        """
        Remove a specific experiment from tracking.
        
        Args:
            config (Dict[str, Any]): Configuration of the experiment to remove.
        
        Returns:
            bool: True if experiment was found and removed, False otherwise.
        """
        config_hash = self._compute_config_hash(config)
        with self._locked():
            experiments = self._load_experiments()
            original_count = len(experiments)

            experiments = [exp for exp in experiments if exp.get("config_hash") != config_hash]

            if len(experiments) < original_count:
                self._save_experiments(experiments)
                return True

        return False
    
    def find_similar_experiments(self, config: Dict[str, Any], exclude_keys: Optional[list] = None) -> list:
        """
        Find experiments that are similar to the given config, excluding specified keys.
        Useful for finding experiments that differ in only a few parameters.
        
        Args:
            config (Dict[str, Any]): Configuration to compare against.
            exclude_keys (Optional[list]): Keys to ignore when comparing. Default: None
        
        Returns:
            list: List of similar experiment records.
        """
        exclude_keys = exclude_keys or []
        filtered_config = {k: v for k, v in config.items() if k not in exclude_keys}
        
        experiments = self._load_experiments()
        similar = []
        
        for exp in experiments:
            exp_config = exp.get("config", {})
            filtered_exp_config = {k: v for k, v in exp_config.items() if k not in exclude_keys}
            
            if filtered_config == filtered_exp_config:
                similar.append(exp)
        
        return similar
