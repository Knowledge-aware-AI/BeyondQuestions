"""
Two-step elicitation prompter parser.

This module handles the paired (predicates, objects) elicitation strategy:
  Step 1: Given a subject, elicit all known predicates/relations.
  Step 2: Given a subject and a predicate, elicit all objects for that predicate.

The two templates work together to produce the same final output as a single-prompt
elicitation (a list of triples with subject, predicate, object).
"""

import json
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from .abstract_prompter_parser import AbstractPrompterParser
from .exceptions import ParsingException


class PromptTwoStep(AbstractPrompterParser):
    """Prompter for a two-step (predicate → object) elicitation strategy.

    Two-step consists of:
      1. predicates.jinja  — returns a JSON object {"predicates": [...]}
      2. objects.jinja     — returns a JSON object {"objects": [...]}

    The objects.jinja template receives both ``subject_name`` and ``predicate``
    variables and uses a custom_id that encodes both for tracking in the batch API.
    """

    # Separator used in custom_id to encode (subject, predicate) for Step 2
    CUSTOM_ID_SEPARATOR = "|||"

    def __init__(
        self,
        predicate_template_path: str,
        object_template_path: str,
        model_elicitation: str,
        reasoning_effort: str = None,
    ):
        self.predicate_template_path = Path(predicate_template_path)
        self.object_template_path = Path(object_template_path)

        assert self.predicate_template_path.exists(), (
            f"Predicate template not found: {predicate_template_path}"
        )
        assert self.object_template_path.exists(), (
            f"Object template not found: {object_template_path}"
        )

        # Set up jinja environments
        self.pred_env = Environment(
            loader=FileSystemLoader(str(self.predicate_template_path.parent))
        )
        self.pred_tmpl = self.pred_env.get_template(self.predicate_template_path.name)

        self.obj_env = Environment(
            loader=FileSystemLoader(str(self.object_template_path.parent))
        )
        self.obj_tmpl = self.obj_env.get_template(self.object_template_path.name)

        self.model_elicitation = model_elicitation
        self.reasoning_effort = reasoning_effort

        # Identical token config to PromptJSONSchema
        self.max_completion_tokens_map = {
            "low": 4096,
            "medium": 10000,
            "high": 15000,
        }
        self.legacy_params = {
            "max_tokens": 4096,
            "top_p": 0,
            "temperature": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

    # --- Step 1: Predicate elicitation ---

    def get_predicate_prompt(self, subject_name: str) -> dict:
        """Render predicates.jinja for a subject. Returns OpenAI batch API payload.

        The returned dict has a ``custom_id`` set to the subject name.
        """
        rendered = self.pred_tmpl.render(
            subject_name=subject_name,
            model=self.model_elicitation,
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=self.max_completion_tokens_map,
            **self.legacy_params,
        )
        return json.loads(rendered)

    def parse_predicate_response(self, response: str) -> tuple[str, list[str]]:
        """Parse a batch response line for Step 1 (predicate elicitation).

        Args:
            response: A single JSON line from the batch results.

        Returns:
            A tuple (subject_name, list_of_predicates).

        Raises:
            ParsingException: If the response cannot be parsed.
        """
        response_object = json.loads(response.strip())

        # The custom_id is the subject name (same as Step 1 input)
        subject_name = response_object["custom_id"]
        choice = response_object["response"]["body"]["choices"][0]

        finish_reason = choice["finish_reason"]
        if finish_reason != "stop":
            raise ParsingException(f"finish_reason={finish_reason}")

        message = choice["message"]
        refusal = message["refusal"]
        if refusal:
            raise ParsingException(f"refusal={refusal}")

        output_string = message["content"]
        generated_json_object = json.loads(output_string)

        # Expect {"predicates": ["...", ...]}
        key = "predicates"
        if (
            type(generated_json_object) != dict
            or key not in generated_json_object
        ):
            raise ParsingException(f"Key '{key}' not found in response")

        raw_predicates = generated_json_object[key]
        if not isinstance(raw_predicates, list):
            raise ParsingException(
                f"'{key}' is not a list (got {type(raw_predicates).__name__})"
            )

        # Filter to only string values
        predicates = [str(p) for p in raw_predicates if p]
        return subject_name, predicates

    # --- Step 2: Object elicitation ---

    def get_object_prompt(self, subject_name: str, predicate: str) -> dict:
        """Render objects.jinja for a subject+predicate pair.

        Returns OpenAI batch API payload with custom_id = subject_name|||predicate
        so we can decode it later during parsing.
        """
        custom_id = f"{subject_name}{self.CUSTOM_ID_SEPARATOR}{predicate}"

        rendered = self.obj_tmpl.render(
            subject_name=subject_name,
            predicate=predicate,
            custom_id=custom_id,
            model=self.model_elicitation,
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=self.max_completion_tokens_map,
            **self.legacy_params,
        )
        return json.loads(rendered)

    def parse_object_response(self, response: str) -> list[dict]:
        """Parse a batch response line for Step 2 (object elicitation).

        Args:
            response: A single JSON line from the batch results.

        Returns:
            A list of triple dicts:
            [{"subject": ..., "predicate": ..., "object": ..., "subject_name": ...}, ...]

        Each object in the response creates a separate triple (flattening the list).

        Raises:
            ParsingException: If the response cannot be parsed.
        """
        response_object = json.loads(response.strip())

        # Decode custom_id to get (subject, predicate)
        custom_id = response_object["custom_id"]
        parts = custom_id.split(self.CUSTOM_ID_SEPARATOR)
        if len(parts) != 2:
            raise ParsingException(
                f"Invalid custom_id format: {custom_id}. "
                f"Expected 'subject{self.CUSTOM_ID_SEPARATOR}predicate'."
            )
        subject_name, predicate = parts[0], parts[1]

        choice = response_object["response"]["body"]["choices"][0]

        finish_reason = choice["finish_reason"]
        if finish_reason != "stop":
            raise ParsingException(f"finish_reason={finish_reason}")

        message = choice["message"]
        refusal = message["refusal"]
        if refusal:
            raise ParsingException(f"refusal={refusal}")

        output_string = message["content"]
        generated_json_object = json.loads(output_string)

        # Expect {"objects": ["...", ...]}
        key = "objects"
        if (
            type(generated_json_object) != dict
            or key not in generated_json_object
        ):
            raise ParsingException(f"Key '{key}' not found in response")

        raw_objects = generated_json_object[key]
        if not isinstance(raw_objects, list):
            raise ParsingException(
                f"'{key}' is not a list (got {type(raw_objects).__name__})"
            )

        # Convert each object into a triple
        triples = []
        for obj in raw_objects:
            if obj:  # Skip empty/None values
                triples.append({
                    "subject": subject_name,
                    "predicate": predicate,
                    "object": str(obj),
                    "subject_name": subject_name,
                })

        return triples

    # --- AbstractPrompterParser interface (legacy compatibility) ---

    def get_elicitation_prompt(self, subject_name: str) -> dict:
        """Alias for get_predicate_prompt (Step 1).

        This exists for backward compatibility with code that expects the
        standard AbstractPrompterParser interface. For two-step, prefer using
        the dedicated get_predicate_prompt() / get_object_prompt() methods.
        """
        return self.get_predicate_prompt(subject_name)

    def parse_elicitation_prompt(self, response: str) -> list[dict]:
        """Parse Step 2 response into triples.

        Note: This parses the *object* response (Step 2), not the predicate response.
        For two-step, you typically call:
          1. parse_predicate_response() → get predicates
          2. parse_object_response() → get triples for each (subject, predicate)

        This method is provided for AbstractPrompterParser compatibility.
        """
        return self.parse_object_response(response)


def extract_system_prompt_from_two_step_template(
    template_path: str,
    model_elicitation: str = "dummy",
    reasoning_effort: str | None = None,
    template_type: str = "predicates",  # "predicates" or "objects"
) -> str:
    """Extract the system prompt from a two-step template (predicates.jinja or objects.jinja).

    Args:
        template_path: Path to the jinja file (predicates.jinja or objects.jinja).
        model_elicitation: Model name for rendering.
        reasoning_effort: Optional reasoning effort setting.
        template_type: Either "predicates" or "objects" — used to provide the right
            variable to the template during rendering.

    Returns:
        The content string of the system message.
    """
    path = Path(template_path)
    env = Environment(loader=FileSystemLoader(str(path.parent)))
    tmpl = env.get_template(path.name)

    max_completion_tokens_map = {
        "low": 4096,
        "medium": 10000,
        "high": 15000,
    }
    legacy_params = {
        "max_tokens": 4096,
        "top_p": 0,
        "temperature": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # For predicates.jinja: render with subject_name only
    # For objects.jinja: render with subject_name and predicate
    render_kwargs = {
        "subject_name": "__PLACEHOLDER__",
        "model": model_elicitation,
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens_map,
        **legacy_params,
    }
    if template_type == "objects":
        render_kwargs["predicate"] = "__PREDICATE_PLACEHOLDER__"

    rendered = tmpl.render(**render_kwargs)
    payload = json.loads(rendered)
    messages = payload["body"]["messages"]
    for msg in messages:
        if msg.get("role") == "system":
            return msg["content"]

    raise ValueError(
        f"No system message found in template {template_path}. "
        f"Messages found: {[m.get('role') for m in messages]}"
    )
