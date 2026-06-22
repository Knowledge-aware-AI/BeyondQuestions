import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from .abstract_prompter_parser import AbstractPrompterParser
from .exceptions import ParsingException
from loguru import logger
from elicitation.template_utils import model_supports_json_schema, model_uses_max_completion_tokens


class PromptJSONSchema(AbstractPrompterParser):

    def __init__(self, template_path_elicitation: str, model_elicitation: str,
                 reasoning_effort: str = None):

        path_elicitation = Path(template_path_elicitation)
        assert path_elicitation.exists(), f"Template file not found: {template_path_elicitation}"
        assert path_elicitation.is_file(), f"Template path is not a file: {template_path_elicitation}"

        folder = path_elicitation.parent
        env = Environment(loader=FileSystemLoader(folder))
        self.template_elicitation = env.get_template(path_elicitation.name)
        self.model_elicitation = model_elicitation
        self.reasoning_effort = reasoning_effort
        self.use_json_schema = model_supports_json_schema(model_elicitation)
        self.use_max_completion_tokens = model_uses_max_completion_tokens(model_elicitation)
        self.max_completion_tokens_map = {
            'low': 4096,
            'medium': 10000,
            'high': 15000,
        }
        self.legacy_params = {
            'max_tokens': 4096,
            'top_p': 0,
            'temperature': 0,
            'frequency_penalty': 0,
            'presence_penalty': 0,
        }

    def get_elicitation_prompt(self, subject_name: str) -> dict:
        return json.loads(
            self.template_elicitation.render(
                subject_name=subject_name,
                model=self.model_elicitation,
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_completion_tokens_map,
                use_json_schema=self.use_json_schema,
                use_max_completion_tokens=self.use_max_completion_tokens,
                **self.legacy_params,
            )
        )
    

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

        if not isinstance(generated_json_object, dict):
            raise ParsingException(f"Expected a dict, got {type(generated_json_object)}")
        key = "facts" if "facts" in generated_json_object else "triples"
        if key not in generated_json_object:
            raise ParsingException(f"Key 'facts' not found in response")

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