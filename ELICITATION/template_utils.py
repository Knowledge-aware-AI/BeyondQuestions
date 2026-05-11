"""
Utilities for resolving and loading elicitation prompt templates.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass
class TwoStepTemplate:
    """Represents a paired two-step elicitation strategy.

    The strategy lives in a subdirectory of the prompts directory containing
    exactly two files:
      - ``predicates.jinja``: Step 1 – given a subject, elicit all predicates
      - ``objects.jinja``:    Step 2 – given a subject + predicate, elicit objects

    The ``name`` attribute is the subdirectory name and is also used as the
    output subdirectory for the elicited triples.
    """
    name: str                 # e.g. "LMCRAWL"
    predicate_template: str   # absolute path to predicates.jinja
    object_template: str      # absolute path to objects.jinja


def _detect_two_step_subdir(subdir_path: str) -> TwoStepTemplate | None:
    """Return a TwoStepTemplate if *subdir_path* is a valid two-step directory,
    otherwise return None.

    A directory is a valid two-step strategy if it contains both
    ``predicates.jinja`` and ``objects.jinja``.
    """
    pred_path = os.path.join(subdir_path, "predicates.jinja")
    obj_path = os.path.join(subdir_path, "objects.jinja")
    if os.path.isfile(pred_path) and os.path.isfile(obj_path):
        return TwoStepTemplate(
            name=os.path.basename(subdir_path),
            predicate_template=os.path.abspath(pred_path),
            object_template=os.path.abspath(obj_path),
        )
    return None


def resolve_templates(
    prompt_template_dir: str,
    use_all_prompts: bool = False,
    prompt_templates: str | None = None,
) -> List[Union[str, TwoStepTemplate]]:
    """Return a list of selected template references.

    Each element is either:
    - a ``str`` (absolute path to a single-prompt ``.jinja`` file), or
    - a :class:`TwoStepTemplate` (a paired two-step strategy from a subdirectory).

    Selection logic (in order of priority):

    1. If *prompt_templates* is given (comma-separated names): use exactly those
       entries.  A name may be either a ``.jinja`` filename (for single-prompt
       templates) or a subdirectory name (for two-step strategies).
    2. Elif *use_all_prompts* is True: include all top-level ``.jinja`` files
       AND all valid two-step subdirectories.
    3. Otherwise: use only the first ``.jinja`` file (backward-compatible;
       two-step subdirectories are NOT included in the default single-template
       mode).

    *prompt_template_dir* may also be a path to a single file – in that case
    the file is returned as-is (backward-compatible path for callers that pass
    a direct file path).
    """

    # Direct file path provided (backward-compat)
    if os.path.isfile(prompt_template_dir):
        return [os.path.abspath(prompt_template_dir)]

    if not os.path.isdir(prompt_template_dir):
        raise FileNotFoundError(
            f"prompt_template_dir not found: {prompt_template_dir}"
        )

    # Collect all top-level .jinja files
    all_jinja = sorted(
        f for f in os.listdir(prompt_template_dir) if f.endswith(".jinja")
    )

    # Collect all valid two-step subdirectories
    two_step_map: dict[str, TwoStepTemplate] = {}
    for entry in sorted(os.listdir(prompt_template_dir)):
        entry_path = os.path.join(prompt_template_dir, entry)
        if os.path.isdir(entry_path):
            ts = _detect_two_step_subdir(entry_path)
            if ts is not None:
                two_step_map[ts.name] = ts

    if prompt_templates is not None:
        # Parse comma-separated list; strip whitespace and ignore empty tokens
        requested = [t.strip() for t in prompt_templates.split(",") if t.strip()]
        result: List[Union[str, TwoStepTemplate]] = []
        for name in requested:
            if name in two_step_map:
                result.append(two_step_map[name])
            elif name in all_jinja:
                result.append(os.path.join(prompt_template_dir, name))
            else:
                raise FileNotFoundError(
                    f"Template '{name}' not found in {prompt_template_dir}. "
                    f"Available single-prompt templates: {all_jinja}. "
                    f"Available two-step strategies: {list(two_step_map.keys())}"
                )
        return result

    if use_all_prompts:
        result = [os.path.join(prompt_template_dir, f) for f in all_jinja]
        result += list(two_step_map.values())
        return result

    # Default: first .jinja file only (unchanged behaviour)
    if not all_jinja:
        if two_step_map:
            raise FileNotFoundError(
                f"No top-level .jinja template files found in {prompt_template_dir}. "
                f"Two-step strategies detected ({list(two_step_map.keys())}) require "
                f"--use_all_prompts or explicit --prompt_templates to be selected."
            )
        raise FileNotFoundError(
            f"No .jinja template files found in {prompt_template_dir}"
        )

    selected = all_jinja[0]
    if len(all_jinja) > 1 or two_step_map:
        extras = list(all_jinja[1:]) + list(two_step_map.keys())
        print(
            f"[template_utils] Multiple templates found in {prompt_template_dir}; "
            f"using {selected}. "
            f"Pass --use_all_prompts or --prompt_templates to select others: {extras}"
        )
    return [os.path.join(prompt_template_dir, selected)]


def extract_system_prompt_from_jinja(
    template_path: str,
    model_elicitation: str = "dummy",
    reasoning_effort: str | None = None,
) -> str:
    """Render a jinja template and extract the **system message content** string.

    The jinja templates produce an OpenAI batch-API JSON payload.  This helper
    renders one with a placeholder subject so we can pull out the system prompt
    text, which is then re-used by the local (non-batched) elicitation path.

    Returns the content string of the first message with ``role == "system"``.
    Raises ``KeyError`` / ``ValueError`` if the template does not contain a
    system message.
    """
    from jinja2 import Environment, FileSystemLoader

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

    rendered = tmpl.render(
        subject_name="__PLACEHOLDER__",
        model=model_elicitation,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens_map,
        **legacy_params,
    )

    payload = json.loads(rendered)
    messages = payload["body"]["messages"]
    for msg in messages:
        if msg.get("role") == "system":
            return msg["content"]

    raise ValueError(
        f"No system message found in template {template_path}. "
        f"Messages found: {[m.get('role') for m in messages]}"
    )
