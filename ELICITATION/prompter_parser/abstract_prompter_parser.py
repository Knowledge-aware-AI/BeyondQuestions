class AbstractPrompterParser:
    def get_elicitation_prompt(self, subject: str) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI.
        :param subject_name: The target subject.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def parse_elicitation_prompt(self, response: str) -> dict:
        """
        Parse the API response from OpenAI to extract triples. The result is a list of dictionaries, each dictionary
        containing a generated triple, with the keys "subject", "predicate" and "object", and an additional key "subject_name",
        which is the original subject name sent to the API. For example:
        {"subject_name": "Vannevar Bush", "subject": "Vannevar Bush", "predicate": "bornIn", "object": "1890"}
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError
    