"""Check a request and its response against a time limit."""

NAME = "Response timing"
DESCRIPTION = "Check whether each request is acknowledged within the configured time."
PARAMETERS = [
    {"name": "request", "label": "Request signal", "type": "signal", "default": "TutorialRequest"},
    {"name": "response", "label": "Response signal", "type": "signal", "default": "TutorialAck"},
    {"name": "seconds", "label": "Maximum delay [s]", "type": "number", "default": 0.5},
]


def run(api, parameters):
    request, response = parameters["request"], parameters["response"]
    api.rule(NAME, trigger=api.becomes(api.S(request) == 1),
             expect=api.S(response) == 1, within=float(parameters["seconds"]))
