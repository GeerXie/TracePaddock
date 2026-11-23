"""List the state changes in a recording."""

NAME = "State transitions"
DESCRIPTION = "Report the timestamp and both values at each state change."
PARAMETERS = [
    {"name": "signal", "label": "State signal", "type": "signal", "default": "TutorialDriveMode"},
]


def run(api, parameters):
    import numpy as np

    name = parameters["signal"]
    time, values = api.data(name)
    valid = np.isfinite(values)
    changes = np.flatnonzero(valid[:-1] & valid[1:] & (values[:-1] != values[1:])) + 1
    for index in changes:
        previous, current = values[index - 1:index + 1]
        api.event(time[index], f"{name}: {previous:g} → {current:g}", verdict="FOUND", rule=NAME,
                  **{name: float(current)})
    api.log(f"{len(changes)} changes of {name}")
