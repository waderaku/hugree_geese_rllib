import json
from main import Parameter
from dataclasses import asdict


MODEL_NAME = "model_name"
ENV_NAME = "env_name"


parameter = Parameter()
with open("./conf/parameter.json", "w") as f:
    json_data = json.dump(asdict(parameter), f)
