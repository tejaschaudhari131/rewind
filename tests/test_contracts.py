import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "contracts"


def test_schemas_are_valid_draft202012():
    for name in ["contract.schema.json", "glimmer-output.schema.json", "wonder-output.schema.json"]:
        schema = json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_example_cycle_messages_validate():
    schema = json.loads((CONTRACTS / "contract.schema.json").read_text(encoding="utf-8"))
    cycle = json.loads((CONTRACTS / "example-cycle.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert len(cycle["messages"]) == 7
    for msg in cycle["messages"]:
        validator.validate(msg)


def test_model_outputs_validate_and_negative_cases_fail():
    schema = json.loads((CONTRACTS / "contract.schema.json").read_text(encoding="utf-8"))
    audit_schema = json.loads((CONTRACTS / "glimmer-output.schema.json").read_text(encoding="utf-8"))
    planner_schema = json.loads((CONTRACTS / "wonder-output.schema.json").read_text(encoding="utf-8"))
    cycle = json.loads((CONTRACTS / "example-cycle.json").read_text(encoding="utf-8"))
    messages = cycle["messages"]

    Draft202012Validator(audit_schema).validate(messages[2]["payload"]["model_output"])
    Draft202012Validator(planner_schema).validate(messages[4]["payload"]["model_output"])

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    bad1 = copy.deepcopy(messages[0]); bad1["extra"] = True
    bad2 = copy.deepcopy(messages[3]); bad2["producer"] = "wonder_adapter"
    bad3 = copy.deepcopy(messages[1]); bad3["payload"]["metrics"][0]["unit"] = "mm"
    bad4 = copy.deepcopy(messages[2]); bad4["payload"]["model_output"]["accepted"] = True
    for bad in (bad1, bad2, bad3, bad4):
        assert not validator.is_valid(bad)
