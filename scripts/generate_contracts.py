"""Generate reviewable architecture contracts, not an application implementation."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

OUT = Path(__file__).parent
def obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
def arr(items, **kw):
    return {"type": "array", "items": items, **kw}
def enum(*values):
    return {"type": "string", "enum": list(values)}
def const(value):
    return {"const": value}
def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}
def ref(name):
    return {"$ref": "#/$defs/" + name}
S = {"type": "string", "minLength": 1}
N = {"type": "number"}
NN = {"type": "number", "minimum": 0}
I = {"type": "integer", "minimum": 0}
B = {"type": "boolean"}
U = {"type": "string", "format": "uuid"}
H = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
T = {"type": "string", "format": "date-time"}
R = {"type": "number", "minimum": 0, "maximum": 1}
XY = arr(N, minItems=2, maxItems=2)
BOX = arr(I, minItems=4, maxItems=4)

defs = {}
defs["Artifact"] = obj({"artifact_id": U, "uri": S, "sha256": H, "mime_type": S})
defs["Frame"] = obj({"frame_id": U, "image": ref("Artifact"), "frame_index": I,
    "timestamp_ms": NN, "width_px": I, "height_px": I,
    "coordinate_space": const("undistorted_source_image_px")})
defs["SegmentationObject"] = obj({"frame_id": U, "sam_response_id": S, "sam_track_id": I,
    "part_id": nullable(S), "identity_status": enum("confirmed", "provisional", "unknown"),
    "bbox_xyxy_inclusive_px": BOX, "mask": ref("Artifact"), "crop": ref("Artifact"),
    "mask_encoding": const("full_frame_binary_png_0_255"),
    "provider_mask_confidence": nullable(R)})
defs["Segmentation"] = obj({"provider_request_id": S, "provider_response_id": S,
    "source_media": ref("Artifact"), "frames": arr(ref("Frame"), minItems=1),
    "objects": arr(ref("SegmentationObject")), "billed_video_frames": nullable(I),
    "response_status": const("completed")})

def metric(name, unit, space):
    return obj({"metric": const(name), "part_id": S, "value": nullable(NN),
        "unit": const(unit), "space": const(space), "plane_id": nullable(S),
        "uncertainty_95": nullable(NN), "policy_threshold_id": S,
        "result": enum("pass", "fail", "unknown"), "evidence_frame_ids": arr(U, minItems=1)})
defs["Metric"] = {"oneOf": [
    metric("mask_iou", "ratio", "image"),
    metric("visible_coverage", "ratio", "image"),
    metric("pivot_error", "px", "image"),
    metric("witness_error", "px", "image"),
    metric("registration_error", "px", "image"),
    metric("boundary_p95", "px", "image"),
    metric("pivot_error", "mm", "calibrated_plane"),
    metric("hole_center_error", "mm", "calibrated_plane"),
    metric("stable_duration", "ms", "time")
]}
defs["Geometry"] = obj({"status": enum("pass", "fail", "insufficient_evidence"),
    "segmentation_message_id": nullable(U), "calibration_valid": B,
    "hands_clear": B, "identity_status": enum("match", "mismatch", "unknown"),
    "whole_component_status": enum("present", "absent", "unknown"),
    "seating_status": enum("aligned", "misaligned", "unknown", "not_applicable"),
    "metrics": arr(ref("Metric")), "reason_codes": arr(S), "evidence_frame_ids": arr(U, minItems=1)})

# Small model-output contracts; service wrappers, not models, supply authoritative metadata.
audit = obj({"observation_id": {"type": "string"},
    "finding": enum("consistent", "mismatch", "insufficient_evidence"),
    "observations": arr(obj({"frame_id": {"type": "string"},
        "region_xyxy_px": nullable(arr({"type": "number"})),
        "statement": {"type": "string"}})),
    "contradicted_predicate_ids": arr({"type": "string"}), "requires_new_view": B})
planner = obj({"observation_id": {"type": "string"},
    "action": enum("show_reference", "request_capture", "request_clarification", "instruct_move", "generate_supported_part", "finish"),
    "part_id": {"type": ["string", "null"]}, "instruction": {"type": "string"},
    "evidence_ids": arr({"type": "string"}), "required_predicate_ids": arr({"type": "string"}),
    "measurement_record_id": {"type": ["string", "null"]}})
defs["VisualAudit"] = obj({"model_id": S, "model_revision": S, "deployment_revision": S,
    "model_output": audit, "finish_reason": const("stop"), "latency_ms": NN})
defs["PlannerProposal"] = obj({"model_id": const("muse-wonder-1.0"),
    "provider_response_id": S, "model_output": planner})
defs["GateDecision"] = obj({"decision": enum("accepted", "rejected", "unknown", "pending"),
    "reason_codes": arr(S), "geometry_message_id": nullable(U), "audit_message_id": nullable(U),
    "segmentation_message_id": nullable(U), "checked_scene_version": I,
    "committed_scene_version": nullable(I), "completed_predicate_ids": arr(S),
    "decision_at": T, "elapsed_ms": NN})
defs["Dimension"] = obj({"value_mm": {"type": "number", "exclusiveMinimum": 0},
    "uncertainty_95_mm": nullable(NN),
    "source": enum("calibrated_flat_scan", "manual_caliper", "declared_stock", "fabrication_policy"),
    "evidence_artifact_ids": arr(U, minItems=1), "plane_id": nullable(S)})
defs["CadRequest"] = obj({"template_id": const("flat_two_hole_link_v1"),
    "template_sha256": H, "measurement_record_id": U,
    "parameters": obj({k: ref("Dimension") for k in ["hole_center_distance", "outer_width", "thickness", "pin_diameter", "diametral_clearance"]}),
    "material_label": S, "fabrication_method": enum("fdm", "laser_cut", "machined", "export_only"),
    "requested_formats": arr(enum("STEP", "STL"), minItems=1, uniqueItems=True),
    "claim_scope": const("dimensional_candidate_for_unloaded_test_rig")})
defs["CadArtifact"] = obj({"cad_request_message_id": U, "parameter_record": ref("Artifact"),
    "template_source": ref("Artifact"), "exports": arr(ref("Artifact"), minItems=1),
    "checks": arr(obj({"check_id": S, "result": enum("pass", "fail", "unknown"), "evidence": ref("Artifact")})),
    "status": enum("geometry_valid", "rejected", "insufficient_evidence"),
    "physical_fit_status": enum("not_tested", "tested_on_this_rig"),
    "physical_test_record": nullable(ref("Artifact"))})
M3 = arr(arr(N, minItems=3, maxItems=3), minItems=3, maxItems=3)
defs["Calibration"] = obj({"camera_id": S, "image_width_px": I, "image_height_px": I,
    "K_raw": M3, "distortion_coefficients": arr(N), "K_undistorted": M3,
    "coordinate_space": const("undistorted_source_image_px"),
    "tag_dictionary": const("APRILTAG_36h11"), "printed_tag_side_mm": NN,
    "planes": arr(obj({"plane_id": S, "z_from_mat_mm": N,
        "H_undistorted_px_to_plane_mm": M3, "origin_description": S,
        "positive_x_description": S, "positive_y_description": S,
        "calibration_frame_ids": arr(U, minItems=1), "holdout_frame_ids": arr(U, minItems=1),
        "holdout_error_95_mm": NN, "valid_image_polygon_px": arr(XY, minItems=3)}), minItems=1),
    "status": enum("validated", "invalid", "unvalidated")})
defs["MeasurementRecord"] = obj({"template_id": const("flat_two_hole_link_v1"),
    "plane_id": S, "flat_contact_confirmed": B, "source_frame_ids": arr(U, minItems=1),
    "dimensions": obj({k: ref("Dimension") for k in ["hole_center_distance", "outer_width", "thickness", "pin_diameter", "diametral_clearance"]}),
    "independent_center_distance_check": ref("Dimension"),
    "status": enum("eligible_for_cad", "insufficient_evidence", "rejected")})

common = {"schema_version": const("1.0.0"), "message_id": U, "session_id": U,
    "scene_version": I, "plan_version": I, "step_id": S, "observation_id": U,
    "observation_sha256": H, "reference_id": U, "reference_sha256": H,
    "calibration_id": S, "calibration_sha256": H, "policy_id": S, "policy_sha256": H,
    "captured_at": T, "produced_at": T, "parent_message_ids": arr(U)}
mapping = {
    "segmentation": ("sam_adapter", "Segmentation"),
    "geometry": ("geometry_service", "Geometry"),
    "visual_audit": ("glimmer_adapter", "VisualAudit"),
    "planner_proposal": ("wonder_adapter", "PlannerProposal"),
    "gate_decision": ("gate_service", "GateDecision"),
    "cad_request": ("coordinator", "CadRequest"),
    "cad_artifact": ("cad_worker", "CadArtifact"),
    "calibration": ("calibration_service", "Calibration"),
    "measurement_record": ("measurement_service", "MeasurementRecord")}
schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:rewind:contract:1.0.0", "title": "REWIND inter-stage contract",
    "description": "Application contract, not a Meta provider response format. Authentication and semantic checks are mandatory.",
    "$defs": defs, "oneOf": [obj({**common, "producer": const(producer), "payload_type": const(kind), "payload": ref(name)})
    for kind, (producer, name) in mapping.items()]}

def uid(i):
    return f"00000000-0000-4000-8000-{i:012d}"
def artifact(i, filename, mime="image/png"):
    return {"artifact_id": uid(i), "uri": "fixtures/synthetic/" + filename, "sha256": "a"*64, "mime_type": mime}
base = {"schema_version": "1.0.0", "session_id": uid(1), "scene_version": 42, "plan_version": 3,
    "step_id": "attach-connecting-rod", "observation_id": uid(2), "observation_sha256": "b"*64,
    "reference_id": uid(3), "reference_sha256": "c"*64, "calibration_id": "fixed-camera-v1",
    "calibration_sha256": "d"*64, "policy_id": "slider-crank-v1", "policy_sha256": "e"*64,
    "captured_at": "2026-10-28T16:00:00Z", "produced_at": "2026-10-28T16:00:01Z", "parent_message_ids": []}
def msg(i, kind, payload, parents=()):
    return {**base, "message_id": uid(i), "producer": mapping[kind][0], "payload_type": kind,
        "parent_message_ids": [uid(p) for p in parents], "payload": payload}
frame = {"frame_id": uid(10), "image": artifact(11,"frame.png"), "frame_index": 0,
    "timestamp_ms": 0, "width_px": 1280, "height_px": 720, "coordinate_space": "undistorted_source_image_px"}
messages = [msg(100,"segmentation", {"provider_request_id": "synthetic-request", "provider_response_id": "synthetic-response",
    "source_media": artifact(11,"frame.png"), "frames": [frame], "objects": [{"frame_id": uid(10),
    "sam_response_id": "synthetic-response", "sam_track_id": 2, "part_id": "connecting-rod", "identity_status": "confirmed",
    "bbox_xyxy_inclusive_px": [100,100,500,180], "mask": artifact(12,"mask.png"), "crop": artifact(13,"crop.png"),
    "mask_encoding": "full_frame_binary_png_0_255", "provider_mask_confidence": 0.95}], "billed_video_frames": None, "response_status": "completed"})]
messages.append(msg(101,"geometry", {"status": "pass", "segmentation_message_id": uid(100), "calibration_valid": True,
    "hands_clear": True, "identity_status": "match", "whole_component_status": "present", "seating_status": "aligned",
    "metrics": [{"metric": "mask_iou", "part_id": "connecting-rod", "value": 0.96, "unit": "ratio", "space": "image",
    "plane_id": None, "uncertainty_95": None, "policy_threshold_id": "iou-rod-evaluated-v1", "result": "pass", "evidence_frame_ids": [uid(10)]}],
    "reason_codes": [], "evidence_frame_ids": [uid(10)]}, [100]))
audit_output = {"observation_id": uid(2), "finding": "consistent", "observations": [{"frame_id": uid(10),
    "region_xyxy_px": [100,100,500,180], "statement": "Synthetic illustration of visible alignment, not a measured finding."}],
    "contradicted_predicate_ids": [], "requires_new_view": False}
messages.append(msg(102,"visual_audit", {"model_id": "muse-glimmer", "model_revision": "example-only", "deployment_revision": "example-only",
    "model_output": audit_output, "finish_reason": "stop", "latency_ms": 1100}, []))
messages.append(msg(103,"gate_decision", {"decision": "accepted", "reason_codes": [], "geometry_message_id": uid(101),
    "audit_message_id": uid(102), "segmentation_message_id": uid(100), "checked_scene_version": 42, "committed_scene_version": 42,
    "completed_predicate_ids": ["rod-visible-alignment"], "decision_at": "2026-10-28T16:00:02Z", "elapsed_ms": 2000}, [100,101,102]))
planner_output = {"observation_id": uid(2), "action": "generate_supported_part", "part_id": "connecting-rod",
    "instruction": "Request an export of the supported link using the recorded dimensions.", "evidence_ids": [uid(10)], "required_predicate_ids": [],
    "measurement_record_id": uid(31)}
messages.append(msg(104,"planner_proposal", {"model_id": "muse-wonder-1.0", "provider_response_id": "synthetic-response-plan", "model_output": planner_output}, [103]))
def dimension(value, source, plane=None):
    return {"value_mm": value, "uncertainty_95_mm": None, "source": source, "evidence_artifact_ids": [uid(30)], "plane_id": plane}
messages.append(msg(105,"cad_request", {"template_id": "flat_two_hole_link_v1", "template_sha256": "f"*64, "measurement_record_id": uid(31),
    "parameters": {"hole_center_distance": dimension(150,"calibrated_flat_scan","mat_z0"),
    "outer_width": dimension(25,"calibrated_flat_scan","mat_z0"), "thickness": dimension(4,"manual_caliper"),
    "pin_diameter": dimension(6,"manual_caliper"), "diametral_clearance": dimension(0.5,"fabrication_policy")},
    "material_label": "example stock; not selected", "fabrication_method": "export_only", "requested_formats": ["STEP","STL"],
    "claim_scope": "dimensional_candidate_for_unloaded_test_rig"}, [104]))
messages.append(msg(106,"cad_artifact", {"cad_request_message_id": uid(105), "parameter_record": artifact(32,"parameters.json","application/json"),
    "template_source": artifact(33,"template.py","text/x-python"), "exports": [artifact(34,"link.step","application/step")],
    "checks": [{"check_id": "synthetic-solid-check", "result": "pass", "evidence": artifact(35,"report.json","application/json")}],
    "status": "geometry_valid", "physical_fit_status": "not_tested", "physical_test_record": None}, [105]))

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
write("contract.schema.json", schema)
write("glimmer-output.schema.json", audit)
write("wonder-output.schema.json", planner)
write("example-cycle.json", {"notice": "SYNTHETIC CONTRACT FIXTURE ONLY. Values, timings, hashes, files and successes are illustrative; no API or hardware was tested.", "messages": messages})
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema, format_checker=FormatChecker())
for m in messages:
    validator.validate(m)
Draft202012Validator(audit).validate(audit_output)
Draft202012Validator(planner).validate(planner_output)
bad_cases = []
bad = copy.deepcopy(messages[0]); bad["extra"] = True; bad_cases.append(bad)
bad = copy.deepcopy(messages[3]); bad["producer"] = "wonder_adapter"; bad_cases.append(bad)
bad = copy.deepcopy(messages[1]); bad["payload"]["metrics"][0]["unit"] = "mm"; bad_cases.append(bad)
bad = copy.deepcopy(messages[2]); bad["payload"]["model_output"]["accepted"] = True; bad_cases.append(bad)
for bad in bad_cases:
    assert not validator.is_valid(bad)
print("Validated full schema, seven synthetic stage messages, both model outputs, and four negative structural cases.")

# Provider request examples use reserved invalid hostnames, never real credentials.
api_dir = OUT / "api-examples"
api_dir.mkdir(exist_ok=True)
media = "https://media.example.invalid/session"
prompts = "<0f>1<|box;x1=100;y1=100;x2=300;y2=250;w=1280;h=720|>,2<|box;x1=400;y1=100;x2=600;y2=250;w=1280;h=720|>"
sam_request = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
    {"type": "input_text", "text": prompts}, {"type": "input_video", "video_url": media+"/clip.mp4"}]}],
    "stream": True, "metadata": {"include_confidence": "true", "propagation_direction": "forward"}}
sam_image = copy.deepcopy(sam_request)
sam_image["input"][0]["content"][1] = {"type": "input_image", "image_url": media+"/checkpoint.png"}
sam_image["metadata"].pop("propagation_direction")
wonder_request = {"model": "muse-wonder-1.0", "store": False, "include": ["reasoning.encrypted_content"],
    "reasoning": {"effort": "low"}, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "Choose the next supported action from REPLACE_WITH_SERIALIZED_EVIDENCE_AND_GATE_RESULT. You propose actions; only the external controller advances the workflow."}]}],
    "text": {"format": {"type": "json_schema", "name": "PlannerProposal", "strict": True, "schema": planner}}}
glimmer_request = {"model": "muse-glimmer", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Audit immutable observation REPLACE_OBSERVATION_ID against the first reference image and second current image. Report only visible evidence. Return JSON matching this schema: "+json.dumps(audit)},
    {"type": "image_url", "image_url": {"url": media+"/reference.png"}},
    {"type": "image_url", "image_url": {"url": media+"/checkpoint.png"}}]}], "max_tokens": 1024}
image_request = {"model": "muse-image-1.0", "prompt": "Create an explanatory illustrated repair card from this reference. This is explanatory artwork, not measurement evidence. Do not depict stress or safety certification.",
    "images": [{"image_url": media+"/reference.png"}], "n": 1, "response_format": "b64_json", "output_format": "png"}
baseten_request = {"model": "meta-models/muse-spark-open", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Review this ambiguous observation offline. Describe visible discrepancies and missing evidence. Your review cannot authorize the current physical workflow."},
    {"type": "image_url", "image_url": {"url": media+"/checkpoint.png"}}]}], "max_tokens": 1024}
voice_handshake = {"authorization": {"accessToken": "Bearer REPLACE_SERVER_SIDE"}, "audioEncoding": "PCM_24KHZ",
    "model": "muse-voice-transcribe-1.0", "mode": "ENDPOINTING", "keywords": ["connecting rod", "slider", "crank"]}
voice_event = {"type": "speechComplete", "turnId": 1, "transcript": "Check the connecting rod.", "audioProcessedMs": 3600}
cua_request = {"model": "muse-wonder-1.0", "tools": [{"type": "computer"}], "input": [{"type": "message", "role": "user", "content": [
    {"type": "input_text", "text": "Inspect the already generated supported link in the dedicated CAD viewport. Fit the view, reveal both holes and record a review screenshot. The external driver restricts available actions; no changes to geometry or acceptance policy are authorized."},
    {"type": "input_image", "image_url": media+"/cad-screenshot.png"}]}]}
examples = {"01-sam-video.json": sam_request, "02-sam-image.json": sam_image,
    "03-wonder-planner.json": wonder_request, "04-glimmer-audit.json": glimmer_request,
    "05-image-edit.json": image_request, "06-baseten-review.json": baseten_request,
    "07-voice-handshake.json": voice_handshake, "08-voice-final-event.json": voice_event,
    "09-wonder-cua.json": cua_request}
for name, value in examples.items():
    (api_dir/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
for p in api_dir.glob("*.json"):
    json.loads(p.read_text(encoding="utf-8"))
print("Validated syntax of nine provider examples. No network calls or live latency measurements performed.")
