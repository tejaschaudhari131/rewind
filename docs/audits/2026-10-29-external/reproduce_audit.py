"""Offline adversarial probes of pinned REWIND source; no live API calls.
Only relocate hard-coded fixture paths. Do not edit the repository.
"""
from pathlib import Path
print('Starting offline probes', flush=True)
import sys, asyncio, copy, json, time, re, dataclasses, traceback, mimetypes
# The desktop runner denies registry reads; this is an OS-only test adaptation.
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
mimetypes.init()

BASE = Path(__file__).resolve().parent
REPO = BASE / 'repo'
sys.path[:0] = [str(BASE / 'audit-deps'), str(REPO)]
sys.dont_write_bytecode = True
import cv2
import numpy as np
from backend.app import calibration as cal, geometry as geo, gate
from backend.app import semantic_validator as sem
from backend.app.state_machine import SessionStateMachine
from backend.app.providers.meta_sam import decode_sam_token_string
from modal_app import glimmer_server as server

GOLDEN = REPO / 'fixtures' / 'golden_rig'
cal.GOLDEN_META_PATH = GOLDEN / 'golden_metadata.json'
cal.load_reference_tag_corners_px.__defaults__ = (cal.GOLDEN_META_PATH,)
geo.GOLDEN_META_PATH = cal.GOLDEN_META_PATH
geo.GOLDEN_REF_MASK_PATH = GOLDEN / 'ref_mask_step2_connecting_rod.png'
PASS = GOLDEN / 'checkpoint_pass_step2.png'
results = {}

def probe(name, fn):
    print('Running ' + name, flush=True)
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = {'error': type(e).__name__, 'detail': str(e), 'trace': traceback.format_exc()}

async def aprobe(name, fn):
    print('Running ' + name, flush=True)
    try:
        results[name] = await fn()
    except Exception as e:
        results[name] = {'error': type(e).__name__, 'detail': str(e), 'trace': traceback.format_exc()}

def verdict_no_image():
    out, verdict = server._inspect_image_for_witness_and_seating([], {})
    return {'output': out, 'verdict': verdict}

def verdict_small_image():
    ok, data = cv2.imencode('.png', np.zeros((32, 32, 3), np.uint8))
    out, verdict = server._inspect_image_for_witness_and_seating([data.tobytes()], {})
    return {'output': out, 'verdict': verdict}

def nested_blinding():
    context = {'metadata': {'geometry': {'decision': 'accepted', 'iou': 1.0}}, 'filename': 'checkpoint_fail_flipped_rod.png'}
    clean = gate.sanitize_blinded_glimmer_context(context)
    sem.assert_glimmer_input_blinded(clean)
    server.verify_blinded_context_dict(clean)
    return {'accepted_context': clean}

def sam_empty():
    return {repr(s): len(decode_sam_token_string(s, image_width_px=1280, image_height_px=800)) for s in ['', '<0f>', '<0f>garbage']}

def sam_encodings_and_multiple():
    import ast
    from meta_sam_parser._mask_codec import _encode_raster
    # Reuse the repository's captured lossless token, not invented wire data.
    tree = ast.parse((REPO/'tests'/'test_providers_unit.py').read_text())
    token = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw_token' for t in n.targets))
    lossless = decode_sam_token_string(token,image_width_px=640,image_height_px=400)
    record=token.removeprefix('<0f>')
    multiple=decode_sam_token_string('<0f>'+record+',2'+record[1:],image_width_px=640,image_height_px=400)
    payload=_encode_raster([1,0,1,0,1,0],3,2)
    onebit=f'<0f>7<|box;x1=4;y1=5;x2=6;y2=6;w=12;h=12|><|mask;x=0;y=0;data=2,3,{payload}|>'
    binary=decode_sam_token_string(onebit,image_width_px=12,image_height_px=12)
    return {'lossless_instances':len(lossless),'multiple_track_ids':[x.sam_track_id for x in multiple], 'one_bit_instances':len(binary),'one_bit_area':binary[0].area_px,'absent_confidence_becomes':binary[0].confidence}

async def voice_probes():
    from backend.app import main
    main.SCENARIO_TO_FRAME={k:GOLDEN/v.name for k,v in main.SCENARIO_TO_FRAME.items()}
    main.session_sm=SessionStateMachine()
    decisions=[]
    for _ in range(3):
        before=main.session_sm.step_id
        out=await main.execute_voice_command(main.VoiceCommandRequest(transcript='check step',mode='live'))
        decisions.append({'checked_step':before,'decision':out['action_result']['checkpoint_decision']})
    try:
        await main.execute_voice_command(main.VoiceCommandRequest(mode='live',wav_path='not-used.wav'))
        error=None
    except Exception as exc:
        error=str(exc)
    return {'repeated_same_golden_frame':decisions,'live_wav_error':error}

def fixture_sam(image, **kw):
    return gate._build_fixture_sam_result(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), part_id=kw['part_id'], prompt_text=kw['prompt_text'])

async def delayed_gate():
    original = gate.MetaSamClient.segment_image
    async def delayed(self, image, **kw):
        await asyncio.sleep(3.10)
        return fixture_sam(image, **kw)
    gate.MetaSamClient.segment_image = delayed
    try:
        start = time.perf_counter()
        outcome = await gate.run_checkpoint_gate(PASS, mode='hybrid')
        return {'wall_ms': round((time.perf_counter()-start)*1000, 2), 'reported_ms': outcome.elapsed_ms, 'decision': outcome.decision}
    finally:
        gate.MetaSamClient.segment_image = original

async def empty_sam_gate():
    original = gate.MetaSamClient.segment_image
    async def empty(self, image, **kw):
        res = fixture_sam(image, **kw)
        res.instances = []
        res.raw_output_text = ''
        return res
    gate.MetaSamClient.segment_image = empty
    try:
        outcome = await gate.run_checkpoint_gate(PASS, mode='hybrid')
        return {'decision': outcome.decision, 'objects': outcome.segmentation_envelope['payload']['objects'], 'mask_iou': outcome.geometry_metrics.mask_iou}
    finally:
        gate.MetaSamClient.segment_image = original

async def movement_during_gate():
    sm = SessionStateMachine()
    original = gate.MetaSamClient.segment_image
    async def moving(self, image, **kw):
        sm.evaluate_stationarity(100.0)
        return fixture_sam(image, **kw)
    gate.MetaSamClient.segment_image = moving
    try:
        outcome = await gate.run_checkpoint_gate(PASS, mode='hybrid', state_machine=sm)
        return {'decision': outcome.decision, 'state': sm.state, 'motion_score': sm.last_motion_score, 'scene_version': sm.scene_version}
    finally:
        gate.MetaSamClient.segment_image = original

async def concurrent_gate():
    sm = SessionStateMachine()
    original = gate.MetaSamClient.segment_image
    async def delayed(self, image, **kw):
        await asyncio.sleep(0.10)
        return fixture_sam(image, **kw)
    gate.MetaSamClient.segment_image = delayed
    try:
        out = await asyncio.gather(*[gate.run_checkpoint_gate(PASS, mode='hybrid', state_machine=sm) for _ in range(2)], return_exceptions=True)
        return {'outcomes': [{'error':str(o)} if isinstance(o, Exception) else {'decision':o.decision} for o in out], 'completed_steps':sm.completed_steps}
    finally:
        gate.MetaSamClient.segment_image = original

async def reset_during_gate():
    from backend.app import main
    main.SCENARIO_TO_FRAME = {k:GOLDEN/v.name for k,v in main.SCENARIO_TO_FRAME.items()}
    main.session_sm = SessionStateMachine()
    old = main.session_sm
    entered, release = asyncio.Event(), asyncio.Event()
    events = []
    original = gate.MetaSamClient.segment_image
    broadcast = main.ws_hub.broadcast
    async def waiting(self, image, **kw):
        entered.set()
        await release.wait()
        return fixture_sam(image, **kw)
    async def capture(event_type, data):
        events.append({'type':event_type, 'decision':data.get('decision'), 'state':data.get('session', {}).get('state', data.get('state'))})
    gate.MetaSamClient.segment_image = waiting
    main.ws_hub.broadcast = capture
    try:
        task = asyncio.create_task(main.verify_checkpoint(main.CheckpointVerifyRequest(mode='hybrid')))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
        except TimeoutError:
            if task.done():
                await task
            task.cancel()
            raise
        reset = await main.reset_session_state()
        release.set()
        outcome = await task
        return {'old_session_id':old.session_id, 'new_session_id':main.session_sm.session_id, 'returned_decision':outcome['decision'], 'old_state':old.state, 'new_state':main.session_sm.state, 'events':events}
    finally:
        gate.MetaSamClient.segment_image = original
        main.ws_hub.broadcast = broadcast

async def validator_cases():
    baseline = await gate.run_checkpoint_gate(PASS)
    cycle = [baseline.segmentation_envelope, baseline.geometry_envelope, baseline.visual_audit_envelope, baseline.gate_decision_envelope]
    checks = {}
    def accepted(name, data):
        try:
            sem.validate_cycle_semantics(data)
            checks[name] = 'accepted_by_validator'
        except Exception as e:
            checks[name] = str(e)
    minimal = copy.deepcopy(cycle[-1]); minimal['parent_message_ids'] = []
    accepted('gate_without_other_stages', [minimal])
    mismatch = copy.deepcopy(cycle)
    mismatch[-1]['session_id'] = '11111111-1111-4111-8111-111111111111'
    mismatch[-1]['policy_sha256'] = 'd'*64
    mismatch[-1]['calibration_sha256'] = 'e'*64
    mismatch[-1]['reference_sha256'] = 'f'*64
    accepted('mismatched_session_and_hashes', mismatch)
    new_view = copy.deepcopy(cycle)
    new_view[2]['payload']['model_output']['requires_new_view'] = True
    accepted('consistent_but_requires_new_view', new_view)
    fresh = SessionStateMachine()
    checks['commit_without_capture'] = fresh.commit_gate_decision(minimal)['state']
    return checks

def flat_measurement():
    m = geo.measure_flat_link_on_mat_z0(GOLDEN/'flat_scan_mat_z0.png')
    return dataclasses.asdict(m)

def step_tampering():
    from backend.app import cad_worker as cad
    out = BASE/'cad-probe'; out.mkdir(exist_ok=True)
    mesh = cad.build_watertight_link_mesh(hole_center_distance=150, outer_width=25, thickness=4, hole_diameter=6.5)
    stl = out/'test.stl'; step=out/'test.step'
    mesh.export(str(stl))
    cad.write_iso10303_21_step_file(step_path=step, mesh=mesh, hole_center_distance=150, outer_width=25, thickness=4, hole_diameter=6.5)
    original = cad.verify_generated_cad_artifacts(step, stl, 150,25,4,6.5)
    txt = step.read_text()
    face_count = len(re.findall(' = FACE_SURFACE', txt))
    # Remove all faces and their loops: retain dimensions and shell references.
    tampered = '\n'.join(line for line in txt.splitlines() if not any(s in line for s in (' = FACE_SURFACE', ' = FACE_OUTER_BOUND', ' = POLY_LOOP')))
    step.write_text(tampered)
    after = cad.verify_generated_cad_artifacts(step, stl,150,25,4,6.5)
    return {'original_status':original.status, 'removed_faces':face_count, 'after_all_faces_removed':after.status}

def print_scale():
    import fitz
    path = REPO/'fixtures'/'printable'/'rewind_apriltag_mat_letter_tiles.pdf'
    doc = fitz.open(path)
    items=[]
    for page in doc:
        for image in page.get_image_info():
            bbox=image['bbox']; width=(bbox[2]-bbox[0])*25.4/72; height=(bbox[3]-bbox[1])*25.4/72
            items.append({'printed_width_mm':round(width,3), 'printed_height_mm':round(height,3), 'source_size_px':[image['width'], image['height']], 'vertical_scale':round(height/250,4), 'tag_side_mm':round(36*height/250,3)})
    return items

async def main():
    for name, fn in [('glimmer_no_image',verdict_no_image),('glimmer_small_blank',verdict_small_image),('nested_blinding',nested_blinding),('sam_empty_and_malformed',sam_empty),('sam_encodings_and_multiple',sam_encodings_and_multiple),('flat_measurement',flat_measurement),('step_tampering',step_tampering),('letter_print_scale',print_scale)]:
        probe(name,fn)
    for name,fn in [('late_provider',delayed_gate),('empty_sam_gate',empty_sam_gate),('movement_during_gate',movement_during_gate),('concurrent_gate',concurrent_gate),('reset_during_gate',reset_during_gate),('semantic_bypasses',validator_cases),('voice_and_step_specificity',voice_probes)]:
        await aprobe(name,fn)
    (BASE/'reproduction-results.json').write_text(json.dumps(results, indent=2, default=str),encoding='utf-8')
    print(json.dumps(results,indent=2,default=str))

if __name__ == '__main__':
    asyncio.run(main())
