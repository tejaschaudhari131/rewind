from __future__ import annotations

import numpy as np

from backend.app.config import settings
from backend.app.providers.baseten_spark import BasetenSparkClient
from backend.app.providers.meta_sam import (
    compute_mask_iou,
    decode_sam_token_string,
    encode_mask_png_bytes,
    format_multi_box_prompt,
)
from backend.app.providers.meta_voice import map_transcript_to_command
from backend.app.providers.meta_wonder import MetaWonderClient
from backend.app.schemas import make_envelope


def test_config_loads_proxy_and_contributor_defaults() -> None:
    assert settings.http_proxy == "http://127.0.0.1:8791"
    assert settings.compiler_model == "muse-spark-1.3-contributor"
    assert settings.model_sam == "sam-3.1"
    assert settings.meta_api_key.startswith("LLM_")


def test_sam_token_decoder_and_iou() -> None:
    # Real token output captured from live sam-3.1 smoke test on synthetic_rig_frame.png
    raw_token = "<0f>0<|box;x1=380;y1=173;x2=472;y2=236;w=640;h=400;c=0.9532163739204407|><|mask;x=0;y=0;c=0.9532163739204407;data=64,93,~!!!0Y!0c[p:4Z-KoyM^7N:8sb+2sheFj]hbKJ)zXZI.9p-MKbnnFBw&!6KWx.L7K)0T=KSal.?k]xM_2a.*YsHCJ]DiufBxJ&R`PIpyu&9Ge+fH/wYCh]=GD)[iBMAGS@CW_gDvTIp=YlI22Xcm88clTcVA]=7#`5Os+)#S3sD_6N*J=^Jmr&(^/sw!z/V/8r8Z0_X!EoW3ds&TD5j8OL*QF#{iTm*Sri7eqf?v[&r_.{c%S2BQ'U2Fy2g6T[nipHxtG:CF0+s@[k!Xx.$`Fir_B)C?+P*Q{f+`YVp=PDY'0N8uv@N4j6$YrYY(D_`-KsiI5wEdHmAxlQ3{h0u(flahHK)A2AxOmPDlL[rYjaz+R_5E:1BB15ZZP)8=Mv3Wo*Xb5uE$J716eyJMjbNy$xO`r2Y^hE6-EV%xy+Z%3OM_%C?GAIrjC74+mpb-M=R'`_h)SdBJ.XoN[pOpZJKoSYd4e#'beG7TsCtUvBnF/!:8@:4D/&cqb!k.XGhu.ua_o&!VoH:Tef^cH0o)#8wd73JbRu6yBvKzy-$s--Vb6UeEWHQlOn[&ijsx6n&Urcd0#'Y0P7+K2#(URrp+XA!-8V_d's%ic$1xMkD-@?$c5UL*DRN`4'o1'BKQZ088h3M%`*oBy+:X^PU#6dr.Og8*s7z!'cS)L@=-mzP/vlTlJnb&`lk9+S?i7J+'%c_{Vj7-C=I{b86W_BkDmeyvL/T7%$l*tRaG63&fY3_/tSrnsR31KtK^ULwg5[O]X9!8R2@[Ym:eP'3VMLURloOe4gPMQ]XD+#HQf9^IG.D^#TVI&a!1uAYwO5]^W.YA%@korFK7ww7?lU51@aK.&p.F!djY3k%/t%L`J0i`-Ais=-TlXyMS=w?7#=X:LI.-5v&={vK.8CV$7@y`4fe`pmfkY!L%kYu=g$&*{b0iB?DpcRH:7u&F])gK7DBp{{5#fJk`'r@y*d]B2'jb{I.hKf7houQ]T81U[ttl{9g3zLhja2]miKpXC-r5PUCF(@(NwI?eO@helLvm9MZHLNcANOo'rAMXNTQW{7#_ipVnD-(&qoIVh-@A'X*@R0`?9[MDT:5[QTJ.1%3Jj.)rghSdAFvPe#-6%w=r%Q9KsmvUhf4:9rRk3$`Ssg6[_%o)q8TN=Wdp#H)q(=)U3bd+O6iIYGJ[c9TWZcac-.g!XhtCqIUJ0z-BdT1*-FOoIDR65wx0O!o)m4LYJHpI%_Ux2Y07X8bDW{=x5v6[zC+a[!]h`jQ(C)nJRZ&k9b1K.#IHmzG-G0Q$x{LOIV82gH8sstTiQoHQ_W()KlDGCL(%+o''WGPlLG^G{1(LsWFMZtDQ`/OIA[fEu&)(H:s2oMGxLu&Q@.%hoSq%-@!y%SgLPp8zWM'2m#b:pig03Q6@!lYdep65&!1CH2R(pal4Im0ggLrj9@Y74d6f5j.UeQyzR%%J/u9rJcS)5pO|>"
    instances = decode_sam_token_string(
        raw_token,
        image_width_px=640,
        image_height_px=400,
        part_id="slider_block",
        prompt_text="blue slider block",
    )
    assert len(instances) == 1
    inst = instances[0]
    assert inst.box_xyxy_px == (380, 173, 472, 236)
    assert inst.full_frame_mask.shape == (400, 640)
    assert inst.area_px > 4000
    assert abs(inst.confidence - 0.9532163739204407) < 1e-6
    assert compute_mask_iou(inst.full_frame_mask, inst.full_frame_mask) == 1.0

    shifted = np.roll(inst.full_frame_mask, shift=10, axis=1)
    iou_shifted = compute_mask_iou(inst.full_frame_mask, shifted)
    assert 0.5 < iou_shifted < 0.95

    png_bytes = encode_mask_png_bytes(inst.full_frame_mask)
    assert png_bytes.startswith(b"\x89PNG")


def test_multi_box_prompt_formatting() -> None:
    prompt = format_multi_box_prompt([(147, 147, 263, 263), (380, 173, 472, 236)])
    assert prompt == "<0f>1<|box;147,147,263,263|>,2<|box;380,173,472,236|>"


def test_voice_command_mapper() -> None:
    assert map_transcript_to_command("check step.").intent == "check_step"
    assert map_transcript_to_command("Please verify step now").intent == "check_step"
    assert map_transcript_to_command("why did that fail?").intent == "explain_failure"
    assert map_transcript_to_command("scan flat part").intent == "scan_flat_part"
    assert map_transcript_to_command("generate replacement link").intent == "generate_replacement"
    assert map_transcript_to_command("repeat instruction").intent == "repeat_instruction"
    assert map_transcript_to_command("random background chatter").intent == "unrecognized"


def test_wonder_cua_payload_builder() -> None:
    payload = MetaWonderClient.build_cua_inspection_payload(
        cad_preview_png=b"fake_png",
        display_width=1280,
        display_height=800,
    )
    assert payload["model"] == "muse-wonder-1.0"
    assert payload["tools"][0]["type"] == "computer"
    assert payload["tools"][0]["display_width"] == 1280


def test_make_envelope_validates_segmentation_stage() -> None:
    raw_token = "<0f>0<|box;x1=380;y1=173;x2=472;y2=236;w=640;h=400;c=0.9532163739204407|><|mask;x=0;y=0;c=0.9532163739204407;data=64,93,~!!!0Y!0c[p:4Z-KoyM^7N:8sb+2sheFj]hbKJ)zXZI.9p-MKbnnFBw&!6KWx.L7K)0T=KSal.?k]xM_2a.*YsHCJ]DiufBxJ&R`PIpyu&9Ge+fH/wYCh]=GD)[iBMAGS@CW_gDvTIp=YlI22Xcm88clTcVA]=7#`5Os+)#S3sD_6N*J=^Jmr&(^/sw!z/V/8r8Z0_X!EoW3ds&TD5j8OL*QF#{iTm*Sri7eqf?v[&r_.{c%S2BQ'U2Fy2g6T[nipHxtG:CF0+s@[k!Xx.$`Fir_B)C?+P*Q{f+`YVp=PDY'0N8uv@N4j6$YrYY(D_`-KsiI5wEdHmAxlQ3{h0u(flahHK)A2AxOmPDlL[rYjaz+R_5E:1BB15ZZP)8=Mv3Wo*Xb5uE$J716eyJMjbNy$xO`r2Y^hE6-EV%xy+Z%3OM_%C?GAIrjC74+mpb-M=R'`_h)SdBJ.XoN[pOpZJKoSYd4e#'beG7TsCtUvBnF/!:8@:4D/&cqb!k.XGhu.ua_o&!VoH:Tef^cH0o)#8wd73JbRu6yBvKzy-$s--Vb6UeEWHQlOn[&ijsx6n&Urcd0#'Y0P7+K2#(URrp+XA!-8V_d's%ic$1xMkD-@?$c5UL*DRN`4'o1'BKQZ088h3M%`*oBy+:X^PU#6dr.Og8*s7z!'cS)L@=-mzP/vlTlJnb&`lk9+S?i7J+'%c_{Vj7-C=I{b86W_BkDmeyvL/T7%$l*tRaG63&fY3_/tSrnsR31KtK^ULwg5[O]X9!8R2@[Ym:eP'3VMLURloOe4gPMQ]XD+#HQf9^IG.D^#TVI&a!1uAYwO5]^W.YA%@korFK7ww7?lU51@aK.&p.F!djY3k%/t%L`J0i`-Ais=-TlXyMS=w?7#=X:LI.-5v&={vK.8CV$7@y`4fe`pmfkY!L%kYu=g$&*{b0iB?DpcRH:7u&F])gK7DBp{{5#fJk`'r@y*d]B2'jb{I.hKf7houQ]T81U[ttl{9g3zLhja2]miKpXC-r5PUCF(@(NwI?eO@helLvm9MZHLNcANOo'rAMXNTQW{7#_ipVnD-(&qoIVh-@A'X*@R0`?9[MDT:5[QTJ.1%3Jj.)rghSdAFvPe#-6%w=r%Q9KsmvUhf4:9rRk3$`Ssg6[_%o)q8TN=Wdp#H)q(=)U3bd+O6iIYGJ[c9TWZcac-.g!XhtCqIUJ0z-BdT1*-FOoIDR65wx0O!o)m4LYJHpI%_Ux2Y07X8bDW{=x5v6[zC+a[!]h`jQ(C)nJRZ&k9b1K.#IHmzG-G0Q$x{LOIV82gH8sstTiQoHQ_W()KlDGCL(%+o''WGPlLG^G{1(LsWFMZtDQ`/OIA[fEu&)(H:s2oMGxLu&Q@.%hoSq%-@!y%SgLPp8zWM'2m#b:pig03Q6@!lYdep65&!1CH2R(pal4Im0ggLrj9@Y74d6f5j.UeQyzR%%J/u9rJcS)5pO|>"
    inst = decode_sam_token_string(
        raw_token,
        image_width_px=640,
        image_height_px=400,
        part_id="slider_block",
        prompt_text="blue slider block",
    )[0]
    from backend.app.providers.meta_sam import SamSegmentationResult

    seg_res = SamSegmentationResult(
        provider_response_id="resp_unit_1",
        provider_request_id="req_unit_1",
        image_width_px=640,
        image_height_px=400,
        raw_output_text=raw_token,
        latency_ms=850.0,
        instances=[inst],
    )
    env = make_envelope(
        payload_type="segmentation",
        payload=seg_res.to_segmentation_payload(),
    )
    assert env["producer"] == "sam_adapter"
    assert env["payload_type"] == "segmentation"


async def test_baseten_local_fallback_session_digest() -> None:
    client = BasetenSparkClient(baseten_api_key="")
    res = await client.summarize_session(
        session_id="sess_unit",
        cycle_events=[{"payload_type": "gate_decision", "decision": "accepted", "elapsed_ms": 1412}],
    )
    assert res.provider == "local-deterministic-fallback"
    assert res.digest["total_cycles"] == 1
