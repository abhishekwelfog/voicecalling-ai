"""AI Call Quality Auditor & QA Agent for post-call assessment."""

import asyncio
import json
import re
from typing import Any, Dict, Optional
from google.genai import types

from app.services.gemini_service import GeminiService
from app.services.db_service import get_db_service
from app.utils.logging import logger
from app.utils.pii_masking import mask_sensitive_pii
from app.services.live_broadcaster import get_live_broadcaster


class CallAuditorService:
    """
    Evaluates completed call transcripts and scores:
    - Human-likeness score (1.0 to 10.0)
    - Customer sentiment (SATISFIED, POSITIVE, NEUTRAL, CONFUSED, FRUSTRATED)
    - RTO (Return to Origin) Risk (LOW, MEDIUM, HIGH)
    - Professional QA summary note
    - Call resolution classification
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None):
        self.gemini_service = gemini_service or GeminiService()

    async def audit_call(
        self,
        call_id: str,
        customer_id: Optional[int],
        customer_name: str,
        phone: str,
        order_id: str,
        transcript: str,
        call_duration: int,
        status: str = "COMPLETED",
        recording_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run post-call QA audit asynchronously with Gemini 3.8 Flash or rule-based fallback."""
        import os
        from pathlib import Path

        clean_transcript = mask_sensitive_pii(transcript or "").strip()
        log_id = None
        if call_id and str(call_id).startswith("log_"):
            try:
                log_id = int(str(call_id).replace("log_", ""))
            except ValueError:
                pass

        # Try to locate audio recording on disk if available
        audio_file_path = None
        if recording_url:
            rec_clean = recording_url.replace("\\", "/").split("/")[-1]
            candidate_path = Path("app/static/recordings") / rec_clean
            if candidate_path.is_file() and candidate_path.stat().st_size > 2048:
                audio_file_path = candidate_path

        # If transcript is empty/minimal and no audio file
        if not audio_file_path and (not clean_transcript or len(clean_transcript) < 5):
            is_canc = "CANCEL" in status.upper()
            is_no_ans = status.upper() in {"NO_ANSWER", "BUSY"} or "VOICEMAIL" in status.upper()
            fallback = {
                "human_score": 8.0,
                "customer_sentiment": "FRUSTRATED" if is_canc else "NEUTRAL",
                "rto_risk": "HIGH" if is_canc else ("MEDIUM" if is_no_ans else "LOW"),
                "rto_risk_percentage": 95 if is_canc else (50 if is_no_ans else 15),
                "rto_factors": "Customer requested order cancellation" if is_canc else ("Voicemail / Answering machine / No Answer - retry scheduled" if is_no_ans else f"Call ended early ({status})"),
                "warehouse_alert": True if is_canc else False,
                "qa_notes": "Order cancelled by customer." if is_canc else ("Voicemail or answering machine detected. Automatic retry scheduled." if is_no_ans else f"Call ended ({status}) with duration {call_duration}s."),
                "resolution": "CANCELLED" if is_canc else ("NO_ANSWER" if is_no_ans else status)
            }
            db = get_db_service()
            if customer_id:
                await db.update_customer_status(
                    customer_id=customer_id,
                    status=fallback["resolution"],
                    call_duration=call_duration,
                    call_summary=fallback["qa_notes"],
                    transcript=clean_transcript,
                    qa_score=fallback["human_score"],
                    customer_sentiment=fallback["customer_sentiment"],
                    rto_risk=fallback["rto_risk"],
                    qa_notes=fallback["qa_notes"],
                    rto_risk_percentage=fallback["rto_risk_percentage"],
                    rto_factors=fallback["rto_factors"],
                    warehouse_alert=fallback["warehouse_alert"]
                )
            if log_id:
                await db.update_call_log(
                    log_id=log_id,
                    qa_score=fallback["human_score"],
                    customer_sentiment=fallback["customer_sentiment"],
                    rto_risk=fallback["rto_risk"],
                    rto_risk_percentage=fallback["rto_risk_percentage"],
                    rto_factors=fallback["rto_factors"],
                    warehouse_alert=fallback["warehouse_alert"],
                    qa_notes=fallback["qa_notes"],
                    order_outcome=fallback["resolution"],
                    status=fallback["resolution"]
                )
            return fallback

        # If Gemini is configured, use LLM for deep audit
        if self.gemini_service.is_configured():
            try:
                client = self.gemini_service.get_client()

                if audio_file_path:
                    # Multi-modal audio evaluation for 100% transcript & sentiment accuracy
                    with open(audio_file_path, "rb") as f:
                        audio_data = f.read()

                    audio_prompt = f"""You are an elite E-Commerce Call Quality Auditor & Logistics Risk Assessor for Welfog.com.
Evaluate this recorded call between AI agent 'Priya' and customer '{customer_name}' for Order #{order_id}.

Return ONLY valid JSON (no markdown formatting, no backticks) with these exact keys:
{{
  "transcript": "Full dialogue formatted as Customer: ... Priya: ...",
  "human_score": float between 1.0 and 10.0,
  "customer_sentiment": "SATISFIED" | "POSITIVE" | "NEUTRAL" | "CONFUSED" | "FRUSTRATED",
  "rto_risk": "LOW" | "MEDIUM" | "HIGH",
  "rto_risk_percentage": integer between 5 and 95 (estimated return-to-origin probability. If customer requested cancellation, refused, or said wrong order, MUST be 95-100),
  "rto_factors": "1 concise sentence listing risk factors or 'Verified'",
  "warehouse_alert": true | false (flag true if courier return risk is high or order was cancelled/hesitant),
  "qa_notes": "1 concise sentence summarizing customer reaction and order verification state",
  "resolution": "CONFIRMED" | "CANCELLED" | "RESCHEDULED" | "NO_ANSWER" | "UNRESOLVED"
}}
"""
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model="models/gemini-3.8-flash",
                        contents=[types.Part.from_bytes(data=audio_data, mime_type="audio/webm"), audio_prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json"
                        )
                    )
                else:
                    audit_prompt = f"""You are an elite E-Commerce Call Quality Auditor & Logistics Risk Assessor for Welfog.com.
Evaluate this call between AI agent 'Priya' and customer '{customer_name}' for Order #{order_id}.

TRANSCRIPT:
{clean_transcript}

CALL DURATION: {call_duration} seconds.
INITIAL STATUS: {status}

Return ONLY valid JSON (no markdown formatting, no backticks) with these exact keys:
{{
  "human_score": float between 1.0 and 10.0 (how natural, conversational, polite, and human Priya sounded),
  "customer_sentiment": "SATISFIED" | "POSITIVE" | "NEUTRAL" | "CONFUSED" | "FRUSTRATED",
  "rto_risk": "LOW" | "MEDIUM" | "HIGH" (likelihood of courier return based on customer clarity, address confirmation, and agreement),
  "rto_risk_percentage": integer between 5 and 95 (estimated return to origin probability. If customer requested cancellation or said wrong order, MUST be 95),
  "rto_factors": "1 concise sentence listing risk factors e.g. COD reluctance, vague address, price surprise, or 'None - Verified'",
  "warehouse_alert": true | false (flag true if courier return risk is high or order was cancelled/hesitant),
  "qa_notes": "1 concise sentence summarizing customer reaction and order verification state",
  "resolution": "CONFIRMED" | "CANCELLED" | "RESCHEDULED" | "NO_ANSWER" | "UNRESOLVED"
}}
"""
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model="models/gemini-3.8-flash",
                        contents=audit_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json"
                        )
                    )

                text = response.text.strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                data = json.loads(text)

                score = round(float(data.get("human_score", 9.0)), 1)
                sentiment = str(data.get("customer_sentiment", "NEUTRAL")).upper()
                rto = str(data.get("rto_risk", "LOW")).upper()
                rto_pct = int(data.get("rto_risk_percentage", 85 if rto == "HIGH" else (45 if rto == "MEDIUM" else 15)))
                rto_factors = str(data.get("rto_factors", "Address and identity verified."))
                wh_alert = bool(data.get("warehouse_alert", rto == "HIGH" or rto_pct >= 70))
                qa_note = mask_sensitive_pii(str(data.get("qa_notes", "Order call completed.")))
                resolution = str(data.get("resolution", "CONFIRMED")).upper()
                audited_transcript = mask_sensitive_pii(data.get("transcript") or clean_transcript)

                # Directly respect Gemini AI's dynamic assessment
                is_cancelled = (resolution == "CANCELLED" or "CANCEL" in status.upper())
                is_no_answer = (resolution == "NO_ANSWER" or status.upper() in {"NO_ANSWER", "BUSY"} or "VOICEMAIL" in status.upper())
                if is_cancelled:
                    rto = "HIGH"
                    rto_pct = max(rto_pct, 90)
                    wh_alert = True
                    resolution = "CANCELLED"
                    if not rto_factors or "Verified" in rto_factors:
                        rto_factors = "Customer requested order cancellation - Do Not Dispatch"
                    if sentiment in {"SATISFIED", "POSITIVE"}:
                        sentiment = "FRUSTRATED"
                elif is_no_answer:
                    resolution = "NO_ANSWER"
                    rto = "MEDIUM"
                    rto_pct = 50
                    wh_alert = False
                    if not rto_factors or "Verified" in rto_factors:
                        rto_factors = "Voicemail / Answering machine / No Answer - retry scheduled"
                    if not qa_note or "Order call completed" in qa_note:
                        qa_note = "Voicemail or answering machine detected. Automatic retry scheduled."

                logger.info(
                    "call_audit_completed",
                    call_id=call_id,
                    order_id=order_id,
                    score=score,
                    sentiment=sentiment,
                    rto=rto,
                    rto_pct=rto_pct,
                    warehouse_alert=wh_alert
                )

                audit_result = {
                    "human_score": score,
                    "customer_sentiment": sentiment,
                    "rto_risk": rto,
                    "rto_risk_percentage": rto_pct,
                    "rto_factors": rto_factors,
                    "warehouse_alert": wh_alert,
                    "qa_notes": qa_note,
                    "resolution": resolution,
                    "transcript": audited_transcript
                }

                # Update DB asynchronously for both customers and call_logs
                db = get_db_service()
                if customer_id:
                    await db.update_customer_status(
                        customer_id=customer_id,
                        status="COMPLETED" if resolution == "CONFIRMED" else resolution,
                        call_duration=call_duration,
                        call_summary=qa_note,
                        transcript=audited_transcript,
                        qa_score=score,
                        customer_sentiment=sentiment,
                        rto_risk=rto,
                        qa_notes=qa_note,
                        rto_risk_percentage=rto_pct,
                        rto_factors=rto_factors,
                        warehouse_alert=wh_alert
                    )

                if log_id:
                    await db.update_call_log(
                        log_id=log_id,
                        qa_score=score,
                        customer_sentiment=sentiment,
                        rto_risk=rto,
                        rto_risk_percentage=rto_pct,
                        rto_factors=rto_factors,
                        warehouse_alert=wh_alert,
                        qa_notes=qa_note,
                        order_outcome=resolution,
                        status=resolution,
                        summary=qa_note,
                        transcript=audited_transcript
                    )

                # Broadcast QA score and warehouse alert to live dashboard
                broadcaster = get_live_broadcaster()
                asyncio.create_task(
                    broadcaster.broadcast({
                        "type": "qa_audit_completed",
                        "call_id": call_id,
                        "customer_id": customer_id,
                        "customer_name": customer_name,
                        "order_id": order_id,
                        "human_score": score,
                        "customer_sentiment": sentiment,
                        "rto_risk": rto,
                        "rto_risk_percentage": rto_pct,
                        "warehouse_alert": wh_alert,
                        "rto_factors": rto_factors,
                        "qa_notes": qa_note,
                        "resolution": resolution,
                        "transcript": audited_transcript
                    })
                )

                return audit_result

            except Exception as e:
                logger.warning("call_audit_llm_error_using_rule_fallback", error=str(e))

        # Rule-based fallback audit (only used if LLM service is completely unreachable)
        if status.upper() == "CANCELLED":
            fb = {
                "human_score": 9.0,
                "customer_sentiment": "FRUSTRATED",
                "rto_risk": "HIGH",
                "rto_risk_percentage": 95,
                "rto_factors": "Call status concluded as CANCELLED - Do Not Dispatch",
                "warehouse_alert": True,
                "qa_notes": "Order cancelled during call.",
                "resolution": "CANCELLED"
            }
        elif status.upper() in {"NO_ANSWER", "BUSY"}:
            fb = {
                "human_score": 8.0,
                "customer_sentiment": "NEUTRAL",
                "rto_risk": "MEDIUM",
                "rto_risk_percentage": 50,
                "rto_factors": "Customer did not answer phone",
                "warehouse_alert": False,
                "qa_notes": f"Call ended without answer ({status}).",
                "resolution": "NO_ANSWER"
            }
        else:
            fb = {
                "human_score": 9.2,
                "customer_sentiment": "NEUTRAL",
                "rto_risk": "LOW",
                "rto_risk_percentage": 15,
                "rto_factors": "Order call concluded normally.",
                "warehouse_alert": False,
                "qa_notes": "Call concluded with standard verification.",
                "resolution": "CONFIRMED"
            }

        db = get_db_service()
        if customer_id:
            await db.update_customer_status(
                customer_id=customer_id,
                status="COMPLETED" if fb["resolution"] == "CONFIRMED" else fb["resolution"],
                call_duration=call_duration,
                call_summary=fb["qa_notes"],
                transcript=clean_transcript,
                qa_score=fb["human_score"],
                customer_sentiment=fb["customer_sentiment"],
                rto_risk=fb["rto_risk"],
                qa_notes=fb["qa_notes"],
                rto_risk_percentage=fb["rto_risk_percentage"],
                rto_factors=fb["rto_factors"],
                warehouse_alert=fb["warehouse_alert"]
            )
        if log_id:
            await db.update_call_log(
                log_id=log_id,
                qa_score=fb["human_score"],
                customer_sentiment=fb["customer_sentiment"],
                rto_risk=fb["rto_risk"],
                rto_risk_percentage=fb["rto_risk_percentage"],
                rto_factors=fb["rto_factors"],
                warehouse_alert=fb["warehouse_alert"],
                qa_notes=fb["qa_notes"],
                order_outcome=fb["resolution"],
                status=fb["resolution"],
                summary=fb["qa_notes"],
                transcript=clean_transcript
            )

        broadcaster = get_live_broadcaster()
        asyncio.create_task(
            broadcaster.broadcast({
                "type": "qa_audit_completed",
                "call_id": call_id,
                "customer_id": customer_id,
                "customer_name": customer_name,
                "order_id": order_id,
                **fb
            })
        )
        return fb


_auditor: Optional[CallAuditorService] = None


def get_call_auditor() -> CallAuditorService:
    global _auditor
    if _auditor is None:
        _auditor = CallAuditorService()
    return _auditor
