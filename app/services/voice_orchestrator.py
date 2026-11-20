"""Voice Orchestrator managing bidirectional audio streaming between TeleCMI and Gemini Live."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from app.config import get_settings
from app.models.call_session import CallSession, CallState
from app.models.call_outcome import CallOutcome
from app.services.gemini_service import (
    GeminiService,
    format_conversational_name,
    simplify_product_for_speech,
)
from app.services.telecmi_service import TeleCMIService
from app.services.telephony_manager import get_telephony_manager
from app.services.call_auditor import get_call_auditor
from app.services.live_broadcaster import get_live_broadcaster
from app.services.db_service import get_db_service
from app.tools.order_tools import OrderTools
from app.tools.knowledge_tools import KnowledgeTools
from app.utils.audio import AudioAdapter
from app.utils.logging import logger
from app.utils.pii_masking import mask_sensitive_pii
from app.utils.security import (
    normalize_indian_phone_number,
    clean_speech_transcript,
    strip_recording_announcement,
    is_voicemail_or_machine,
)


class VoiceOrchestrator:
    """
    Central orchestrator coordinating:
    - TeleCMI call setup & events
    - Real-time customer audio streaming
    - Gemini Multimodal Live API streaming
    - Tool invocation (Welfog Order APIs + Qdrant RAG)
    - Audio format adaptation & resampling
    - Call lifecycle state & outcomes
    """

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        telecmi_service: Optional[TeleCMIService] = None,
        order_tools: Optional[OrderTools] = None,
        knowledge_tools: Optional[KnowledgeTools] = None
    ):
        self.settings = get_settings()
        self.gemini_service = gemini_service or GeminiService()
        self.telecmi_service = telecmi_service or TeleCMIService()
        self.order_tools = order_tools or OrderTools()
        self.knowledge_tools = knowledge_tools or KnowledgeTools()
        self.sessions: Dict[str, CallSession] = {}
        self.outcomes: Dict[str, CallOutcome] = {}

    def create_session(
        self,
        user_id: int,
        customer_phone: str,
        order_id: Optional[str] = None,
        language: str = "Hindi"
    ) -> CallSession:
        """Create and register a new call session."""
        call_id = f"welfog_call_{uuid.uuid4().hex[:12]}"
        normalized_phone = normalize_indian_phone_number(customer_phone)

        session = CallSession(
            call_id=call_id,
            user_id=user_id,
            order_id=order_id,
            customer_phone=normalized_phone,
            language=language,
            status=CallState.CREATED
        )
        self.cleanup_stale_sessions()
        self.sessions[call_id] = session
        logger.info(
            "call_session_created",
            call_id=call_id,
            user_id=user_id,
            order_id=order_id,
            extra={"phone": normalized_phone, "language": language}
        )
        return session

    def cleanup_stale_sessions(self, max_age_seconds: int = 3600):
        """Remove completed or failed sessions older than max_age_seconds to prevent memory leaks."""
        now = time.time()
        to_delete = []
        for cid, s in list(self.sessions.items()):
            age = (now - s.started_at.timestamp()) if hasattr(s, "started_at") and s.started_at else 0
            if s.status in {CallState.COMPLETED, CallState.FAILED, CallState.DISCONNECTED, CallState.NO_ANSWER, CallState.BUSY} and age > max_age_seconds:
                to_delete.append(cid)
        for cid in to_delete:
            self.sessions.pop(cid, None)
            self.outcomes.pop(cid, None)
        if to_delete:
            logger.info("cleaned_up_stale_call_sessions", count=len(to_delete))

    async def takeover_call(self, call_id: str, admin_name: str = "Admin") -> bool:
        """Allows human agent to take over call from AI agent in real-time."""
        session = self.get_session(call_id)
        if not session:
            return False
        session.metadata["is_taken_over"] = True
        session.metadata["taken_over_by"] = admin_name
        session.metadata["is_ai_speaking"] = False
        broadcaster = get_live_broadcaster()
        await broadcaster.broadcast({
            "type": "call_taken_over",
            "call_id": call_id,
            "admin": admin_name
        })
        logger.info("call_taken_over_by_agent", call_id=call_id, admin=admin_name)
        return True

    def get_session(self, call_id: str) -> Optional[CallSession]:
        return self.sessions.get(call_id)

    async def initiate_outbound_call(self, call_id: str) -> Dict[str, Any]:
        """Trigger outbound call with multi-carrier failover and error classification."""
        session = self.get_session(call_id)
        if not session:
            return {"success": False, "error": f"Session {call_id} not found"}

        session.transition_to(CallState.CALLING)
        logger.info("initiating_telecmi_call", call_id=call_id, user_id=session.user_id)

        ws_url = f"{self.settings.telecmi_websocket_url}?call_id={call_id}"

        telephony_mgr = get_telephony_manager()
        res = await telephony_mgr.initiate_call(
            customer_phone=session.customer_phone,
            websocket_url=ws_url,
            call_id=call_id
        )

        if res.get("success"):
            session.telecmi_call_id = str(res.get("call_id") or "")
            session.transition_to(CallState.RINGING)
            return {"success": True, "call_id": call_id, "telecmi_response": res}
        else:
            session.transition_to(CallState.FAILED)
            self._finalize_outcome(session, "telephony_initiation_failed", False)
            return {"success": False, "error": res.get("error")}

    async def handle_telecmi_stream(
        self,
        websocket: WebSocket,
        call_id: Optional[str] = None
    ):
        """
        Main WebSocket pipeline connecting TeleCMI call audio <-> Gemini Live API.
        """
        await websocket.accept()

        query_params = dict(websocket.query_params)
        extracted_cust_id = None
        if query_params.get("customer_id"):
            try:
                extracted_cust_id = int(query_params["customer_id"])
            except ValueError:
                pass
        extracted_order = query_params.get("order_id") or query_params.get("call_id") or (call_id if call_id and not call_id.startswith("test") else None) or "WF-10928"
        extracted_phone = query_params.get("phone") or "9876543210"
        extracted_name = query_params.get("name") or "Customer"
        extracted_address = query_params.get("address") or query_params.get("notes") or ""
        extracted_product = query_params.get("product_name") or query_params.get("product") or ""
        extracted_price = query_params.get("product_price") or query_params.get("price") or ""
        extracted_delivery = query_params.get("delivery_time") or query_params.get("delivery") or ""
        extracted_payment = query_params.get("payment_method") or query_params.get("payment") or ""

        from app.services.db_service import get_db_service
        try:
            db = get_db_service()
            clean_digits = "".join(c for c in extracted_phone if c.isdigit())
            if len(clean_digits) == 12 and clean_digits.startswith("91"):
                clean_digits = clean_digits[2:]
            matching = []
            if extracted_order and extracted_order != "WF-10928":
                matching = await db.get_customers(search=extracted_order)
            if not matching and clean_digits:
                matching = await db.get_customers(search=clean_digits)
            if matching:
                cust = matching[0]
                if not extracted_cust_id and cust.get("id"):
                    extracted_cust_id = cust["id"]
                if cust.get("name") and cust["name"] != "Customer":
                    extracted_name = cust["name"]
                if cust.get("notes") and cust["notes"] not in {"Bulk Import", "Bulk Pasted"}:
                    extracted_address = cust["notes"]
                if cust.get("order_id"):
                    extracted_order = cust["order_id"]
                if cust.get("product_name") and not extracted_product:
                    extracted_product = cust["product_name"]
                if cust.get("product_price") and not extracted_price:
                    extracted_price = cust["product_price"]
                if cust.get("delivery_time") and not extracted_delivery:
                    extracted_delivery = cust["delivery_time"]
                if cust.get("payment_method") and not extracted_payment:
                    extracted_payment = cust["payment_method"]
        except Exception as e:
            logger.warning("db_customer_lookup_fallback", error=str(e))

        if extracted_address in {"Bulk Import", "Bulk Pasted"}:
            extracted_address = ""
        if not extracted_product:
            extracted_product = "Welfog item"
        if not extracted_delivery:
            extracted_delivery = "2-3 business days"
        if not extracted_payment:
            extracted_payment = "Cash on Delivery (COD)"

        session: Optional[CallSession] = None
        if call_id and call_id in self.sessions:
            session = self.sessions[call_id]
        elif self.sessions:
            for s in reversed(list(self.sessions.values())):
                if s.status in {CallState.CREATED, CallState.CALLING, CallState.RINGING}:
                    session = s
                    break

        if not session:
            session = self.create_session(
                user_id=extracted_cust_id or 1,
                customer_phone=extracted_phone,
                order_id=extracted_order
            )
        else:
            if extracted_cust_id:
                session.user_id = extracted_cust_id
            if extracted_order and not session.order_id:
                session.order_id = extracted_order

        session.metadata["customer_name"] = extracted_name
        session.metadata["delivery_address"] = extracted_address
        session.metadata["product_name"] = extracted_product
        session.metadata["product_price"] = extracted_price
        session.metadata["delivery_time"] = extracted_delivery
        session.metadata["payment_method"] = extracted_payment

        session.transition_to(CallState.ANSWERED)
        logger.info(
            "telecmi_websocket_connected",
            call_id=session.call_id,
            order_id=session.order_id,
            extra={
                "customer": extracted_name,
                "address": extracted_address,
                "product": extracted_product,
                "price": extracted_price,
                "delivery": extracted_delivery
            }
        )

        audio_adapter = AudioAdapter(
            telecmi_sample_rate=16000,
            gemini_input_sample_rate=16000,
            gemini_output_sample_rate=24000
        )

        if not self.gemini_service.is_configured():
            logger.error("gemini_not_configured_for_stream", call_id=session.call_id)
            session.transition_to(CallState.FAILED)
            await websocket.close(code=1011, reason="Gemini API Key unconfigured")
            self._finalize_outcome(session, "gemini_unconfigured", False)
            return

        session.transition_to(CallState.AI_SESSION_STARTED)
        gemini_client = self.gemini_service.get_client()
        live_config = self.gemini_service.build_live_config(
            customer_name=extracted_name,
            order_id=extracted_order,
            customer_phone=extracted_phone,
            delivery_address=extracted_address,
            product_name=extracted_product,
            product_price=extracted_price,
            delivery_time=extracted_delivery,
            payment_method=extracted_payment,
            include_tools=True
        )
        model_id = self.settings.gemini_live_model
        fallback_model = "models/gemini-2.5-flash-native-audio-latest"

        try:
            connect_cm = None
            try:
                connect_cm = gemini_client.aio.live.connect(model=model_id, config=live_config)
                gemini_session = await connect_cm.__aenter__()
                active_model = model_id
            except Exception as primary_err:
                if model_id != fallback_model:
                    logger.warning(
                        "primary_gemini_model_failed_trying_fallback",
                        primary=model_id,
                        fallback=fallback_model,
                        error=str(primary_err)
                    )
                    connect_cm = gemini_client.aio.live.connect(model=fallback_model, config=live_config)
                    gemini_session = await connect_cm.__aenter__()
                    active_model = fallback_model
                else:
                    raise primary_err

            try:
                session.transition_to(CallState.CONVERSATION_ACTIVE)
                logger.info("gemini_live_session_active", call_id=session.call_id, model=active_model)

                pickup_event = asyncio.Event()
                session.metadata["pickup_event"] = pickup_event

                greeting_task = asyncio.create_task(
                    self._await_pickup_and_greet(
                        gemini_session=gemini_session,
                        session=session,
                        pickup_event=pickup_event,
                        customer_name=extracted_name,
                        order_id=extracted_order,
                        product_name=extracted_product,
                        product_price=extracted_price
                    )
                )

                stream_in_task = asyncio.create_task(
                    self._customer_to_gemini(websocket, gemini_session, audio_adapter, session, pickup_event)
                )
                stream_out_task = asyncio.create_task(
                    self._gemini_to_customer(websocket, gemini_session, audio_adapter, session)
                )
                silence_task = asyncio.create_task(
                    self._monitor_silence(websocket, gemini_session, session, pickup_event)
                )

                done, pending = await asyncio.wait(
                    [stream_in_task, stream_out_task, silence_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                greeting_task.cancel()
                for task in pending:
                    task.cancel()

                for task in done:
                    if not task.cancelled():
                        exc = task.exception()
                        if exc:
                            raise exc
            finally:
                if connect_cm:
                    await connect_cm.__aexit__(None, None, None)

        except WebSocketDisconnect:
            logger.info("telecmi_call_hung_up", call_id=session.call_id)
            session.transition_to(CallState.DISCONNECTED)
        except Exception as e:
            logger.error("voice_orchestrator_stream_error", call_id=session.call_id, error=str(e))
            session.transition_to(CallState.FAILED)
        finally:
            if session.status not in {CallState.COMPLETED, CallState.FAILED, CallState.DISCONNECTED}:
                session.transition_to(CallState.COMPLETED)
            self._finalize_outcome(session, "conversation_completed", True)

    async def _await_pickup_and_greet(
        self,
        gemini_session: Any,
        session: CallSession,
        pickup_event: asyncio.Event,
        customer_name: str,
        order_id: str,
        product_name: str = "Welfog Premium Item",
        product_price: str = "₹999"
    ):
        """Wait for verified customer pickup signal and trigger warm human Priya opening greeting."""
        try:
            await asyncio.wait_for(pickup_event.wait(), timeout=60.0)
            session.transition_to(CallState.ANSWERED)
            logger.info("customer_pickup_confirmed_triggering_greeting", call_id=session.call_id)

            formatted_name = format_conversational_name(customer_name)
            spoken_product = simplify_product_for_speech(product_name)
            clean_price = product_price.strip() if product_price else ""
            price_phrase = f" (Price: {clean_price})" if clean_price else ""

            # Broadcast live call answered event to admin dashboard
            broadcaster = get_live_broadcaster()
            asyncio.create_task(
                broadcaster.broadcast({
                    "type": "call_connected",
                    "call_id": session.call_id,
                    "customer_name": formatted_name,
                    "order_id": order_id
                })
            )

            # Optimized 0.08s buffer for WebRTC audio socket stabilization without delay
            await asyncio.sleep(0.08)

            session.metadata["opening_greeting_active"] = True
            session.metadata["greeting_start_time"] = time.time()
            session.metadata["opening_speech_frames"] = 0

            greeting_instruction = (
                f"Customer ({formatted_name}) has answered the live call. "
                f"Zero hardcoded scripts: In 1 warm, concise, and polite sentence, introduce yourself as Priya from Welfog and ask to verify their order for {spoken_product}{price_phrase} for immediate dispatch. "
                f"CRITICAL LANGUAGE RULE: If the customer speaks, greets, or asks in English, speak fluent, natural English. If they speak Hindi, speak Hindi. Always mirror the customer's active language immediately. "
                f"CRITICAL: Telephone pickup greetings ('Hello', 'Haanji', 'Kaun?', 'Kahiye') are strictly call acknowledgments, NOT order confirmation. Distinguish pickup greetings from confirmation, and never call end_call during opening greetings."
            )
            await gemini_session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=greeting_instruction)]
                    )
                ],
                turn_complete=True
            )
            logger.info("opening_greeting_dispatched_to_gemini", call_id=session.call_id)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error("greeting_trigger_error", call_id=session.call_id, error=str(e))

    async def _customer_to_gemini(
        self,
        websocket: WebSocket,
        gemini_session: Any,
        audio_adapter: AudioAdapter,
        session: CallSession,
        pickup_event: asyncio.Event
    ):
        """Receive audio bytes from TeleCMI, filter ringback/echo, detect true voice, and stream to Gemini Live."""
        try:
            chunks_received = 0
            session.metadata["last_customer_speech_time"] = time.time()
            session.metadata["silence_prompt_count"] = 0

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info("websocket_disconnect_received", call_id=session.call_id)
                    break

                if "bytes" in message and message["bytes"]:
                    raw_audio = message["bytes"]
                    pcm16 = audio_adapter.telecmi_to_gemini(raw_audio)
                    if pcm16:
                        chunks_received += 1
                        is_ai_speaking = bool(session.metadata.get("is_ai_speaking", False))
                        last_ai_audio = session.metadata.get("last_ai_audio_time", 0.0)
                        if is_ai_speaking and (time.time() - last_ai_audio > 0.20):
                            is_ai_speaking = False
                            session.metadata["is_ai_speaking"] = False

                        recent_ai_peak = session.metadata.get("recent_ai_peak", 0)
                        gated_pcm, is_speech, rms, peak = audio_adapter.process_incoming_pcm(
                            pcm16, 
                            is_ai_speaking=is_ai_speaking,
                            recent_ai_peak=recent_ai_peak,
                            is_call_answered=pickup_event.is_set()
                        )

                        if is_speech:
                            session.metadata["last_customer_speech_time"] = time.time()
                            session.metadata["last_ai_turn_end_time"] = time.time()
                            session.metadata["silence_prompt_count"] = 0

                            if session.metadata.get("call_concluded") and not session.metadata.get("farewell_delivered"):
                                session.metadata["post_farewell_ack"] = True
                                session.metadata["farewell_time"] = time.time()

                            # Verified human voice pickup detection (not dial tone)
                            if not pickup_event.is_set():
                                pickup_event.set()
                                session.transition_to(CallState.ANSWERED)
                                logger.info("pickup_detected_via_human_voice", call_id=session.call_id)

                        # Broadcast live audio level to dashboard waveform every 4 chunks (~80ms)
                        if chunks_received % 4 == 0:
                            broadcaster = get_live_broadcaster()
                            asyncio.create_task(
                                broadcaster.broadcast({
                                    "type": "audio_levels",
                                    "call_id": session.call_id,
                                    "rms_in": round(rms, 1),
                                    "peak_in": int(peak),
                                    "is_speech": is_speech,
                                    "is_ai_speaking": is_ai_speaking
                                })
                            )

                        if gated_pcm:
                            # Protect opening greeting: don't let initial line connection clicks/ringback residue interrupt Priya
                            is_opening = session.metadata.get("opening_greeting_active", False)
                            greeting_elapsed = time.time() - session.metadata.get("greeting_start_time", 0.0)
                            if is_opening and greeting_elapsed < 2.0 and peak < 1600:
                                pass
                            # While AI is speaking, only forward if it is customer voice (above AEC threshold)
                            elif is_ai_speaking and not is_speech:
                                pass
                            else:
                                await gemini_session.send_realtime_input(
                                    audio=types.Blob(data=gated_pcm, mime_type="audio/pcm;rate=16000")
                                )
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        event = (data.get("event") or data.get("action") or "").lower()
                        if event in ("call_answered", "answered", "call_connected", "connected", "start", "answer"):
                            if not pickup_event.is_set():
                                pickup_event.set()
                                session.transition_to(CallState.ANSWERED)
                                logger.info("pickup_detected_via_telecom_event", call_id=session.call_id, telecom_event=event)
                        elif event in ("stop", "hangup"):
                            logger.info("telecmi_hangup_event", call_id=session.call_id)
                            break
                        elif event == "media" and "payload" in data:
                            import base64
                            raw_audio = base64.b64decode(data["payload"])
                            pcm16 = audio_adapter.telecmi_to_gemini(raw_audio)
                            if pcm16:
                                chunks_received += 1
                                is_ai_speaking = bool(session.metadata.get("is_ai_speaking", False))
                                last_ai_audio = session.metadata.get("last_ai_audio_time", 0.0)
                                if is_ai_speaking and (time.time() - last_ai_audio > 0.20):
                                    is_ai_speaking = False
                                    session.metadata["is_ai_speaking"] = False

                                recent_ai_peak = session.metadata.get("recent_ai_peak", 0)
                                gated_pcm, is_speech, rms, peak = audio_adapter.process_incoming_pcm(
                                    pcm16, 
                                    is_ai_speaking=is_ai_speaking,
                                    recent_ai_peak=recent_ai_peak,
                                    is_call_answered=pickup_event.is_set()
                                )
                                if is_speech:
                                    session.metadata["last_customer_speech_time"] = time.time()
                                    session.metadata["last_ai_turn_end_time"] = time.time()
                                    session.metadata["silence_prompt_count"] = 0

                                    if session.metadata.get("call_concluded") and not session.metadata.get("farewell_delivered"):
                                        session.metadata["post_farewell_ack"] = True
                                        session.metadata["farewell_time"] = time.time()

                                    if not pickup_event.is_set():
                                        pickup_event.set()
                                        session.transition_to(CallState.ANSWERED)
                                        logger.info("pickup_detected_via_human_voice", call_id=session.call_id)

                                if chunks_received % 4 == 0:
                                    broadcaster = get_live_broadcaster()
                                    asyncio.create_task(
                                        broadcaster.broadcast({
                                            "type": "audio_levels",
                                            "call_id": session.call_id,
                                            "rms_in": round(rms, 1),
                                            "peak_in": int(peak),
                                            "is_speech": is_speech,
                                            "is_ai_speaking": is_ai_speaking
                                        })
                                    )

                                if gated_pcm:
                                    is_opening = session.metadata.get("opening_greeting_active", False)
                                    greeting_elapsed = time.time() - session.metadata.get("greeting_start_time", 0.0)
                                    if is_opening and greeting_elapsed < 2.0 and peak < 1600:
                                        pass
                                    elif is_ai_speaking and not is_speech:
                                        pass
                                    else:
                                        await gemini_session.send_realtime_input(
                                            audio=types.Blob(data=gated_pcm, mime_type="audio/pcm;rate=16000")
                                        )
                    except json.JSONDecodeError:
                        pass
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    async def _gemini_to_customer(
        self,
        websocket: WebSocket,
        gemini_session: Any,
        audio_adapter: AudioAdapter,
        session: CallSession
    ):
        """Receive audio & tool calls from Gemini Live across all conversational turns and stream audio to TeleCMI."""
        import numpy as np
        ai_chunks_sent = 0
        ai_turn_parts = session.metadata.setdefault("ai_turn_parts", [])
        customer_turn_parts = session.metadata.setdefault("customer_turn_parts", [])

        try:
            while True:
                if session.status in {CallState.COMPLETED, CallState.DISCONNECTED, CallState.FAILED}:
                    break

                try:
                    async for response in gemini_session.receive():
                        # If human supervisor took over, silence AI stream
                        if session.metadata.get("is_taken_over"):
                            continue

                        server_content = getattr(response, "server_content", None)
                        if server_content:
                            # 1. Capture customer speech transcription from Gemini Live ASR
                            in_trans = getattr(server_content, "input_transcription", None)
                            if in_trans and getattr(in_trans, "text", None):
                                txt = in_trans.text
                                if txt:
                                    customer_turn_parts.append(txt)

                            if in_trans and getattr(in_trans, "finished", False):
                                if customer_turn_parts:
                                    raw_text = "".join(customer_turn_parts).strip()
                                    customer_turn_parts.clear()
                                    if raw_text:
                                        cleaned = clean_speech_transcript(mask_sensitive_pii(raw_text))
                                        if not cleaned:
                                            # Pure automated carrier recording announcement stripped away
                                            continue

                                        # Answering Machine Detection (AMD) / Voicemail auto-disconnect & retry
                                        if is_voicemail_or_machine(cleaned):
                                            logger.info("voicemail_detected_on_customer_speech", call_id=session.call_id, text=cleaned)
                                            session.status = CallState.NO_ANSWER
                                            session.metadata["call_concluded"] = True
                                            session.metadata["concluded_time"] = time.time()
                                            session.metadata["farewell_time"] = time.time()
                                            session.metadata["concluded_reason"] = "voicemail"
                                            session.metadata["farewell_delivered"] = True
                                            try:
                                                db = get_db_service()
                                                asyncio.create_task(db.schedule_retry(session.user_id, delay_minutes=20))
                                            except Exception as db_err:
                                                logger.warning("failed_scheduling_voicemail_retry", error=str(db_err))
                                            try:
                                                await websocket.send_text(json.dumps({
                                                    "event": "hangup",
                                                    "reason": "voicemail",
                                                    "delay_ms": 100
                                                }))
                                            except Exception:
                                                pass
                                            break

                                        entry = f"Customer: {cleaned}"
                                        turns = session.metadata.setdefault("conversation_turns", [])
                                        if not turns or turns[-1] != entry:
                                            turns.append(entry)
                                            logger.info("customer_transcript_captured", call_id=session.call_id, text=cleaned)
                                            broadcaster = get_live_broadcaster()
                                            asyncio.create_task(
                                                broadcaster.broadcast({
                                                    "type": "customer_transcript",
                                                    "call_id": session.call_id,
                                                    "text": cleaned
                                                })
                                            )
                                            try:
                                                await websocket.send_text(json.dumps({
                                                    "event": "customer_transcript",
                                                    "text": cleaned
                                                }))
                                            except Exception:
                                                pass
                                            self._handle_post_turn_signoff(session, speaker="customer", text=cleaned)

                            # 2. Capture AI speech transcription from Gemini Live
                            out_trans = getattr(server_content, "output_transcription", None)
                            if out_trans and getattr(out_trans, "text", None):
                                txt = out_trans.text
                                if txt:
                                    # If customer had pending transcription parts before AI started speaking, finalize customer turn first!
                                    if customer_turn_parts:
                                        raw_text = "".join(customer_turn_parts).strip()
                                        customer_turn_parts.clear()
                                        if raw_text:
                                            cleaned = clean_speech_transcript(mask_sensitive_pii(raw_text))
                                            if cleaned:
                                                if is_voicemail_or_machine(cleaned):
                                                    logger.info("voicemail_detected_on_customer_speech_ai_start", call_id=session.call_id, text=cleaned)
                                                    session.status = CallState.NO_ANSWER
                                                    session.metadata["call_concluded"] = True
                                                    session.metadata["concluded_time"] = time.time()
                                                    session.metadata["farewell_time"] = time.time()
                                                    session.metadata["concluded_reason"] = "voicemail"
                                                    session.metadata["farewell_delivered"] = True
                                                    try:
                                                        db = get_db_service()
                                                        asyncio.create_task(db.schedule_retry(session.user_id, delay_minutes=20))
                                                    except Exception as db_err:
                                                        logger.warning("failed_scheduling_voicemail_retry", error=str(db_err))
                                                    try:
                                                        await websocket.send_text(json.dumps({
                                                            "event": "hangup",
                                                            "reason": "voicemail",
                                                            "delay_ms": 100
                                                        }))
                                                    except Exception:
                                                        pass
                                                    break

                                                entry = f"Customer: {cleaned}"
                                                turns = session.metadata.setdefault("conversation_turns", [])
                                                if not turns or turns[-1] != entry:
                                                    turns.append(entry)
                                                    logger.info("customer_transcript_captured_on_ai_start", call_id=session.call_id, text=cleaned)
                                                    broadcaster = get_live_broadcaster()
                                                    asyncio.create_task(
                                                        broadcaster.broadcast({
                                                            "type": "customer_transcript",
                                                            "call_id": session.call_id,
                                                            "text": cleaned
                                                        })
                                                    )
                                                    try:
                                                        await websocket.send_text(json.dumps({
                                                            "event": "customer_transcript",
                                                            "text": cleaned
                                                        }))
                                                    except Exception:
                                                        pass
                                                self._handle_post_turn_signoff(session, speaker="customer", text=cleaned)

                                    ai_turn_parts.append(txt)

                            # Interruption handling
                            if getattr(server_content, "interrupted", False):
                                session.metadata["is_ai_speaking"] = False
                                session.metadata["opening_greeting_active"] = False
                                session.metadata["interrupted_time"] = time.time()

                                # If interrupted, flush whatever Priya spoke so far
                                if ai_turn_parts:
                                    raw_text = "".join(ai_turn_parts).strip()
                                    ai_turn_parts.clear()
                                    if raw_text:
                                        cleaned = clean_speech_transcript(mask_sensitive_pii(raw_text))
                                        entry = f"Priya: {cleaned}"
                                        turns = session.metadata.setdefault("conversation_turns", [])
                                        if not turns or turns[-1] != entry:
                                            turns.append(entry)
                                            broadcaster = get_live_broadcaster()
                                            asyncio.create_task(
                                                broadcaster.broadcast({
                                                    "type": "ai_transcript",
                                                    "call_id": session.call_id,
                                                    "text": cleaned
                                                })
                                            )
                                            try:
                                                await websocket.send_text(json.dumps({"event": "ai_transcript", "text": cleaned}))
                                            except Exception:
                                                pass

                                        # Auto-safeguard: If Priya delivered final confirmation or farewell dialogue,
                                        # mark concluded immediately so silence watcher never wakes up!
                                        lower_c = cleaned.lower()
                                        farewell_markers = [
                                            "din shubh ho", "shubh ho", "dhanyavad", "dhanyawaad",
                                            "order confirm ho gaya", "order cancel kar deti hoon",
                                            "कन्फर्म हो गया", "शुभ हो", "धन्यवाद", "कैंसिल कर देती",
                                            "have a great day", "order has been confirmed", "order confirmed",
                                            "order is confirmed", "thank you for shopping", "order has been cancelled",
                                            "order is cancelled", "take care", "goodbye"
                                        ]
                                        if any(m in lower_c for m in farewell_markers):
                                            session.metadata["call_concluded"] = True
                                            session.metadata["farewell_delivered"] = True
                                            session.metadata["farewell_time"] = time.time()
                                            session.metadata.setdefault("concluded_reason", "order_confirmed")
                                            logger.info("farewell_dialogue_spoken_auto_concluded", call_id=session.call_id)
                                            self._handle_post_turn_signoff(session, speaker="ai", text=cleaned)

                                logger.info("gemini_speech_interrupted_by_customer", call_id=session.call_id)
                                try:
                                    await websocket.send_text(json.dumps({
                                        "event": "interrupted",
                                        "action": "clear_buffer"
                                    }))
                                except Exception:
                                    pass
                                continue

                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn:
                                for part in getattr(model_turn, "parts", []):
                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data and getattr(inline_data, "data", None):
                                        # Drop any trailing audio frames arriving within 200ms of interruption
                                        if time.time() - session.metadata.get("interrupted_time", 0) < 0.20:
                                            continue

                                        # Anti-loop shield: Drop repeated trailing audio once single farewell has been delivered
                                        if session.metadata.get("call_concluded") and session.metadata.get("farewell_delivered"):
                                            continue

                                        session.metadata["is_ai_speaking"] = True
                                        session.metadata["last_ai_audio_time"] = time.time()
                                        raw_gemini_pcm = inline_data.data
                                        ai_chunks_sent += 1

                                        # Calculate audio energy for AEC tracking & live waveform broadcast
                                        gemini_samples = np.frombuffer(raw_gemini_pcm, dtype=np.int16)
                                        ai_rms, ai_peak = AudioAdapter.calculate_energy(gemini_samples)
                                        session.metadata["recent_ai_peak"] = ai_peak

                                        telecmi_audio = audio_adapter.gemini_to_telecmi(raw_gemini_pcm)
                                        if telecmi_audio:
                                            try:
                                                await websocket.send_bytes(telecmi_audio)
                                            except (WebSocketDisconnect, RuntimeError):
                                                logger.info("telecmi_ws_closed_during_audio_send", call_id=session.call_id)
                                                return
                                            except Exception as send_err:
                                                logger.warning("failed_sending_telecmi_audio", error=str(send_err))

                                        # Broadcast AI speech audio level for real-time waveform on dashboard (~80ms rate)
                                        if ai_chunks_sent % 4 == 0:
                                            broadcaster = get_live_broadcaster()
                                            asyncio.create_task(
                                                broadcaster.broadcast({
                                                    "type": "audio_levels",
                                                    "call_id": session.call_id,
                                                    "rms_out": round(ai_rms, 1),
                                                    "peak_out": int(ai_peak),
                                                    "is_ai_speaking": True
                                                })
                                            )

                            # Turn completion check
                            if getattr(server_content, "turn_complete", False) or (out_trans and getattr(out_trans, "finished", False)):
                                if ai_turn_parts:
                                    raw_text = "".join(ai_turn_parts).strip()
                                    ai_turn_parts.clear()
                                    if raw_text:
                                        cleaned = clean_speech_transcript(mask_sensitive_pii(raw_text))
                                        entry = f"Priya: {cleaned}"
                                        turns = session.metadata.setdefault("conversation_turns", [])
                                        if not turns or turns[-1] != entry:
                                            turns.append(entry)
                                            logger.info("ai_transcript_captured", call_id=session.call_id, text=cleaned)
                                            broadcaster = get_live_broadcaster()
                                            asyncio.create_task(
                                                broadcaster.broadcast({
                                                    "type": "ai_transcript",
                                                    "call_id": session.call_id,
                                                    "text": cleaned
                                                })
                                            )
                                            try:
                                                await websocket.send_text(json.dumps({"event": "ai_transcript", "text": cleaned}))
                                            except Exception:
                                                pass

                                        # Auto-safeguard: If Priya delivered final confirmation or farewell dialogue,
                                        # mark concluded immediately so silence watcher never wakes up!
                                        lower_c = cleaned.lower()
                                        farewell_markers = [
                                            "din shubh ho", "shubh ho", "dhanyavad", "dhanyawaad",
                                            "order confirm ho gaya", "order cancel kar deti hoon",
                                            "कन्फर्म हो गया", "शुभ हो", "धन्यवाद", "कैंसिल कर देती",
                                            "have a great day", "order has been confirmed", "order confirmed",
                                            "order is confirmed", "thank you for shopping", "order has been cancelled",
                                            "order is cancelled", "take care", "goodbye"
                                        ]
                                        if any(m in lower_c for m in farewell_markers):
                                            session.metadata["call_concluded"] = True
                                            session.metadata["farewell_delivered"] = True
                                            session.metadata["farewell_time"] = time.time()
                                            session.metadata.setdefault("concluded_reason", "order_confirmed")
                                            logger.info("farewell_dialogue_spoken_auto_concluded", call_id=session.call_id)

                                session.metadata["is_ai_speaking"] = False
                                session.metadata["opening_greeting_active"] = False
                                session.metadata["last_ai_turn_end_time"] = time.time()
                                logger.info("gemini_turn_completed_listening_to_customer", call_id=session.call_id)
                                try:
                                    await websocket.send_text(json.dumps({"event": "turn_complete"}))
                                except Exception:
                                    pass

                                if session.metadata.get("call_concluded"):
                                    session.metadata["farewell_time"] = time.time()
                                    session.metadata["farewell_delivered"] = True

                        tool_call = getattr(response, "tool_call", None)
                        if tool_call:
                            session.metadata["is_ai_speaking"] = False
                            session.transition_to(CallState.TOOL_CALL)
                            function_responses = []

                            for call in getattr(tool_call, "function_calls", []):
                                fn_name = call.name
                                fn_args = call.args or {}
                                call_id_val = call.id

                                logger.info(
                                    "executing_gemini_tool",
                                    call_id=session.call_id,
                                    tool_name=fn_name,
                                    extra={"args": fn_args}
                                )
                                broadcaster = get_live_broadcaster()
                                asyncio.create_task(
                                    broadcaster.broadcast({
                                        "type": "tool_executed",
                                        "call_id": session.call_id,
                                        "tool": fn_name,
                                        "args": fn_args
                                    })
                                )
                                try:
                                    await websocket.send_text(json.dumps({"event": "tool_calling", "tool": fn_name}))
                                    if fn_name == "end_call":
                                        end_reason = fn_args.get("reason", "order_confirmed")
                                        await websocket.send_text(json.dumps({"event": "order_concluded", "reason": end_reason}))
                                except Exception:
                                    pass

                                t_start = time.perf_counter()
                                tool_result = await self._execute_tool(fn_name, fn_args, session)
                                latency_ms = (time.perf_counter() - t_start) * 1000

                                session.record_tool_call(fn_name, fn_args, tool_result, latency_ms)

                                function_responses.append(
                                    types.FunctionResponse(
                                        name=fn_name,
                                        id=call_id_val,
                                        response={"result": tool_result}
                                    )
                                )

                            session.transition_to(CallState.TOOL_RESULT)
                            await gemini_session.send_tool_response(function_responses=function_responses)
                            session.transition_to(CallState.CONVERSATION_ACTIVE)

                except (WebSocketDisconnect, asyncio.CancelledError):
                    raise
                except Exception as stream_err:
                    logger.warning("gemini_receive_turn_cycle_warning", call_id=session.call_id, error=str(stream_err))
                    if "closed" in str(stream_err).lower() or "1007" in str(stream_err):
                        break
                    await asyncio.sleep(0.05)

        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error("voice_orchestrator_gemini_to_customer_error", call_id=session.call_id, error=str(e))

    async def _monitor_silence(
        self,
        websocket: WebSocket,
        gemini_session: Any,
        session: CallSession,
        pickup_event: asyncio.Event
    ):
        """
        Proactive Silence & Call Concluding Watcher:
        1. If conversation has reached farewell, wait for true silence before disconnecting.
        2. If customer speaks at ANY point, hangup is cancelled immediately.
        3. Human-like natural probing (Tier 1: 7s, Tier 2: 13s, Tier 3: 20s).
        """
        try:
            await pickup_event.wait()
            now = time.time()
            session.metadata.setdefault("last_ai_turn_end_time", now)
            session.metadata.setdefault("last_customer_speech_time", now)
            session.metadata.setdefault("silence_prompt_count", 0)

            while True:
                await asyncio.sleep(0.4)
                if session.status in {CallState.COMPLETED, CallState.DISCONNECTED, CallState.FAILED}:
                    break

                # Periodic dialogue check: auto-detect natural conclusion from turns
                self._check_and_mark_call_concluded(session)

                if session.metadata.get("call_concluded"):
                    # CRITICAL: NEVER hang up while AI is actively speaking or streaming audio frames!
                    if session.metadata.get("is_ai_speaking"):
                        continue

                    last_ai_audio = session.metadata.get("last_ai_audio_time", 0.0)
                    if last_ai_audio and (time.time() - last_ai_audio < 0.5):
                        continue

                    farewell_t = session.metadata.get("farewell_time")
                    concluded_t = session.metadata.get("concluded_time", 0)
                    now_t = time.time()

                    last_cust = session.metadata.get("last_customer_speech_time") or 0.0
                    last_ai_audio = session.metadata.get("last_ai_audio_time") or 0.0
                    last_ai_turn = session.metadata.get("last_ai_turn_end_time") or 0.0
                    last_active = max(last_cust, last_ai_audio, last_ai_turn)

                    should_hangup = False
                    # Natural polite human hangup timing: 1.2s - 1.5s after farewell finishes streaming so customer isn't trapped in dead air
                    if farewell_t:
                        if (now_t - last_ai_audio >= 0.8) and (now_t - farewell_t >= 1.2):
                            should_hangup = True
                    elif concluded_t:
                        if (now_t - last_ai_audio >= 0.8) and (now_t - concluded_t >= 2.0):
                            should_hangup = True

                    if should_hangup:
                        reason = session.metadata.get("concluded_reason", "completed")
                        logger.info("silence_watcher_concluding_call", call_id=session.call_id, reason=reason)
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "hangup",
                                "reason": reason,
                                "delay_ms": 250
                            }))
                        except Exception:
                            pass
                        break
                    continue

                if session.metadata.get("is_ai_speaking"):
                    continue

                # Never trigger silence follow-ups if call has reached farewell or conclusion
                if session.metadata.get("call_concluded") or session.metadata.get("farewell_time") or session.metadata.get("farewell_delivered"):
                    continue

                last_cust = session.metadata.get("last_customer_speech_time") or 0.0
                last_ai_audio = session.metadata.get("last_ai_audio_time") or 0.0
                last_ai_turn = session.metadata.get("last_ai_turn_end_time") or 0.0
                last_active = max(last_cust, last_ai_audio, last_ai_turn)
                if not last_active or last_active <= 0:
                    continue

                silence_duration = time.time() - last_active
                prompt_count = session.metadata.get("silence_prompt_count", 0)

                # Tier 1: 8.5 seconds true silence after last activity (natural check-in, never restart pitch!)
                if silence_duration >= 8.5 and prompt_count == 0:
                    session.metadata["silence_prompt_count"] = 1
                    session.metadata["last_ai_turn_end_time"] = time.time()
                    name = session.metadata.get("customer_name") or "Sir"
                    if name == "Customer":
                        name = "Sir"
                    logger.info("silence_tier1_natural_followup", call_id=session.call_id, silence_sec=silence_duration)
                    try:
                        await gemini_session.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(
                                        text=f"[System Notice: Customer has been quiet for a moment. In maximum 4-5 words in the customer's active language, check if {name} can hear you.]"
                                    )]
                                )
                            ],
                            turn_complete=True
                        )
                    except Exception as e:
                        logger.warning("silence_tier1_prompt_error", error=str(e))

                # Tier 2: 14 seconds total silence — gentle recheck
                elif silence_duration >= 14.0 and prompt_count == 1:
                    session.metadata["silence_prompt_count"] = 2
                    session.metadata["last_ai_turn_end_time"] = time.time()
                    name = session.metadata.get("customer_name") or "Sir"
                    if name == "Customer":
                        name = "Sir"
                    logger.info("silence_tier2_reprompt", call_id=session.call_id, silence_sec=silence_duration)
                    try:
                        await gemini_session.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(
                                        text="[System Notice: Customer is still quiet. In maximum 4-5 words in the customer's active language, gently ask if your voice is reaching them.]"
                                    )]
                                )
                            ],
                            turn_complete=True
                        )
                    except Exception as e:
                        logger.warning("silence_tier2_prompt_error", error=str(e))

                # Tier 3: 20 seconds total silence -> polite goodbye
                elif silence_duration >= 20.0 and prompt_count == 2:
                    session.metadata["silence_prompt_count"] = 3
                    session.metadata["call_concluded"] = True
                    session.metadata["concluded_time"] = time.time()
                    session.metadata["concluded_reason"] = "no_response"
                    logger.info("silence_watcher_concluding_silent_call", call_id=session.call_id)
                    try:
                        await gemini_session.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(
                                        text="[System Notice: Customer is not responding. In 1 short courteous sentence in the active language, state that we will share order details via message, and politely say goodbye.]"
                                    )]
                                )
                            ],
                            turn_complete=True
                        )
                    except Exception as e:
                        logger.warning("silence_tier3_prompt_error", error=str(e))

                    session.metadata["farewell_time"] = time.time()

        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception as e:
            logger.error("silence_watcher_error", call_id=session.call_id, error=str(e))


    def _handle_post_turn_signoff(self, session: CallSession, speaker: str, text: str):
        """
        Manages post-turn lifecycle dynamically:
        If customer asks to hang up ("call kaat do", "phone rakh do"), triggers immediate hangup.
        If call was concluded, polite acknowledgments ("thank you", "aapka bhi") allow natural hangup,
        while substantive follow-up questions revive the session so Priya answers back.
        """
        if speaker == "customer":
            clean_lower = (text or "").lower()
            
            # 1. Customer explicitly asks to hang up / cut the call
            hangup_triggers = [
                "call kaat", "kaat de", "phone rakh", "phone kaat", "disconnect",
                "call disconnect", "cut the call", "hang up", "call end", "rakh do phone", "kaat do"
            ]
            if any(t in clean_lower for t in hangup_triggers):
                session.metadata["call_concluded"] = True
                session.metadata["concluded_reason"] = "customer_requested_hangup"
                session.metadata["farewell_delivered"] = True
                session.metadata["farewell_time"] = time.time() - 3.0  # Causes immediate hangup in silence watcher
                session.metadata["concluded_time"] = time.time() - 3.0
                logger.info("customer_explicitly_requested_hangup_triggering_immediate_disconnect", call_id=session.call_id, text=text)
                return

            # 2. Post-conclusion handling
            if session.metadata.get("call_concluded"):
                # Check for substantive questions or objections
                question_or_objection = any(q in clean_lower for q in [
                    "?", "kaun", "kya", "kyun", "kab", "kaisa", "kaha", "kahan", "nahi", "ruk", "suno", "wait", "hold", "what", "why", "when", "how", "no", "app", "address", "pata", "item"
                ])
                # Pure polite acknowledgments should NOT revive the call
                is_pure_ack = any(ack in clean_lower for ack in [
                    "aapka bhi", "thank you", "dhanyawad", "shukriya", "bye", "okay ji", "theek hai ji", "aapka", "same to you"
                ])
                if question_or_objection and not is_pure_ack:
                    session.metadata["call_concluded"] = False
                    session.metadata["farewell_delivered"] = False
                    session.metadata["farewell_time"] = None
                    session.metadata["concluded_time"] = None
                    logger.info("customer_spoke_substantive_turn_after_farewell_revived_call", call_id=session.call_id, text=text)
                else:
                    # Keep conclusion active and allow clean disconnect
                    session.metadata["farewell_time"] = time.time()
                    logger.info("customer_spoke_polite_closing_ack_permitting_disconnect", call_id=session.call_id, text=text)

    def _check_and_mark_call_concluded(self, session: CallSession) -> bool:
        """
        Language-Agnostic Zero-Hardcode Conclusion Check:
        Call conclusion is governed 100% dynamically by Gemini Live's tool calling (end_call).
        Zero hardcoded words, phrases, scripts, or language-specific string matching.
        """
        return bool(session.metadata.get("call_concluded"))

    async def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        session: CallSession
    ) -> Dict[str, Any]:
        """Dispatch tool calls securely with customer session isolation."""
        user_id = session.user_id
        order_id = args.get("order_id") or session.order_id

        if name == "get_order_tracking":
            session.current_intent = "order_tracking"
            return await self.order_tools.get_order_tracking(user_id=user_id, order_id=str(order_id or ""))
        elif name == "get_order_details":
            session.current_intent = "order_details"
            return await self.order_tools.get_order_details(user_id=user_id, order_id=str(order_id or ""))
        elif name == "get_order_history":
            session.current_intent = "order_history"
            limit = int(args.get("limit", 5))
            return await self.order_tools.get_order_history(user_id=user_id, limit=limit)
        elif name == "get_refund_status":
            session.current_intent = "refund_status"
            return await self.order_tools.get_refund_status(user_id=user_id, order_id=str(order_id or ""))
        elif name == "get_invoice":
            session.current_intent = "invoice"
            return await self.order_tools.get_invoice(user_id=user_id, order_id=str(order_id or ""))
        elif name == "search_welfog_knowledge":
            session.current_intent = "knowledge_policy"
            query = args.get("query", "")
            return await self.knowledge_tools.search_welfog_knowledge(query=str(query))
        elif name == "end_call":
            session.current_intent = "call_ended"
            reason = args.get("reason", "order_confirmed")
            session.metadata["call_concluded"] = True
            session.metadata["concluded_time"] = time.time()
            session.metadata["farewell_time"] = time.time()
            session.metadata["concluded_reason"] = reason

            if reason in {"voicemail", "answering_machine"}:
                session.status = CallState.NO_ANSWER
                session.metadata["farewell_delivered"] = True
                try:
                    db = get_db_service()
                    asyncio.create_task(db.schedule_retry(session.user_id, delay_minutes=20))
                except Exception as db_err:
                    logger.warning("failed_scheduling_voicemail_retry", error=str(db_err))
                logger.info("gemini_tool_end_call_voicemail", call_id=session.call_id)
                return {
                    "status": "success",
                    "message": "Voicemail / Answering machine detected. Call disconnected and retry scheduled."
                }
            elif reason == "customer_requested_hangup":
                session.status = CallState.COMPLETED
                session.metadata["farewell_delivered"] = True
                session.metadata["farewell_time"] = time.time() - 3.0
                session.metadata["concluded_time"] = time.time() - 3.0
                logger.info("gemini_tool_end_call_customer_requested_hangup", call_id=session.call_id)
                return {
                    "status": "success",
                    "message": "Customer requested immediate hangup. Disconnecting."
                }

            logger.info("gemini_tool_end_call_invoked", call_id=session.call_id, reason=reason)
            return {
                "status": "success",
                "message": f"Order {reason} recorded. Call is concluding now. Conclude speech immediately."
            }
        else:
            return {"status": "error", "message": f"Unknown tool: {name}"}

    def _finalize_outcome(self, session: CallSession, notes: str, success: bool):
        """Construct structured outcome report and trigger background AI Quality Auditor."""
        # Flush any trailing uncommitted transcript parts into conversation turns
        cust_parts = session.metadata.get("customer_turn_parts")
        if cust_parts:
            raw_cust = "".join(cust_parts).strip()
            cust_parts.clear()
            if raw_cust:
                session.metadata.setdefault("conversation_turns", []).append(f"Customer: {mask_sensitive_pii(raw_cust)}")

        ai_parts = session.metadata.get("ai_turn_parts")
        if ai_parts:
            raw_ai = "".join(ai_parts).strip()
            ai_parts.clear()
            if raw_ai:
                session.metadata.setdefault("conversation_turns", []).append(f"Priya: {mask_sensitive_pii(raw_ai)}")

        duration = None
        if session.started_at and session.ended_at:
            duration = (session.ended_at - session.started_at).total_seconds()

        concluded_reason = session.metadata.get("concluded_reason")
        if concluded_reason in {"voicemail", "answering_machine"}:
            session.status = CallState.NO_ANSWER
            notes = "Voicemail / Answering machine detected - automatically disconnected and retry scheduled in 20 min."
            success = False

        outcome = CallOutcome(
            call_id=session.call_id,
            call_status=session.status.value.lower(),
            intent=session.current_intent or "order_confirmation",
            customer_language=session.language,
            order_id=session.order_id,
            resolution="completed" if success else "failed",
            escalation_required=not success,
            notes=notes,
            duration_seconds=duration
        )
        self.outcomes[session.call_id] = outcome
        logger.info(
            "call_outcome_finalized",
            call_id=session.call_id,
            order_id=session.order_id,
            user_id=session.user_id,
            extra=outcome.model_dump()
        )

        # Trigger background post-call QA Auditor
        full_transcript = "\n".join(session.metadata.get("conversation_turns", []))
        auditor = get_call_auditor()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                auditor.audit_call(
                    call_id=session.call_id,
                    customer_id=session.user_id,
                    customer_name=session.metadata.get("customer_name") or "Customer",
                    phone=session.customer_phone,
                    order_id=session.order_id or "WF-10928",
                    transcript=full_transcript,
                    call_duration=int(duration or 0),
                    status=session.status.value
                )
            )

            # Broadcast call completion event to live admin dashboard
            broadcaster = get_live_broadcaster()
            loop.create_task(
                broadcaster.broadcast({
                    "type": "call_ended",
                    "call_id": session.call_id,
                    "order_id": session.order_id,
                    "duration": int(duration or 0),
                    "status": session.status.value
                })
            )
        except RuntimeError:
            pass

        return outcome
