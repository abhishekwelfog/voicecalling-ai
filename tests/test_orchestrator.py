"""Tests for Voice Orchestrator lifecycle and tool dispatching."""

import pytest
from app.models.call_session import CallState
from app.services.voice_orchestrator import VoiceOrchestrator
from app.services.gemini_service import GeminiService, get_gemini_tools
from google.genai import types


def test_session_creation_and_state_transitions():
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=1, customer_phone="9876543210", order_id="ORD123")

    assert session.customer_phone == "919876543210"
    assert session.status == CallState.CREATED
    assert session.user_id == 1
    assert session.order_id == "ORD123"

    session.transition_to(CallState.CALLING)
    assert session.status == CallState.CALLING

    session.transition_to(CallState.ANSWERED)
    assert session.status == CallState.ANSWERED

    session.transition_to(CallState.COMPLETED)
    assert session.status == CallState.COMPLETED
    assert session.ended_at is not None

    outcome = orch._finalize_outcome(session, "test completion", True)
    assert outcome.call_id == session.call_id
    assert outcome.call_status == "completed"
    assert outcome.resolution == "completed"
    assert outcome.escalation_required is False


@pytest.mark.asyncio
async def test_end_call_tool_execution():
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=1, customer_phone="9876543210", order_id="WF-10928")
    
    # Execute end_call tool
    result = await orch._execute_tool("end_call", {"reason": "order_confirmed"}, session)
    assert result["status"] == "success"
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("concluded_reason") == "order_confirmed"
    assert session.metadata.get("concluded_time") is not None


def test_gemini_tools_and_live_config_latency():
    # Verify end_call is in declared tools
    tools = get_gemini_tools()
    tool_names = [fn.name for t in tools for fn in t.function_declarations]
    assert "end_call" in tool_names
    assert "get_order_details" in tool_names
    assert "get_order_tracking" in tool_names

    # Verify live config has low-latency VAD settings
    service = GeminiService(api_key="dummy_key_for_test")
    config = service.build_live_config(
        customer_name="Abhishek",
        order_id="WF-10928"
    )
    assert config.realtime_input_config is not None
    vad = config.realtime_input_config.automatic_activity_detection
    assert vad.end_of_speech_sensitivity in (types.EndSensitivity.END_SENSITIVITY_HIGH, types.EndSensitivity.END_SENSITIVITY_LOW)
    assert vad.silence_duration_ms in (200, 250, 300, 350, 400, 450, 500)


@pytest.mark.asyncio
async def test_natural_hangup_signoff_and_safety():
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=1, customer_phone="9876543210", order_id="WF-10928")

    # Case 1: In the middle of conversation (Gemini hasn't called end_call) -> Must NOT conclude
    assert orch._check_and_mark_call_concluded(session) is False
    assert not session.metadata.get("call_concluded")

    # Case 2: Gemini invokes end_call tool -> Concludes dynamically without hardcoded keywords
    await orch._execute_tool("end_call", {"reason": "order_confirmed"}, session)
    assert orch._check_and_mark_call_concluded(session) is True
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("farewell_time") is not None

    # Case 3: Customer acknowledges after conclusion -> Keeps concluded, refreshes farewell timer
    orch._handle_post_turn_signoff(session, speaker="customer", text="Thank you ji")
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("farewell_time") is not None

    # Case 4: Customer asks question after conclusion -> Revives call so Priya answers and doesn't hang up abruptly
    orch._handle_post_turn_signoff(session, speaker="customer", text="Kaun sa app?")
    assert session.metadata.get("call_concluded") is False
    assert session.metadata.get("farewell_time") is None

    # Case 5: Customer says 'Aapka bhi' polite acknowledgment -> Stays concluded
    session.metadata["call_concluded"] = True
    session.metadata["farewell_time"] = 1000.0
    orch._handle_post_turn_signoff(session, speaker="customer", text="Aapka bhi.")
    assert session.metadata.get("call_concluded") is True

    # Case 6: Customer explicitly asks to hang up -> Concludes immediately
    session.metadata["call_concluded"] = False
    orch._handle_post_turn_signoff(session, speaker="customer", text="Call kaat de na ab")
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("concluded_reason") == "customer_requested_hangup"

@pytest.mark.asyncio
async def test_pickup_greeting_does_not_conclude_prematurely():
    """Verify telephone pickup greetings without Gemini end_call invocation never conclude."""
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=1, customer_phone="9876543210", order_id="WF-10929")

    # In turn 1 pickup, Gemini does not invoke end_call
    assert orch._check_and_mark_call_concluded(session) is False
    assert not session.metadata.get("call_concluded")


@pytest.mark.asyncio
async def test_call_10_dialogue_concludes_cleanly():
    """Verify Gemini calling end_call tool concludes call cleanly in any language."""
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=2, customer_phone="9521399708", order_id="WF-10929")

    await orch._execute_tool("end_call", {"reason": "order_confirmed"}, session)
    assert orch._check_and_mark_call_concluded(session) is True
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("concluded_reason") == "order_confirmed"
    assert session.metadata.get("farewell_time") is not None



@pytest.mark.asyncio
async def test_voicemail_and_answering_machine_orchestrator_handling():
    """Verify voicemail / answering machine detection automatically marks NO_ANSWER and schedules retry."""
    orch = VoiceOrchestrator()
    session = orch.create_session(user_id=1, customer_phone="9876543210", order_id="WF-10930")

    result = await orch._execute_tool("end_call", {"reason": "voicemail"}, session)
    assert result["status"] == "success"
    assert "Voicemail" in result["message"]
    assert session.status == CallState.NO_ANSWER
    assert session.metadata.get("call_concluded") is True
    assert session.metadata.get("concluded_reason") == "voicemail"
    assert session.metadata.get("farewell_delivered") is True

    outcome = orch._finalize_outcome(session, notes="Initial test notes", success=True)
    assert outcome.call_status == "no_answer"
    assert outcome.resolution == "failed"
    assert "Voicemail" in outcome.notes
