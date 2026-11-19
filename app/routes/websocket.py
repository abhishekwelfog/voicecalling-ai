"""WebSocket endpoints for TeleCMI bidirectional audio streaming and Admin Live Monitoring."""

import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.routes.calls import get_orchestrator
from app.services.live_broadcaster import get_live_broadcaster
from app.utils.logging import logger

router = APIRouter(tags=["Streaming"])


@router.websocket("/ws/telecmi")
async def telecmi_audio_stream(
    websocket: WebSocket,
    call_id: Optional[str] = Query(None)
):
    """
    WebSocket streaming endpoint.
    TeleCMI streams live customer audio to this endpoint, and receives Gemini response audio.
    """
    logger.info("incoming_telecmi_websocket", call_id=call_id)
    orch = get_orchestrator()
    await orch.handle_telecmi_stream(websocket, call_id=call_id)


from app.services.auth_service import verify_session_token, SESSION_COOKIE_NAME


@router.websocket("/ws/admin/live-monitor")
async def admin_live_monitor(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for admin dashboard:
    Receives real-time waveform energy levels, word-by-word streaming transcript,
    tool execution notifications, and live post-call QA metrics.
    Requires authenticated admin session.
    """
    auth_token = token or websocket.cookies.get(SESSION_COOKIE_NAME)
    if not auth_token or not verify_session_token(auth_token):
        await websocket.close(code=1008, reason="Authentication required")
        return

    broadcaster = get_live_broadcaster()
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection open; receive ping or commands from admin
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except (WebSocketDisconnect, Exception):
        broadcaster.disconnect(websocket)
