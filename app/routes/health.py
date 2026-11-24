"""Health and integration diagnostics endpoints."""

from fastapi import APIRouter
from typing import Any, Dict

from app.config import get_settings
from app.services.telecmi_service import TeleCMIService
from app.services.welfog_service import WelfogService
from app.services.qdrant_service import QdrantService
from app.services.gemini_service import GeminiService

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Primary liveness check."""
    return {"status": "ok"}


@router.get("/health/integrations")
async def integrations_diagnostic() -> Dict[str, Any]:
    """
    Non-destructive integration connectivity check.
    Does NOT leak any secrets or sensitive keys.
    """
    settings = get_settings()
    results: Dict[str, Any] = {
        "status": "ok",
        "integrations": {}
    }

    # 1. TeleCMI Connectivity
    telecmi = TeleCMIService()
    try:
        token = await telecmi.get_app_token()
        results["integrations"]["telecmi"] = {
            "status": "connected" if token else "failed",
            "app_id": settings.telecmi_app_id,
            "sbc": settings.telecmi_sbc
        }
    except Exception as e:
        results["integrations"]["telecmi"] = {"status": "error", "message": str(e)}
    finally:
        await telecmi.close()

    # 2. Welfog API Connectivity
    welfog = WelfogService()
    try:
        welfog_res = await welfog.get_order_history(user_id=1, limit=1)
        results["integrations"]["welfog_api"] = {
            "status": "connected" if welfog_res.get("success") else "unreachable",
            "base_url": settings.welfog_api_base
        }
    except Exception as e:
        results["integrations"]["welfog_api"] = {"status": "error", "message": str(e)}
    finally:
        await welfog.close()

    # 3. Welfog Knowledge Base (aidata.welfog.com / Qdrant)
    qdrant = QdrantService()
    try:
        if settings.welfog_knowledge_api_key or settings.qdrant_url:
            q_res = await qdrant.search_knowledge("test")
            results["integrations"]["knowledge_base"] = {
                "status": "connected" if q_res.get("success") else "error",
                "provider": q_res.get("provider", "unknown"),
                "collection": settings.qdrant_collection
            }
        else:
            results["integrations"]["knowledge_base"] = {
                "status": "unconfigured",
                "message": "Awaiting WELFOG_KNOWLEDGE_API_KEY or QDRANT_URL. Production RAG will activate when key is provided.",
                "existing_admin_ui": "https://aidata.welfog.com/admin"
            }
    except Exception as e:
        results["integrations"]["knowledge_base"] = {"status": "error", "message": str(e)}
    finally:
        await qdrant.close()

    # 4. Gemini Configuration & Quick Connectivity Check
    gemini = GeminiService()
    gemini_status = "configured" if gemini.is_configured() else "unconfigured"
    gemini_msg = ""
    if gemini.is_configured():
        try:
            client = gemini.get_client()
            # Fast ping: list models (validates API key instantly without tenacity retry lag)
            next(client.models.list(config={"page_size": 1}))
            gemini_status = "connected"
        except Exception as ge:
            err_str = str(ge)
            if "prepayment credits are depleted" in err_str or "RESOURCE_EXHAUSTED" in err_str or "402" in err_str:
                gemini_status = "credits_depleted"
                gemini_msg = "Google AI Studio prepayment credits are depleted. Recharge at https://ai.studio/projects"
            elif "401" in err_str or "API_KEY_INVALID" in err_str:
                gemini_status = "invalid_key"
                gemini_msg = "Invalid Gemini API key. Check GEMINI_API_KEY in .env"
            else:
                gemini_status = "error"
                gemini_msg = err_str[:120]

    results["integrations"]["gemini"] = {
        "status": gemini_status,
        "live_model": settings.gemini_live_model,
        "voice": settings.gemini_voice_name,
        **({"message": gemini_msg} if gemini_msg else {})
    }

    return results
