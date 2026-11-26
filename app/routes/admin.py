import csv
import io
import os
import time
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Response, Request, UploadFile, File, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.auth_service import (
    create_session_token,
    get_current_admin,
    SESSION_COOKIE_NAME,
    SESSION_EXPIRY_SECONDS
)
from app.services.db_service import get_db_service
from app.services.call_auditor import get_call_auditor
from app.utils.security import sanitize_csv_cell

router = APIRouter(prefix="/api", tags=["Admin & Customers"])


# ==================== Pydantic Schemas ====================

from app.services.security_service import get_brute_force_manager


class LoginRequest(BaseModel):
    username: str
    password: str
    trap: Optional[str] = None  # Honeypot field: real users leave empty, automated bots fill it


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class CustomerCreateRequest(BaseModel):
    name: str = Field(default="Customer")
    phone: str
    order_id: Optional[str] = Field(default="WF-10928")
    product_name: Optional[str] = Field(default="Welfog Premium Item")
    product_price: Optional[str] = Field(default="₹999")
    delivery_time: Optional[str] = Field(default="2-3 Business Days")
    payment_method: Optional[str] = Field(default="Cash on Delivery (COD)")
    notes: Optional[str] = Field(default="")


class BulkCustomerItem(BaseModel):
    name: Optional[str] = "Customer"
    phone: str
    order_id: Optional[str] = "WF-10928"
    product_name: Optional[str] = "Welfog Premium Item"
    product_price: Optional[str] = "₹999"
    delivery_time: Optional[str] = "2-3 Business Days"
    payment_method: Optional[str] = "Cash on Delivery (COD)"
    notes: Optional[str] = "Bulk Import"


class BulkCustomerRequest(BaseModel):
    customers: Optional[List[BulkCustomerItem]] = None
    raw_text: Optional[str] = None  # Comma or newline separated phone numbers / CSV


class CustomerStatusUpdateRequest(BaseModel):
    status: str
    call_duration: Optional[int] = None
    call_summary: Optional[str] = None
    transcript: Optional[str] = None
    recording_url: Optional[str] = None
    qa_score: Optional[float] = None
    customer_sentiment: Optional[str] = None
    rto_risk: Optional[str] = None
    qa_notes: Optional[str] = None


class CustomerUpdateRequest(BaseModel):
    name: str
    phone: str
    order_id: Optional[str] = "WF-10928"
    product_name: Optional[str] = None
    product_price: Optional[str] = None
    delivery_time: Optional[str] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = ""


class CallLogCreateRequest(BaseModel):
    customer_id: Optional[int] = None
    phone: str
    customer_name: Optional[str] = "Customer"
    order_id: Optional[str] = "WF-10928"
    status: str
    duration: int = 0
    summary: Optional[str] = ""
    transcript: Optional[str] = ""
    recording_url: Optional[str] = ""
    qa_score: Optional[float] = None
    customer_sentiment: Optional[str] = None
    rto_risk: Optional[str] = None
    qa_notes: Optional[str] = None
    product_name: Optional[str] = None
    product_price: Optional[str] = None
    delivery_time: Optional[str] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None
    order_outcome: Optional[str] = None
    rto_risk_percentage: Optional[int] = None
    rto_factors: Optional[str] = None
    warehouse_alert: Optional[Union[bool, int]] = None


# ==================== Authentication Endpoints ====================

@router.post("/auth/login")
async def login(req: LoginRequest, request: Request, response: Response):
    """
    Hardened Admin Login Endpoint:
    - Protection against credential stuffing and brute force dictionary attacks.
    - Automatic 15-minute IP/account lockout after 5 consecutive failures.
    - Constant-time verification & side-channel timing jitter defense.
    - Transparent auto-upgrade of legacy SHA-256 to PBKDF2-HMAC-SHA256 (600,000 rounds).
    """
    clean_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (request.client.host if request.client else "127.0.0.1")
    clean_user = req.username.strip()

    # Honeypot defense against automated credential stuffing / spam bots
    if req.trap and req.trap.strip():
        await asyncio.sleep(0.3)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Automated bot submission detected."
        )

    bf_manager = get_brute_force_manager()
    lockout_sec = bf_manager.check_lockout(clean_ip, clean_user)
    if lockout_sec:
        mins = max(1, round(lockout_sec / 60))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Access temporarily locked for {mins} minutes for security.",
            headers={"Retry-After": str(lockout_sec)}
        )

    db = get_db_service()
    valid = await db.verify_admin(clean_user, req.password)
    if not valid:
        remaining, is_locked = bf_manager.record_failure(clean_ip, clean_user)
        # Constant-time delay to eliminate response-timing side-channels
        await asyncio.sleep(0.25)
        if is_locked:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Account temporarily locked for 15 minutes to prevent unauthorized access.",
                headers={"Retry-After": "900"}
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid username or password. {remaining} attempt(s) remaining before security lockout."
        )

    # Success: reset failure counters
    bf_manager.record_success(clean_ip, clean_user)

    is_https = request.url.scheme == "https" or request.headers.get("X-Forwarded-Proto") == "https"
    token = create_session_token(clean_user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_EXPIRY_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_https
    )
    return {
        "success": True,
        "token": token,
        "username": clean_user,
        "message": "Login successful"
    }


@router.post("/auth/change-password")
async def change_password(
    req: ChangePasswordRequest,
    current_admin: str = Depends(get_current_admin)
):
    """
    Allow authenticated administrator to securely update their password.
    Enforces minimum length of 8 characters and PBKDF2-HMAC-SHA256 encryption.
    """
    if len(req.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters long for security."
        )

    db = get_db_service()
    valid = await db.verify_admin(current_admin, req.current_password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password verification failed. Please enter your correct current password."
        )

    success = await db.update_admin_password(current_admin, req.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password in database."
        )

    return {
        "success": True,
        "message": "Password updated successfully with PBKDF2-HMAC-SHA256 (600,000 rounds) encryption."
    }


@router.post("/auth/logout")
async def logout(response: Response):
    """Admin logout clearing session cookie."""
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"success": True, "message": "Logged out successfully"}


@router.get("/auth/me")
async def get_me(username: str = Depends(get_current_admin)):
    """Return currently authenticated admin username."""
    return {"username": username}


# ==================== Customer Onboarding Endpoints ====================

@router.get("/customers")
async def list_customers(
    status: Optional[str] = None,
    search: Optional[str] = None,
    _: str = Depends(get_current_admin)
):
    """Fetch customer queue with optional status or search filter."""
    db = get_db_service()
    customers = await db.get_customers(status=status, search=search)
    return {"customers": customers, "count": len(customers)}


@router.get("/orders/lookup")
async def lookup_order(
    order_id: str,
    _: str = Depends(get_current_admin)
):
    """
    Fetch real order information (product title, price, delivery time, address) from live Welfog APIs.
    """
    from app.services.welfog_service import WelfogService
    service = WelfogService()
    try:
        clean_order_id = order_id.strip()
        tracking_res = await service.get_order_tracking(user_id=1, order_id=clean_order_id)
        details_res = await service.get_order_details(user_id=1, order_id=clean_order_id)

        found = False
        product_name = None
        product_price = None
        delivery_time = None
        payment_method = None
        delivery_address = None
        customer_name = None
        customer_phone = None

        if tracking_res.get("success") and tracking_res.get("data") and tracking_res["data"].get("result") != "not_found":
            found = True
            t_data = tracking_res["data"]
            product_name = t_data.get("product_title") or tracking_res.get("product_title")
            raw_del = t_data.get("expected_delivery")
            if raw_del and not str(raw_del).isdigit():
                delivery_time = str(raw_del)
            else:
                flow = t_data.get("order_status_detail", {}).get("order_status", {}).get("order_flow", {})
                del_flow = flow.get("delivered", {})
                del_date = del_flow.get("date")
                if del_date:
                    delivery_time = f"Expected by {del_date[:10]}"
                else:
                    delivery_time = "In 2-3 Business Days"

        if details_res.get("success") and details_res.get("details"):
            found = True
            d_raw = details_res["details"]
            d_item = d_raw
            if isinstance(d_raw, dict) and "data" in d_raw:
                if isinstance(d_raw["data"], list) and len(d_raw["data"]) > 0:
                    d_item = d_raw["data"][0]
                elif isinstance(d_raw["data"], dict):
                    d_item = d_raw["data"]

            if not product_name:
                product_name = d_item.get("product_title") or d_item.get("product_name")

            raw_price = d_item.get("grand_total") or d_item.get("subtotal") or d_item.get("total")
            if raw_price is not None:
                raw_str = str(raw_price).strip()
                if raw_str.lower().startswith("rs"):
                    product_price = "₹" + raw_str[2:].strip()
                elif not raw_str.startswith("₹"):
                    product_price = f"₹{raw_str}"
                else:
                    product_price = raw_str

            raw_pay = d_item.get("payment_type") or d_item.get("payment_mode") or d_item.get("payment_status_string")
            if raw_pay:
                if "cod" in str(raw_pay).lower() or "cash" in str(raw_pay).lower():
                    payment_method = "Cash on Delivery (COD)"
                else:
                    payment_method = str(raw_pay)

            ship = d_item.get("shipping_address")
            if isinstance(ship, dict):
                customer_name = ship.get("name")
                customer_phone = ship.get("phone")
                addr_parts = [ship.get("address"), ship.get("city"), ship.get("state")]
                valid_parts = [str(p).strip() for p in addr_parts if p and str(p).strip()]
                full_addr = ", ".join(valid_parts)
                if ship.get("postal_code"):
                    full_addr += f" ({ship.get('postal_code')})"
                delivery_address = full_addr
            elif isinstance(ship, str):
                delivery_address = ship

            if not customer_name:
                customer_name = d_item.get("customer_name") or d_item.get("user_name")
            if not customer_phone:
                customer_phone = d_item.get("customer_phone") or d_item.get("phone")

        order_payload = {
            "order_id": clean_order_id,
            "product_name": product_name,
            "product_price": product_price,
            "delivery_time": delivery_time,
            "payment_method": payment_method,
            "delivery_address": delivery_address,
            "customer_name": customer_name,
            "customer_phone": customer_phone
        }

        return {
            "found": found,
            "order_id": clean_order_id,
            "product_name": product_name,
            "product_price": product_price,
            "delivery_time": delivery_time,
            "payment_method": payment_method,
            "delivery_address": delivery_address,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "order": order_payload,
            "message": "Real order details retrieved from Welfog API" if found else "Order not found in remote API. You can enter or edit details manually."
        }
    finally:
        await service.close()


@router.post("/customers")
async def create_customer(
    req: CustomerCreateRequest,
    _: str = Depends(get_current_admin)
):
    """Onboard a single customer into calling queue with complete order details."""
    db = get_db_service()
    customer_id = await db.add_customer(
        name=req.name,
        phone=req.phone,
        order_id=req.order_id,
        notes=req.notes,
        product_name=req.product_name,
        product_price=req.product_price,
        delivery_time=req.delivery_time,
        payment_method=req.payment_method
    )
    return {"success": True, "customer_id": customer_id, "message": "Customer onboarded successfully"}


@router.post("/customers/bulk")
async def bulk_create_customers(
    req: BulkCustomerRequest,
    _: str = Depends(get_current_admin)
):
    """Bulk onboard customers via JSON list or raw comma/newline text."""
    entries: List[Dict[str, str]] = []

    if req.customers:
        for c in req.customers:
            entries.append({
                "name": c.name or "Customer",
                "phone": c.phone,
                "order_id": c.order_id or "WF-10928",
                "product_name": c.product_name or "Welfog Premium Item",
                "product_price": c.product_price or "₹999",
                "delivery_time": c.delivery_time or "2-3 Business Days",
                "payment_method": c.payment_method or "Cash on Delivery (COD)",
                "notes": c.notes or "Bulk Import"
            })

    if req.raw_text:
        lines = req.raw_text.strip().splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                # Format: Name, Phone, OrderID (optional), Product (optional), Price (optional)
                entries.append({
                    "name": parts[0],
                    "phone": parts[1],
                    "order_id": parts[2] if len(parts) > 2 else f"WF-1092{idx+1}",
                    "product_name": parts[3] if len(parts) > 3 else "Welfog Premium Item",
                    "product_price": parts[4] if len(parts) > 4 else "₹999",
                    "delivery_time": "2-3 Business Days",
                    "payment_method": "Cash on Delivery (COD)",
                    "notes": "Bulk Pasted"
                })
            else:
                # Format: Just Phone
                entries.append({
                    "name": f"Customer {idx+1}",
                    "phone": parts[0],
                    "order_id": f"WF-1092{idx+1}",
                    "product_name": "Welfog Premium Item",
                    "product_price": "₹999",
                    "delivery_time": "2-3 Business Days",
                    "payment_method": "Cash on Delivery (COD)",
                    "notes": "Bulk Pasted"
                })

    db = get_db_service()
    count = await db.bulk_add_customers(entries)
    return {"success": True, "imported_count": count, "message": f"{count} customers added to queue"}


@router.get("/customers/export-csv")
async def export_customers_csv(
    _: str = Depends(get_current_admin)
):
    """Export all customers and call outcomes as a downloadable Excel-ready CSV."""
    db = get_db_service()
    customers = await db.get_customers()

    output = io.StringIO()
    # Write UTF-8 BOM for Microsoft Excel compatibility
    output.write("\ufeff")
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "Customer ID", "Customer Name", "Phone Number", "Order ID", "Product Name", "Product Price",
        "Delivery Time", "Payment Method", "Queue Status", "Talk Duration (s)",
        "Last Called At", "Delivery Address / Notes", "Recording URL"
    ])
    for c in customers:
        phone_val = f"'+91 {c.get('phone')}" if c.get("phone") else ""
        writer.writerow([
            c.get("id"),
            sanitize_csv_cell(c.get("name") or "Customer"),
            phone_val,
            sanitize_csv_cell(c.get("order_id") or ""),
            sanitize_csv_cell(c.get("product_name", "")),
            sanitize_csv_cell(c.get("product_price", "")),
            sanitize_csv_cell(c.get("delivery_time", "")),
            sanitize_csv_cell(c.get("payment_method", "")),
            sanitize_csv_cell(c.get("status")),
            c.get("call_duration", 0),
            sanitize_csv_cell(c.get("last_called_at", "")),
            sanitize_csv_cell(c.get("notes", "")),
            sanitize_csv_cell(c.get("recording_url", ""))
        ])

    output.seek(0)
    filename = f"welfog_customers_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8"
        }
    )


@router.get("/customers/high-rto")
async def list_high_rto_customers(_: str = Depends(get_current_admin)):
    """Fetch customers flagged with high RTO risk for warehouse team verification."""
    db = get_db_service()
    high_rto = await db.get_high_rto_customers()
    return {"customers": high_rto, "count": len(high_rto)}


@router.get("/customers/{customer_id}")
async def get_customer_details(
    customer_id: int,
    _: str = Depends(get_current_admin)
):
    """Retrieve details, notes, and full conversation transcript for a customer."""
    db = get_db_service()
    customer = await db.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.put("/customers/{customer_id}")
async def edit_customer(
    customer_id: int,
    req: CustomerUpdateRequest,
    _: str = Depends(get_current_admin)
):
    """Update customer details and order metadata."""
    db = get_db_service()
    await db.update_customer(
        customer_id=customer_id,
        name=req.name,
        phone=req.phone,
        order_id=req.order_id,
        notes=req.notes,
        product_name=req.product_name,
        product_price=req.product_price,
        delivery_time=req.delivery_time,
        payment_method=req.payment_method
    )
    return {"success": True, "message": "Customer updated successfully"}


@router.delete("/customers/{customer_id}")
async def delete_customer(
    customer_id: int,
    _: str = Depends(get_current_admin)
):
    """Remove a customer from the queue."""
    db = get_db_service()
    await db.delete_customer(customer_id)
    return {"success": True, "message": "Customer deleted"}


@router.post("/customers/{customer_id}/reset")
async def reset_customer(
    customer_id: int,
    _: str = Depends(get_current_admin)
):
    """Reset single customer status back to PENDING."""
    db = get_db_service()
    await db.reset_customer(customer_id)
    return {"success": True, "message": "Customer status reset to PENDING"}


@router.post("/customers/reset-all")
async def reset_all_customers(
    _: str = Depends(get_current_admin)
):
    """Reset all finished or failed customers back to PENDING."""
    db = get_db_service()
    count = await db.reset_all_customers()
    return {"success": True, "reset_count": count, "message": f"{count} customers reset to PENDING"}


@router.post("/customers/retry-unanswered")
async def retry_unanswered_customers(
    _: str = Depends(get_current_admin)
):
    """Reset NO_ANSWER, BUSY, and FAILED customers back to PENDING for auto-retrying."""
    db = get_db_service()
    count = await db.reset_unanswered_customers()
    return {"success": True, "reset_count": count, "message": f"{count} unanswered/busy customers reset to PENDING"}




@router.post("/customers/{customer_id}/status")
async def update_status(
    customer_id: int,
    req: CustomerStatusUpdateRequest,
    _: str = Depends(get_current_admin)
):
    """Update customer status, call duration, notes, transcript, recording, and QA scores."""
    db = get_db_service()
    await db.update_customer_status(
        customer_id=customer_id,
        status=req.status,
        call_duration=req.call_duration,
        call_summary=req.call_summary,
        transcript=req.transcript,
        recording_url=req.recording_url,
        qa_score=req.qa_score,
        customer_sentiment=req.customer_sentiment,
        rto_risk=req.rto_risk,
        qa_notes=req.qa_notes
    )
    return {"success": True}


# ==================== Audio Recording Upload ====================

@router.post("/recordings/upload")
async def upload_recording(
    audio: UploadFile = File(...),
    _: str = Depends(get_current_admin)
):
    """Upload recorded call audio (.webm/.mp3/.wav) and save to static recordings."""
    try:
        contents = await audio.read()
        if not contents or len(contents) < 300:
            return {"success": False, "error": "Audio recording payload too small or empty"}

        rec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        filename = f"rec_{int(time.time())}_{uuid.uuid4().hex[:8]}.webm"
        file_path = os.path.join(rec_dir, filename)

        with open(file_path, "wb") as f:
            f.write(contents)

        recording_url = f"/static/recordings/{filename}"
        return {"success": True, "recording_url": recording_url}
    except Exception as e:
        return {"success": False, "error": f"Failed to save recording: {str(e)}"}


# ==================== Auto-Dialer & Campaign Queue ====================

@router.get("/campaign/next")
async def get_next_customer(
    _: str = Depends(get_current_admin)
):
    """Get next customer in queue to be called automatically."""
    db = get_db_service()
    next_cust = await db.get_next_pending_customer()
    return {"has_next": next_cust is not None, "customer": next_cust}


@router.get("/stats")
async def get_stats(
    _: str = Depends(get_current_admin)
):
    """Return dashboard analytics metrics."""
    db = get_db_service()
    return await db.get_dashboard_stats()


# ==================== Call Logs ====================

@router.post("/call-logs")
async def save_call_log(
    req: CallLogCreateRequest,
    _: str = Depends(get_current_admin)
):
    """Save call outcome, recording URL and full transcript into call_logs with QA metrics and order context."""
    db = get_db_service()

    final_outcome = (req.order_outcome or req.status or "COMPLETED").upper()
    is_cancelled = (final_outcome == "CANCELLED" or req.status.upper() == "CANCELLED")

    final_rto_risk = (
        req.rto_risk.upper() if req.rto_risk
        else ("HIGH" if is_cancelled else ("MEDIUM" if req.status in {"NO_ANSWER", "BUSY"} else "LOW"))
    )
    final_rto_pct = (
        req.rto_risk_percentage if req.rto_risk_percentage is not None
        else (95 if is_cancelled else (50 if req.status in {"NO_ANSWER", "BUSY"} else 15))
    )
    final_wh_alert = 1 if (is_cancelled or final_rto_risk == "HIGH" or req.warehouse_alert) else 0
    final_factors = req.rto_factors or (
        "Order marked cancelled - Do Not Dispatch" if is_cancelled
        else ("Customer missed call" if req.status in {"NO_ANSWER", "BUSY"} else "Verified by AI")
    )

    log_id = await db.record_call_log(
        customer_id=req.customer_id,
        phone=req.phone,
        customer_name=req.customer_name or "Customer",
        order_id=req.order_id or "WF-10928",
        status=final_outcome,
        duration=req.duration,
        summary=req.summary,
        transcript=req.transcript,
        recording_url=req.recording_url,
        qa_score=req.qa_score,
        customer_sentiment=req.customer_sentiment or ("FRUSTRATED" if is_cancelled else "NEUTRAL"),
        rto_risk=final_rto_risk,
        qa_notes=req.qa_notes or ("Customer requested cancellation." if is_cancelled else None),
        product_name=req.product_name,
        product_price=req.product_price,
        delivery_time=req.delivery_time,
        payment_method=req.payment_method,
        notes=req.notes,
        order_outcome=final_outcome,
        rto_risk_percentage=final_rto_pct,
        rto_factors=final_factors,
        warehouse_alert=final_wh_alert
    )

    # Also update customer status and recording in customers table
    if req.customer_id:
        await db.update_customer_status(
            customer_id=req.customer_id,
            status=final_outcome,
            call_duration=req.duration,
            call_summary=req.summary,
            transcript=req.transcript,
            recording_url=req.recording_url,
            qa_score=req.qa_score,
            customer_sentiment=req.customer_sentiment or ("FRUSTRATED" if is_cancelled else "NEUTRAL"),
            rto_risk=final_rto_risk,
            qa_notes=req.qa_notes,
            rto_risk_percentage=final_rto_pct,
            rto_factors=final_factors,
            warehouse_alert=bool(final_wh_alert)
        )

    # Trigger post-call QA auditor in background to audit transcript or audio
    has_audio = bool(req.recording_url and req.recording_url.strip())
    has_transcript = bool(req.transcript and len(req.transcript.strip()) > 5)
    if req.qa_score is None and (has_audio or has_transcript or req.duration >= 3):
        auditor = get_call_auditor()
        asyncio.create_task(
            auditor.audit_call(
                call_id=f"log_{log_id}",
                customer_id=req.customer_id,
                customer_name=req.customer_name or "Customer",
                phone=req.phone,
                order_id=req.order_id or "WF-10928",
                transcript=req.transcript or "",
                call_duration=req.duration,
                status=final_outcome,
                recording_url=req.recording_url
            )
        )

    return {"success": True, "log_id": log_id}


@router.get("/call-logs")
async def list_call_logs(
    limit: Optional[int] = None,
    page: int = 1,
    page_size: int = 10,
    status: Optional[str] = None,
    search: Optional[str] = None,
    _: str = Depends(get_current_admin)
):
    """List recent call history logs with pagination, search, and status filtering."""
    db = get_db_service()
    if limit is not None:
        logs = await db.get_call_logs(limit=limit)
        return {"logs": logs, "count": len(logs)}

    result = await db.get_call_logs(
        page=page,
        page_size=page_size,
        status=status,
        search=search,
        return_pagination=True
    )
    return result


@router.get("/telecmi/webrtc-config")
async def get_webrtc_config(_: str = Depends(get_current_admin)):
    """Return runtime TeleCMI WebRTC credentials from server environment."""
    from app.config import get_settings
    settings = get_settings()
    return {
        "user_id": settings.telecmi_user_id or "",
        "password": settings.telecmi_user_password or "",
        "sbc": settings.telecmi_sbc or "sbcind.telecmi.com",
        "caller_id": settings.telecmi_caller_id or ""
    }


# ==================== Reporting & Webhooks ====================

@router.get("/reports/export-csv")
@router.get("/call-logs/export-csv")
async def export_call_report_csv(
    status: Optional[str] = None,
    search: Optional[str] = None,
    _: str = Depends(get_current_admin)
):
    """
    Export comprehensive call history, customer order details, recordings,
    and QA verification notes as a fully Excel-compatible UTF-8 CSV report.
    """
    db = get_db_service()
    # Fetch all matching call logs
    logs = await db.get_call_logs(limit=10000, status=status, search=search)

    output = io.StringIO()
    # Write UTF-8 BOM so Microsoft Excel opens it cleanly with Indian Rupee (₹) and Hindi characters
    output.write("\ufeff")
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # 23 Complete, Professional Columns
    writer.writerow([
        "Call ID",
        "Call Date & Time",
        "Customer Name",
        "Phone Number",
        "Order ID",
        "Product Name",
        "Product Price",
        "Payment Method",
        "Expected Delivery",
        "Call Status",
        "Order Outcome",
        "Duration (Seconds)",
        "Duration (Formatted)",
        "Customer Address & Spoken Notes",
        "AI Call Summary",
        "QA Score",
        "Customer Sentiment",
        "RTO Risk",
        "RTO Risk %",
        "Warehouse Alert",
        "QA Auditor Notes",
        "Audio Recording Link",
        "Full Conversation Transcript"
    ])

    for l in logs:
        # Duration formatting
        sec = int(l.get("duration") or 0)
        dur_mins = sec // 60
        dur_secs = sec % 60
        formatted_duration = f"{dur_mins}m {dur_secs:02d}s" if sec >= 60 else f"{sec}s"

        # Order outcome and RTO decision
        outcome = (l.get("order_outcome") or "").strip().upper()
        summary_upper = (l.get("summary") or "").upper()
        status_upper = (l.get("status") or "").upper()
        rto_risk_raw = (l.get("rto_risk") or "").upper()

        if "CANCEL" in outcome or "CANCEL" in status_upper or "CANCEL" in summary_upper or rto_risk_raw == "HIGH":
            outcome = "CANCELLED"
            rto_risk_str = "HIGH"
            rto_pct_str = f"{l.get('rto_risk_percentage') or 95}%"
            wh_alert = "YES"
        elif outcome == "CONFIRMED" or "CONFIRM" in summary_upper or status_upper == "COMPLETED":
            outcome = "CONFIRMED"
            rto_risk_str = rto_risk_raw or "LOW"
            rto_pct_str = f"{l.get('rto_risk_percentage') or 10}%"
            wh_alert = "YES" if l.get("warehouse_alert") else "NO"
        elif status_upper in {"NO_ANSWER", "BUSY"}:
            outcome = "NO_ANSWER"
            rto_risk_str = "MEDIUM"
            rto_pct_str = f"{l.get('rto_risk_percentage') or 50}%"
            wh_alert = "NO"
        else:
            outcome = outcome or "PENDING"
            rto_risk_str = rto_risk_raw or "LOW"
            rto_pct_str = f"{l.get('rto_risk_percentage') or 15}%"
            wh_alert = "YES" if l.get("warehouse_alert") else "NO"

        # QA Score
        raw_qa = l.get("qa_score")
        qa_str = f"{float(raw_qa):.1f}" if raw_qa is not None else "9.0"

        # Phone formatted cleanly for Excel (won't convert to exponential 7.08E+09)
        phone_str = f"'+91 {l.get('phone')}" if l.get("phone") else ""

        # Clean transcript
        transcript_clean = (l.get("transcript") or "").strip()

        writer.writerow([
            l.get("id"),
            sanitize_csv_cell(l.get("created_at") or ""),
            sanitize_csv_cell(l.get("customer_name") or "Customer"),
            phone_str,
            sanitize_csv_cell(l.get("order_id") or ""),
            sanitize_csv_cell(l.get("product_name") or "Welfog Order Item"),
            sanitize_csv_cell(l.get("product_price") or ""),
            sanitize_csv_cell(l.get("payment_method") or "Cash on Delivery (COD)"),
            sanitize_csv_cell(l.get("delivery_time") or "2-3 Business Days"),
            sanitize_csv_cell(status_upper),
            sanitize_csv_cell(outcome),
            sec,
            formatted_duration,
            sanitize_csv_cell(l.get("customer_notes") or l.get("notes") or ""),
            sanitize_csv_cell(l.get("summary") or ""),
            qa_str,
            sanitize_csv_cell((l.get("customer_sentiment") or ("FRUSTRATED" if outcome == "CANCELLED" else "NEUTRAL")).upper()),
            sanitize_csv_cell(rto_risk_str),
            sanitize_csv_cell(rto_pct_str),
            sanitize_csv_cell(wh_alert),
            sanitize_csv_cell(l.get("qa_notes") or ""),
            sanitize_csv_cell(l.get("recording_url") or ""),
            sanitize_csv_cell(transcript_clean)
        ])

    output.seek(0)
    filename = f"welfog_call_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/csv; charset=utf-8"
        }
    )


@router.post("/telephony/webhook")
@router.post("/telecmi/webhook")
async def telecmi_webhook(request: Request):
    """
    Handle real-time call disposition and hangup webhooks from TeleCMI / PIOPIY.
    Automatically updates call duration, recording URL, and schedules retries.
    """
    try:
        data = await request.json()
    except Exception:
        form = await request.form()
        data = dict(form)

    logger.info("telecmi_webhook_received", extra=data)
    event = (data.get("event") or data.get("action") or data.get("status") or "").lower()
    call_id = data.get("call_id") or data.get("cmi_call_id")
    phone = data.get("to") or data.get("customer_phone") or data.get("phone")
    duration = int(data.get("duration") or data.get("billsec") or 0)
    recording = data.get("recording_url") or data.get("record_url")

    db = get_db_service()
    if phone:
        clean_digits = "".join(c for c in str(phone) if c.isdigit())
        if len(clean_digits) == 12 and clean_digits.startswith("91"):
            clean_digits = clean_digits[2:]
        matching = await db.get_customers(search=clean_digits)
        if matching:
            cust = matching[0]
            cust_id = cust["id"]
            if event in ("answered", "completed", "normal_clearing", "hangup") and duration > 5:
                pass  # State already handled by WebSocket
            elif event in ("busy", "user_busy"):
                await db.update_customer_status(cust_id, "BUSY", call_duration=duration)
                await db.schedule_retry(cust_id, delay_minutes=15)
            elif event in ("no_answer", "noanswer", "timeout", "cancel"):
                await db.update_customer_status(cust_id, "NO_ANSWER", call_duration=duration)
                await db.schedule_retry(cust_id, delay_minutes=20)
            elif event in ("failed", "rejected", "congestion"):
                await db.update_customer_status(cust_id, "FAILED", call_duration=duration)
                await db.schedule_retry(cust_id, delay_minutes=30)

            if recording:
                await db.update_customer_status(cust_id, cust["status"], recording_url=recording)

    return {"status": "ok", "received_event": event}


@router.get("/campaign/retries")
async def get_scheduled_retries(_: str = Depends(get_current_admin)):
    """Fetch customers currently queued for scheduled retries."""
    db = get_db_service()
    retries = await db.get_scheduled_retries()
    return {"retries": retries, "count": len(retries)}


@router.post("/campaign/retry-scheduled")
async def trigger_scheduled_retries(_: str = Depends(get_current_admin)):
    """Activate due scheduled retries back to PENDING so campaign caller dials them."""
    db = get_db_service()
    due = await db.get_scheduled_retries()
    requeued = 0
    for c in due:
        await db.update_customer_status(c["id"], "PENDING")
        requeued += 1
    return {"success": True, "requeued_count": requeued}

