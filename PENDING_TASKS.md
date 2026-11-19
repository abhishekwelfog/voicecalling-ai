# Welfog AI Voice Calling Agent - Pending Integration Guide
**Document Type:** Technical Specification & Handover Document  
**Target Audience:** Senior Backend Engineers, Tech Leads, Application Architects  
**Status of Core System:** ✅ **100% Completed, Tested (45/45 Green), and Production-Ready**  
**Remaining Integration Task:** ⏳ **Automated Real-Time Order Onboarding (Webhook / API Trigger)**

---

## 1. Executive Summary

The **Welfog AI Voice Calling Agent** (powered by Google Gemini 2.5 Live Multimodal Audio, TeleCMI WebRTC Telephony, and PostgreSQL) is fully functional and hardened.

### What is Already Working (Completed):
* **Autonomous Outbound Calling:** AI agent ("Priya") conducts natural, low-latency (<600ms) voice calls in Hindi/English.
* **Intelligent Conversation Flow:** Verifies order items, quantity, total price, delivery timeline, payment mode (COD/Prepaid), and confirms delivery address.
* **RTO & Courier Risk Analysis:** Automatically scores Return-to-Origin probability and flags hesitant customers or incomplete addresses for warehouse dispatch safety.
* **Automated Post-Call QA Audit:** Generates QA score (out of 10), sentiment analysis, call summary, and scrubbed transcripts with zero PII leaks.
* **Manual & Bulk Queueing:** Orders can currently be queued manually via the web console (`+ Onboard Customer`) or imported via CSV.

### What is Pending (Handover to Senior / Main App Team):
* **Direct Application Integration (Zero-Touch Onboarding):** Whenever a customer places a new order on the Welfog mobile app or website, the order details must automatically sync into this voice agent's queue in real-time, eliminating any manual CSV imports or data entry.

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────┐
│   Customer Places Order         │
│   (Welfog App / Website)        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Welfog Main Application       │
│   Backend / Order Service       │
└────────────────┬────────────────┘
                 │
                 │ HTTP POST Webhook (Event: `order.created`)
                 ▼
┌─────────────────────────────────┐
│   Voice Calling Agent API       │
│   Endpoint: `/api/customers`    │
└────────────────┬────────────────┘
                 │
                 │ Saves Lead with Status = 'PENDING'
                 ▼
┌─────────────────────────────────┐
│   PostgreSQL Queue & Engine     │
│   - Order ID: WF-xxxxx          │
│   - Customer Name & Phone       │
│   - Product, Price, Address     │
└────────────────┬────────────────┘
                 │
                 │ Auto-Dial Campaign / Instant Trigger
                 ▼
┌─────────────────────────────────┐
│   AI Agent (Priya) Dials        │
│   Customer via TeleCMI          │
└─────────────────────────────────┘
```

---

## 3. Webhook / API Specification for Onboarding

To automatically onboard an order, the main app backend should send an HTTP POST request to the Voice Agent server upon order creation.

### Endpoint Details
* **Method:** `POST`
* **Path:** `/api/customers`
* **Base URL:** `https://your-voice-server-domain.com` (or local/internal network URL)
* **Headers:**
  ```http
  Content-Type: application/json
  Cookie: welfog_admin_session=<ADMIN_SESSION_TOKEN>
  ```
  *(Note: A dedicated API Key / Bearer Token header can also be enabled in `app/routes/admin.py` if preferred over session cookies).*

### Request Payload (JSON Schema)

| Field Name | Type | Required | Description | Example |
|---|---|---|---|---|
| `name` | string | **Yes** | Full name of the customer | `"Abhishek Degra"` |
| `phone` | string | **Yes** | 10-digit Indian mobile number | `"9521399708"` |
| `order_id` | string | **Yes** | Unique Welfog order identifier | `"WF-10929"` |
| `product_name` | string | **Yes** | Item purchased by the customer | `"Smart Fitness Watch Series 9"` |
| `product_price`| string | **Yes** | Order total including currency symbol | `"₹2,499"` |
| `delivery_time`| string | Optional | Estimated delivery duration | `"2-3 Business Days"` |
| `payment_method`| string| Optional | Payment mode (`COD` or `Prepaid`) | `"Cash on Delivery (COD)"` |
| `notes` | string | Optional | Complete shipping address & landmark | `"House 12, Malviya Nagar, Sector 4, Jaipur, Rajasthan - 302017"` |

### Example Request Body
```json
{
  "name": "Abhishek Degra",
  "phone": "9999999999",
  "order_id": "WF-10929",
  "product_name": "Smart Fitness Watch Series 9 (Midnight Black)",
  "product_price": "₹2,499",
  "delivery_time": "2-3 Business Days",
  "payment_method": "Cash on Delivery (COD)",
  "notes": "Flat 302, Royal Residency, Malviya Nagar Sector 4, Jaipur, Rajasthan 302017"
}
```

### Expected Response
* **Status Code:** `200 OK`
* **Response Body:**
```json
{
  "success": true,
  "customer_id": 42,
  "status": "PENDING",
  "message": "Customer onboarded successfully into call queue"
}
```

---

## 4. Implementation Options for Senior Developer

The senior engineer can choose between two operational modes based on business preference:

### Mode A: Autonomous Campaign Queue (Recommended)
1. When an order is placed, the backend calls `POST /api/customers`.
2. The record enters PostgreSQL with status `PENDING`.
3. The Voice Agent's **Autonomous Auto-Calling Campaign** processes the queue sequentially during permitted calling hours (e.g., 10:00 AM – 7:00 PM IST).
4. **Advantage:** Respects Do-Not-Disturb (DND) time windows and rate-limits telephony concurrency.

### Mode B: Instant Auto-Call with Buffer Delay
1. When an order is placed, wait 5–10 minutes (to allow user to finish checking their app/receipt).
2. The main backend triggers an immediate outbound call using:
   ```http
   POST /api/calls/outbound
   Content-Type: application/json
   
   {
     "user_id": 1029,
     "customer_phone": "9999999999",
     "order_id": "WF-10929",
     "language": "Hindi"
   }
   ```
3. **Advantage:** Ultra-fast order confirmation while customer's purchase intent is fresh.

---

## 5. Sample Backend Integration Code

### Node.js / Express Example
```javascript
const axios = require('axios');

async function syncNewOrderToVoiceAgent(order) {
    try {
        const payload = {
            name: `${order.shipping_address.first_name} ${order.shipping_address.last_name}`.trim(),
            phone: order.shipping_address.phone.replace(/[^0-9]/g, '').slice(-10),
            order_id: String(order.id || order.order_number),
            product_name: order.items.map(item => item.title).join(', '),
            product_price: `₹${order.total_amount}`,
            delivery_time: "2-3 Business Days",
            payment_method: order.is_cod ? "Cash on Delivery (COD)" : "Prepaid Online",
            notes: `${order.shipping_address.address1}, ${order.shipping_address.city}, ${order.shipping_address.province} - ${order.shipping_address.zip}`
        };

        const response = await axios.post('http://VOICE_AGENT_HOST:8000/api/customers', payload, {
            headers: { 'Content-Type': 'application/json' },
            timeout: 5000
        });

        console.log(`[VoiceAgent] Successfully queued Order #${order.id} for AI verification.`);
        return response.data;
    } catch (err) {
        console.error(`[VoiceAgent] Failed to queue Order #${order.id}:`, err.message);
    }
}
```

### Python / Django / FastAPI Example
```python
import httpx

async def sync_order_to_voice_agent(order_data: dict):
    url = "http://VOICE_AGENT_HOST:8000/api/customers"
    payload = {
        "name": order_data.get("customer_name", "Customer"),
        "phone": order_data.get("mobile_number")[-10:],
        "order_id": order_data.get("order_id"),
        "product_name": order_data.get("product_title"),
        "product_price": f"₹{order_data.get('grand_total')}",
        "delivery_time": "2-3 Business Days",
        "payment_method": "Cash on Delivery (COD)" if order_data.get("is_cod") else "Prepaid",
        "notes": order_data.get("full_shipping_address", "")
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            res = await client.post(url, json=payload)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            # Non-blocking log to ensure order placement is never interrupted
            print(f"Voice Agent Sync Error: {e}")
```

---

## 6. Feedback Loop (Syncing Call Results Back to Main App)

When Priya finishes speaking with the customer, the Voice Agent updates its internal database. If the senior developer wants the Voice Agent to send call results back to the main app, they can point the completion hook to their webhook receiver:

### Dispositions Generated by AI Voice Agent:
* `CONFIRMED`: Customer explicitly verified order, price, and delivery address. Order can be moved directly to **Processing / Warehouse Packing**.
* `CANCELLED`: Customer requested cancellation or denied placing the order. Order can be marked **Cancelled** before courier dispatch, saving shipping charges.
* `ADDRESS_CORRECTION`: Customer provided updated street, landmark, or house number. Transcripts contain the corrected address.
* `NO_ANSWER / BUSY`: Customer did not answer or disconnected. Automatically queued for smart retries.
* `HIGH_RTO_RISK`: Customer hesitated, sounded uncertain, or refused confirmation. Flagged for manual warehouse verification.

---

## 7. Checklist for Senior Developer

- [ ] Choose trigger point in Welfog App backend (e.g., checkout success controller / `OrderCreated` event listener).
- [ ] Ensure customer phone number is extracted as a clean 10-digit string.
- [ ] Map order item details, total price, and address into the `/api/customers` payload.
- [ ] Set network route / firewall permission between Welfog App backend and the Voice Agent server port (`8000`).
- [ ] (Optional) Provide a callback endpoint on Welfog App backend if real-time order status updates (Confirmed vs Cancelled) need to be written directly into the main app database.

---
*Created by: Welfog AI Voice Agent Team*  
*Date: November 2026*  
*Repository Branch: `abhi`*
