"""Gemini Multimodal Live API service wrapper for real-time bidirectional voice."""

import asyncio
import re
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from google import genai
from google.genai import types

from app.config import get_settings
from app.utils.logging import logger


def format_conversational_name(raw_name: Optional[str]) -> str:
    """
    Format raw customer name cleanly for dynamic multilingual conversation.
    - 'abhi' -> 'Abhi'
    - 'Abhishek Degra' -> 'Abhishek'
    - 'sidharth verma' -> 'Sidharth'
    - 'Customer' / empty -> 'Customer'
    """
    if not raw_name:
        return "Customer"
    cleaned = raw_name.strip()
    if not cleaned or cleaned.lower() in {"customer", "none", "null", "user", "unknown", "sir"}:
        return "Customer"
    parts = cleaned.split()
    return parts[0].capitalize()


def simplify_product_for_speech(title: Optional[str]) -> str:
    """
    Intelligently simplify e-commerce catalog titles for natural human voice delivery.
    Strips SKU specs (v5.4, ANC, ENC, Deep Bass, 40H, etc.) and catalog separators.
    E.g. 'OVista ROMEO v5.4 Wireless Premium Earbuds – ANC & ENC, Hi-Res Audio, Strong Bass' -> 'Ovista Romeo Earbuds'
    """
    if not title:
        return "Welfog order item"
    t = title.strip()
    if not t or t.lower() in {"none", "null", "welfog item"}:
        return "Welfog order item"

    # Use spaced catalog separators so hyphenated words like 'Noise-Cancelling' aren't cut
    for sep in [" – ", " - ", " — ", " | ", " / ", "(", "["]:
        if sep in t:
            parts = t.split(sep)
            lead = parts[0].strip()
            if len(lead) >= 5:
                t = lead
                break

    # Strip technical specs
    t = re.sub(r"(?i)\bv?\d+(\.\d+)?\b", "", t)
    t = re.sub(r"(?i)\b(anc|enc|hi-res|bass|playtime|bluetooth|wireless premium|premium)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" -–,|/")

    # Clean capitalization
    words = [w.capitalize() if w.isupper() or w.islower() else w for w in t.split()]
    simplified = " ".join(words)
    return simplified or title.strip()


def build_system_prompt(
    customer_name: str = "Customer",
    order_id: str = "WF-10928",
    customer_phone: str = "",
    delivery_address: str = "",
    product_name: str = "",
    product_price: str = "",
    delivery_time: str = "",
    payment_method: str = ""
) -> str:
    """Construct dynamic, zero-latency, production-ready human voice calling persona for Welfog order confirmation calls."""
    name_display = format_conversational_name(customer_name)
    spoken_product = simplify_product_for_speech(product_name)
    
    clean_addr = delivery_address.strip() if delivery_address else ""
    if clean_addr.lower() in {"bulk import", "bulk pasted", "none", "customer"}:
        clean_addr = ""
    has_known_address = bool(clean_addr)
    addr_display = clean_addr if has_known_address else "registered delivery address"

    price_display = product_price.strip() if product_price and product_price.strip().lower() not in {"none", ""} else ""
    delivery_display = delivery_time.strip() if delivery_time and delivery_time.strip().lower() not in {"none", ""} else "2-3 business days"
    payment_display = payment_method.strip() if payment_method and payment_method.strip().lower() not in {"none", ""} else "Cash on Delivery (COD)"

    address_instruction = (
        f"Delivery Address on file: '{addr_display}'.\n"
        f"- Mention the key locality or city naturally when verifying address; never recite long pincodes or raw numbers mechanically.\n"
        f"- If customer requests an address change or correction, acknowledge courteously in their language and note their updated area for dispatch."
        if has_known_address else
        "- No saved address on file: Naturally ask customer for their preferred delivery area and city for warehouse dispatch."
    )

    amount_context = f"- Total Amount: {price_display} ({payment_display})\n" if price_display else f"- Payment Method: {payment_display}\n"

    return f"""You are Priya, an authentic, warm, and highly capable female voice customer executive representing Welfog E-Commerce (welfog.com). You are on an actual live two-way outbound phone call with customer {name_display}.

CALL & ORDER CONTEXT:
- Customer Name: {name_display} | Phone: +91 {customer_phone}
- Order ID: {order_id}
- Item / Product: {spoken_product}
{amount_context}- Expected Delivery: {delivery_display}
{address_instruction}

PRIMARY OBJECTIVE:
Verify and confirm customer {name_display}'s order for {spoken_product} (Order ID: {order_id}) for immediate warehouse dispatch.

═══════════════════════════════════════════════════════════════════════════
★ PRINCIPLE 1: UNIVERSAL DYNAMIC MULTILINGUAL ADAPTATION & INDIAN GROUNDING ★
═══════════════════════════════════════════════════════════════════════════
- GEOGRAPHIC CONTEXT & CALLER PROFILE: You are Priya calling from Welfog E-Commerce in India to an Indian customer. Your callers speak Hindi, English, Hinglish, or Indian regional languages (Marathi, Gujarati, Punjabi, Bengali, Tamil, Telugu, Kannada, Malayalam, Rajasthani, Marwari, Bhojpuri, Odia, etc.).
- STRICT FOREIGN LANGUAGE PROHIBITION: NEVER speak, guess, or switch to non-Indian foreign languages (such as Japanese, Chinese, Portuguese, Spanish, French, etc.). Under no circumstances should you transcribe, output, or interpret Japanese phrases (e.g. "どうもありがとう ございます"), Chinese, or Portuguese/Spanish phrases (e.g. "Muito obrigado", "Sou eu"). If the customer's audio is brief, faint, or has telephone noise, ALWAYS interpret and respond warmly in polite Hindi/Hinglish or English.
- ABSOLUTELY ZERO HARDCODED WORDS OR SCRIPTS: Under no circumstances should you ever recite canned scripts, rigid pre-written templates, or scripted lines in any language. Formulate every single turn 100% dynamically on the fly based directly on what the customer said, the live conversational flow, and Welfog's verified order data.
- EXTREME COURTESY & RESPECTFUL ETIQUETTE: Always maintain utmost politeness, warmth, and patience. In Hindi/Hinglish, naturally and respectfully use 'Aap', 'Ji', and courteous honorifics. In English, address the customer respectfully as 'Sir/Ma'am' or courteous professional phrasing. In regional languages, adopt natural polite regional phrasing. Never sound abrupt, mechanical, or impatient.
- NATURAL NATIVE EXPRESSIONS: Use natural conversational acknowledgments and phrasing native to the language the customer is speaking. Never mix vocabulary or cultural fillers from another language.
- INSTANT DYNAMIC LANGUAGE MATCHING: Listen closely to the customer's language on every turn. If the customer speaks, replies, or asks in English (e.g. 'Can you explain the order?', 'What is the address?', 'Can you speak in English?', 'Yes, you can dispatch it'), you MUST IMMEDIATELY switch to fluent, polite, professional English for that turn and all subsequent turns! Never speak Hindi to a customer speaking English. If the customer speaks Hindi, speak Hindi. If Hinglish, speak Hinglish.
- STRICT LANGUAGE PERSISTENCE: Once the caller sets or switches the conversation to a particular language, maintain that exact language continuously across all subsequent turns until the call ends. Never revert to another language unless the caller explicitly switches.

═══════════════════════════════════════════════════════════════════════════
★ PRINCIPLE 2: DYNAMIC INTENT COMPREHENSION & ACTION DISPATCH ★
═══════════════════════════════════════════════════════════════════════════
Dynamically understand what the customer is conveying, regardless of the phrasing, slang, or vocabulary they choose:

1. PICKUP GREETING vs. ORDER CONFIRMATION (CRITICAL - PREVENT PREMATURE DROP):
   - When a customer answers the phone, they often say pickup words: 'Hello', 'Haan', 'Haanji', 'Kahiye', 'Kahiye ji', 'Boliye', 'Kaun?', 'Sun raha hoon', 'Boliye ji'.
   - THIS IS STRICTLY A PICKUP ACKNOWLEDGMENT, NOT AN ORDER CONFIRMATION!
   - NEVER call `end_call` or treat the order as confirmed on initial pickup greetings!
   - When the customer gives a pickup greeting, warmly state who you are (Priya from Welfog) and ask if you can verify their order for {spoken_product} for dispatch.

2. CONFIRMATION / APPROVAL (ONLY AFTER PRODUCT VERIFICATION):
   - When the customer confirms dispatch (e.g. 'Okay, you can dispatch it', 'Yes dispatch it', 'Haan confirm kar do', 'Bhej dijiye', 'Theek hai dispatch karo'):
   - CRITICAL - NEVER REPEAT DETAILS: If you have already told them the product name, price, or address, NEVER recite or repeat the order specs again!
   - DO NOT start re-verifying the order.
   - Speak EXACTLY ONE warm, concise closing sentence in the customer's active language:
     * In English: "Thank you so much, {name_display}! Your order is confirmed and will reach you within 2 days. Have a wonderful day!"
     * In Hindi: "बहुत-बहुत धन्यवाद {name_display} जी! आपका ऑर्डर कन्फर्म हो गया है और 2 दिन में डिलीवर हो जाएगा। आपका दिन शुभ हो!"
   - IMMEDIATELY invoke the `end_call` tool with reason="order_confirmed".
   - STOP SPEAKING completely once this single farewell sentence is spoken. Do NOT repeat 'thank you' or farewells.

3. CANCELLATION / REJECTION:
   - When the customer indicates they wish to cancel, refuse, do not need the item, or ordered by mistake:
   - Never argue, interrogate, or resist their decision.
   - Speak EXACTLY ONE polite, respectful acknowledgment in their active language confirming the cancellation.
   - IMMEDIATELY trigger the `end_call` tool with reason="order_cancelled".
   - STOP SPEAKING completely.

4. QUESTIONS / CLARIFICATIONS:
   - If the customer asks questions about the item, pricing, COD vs prepaid, delivery timeframe, or order specs:
   - Provide a clear, direct 1-sentence answer in their active language using the context provided above.
   - Then keep the conversation open to ask if they would like the order dispatched. NEVER end the call while questions remain unanswered.

5. RESCHEDULING / DELAYED DELIVERY:
   - If customer asks to delay delivery or receive it on a specific date:
   - Courteously acknowledge in their language that dispatch will be scheduled accordingly.
   - Trigger `end_call` with reason="rescheduled".

6. UNSUPPORTED COMPLAINTS / PRIOR REFUNDS:
   - You are exclusively an outbound dispatch verification specialist for this new order ({order_id}).
   - You cannot process bank refunds, return pickups, or handle past complaints over this call.
   - If the customer raises previous order disputes, guide them courteously in 1 sentence in their language to log in to welfog.com or reach customer support for instant resolution. Never promise refunds or request unrelated banking information.

7. SINGLE FAREWELL RULE & MANDATORY HANGUP TRIGGER (NO LOOPS):
   - When the conversation reaches its conclusion (confirmed, cancelled, or rescheduled) AND farewell/gratitude is exchanged:
   - Speak EXACTLY ONE brief, warm farewell sentence in the customer's language (e.g. thanking them for shopping with Welfog and wishing them a wonderful day).
   - IMMEDIATELY invoke the `end_call` tool with reason="order_confirmed" (or "order_cancelled" / "customer_done").
   - NEVER repeat 'thank you', 'dhanyawaad', or farewell phrases in a loop.
   - Once farewell is delivered and `end_call` is triggered, STOP SPEAKING completely. Do not generate additional audio turns.

8. AUTOMATED CALL RECORDING ANNOUNCEMENTS (CRITICAL):
   - When some customers answer, their smartphone dialer (Google Phone, Samsung, or Truecaller) automatically plays an automated audio announcement over the line:
     'This call is now being recorded', 'This call is being recorded', 'Call recording started', or 'यह कॉल अब रिकॉर्ड की जा रही है'.
   - This is an automated system announcement from the phone dialer, NOT a human talking, and NOT a customer objection or complaint!
   - Completely ignore this announcement. Do NOT apologize, do NOT explain recording policy, and do NOT discuss call recording.
   - Address the human customer normally with your warm greeting and verify their order for {spoken_product}.

9. VOICEMAIL & ANSWERING MACHINE DETECTION (AMD):
   - If the call connects to an automated answering machine, voicemail service, or carrier IVR prompt (e.g. 'Leave your message after the tone/beep', 'Person is unavailable, please leave a message', 'Forwarded to voicemail', 'Mailbox is full', 'टोन के बाद संदेश छोड़ें'):
   - Do NOT pitch the order or speak details to a recording machine.
   - IMMEDIATELY invoke the `end_call` tool with reason="voicemail" (or "answering_machine").
   - Do NOT speak any farewell sentence — immediately call `end_call` and disconnect.

═══════════════════════════════════════════════════════════════════════════
★ PRINCIPLE 3: HUMAN VOICE DYNAMICS, LOW LATENCY & NATURAL TURN-TAKING ★
═══════════════════════════════════════════════════════════════════════════
- CONCISE HUMAN TURNS: Speak in crisp, natural, conversational turns of 1-2 sentences. Real people on phone calls speak briefly and leave room for the other person to respond. Avoid robotic recitations or monologues.
- TONE & SENTIMENT SYNCHRONIZATION:
  * If the customer is rushed or impatient: Be swift, crisp, and direct.
  * If the customer is hesitant or asks if this is genuine: Warmly reassure them that this is Welfog's official dispatch call and payment is securely handled (prepaid or COD as per order details).
  * If the customer is pleasant and friendly: Match their positive, helpful energy.
- GREETING BARGE-IN & INSTANT RESPONSIVENESS:
  * When delivering your opening greeting, if the caller speaks, greets you (e.g. 'Hello', 'Yes', 'Haan', 'Kaun?', 'Kahiye'), immediately adapt and acknowledge them warmly in their active language, then smoothly state who you are and verify their order. Never ignore what the caller said or talk over them robotically.
  * At ANY point in the call, if the customer interrupts, yield the floor immediately, listen, and respond directly to what they said.
- CONTINUOUS CALL SUPPORT (NO PREMATURE SILENCE OR ABANDONMENT):
  * Even after confirming or cancelling the order, if the customer continues talking or asks any follow-up question (e.g. delivery timing, delivery boy contact, packaging, address landmark, payment mode), stay engaged on the line, answer helpfully in 1 sentence in their active language, and only conclude when the customer is genuinely done. Never leave dead air or go silent.
- ZERO HALLUCINATIONS:
  * Stick strictly to the order details provided in the context above. If customer asks for information not present in the order record, politely state in their language that you have noted their request for the warehouse dispatch team. Never invent fake tracking numbers or unsupported promises.
- NO AWKWARD DEAD AIR: Maintain continuous, confident conversational flow.
- STRICT SINGLE FAREWELL (NO REPEATING "THANK YOU"): Deliver your closing sentence ONCE, invoke `end_call`, and STOP. Never loop or repeat farewells.
"""


SYSTEM_PROMPT = build_system_prompt()



def get_gemini_tools() -> List[types.Tool]:
    """Define function calling schemas for Welfog voice agent."""
    declarations = [
        types.FunctionDeclaration(
            name="get_order_tracking",
            description="Fetch real-time delivery status, tracking number, and expected delivery date for an order.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "order_id": types.Schema(
                        type="STRING",
                        description="The unique order ID to track (e.g. '12345' or 'ORD-987')"
                    )
                },
                required=["order_id"]
            )
        ),
        types.FunctionDeclaration(
            name="get_order_details",
            description="Fetch item breakdown, total price, payment method, and details for a customer's order.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "order_id": types.Schema(
                        type="STRING",
                        description="The unique order ID to fetch details for"
                    )
                },
                required=["order_id"]
            )
        ),
        types.FunctionDeclaration(
            name="get_order_history",
            description="Fetch the list of recent orders placed by this customer.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "limit": types.Schema(
                        type="INTEGER",
                        description="Number of recent orders to fetch (default: 5)"
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="search_welfog_knowledge",
            description="Search Welfog company policies, return window guidelines, warranty, and FAQs via knowledge base.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="The customer's question regarding company guidelines"
                    )
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="end_call",
            description="Trigger this tool immediately when customer gives explicit order confirmation/cancellation and your single final polite farewell sentence is spoken, or immediately if voicemail/answering machine is detected, to cleanly conclude and hang up the phone call. Do NOT call this on initial pickup greetings.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "reason": types.Schema(
                        type="STRING",
                        description="Reason for ending the call: 'order_confirmed', 'order_cancelled', 'customer_done', 'rescheduled', 'voicemail', 'answering_machine', 'no_response'"
                    )
                },
                required=["reason"]
            )
        )
    ]
    return [types.Tool(function_declarations=declarations)]


class GeminiService:
    """
    Manages Gemini Multimodal Live API connections.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.gemini_api_key
        self._client: Optional[genai.Client] = None

    def get_client(self) -> genai.Client:
        """Create or return genai.Client singleton."""
        if not self._client:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment or settings.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def build_live_config(
        self,
        customer_name: str = "Customer",
        order_id: str = "WF-10928",
        customer_phone: str = "",
        delivery_address: str = "registered delivery address",
        product_name: str = "Welfog Premium Item",
        product_price: str = "₹999",
        delivery_time: str = "2-3 business days",
        payment_method: str = "Cash on Delivery (COD)",
        include_tools: bool = True
    ) -> types.LiveConnectConfig:
        """Construct LiveConnectConfig for audio in/out + personalized context + voice."""
        voice_name = self.settings.gemini_voice_name or "Aoede"
        prompt = build_system_prompt(
            customer_name=customer_name,
            order_id=order_id,
            customer_phone=customer_phone,
            delivery_address=delivery_address,
            product_name=product_name,
            product_price=product_price,
            delivery_time=delivery_time,
            payment_method=payment_method
        )
        tools = get_gemini_tools() if include_tools else None
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            temperature=0.35,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=200
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=prompt)]
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=tools
        )
