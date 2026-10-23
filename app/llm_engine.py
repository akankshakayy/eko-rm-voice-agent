"""LLM-powered conversation engine (Claude, or a free OpenAI-compatible provider like Groq or Gemini).

Every turn the model is forced to call a single `agent_turn` tool, so its output
is always structured: the spoken reply + intent + extracted details + the next
action. The orchestrator (agent.py) applies hard safety rules on top.
"""
import json
import os
import time

from .taxonomy import CATEGORIES, SLOT_LABELS, INTENT_CONFIDENCE_MIN

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

# Free OpenAI-compatible providers. Each needs only its own API key.
PROVIDERS = {
    "groq": {"label": "Groq", "key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1",
             # Comma-separated fallback chain: each model has its own free-tier rate limit
             "model": os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b")},
    "gemini": {"label": "Gemini", "key_env": "GEMINI_API_KEY",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")},
}

INTENTS = list(CATEGORIES.keys()) + ["unknown"]

AGENT_TURN_TOOL = {
    "name": "agent_turn",
    "description": "Return the agent's next spoken reply and its structured understanding of the call so far.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string",
                      "description": "What the agent says next, read aloud by a voice. Short (max 2 sentences), "
                                     "polite, in the call language and script given in the instructions."},
            "understood": {"type": "boolean",
                           "description": "False if the caller's last message was unclear, garbled or off-topic."},
            "language": {"type": "string", "enum": ["hi", "hinglish", "en", "other"]},
            "intent": {"type": "string", "enum": INTENTS},
            "intent_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "slots": {"type": "object", "additionalProperties": {"type": "string"},
                      "description": "All issue details collected so far (cumulative), using the slot keys listed."},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "frustrated", "angry"]},
            "action": {"type": "string",
                       "enum": ["continue", "create_ticket", "escalate", "resolved_by_agent", "end_call"]},
            "escalation_reason": {"type": "string"},
            "summary": {"type": "string",
                        "description": "One-line English summary of the issue for the RM's ticket."},
        },
        "required": ["reply", "understood", "intent", "intent_confidence", "slots", "action", "sentiment"],
    },
}


def _language_rules(lang: str | None) -> str:
    if lang == "en":
        return ("LANGUAGE: The caller chose English. Reply only in simple, clear Indian English, even if the caller "
                "mixes in Hindi words. Address them respectfully (e.g. 'Ramesh ji').")
    return ("LANGUAGE: The caller chose Hindi. Write every reply in Hindi using DEVANAGARI script - it is read "
            "aloud by a Hindi voice, and Roman-script Hindi is mispronounced. Keep banking/app terms in English "
            "letters as people say them: DMT, AePS, transaction ID, UTR, OTP, KYC, app, ticket, RM, bank, UPI. "
            "Use 'जी' and 'आप'. You are a woman: use feminine first-person forms (समझ गई, कर रही हूँ, नोट कर लेती हूँ). "
            "Example: 'समझ गई जी, DMT transfer fail हुआ है। कृपया transaction ID बताइए।' "
            "Never write Hindi in Roman script. If the caller asks to switch to English, reply in English.")


def _system_prompt(state: dict) -> str:
    cats = "\n".join(
        f"- {k}: {v['label']} | required details: {', '.join(v['slots']) or 'none'} | priority {v['priority']}"
        + (" | ALWAYS escalate" if v.get("always_escalate") else "")
        + (" | self-serve: answer from the CSP's ticket list, action=resolved_by_agent" if v.get("self_serve") else "")
        for k, v in CATEGORIES.items())
    csp = state.get("csp") or {}
    tickets = state.get("csp_tickets") or []
    tlist = "\n".join(f"  - {t['ticket_no']}: {t['category']} | status {t['status']} | {t['summary']}"
                      for t in tickets) or "  (none)"
    slots = {k: v for k, v in (state.get("slots") or {}).items() if not k.startswith("_")}
    return f"""You are "Eko Sahayak", a voice assistant that answers and makes phone calls on behalf of
Eko Relationship Managers (RMs). You talk to CSPs (Customer Service Points - retailers/shopkeepers who
offer Eko services like DMT money transfer, AePS cash withdrawal, BBPS bill payment and recharge).

YOUR JOB ON THIS CALL
1. Understand the CSP's issue and classify it into exactly one category below.
2. Collect the required details for that category, ONE question at a time.
3. Once all details are present, read back a very short summary and ask the CSP to confirm.
4. After the CSP confirms -> action=create_ticket (the system will tell them the ticket number).
5. Escalate (action=escalate, give escalation_reason) when: the CSP asks for a human/their RM, the CSP is
   angry, the issue is suspected fraud, the issue is outside these categories, or you cannot make progress.
6. For ticket_status questions, answer from the CSP's existing tickets below -> action=resolved_by_agent.
7. If the caller only says thanks/bye with no issue -> action=end_call.

CATEGORIES
{cats}

DETAIL (SLOT) KEYS: {", ".join(f"{k} ({v})" for k, v in SLOT_LABELS.items())}
Record in `slots` EVERY detail the caller has stated anywhere in the call, starting with their very first
message - e.g. "I deposited 10000 by NEFT, UTR 412398765432" -> amount=10000, payment_mode=NEFT,
utr_or_ref=412398765432. Use exactly these keys with just the value. Never ask for a detail already given.
If the CSP says they do not know a transaction ID / UTR, store "not known" and move on.

STYLE: You are speaking on a phone line. Replies must be 1-2 short sentences, warm and respectful.
{_language_rules(state.get("lang"))}
No markdown, lists or emojis. Never ask for OTP, PIN, password or Aadhaar number. Never promise a refund
amount or timeline you do not know - say the RM team will check.
Never say a ticket number yourself - when you set action=create_ticket the system reads out the real number.
If the last message was unclear or garbled, set understood=false and politely ask them to repeat.

CALL CONTEXT
- Direction: {state.get('direction')} | Purpose: {state.get('purpose')}
- CSP: {csp.get('name', 'UNKNOWN - ask for CSP ID')} ({csp.get('csp_code', '-')}), shop {csp.get('shop_name', '-')}, {csp.get('district', '-')}
- RM: {state.get('rm_name', '-')}
- CSP's recent tickets:
{tlist}
- Current understanding: intent={state.get('intent')} confidence={state.get('intent_confidence')} details={json.dumps(slots, ensure_ascii=False)}
- Confirmation already asked: {'yes' if (state.get('slots') or {}).get('_asked') == '_confirm' else 'no'}
- Intent confidence below {INTENT_CONFIDENCE_MIN} means you should ask a clarifying question instead of guessing.
"""


def _build_messages(state: dict, user_text: str) -> list[dict]:
    messages = [{"role": "user", "content": "[Call connected]"}]
    for turn in state.get("history", []):
        role = "assistant" if turn["role"] == "agent" else "user"
        if messages[-1]["role"] == role:  # keep strict alternation
            messages[-1]["content"] += "\n" + turn["text"]
        else:
            messages.append({"role": role, "content": turn["text"]})
    if messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n" + user_text
    else:
        messages.append({"role": "user", "content": user_text})
    return messages


ID_SLOTS = ("txn_id", "utr_or_ref")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_id(value: str) -> str:
    """IDs read out in pieces come back as "TXN 482-913-577" or "४१२ ३९८"; joined up they match the real ID.
    Anything that isn't an ID once joined ("not known") is left alone."""
    joined = "".join(str(value).translate(_DEVANAGARI_DIGITS).split()).replace("-", "").replace(".", "").upper()
    return joined if len(joined) >= 6 and joined.isalnum() and any(ch.isdigit() for ch in joined) else str(value).strip()


def _clean_slots(raw) -> dict:
    """Models sometimes garble a key ("payment_mode NEFT"); map it back to the slot it starts with."""
    out = {}
    for k, v in (raw if isinstance(raw, dict) else {}).items():
        if v in (None, ""):
            continue
        key = str(k).strip()
        if key not in SLOT_LABELS:
            key = next((s for s in SLOT_LABELS if key.lower().startswith(s)), key)
        out[key] = normalize_id(v) if key in ID_SLOTS else str(v).strip()
    return out


def _parse(out: dict, state: dict, engine: str, latency: float, tin: int, tout: int, model: str | None = None) -> dict:
    slots = _clean_slots(out.get("slots"))
    prev = state.get("slots") or {}
    merged = {**{k: v for k, v in prev.items() if not k.startswith("_")}, **slots}
    reply_l = (out.get("reply") or "").lower()
    if out.get("action") == "continue" and any(w in reply_l for w in ("confirm", "raise kar", "sahi hai", "सही है",
                                                                      "raise कर", "shall i raise", "is that correct")):
        merged["_asked"] = "_confirm"
    intent = out.get("intent", "unknown")
    return {
        "engine": engine,
        "reply": out.get("reply", ""),
        "understood": bool(out.get("understood", True)),
        "language": out.get("language", "hinglish"),
        "intent": intent if intent in INTENTS else "unknown",
        "intent_confidence": float(out.get("intent_confidence") or 0),
        "slots": merged,
        "sentiment": out.get("sentiment", "neutral"),
        "action": out.get("action", "continue"),
        "escalation_reason": out.get("escalation_reason"),
        "summary": out.get("summary"),
        "latency_ms": round(latency, 1),
        "input_tokens": tin,
        "output_tokens": tout,
        "model": model,
    }


class ClaudeEngine:
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str = MODEL):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def decide(self, state: dict, user_text: str) -> dict:
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=_system_prompt(state),
            messages=_build_messages(state, user_text),
            tools=[AGENT_TURN_TOOL],
            tool_choice={"type": "tool", "name": "agent_turn"},
        )
        latency = (time.perf_counter() - t0) * 1000
        block = next(b for b in resp.content if b.type == "tool_use")
        return _parse(dict(block.input), state, self.name, latency, resp.usage.input_tokens, resp.usage.output_tokens,
                      self.model)


def _compat_tool() -> dict:
    """agent_turn in OpenAI function format. Slots get explicit properties because some
    providers (Gemini) reject an object schema with no properties."""
    params = json.loads(json.dumps(AGENT_TURN_TOOL["input_schema"]))
    params["properties"]["slots"] = {
        "type": "object", "description": params["properties"]["slots"]["description"],
        "properties": {k: {"type": "string", "description": v} for k, v in SLOT_LABELS.items()}}
    return {"type": "function", "function": {"name": AGENT_TURN_TOOL["name"],
                                             "description": AGENT_TURN_TOOL["description"], "parameters": params}}


class OpenAICompatEngine:
    """Groq / Gemini through their OpenAI-compatible APIs (free tiers, rate-limited).

    The model setting can be a comma-separated chain. When one model hits its rate limit
    the turn goes straight to the next one (no waiting), and that model is skipped until
    its limit resets. A model that answers without calling the tool also hands over to the
    next one. If every model fails, the error falls through to the rules engine.
    """

    def __init__(self, provider: str):
        from openai import OpenAI
        cfg = PROVIDERS[provider]
        self.name = provider
        self.models = [m.strip() for m in cfg["model"].split(",") if m.strip()]
        self.model = self.models[0]
        self.client = OpenAI(api_key=os.environ.get(cfg["key_env"]), base_url=cfg["base_url"],
                             timeout=15, max_retries=0)
        self.tool = _compat_tool()
        self._cooldown: dict[str, float] = {}

    def decide(self, state: dict, user_text: str) -> dict:
        from openai import BadRequestError, RateLimitError
        messages = [{"role": "system", "content": _system_prompt(state)}] + _build_messages(state, user_text)
        now = time.time()
        ready = [m for m in self.models if self._cooldown.get(m, 0) <= now] or self.models[-1:]
        t0 = time.perf_counter()
        for i, model in enumerate(ready):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    max_tokens=600,
                    temperature=0.3,
                    messages=messages,
                    tools=[self.tool],
                    tool_choice={"type": "function", "function": {"name": "agent_turn"}},
                )
                break
            except RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", 20)) if e.response is not None else 20
                self._cooldown[model] = time.time() + min(max(wait, 5), 60)
                if i == len(ready) - 1:
                    raise
            except BadRequestError as e:
                # The model answered without calling agent_turn, or with arguments that break its schema:
                # a model-specific slip, so let the next model try before giving up on the LLM
                msg = str(e).lower()
                if not ("tool_use_failed" in msg or "tool call validation" in msg) or i == len(ready) - 1:
                    raise
        latency = (time.perf_counter() - t0) * 1000
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise ValueError("model returned no agent_turn tool call")
        out = json.loads(calls[0].function.arguments)
        usage = resp.usage
        return _parse(out, state, self.name, latency,
                      usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0, model)


def provider() -> str | None:
    """Which LLM to use: LLM_PROVIDER if set, else the first one with a key (free providers first)."""
    want = os.environ.get("LLM_PROVIDER", "auto").lower()
    keys = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    if want in keys:
        return want if os.environ.get(keys[want]) else None
    return next((p for p, env in keys.items() if os.environ.get(env)), None)


def model_for(p: str | None) -> str:
    if p == "claude":
        return MODEL
    return PROVIDERS[p]["model"].split(",")[0].strip() if p in PROVIDERS else "rules-v1"


def label_for(p: str | None) -> str:
    return "Claude" if p == "claude" else PROVIDERS[p]["label"] if p in PROVIDERS else "Rules"


def make_engine(p: str):
    return ClaudeEngine() if p == "claude" else OpenAICompatEngine(p)
