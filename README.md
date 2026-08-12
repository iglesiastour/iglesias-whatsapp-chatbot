# Iglesias WhatsApp Chatbot

A small FastAPI boundary for the Iglesias Tour Turkey WhatsApp AI chatbot. It
validates customer messages and can forward them to the existing n8n automation.

Meta WhatsApp Cloud API is **not connected yet**. This backend does not call
Meta, OpenAI, or Google Sheets directly.

## Current architecture

```text
Test Client
   ↓
FastAPI
   ↓
n8n
   ↓
AI + Tour Lookup + Lead Save
```

### Existing n8n functionality

n8n remains the current orchestration layer and working MVP. It owns bilingual
conversations, memory, tour selection and lookup, lead confirmation and storage,
human handoff, and lead ID generation. This repository does not reimplement or
modify that workflow.

### Python backend functionality

- `GET /health` reports API health.
- `POST /api/v1/messages/test` validates and normalizes a message locally.
- `POST /api/v1/messages/process` normalizes a message, forwards it to n8n, and
  returns only the parsed chatbot reply.

The process endpoint accepts n8n JSON responses with a non-empty top-level
`reply`, `output`, or `message` string. Unsupported shapes, invalid JSON, and
upstream errors produce a controlled gateway error; raw n8n responses are never
returned to the client.

## Local setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Set `N8N_WEBHOOK_URL` in the local `.env` file before using the process
endpoint. Never commit the webhook URL or other credentials.

## Run API

```bash
uvicorn app.main:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

Run tests with:

```bash
python -m pytest -v
```

Tests mock n8n and never call a real webhook.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/messages/process \
  -H "Content-Type: application/json" \
  -d '{
    "from": "+905551112233",
    "name": "Maria",
    "message": "Efes turları hakkında bilgi almak istiyorum"
  }'
```

Example response:

```json
{
  "success": true,
  "data": {
    "customer_phone": "+905551112233",
    "reply": "..."
  }
}
```

## Environment variables

| Variable | Purpose |
| --- | --- |
| `APP_ENV` | Runtime environment name |
| `APP_HOST` | API bind host |
| `APP_PORT` | API bind port |
| `N8N_WEBHOOK_URL` | Required locally for message processing; n8n webhook target |
| `N8N_TIMEOUT_SECONDS` | Maximum n8n wait in seconds; defaults to `20` |
| `WHATSAPP_VERIFY_TOKEN` | Unused placeholder for future Meta integration |
| `WHATSAPP_ACCESS_TOKEN` | Unused placeholder for future Meta integration |
| `WHATSAPP_PHONE_NUMBER_ID` | Unused placeholder for future Meta integration |
| `OPENAI_API_KEY` | Unused placeholder; OpenAI remains behind n8n |
| `GOOGLE_SHEETS_ID` | Unused placeholder; Sheets remains behind n8n |

`.env` is ignored by Git. If `N8N_WEBHOOK_URL` is empty, the process endpoint
returns `503 Service Unavailable`.
