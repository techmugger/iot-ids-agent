"""
Real Google Calendar integration skeleton.

Setup (one-time):
  1. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  2. Create OAuth credentials in Google Cloud Console, download as credentials.json
     into project root.
  3. First run will open a browser for consent and cache a token.json locally.

This module is intentionally NOT imported until google_calendar_event() in
tools.py needs it, so the project runs without Google credentials configured
(falls back to the stub response in tools.py).
"""
from datetime import datetime, timedelta
from typing import Dict, Any

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    import os

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def create_event(args: Dict[str, Any]) -> Dict[str, Any]:
    service = _get_service()

    date = args.get("date")
    start_time = args.get("start_time", "09:00:00")
    duration_minutes = int(args.get("duration_minutes") or 60)

    start_dt = datetime.fromisoformat(f"{date}T{start_time}")
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    event_body = {
        "summary": args.get("title", "Untitled Meeting"),
        "location": args.get("location", ""),
        "start": {"dateTime": start_dt.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
        "attendees": [{"email": e} for e in (args.get("attendees") or [])],
    }
    created = service.events().insert(calendarId="primary", body=event_body).execute()
    return {"status": 200, "event_id": created.get("id"), "html_link": created.get("htmlLink")}
