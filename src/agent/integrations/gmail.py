"""
Real Gmail integration skeleton. Shares OAuth flow/token with google_calendar.py
(add https://www.googleapis.com/auth/gmail.send to SCOPES in credentials setup).
"""
import base64
from email.mime.text import MIMEText
from typing import Dict, Any

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _get_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    import os

    creds = None
    if os.path.exists("gmail_token.json"):
        creds = Credentials.from_authorized_user_file("gmail_token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("gmail_token.json", "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_message(args: Dict[str, Any]) -> Dict[str, Any]:
    service = _get_service()

    to_list = args.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]

    message = MIMEText(args.get("body", ""))
    message["to"] = ", ".join(to_list)
    message["subject"] = args.get("subject", "")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"status": "sent", "message_id": sent.get("id")}
