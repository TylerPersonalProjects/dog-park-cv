"""
Google Drive uploader for incident clips.

Setup:
  1. Go to https://console.cloud.google.com
  2. Create project, enable Google Drive API
  3. Create OAuth2 credentials, download as credentials.json
  4. Place credentials.json in project root
  5. Run once to authorize:
     python -c "from src.drive_uploader import DriveUploader; DriveUploader()"
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logger.warning("Google Drive libraries not installed. Run: pip install google-auth google-auth-oauthlib google-api-python-client")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"


class DriveUploader:
    def __init__(self, folder_name: str = "DogParkCV_Incidents",
                 credentials_path: str = CREDS_FILE):
        self.folder_name = folder_name
        self.credentials_path = credentials_path
        self.folder_id: Optional[str] = None
        self._service = None

        if not GDRIVE_AVAILABLE:
            return

        self._service = self._authenticate()
        if self._service:
            self.folder_id = self._get_or_create_folder(folder_name)
            logger.info(f"Drive uploader ready. Folder: {folder_name} ({self.folder_id})")

    def _authenticate(self):
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif os.path.exists(self.credentials_path):
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                logger.warning(f"No credentials.json found. Drive upload disabled.")
                return None
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        return build("drive", "v3", credentials=creds)

    def _get_or_create_folder(self, name: str) -> Optional[str]:
        try:
            results = self._service.files().list(
                q=f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)"
            ).execute()
            items = results.get("files", [])
            if items:
                return items[0]["id"]
            meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
            folder = self._service.files().create(body=meta, fields="id").execute()
            return folder.get("id")
        except Exception as e:
            logger.error(f"Drive folder setup failed: {e}")
            return None

    def upload(self, filepath: str) -> Optional[str]:
        if not self._service or not self.folder_id:
            return None
        path = Path(filepath)
        if not path.exists():
            return None
        try:
            meta = {"name": path.name, "parents": [self.folder_id]}
            media = MediaFileUpload(str(path), mimetype="video/mp4", resumable=True)
            file = self._service.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
            link = file.get("webViewLink", "")
            logger.info(f"Uploaded to Drive: {path.name} -> {link}")
            return link
        except Exception as e:
            logger.error(f"Drive upload failed: {e}", exc_info=True)
            return None

    def is_ready(self) -> bool:
        return self._service is not None and self.folder_id is not None
