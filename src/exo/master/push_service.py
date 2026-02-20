import json
import time
from pathlib import Path

import httpx
import jwt
from loguru import logger
from pydantic import BaseModel

from exo.shared.constants import (
    EXO_APNS_BUNDLE_ID,
    EXO_APNS_KEY_ID,
    EXO_APNS_KEY_PATH,
    EXO_APNS_TEAM_ID,
)


class DeviceRegistration(BaseModel, frozen=True):
    device_token: str
    device_name: str
    bundle_id: str
    registered_at: float


class DeviceRegistrationRequest(BaseModel):
    device_token: str
    device_name: str
    bundle_id: str


_APNS_HOST = "https://api.push.apple.com"
_JWT_EXPIRY_SECONDS = 55 * 60  # Refresh JWT every 55 minutes (APNs allows 60)


class PushService:
    """Sends APNs push notifications to registered devices.

    Gracefully no-ops when APNs environment variables are not configured.
    """

    def __init__(self, devices_file: Path) -> None:
        self._devices_file = devices_file
        self._devices: dict[str, DeviceRegistration] = {}
        self._client: httpx.AsyncClient | None = None
        self._cached_jwt: str | None = None
        self._jwt_issued_at: float = 0.0
        self._load_devices()

    @property
    def is_configured(self) -> bool:
        """True when all APNs environment variables are set and the .p8 key exists."""
        if not all([EXO_APNS_KEY_PATH, EXO_APNS_KEY_ID, EXO_APNS_TEAM_ID, EXO_APNS_BUNDLE_ID]):
            return False
        return Path(EXO_APNS_KEY_PATH).exists() if EXO_APNS_KEY_PATH else False

    def register_device(self, token: str, name: str, bundle_id: str) -> DeviceRegistration:
        """Register a device for push notifications. Deduplicates by token."""
        existing = self._devices.get(token)
        if existing is not None:
            return existing

        reg = DeviceRegistration(
            device_token=token,
            device_name=name,
            bundle_id=bundle_id,
            registered_at=time.time(),
        )
        self._devices[token] = reg
        self._persist_devices()
        logger.info(f"Registered device for push: {name} ({token[:8]}...)")
        return reg

    def unregister_device(self, token: str) -> bool:
        """Remove a device registration. Returns True if the device was found."""
        if token not in self._devices:
            return False
        del self._devices[token]
        self._persist_devices()
        logger.info(f"Unregistered device: {token[:8]}...")
        return True

    async def send_notification(
        self,
        title: str,
        body: str,
        conversation_id: str | None = None,
        assistant_content: str | None = None,
    ) -> None:
        """Send a push notification to all registered devices.

        Sends both a visible alert and a silent background notification so the
        client can sync the conversation immediately. If ``assistant_content``
        fits within the 4KB APNs payload limit it is included directly;
        otherwise the client fetches via ``conversation_id``.

        No-ops gracefully if APNs is not configured or no devices are registered.
        Removes tokens that receive HTTP 410 (expired/invalid).
        """
        if not self.is_configured or not self._devices:
            return

        client = self._get_client()
        token = self._get_jwt()
        bundle_id = EXO_APNS_BUNDLE_ID or ""

        payload: dict[str, object] = {
            "aps": {
                "alert": {"title": title, "body": body},
                "sound": "default",
                "content-available": 1,
            },
        }
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id

        # Include the full assistant response if it fits in the 4KB APNs limit.
        # Reserve 512 bytes for the rest of the payload structure.
        if assistant_content is not None:
            content_bytes = assistant_content.encode("utf-8")
            if len(content_bytes) <= 3584:
                payload["assistant_content"] = assistant_content

        payload_bytes = json.dumps(payload).encode("utf-8")
        expired_tokens: list[str] = []

        for device_token in list(self._devices.keys()):
            url = f"{_APNS_HOST}/3/device/{device_token}"
            headers = {
                "authorization": f"bearer {token}",
                "apns-topic": bundle_id,
                "apns-push-type": "alert",
                "apns-priority": "10",
            }
            try:
                resp = await client.post(url, content=payload_bytes, headers=headers)
                if resp.status_code == 410:
                    expired_tokens.append(device_token)
                    logger.info(f"Device token expired, removing: {device_token[:8]}...")
                elif resp.status_code != 200:
                    logger.warning(
                        f"APNs push failed for {device_token[:8]}...: "
                        f"HTTP {resp.status_code} {resp.text}"
                    )
            except Exception as exc:
                logger.warning(f"APNs push error for {device_token[:8]}...: {exc}")

        for token_to_remove in expired_tokens:
            self._devices.pop(token_to_remove, None)
        if expired_tokens:
            self._persist_devices()

    async def close(self) -> None:
        """Close the HTTP/2 client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # MARK: - Private

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(http2=True)
        return self._client

    def _get_jwt(self) -> str:
        """Return a cached JWT or create a fresh one (ES256 signed)."""
        now = time.time()
        if self._cached_jwt is not None and (now - self._jwt_issued_at) < _JWT_EXPIRY_SECONDS:
            return self._cached_jwt

        key_path = EXO_APNS_KEY_PATH
        key_id = EXO_APNS_KEY_ID
        team_id = EXO_APNS_TEAM_ID

        if key_path is None or key_id is None or team_id is None:
            raise RuntimeError("APNs environment variables not fully configured")

        with open(key_path) as f:
            signing_key = f.read()

        issued_at = int(now)
        headers = {"alg": "ES256", "kid": key_id}
        claims = {"iss": team_id, "iat": issued_at}
        token: str = jwt.encode(claims, signing_key, algorithm="ES256", headers=headers)

        self._cached_jwt = token
        self._jwt_issued_at = now
        return token

    def _load_devices(self) -> None:
        if not self._devices_file.exists():
            return
        try:
            data = json.loads(self._devices_file.read_text(encoding="utf-8"))
            for item in data:
                reg = DeviceRegistration.model_validate(item)
                self._devices[reg.device_token] = reg
            if self._devices:
                logger.info(f"Loaded {len(self._devices)} registered devices")
        except Exception as exc:
            logger.warning(f"Failed to load devices file: {exc}")

    def _persist_devices(self) -> None:
        self._devices_file.parent.mkdir(parents=True, exist_ok=True)
        data = [reg.model_dump() for reg in self._devices.values()]
        self._devices_file.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
