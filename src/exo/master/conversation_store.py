import json
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel


class ConversationMessage(BaseModel, frozen=True):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float
    image_ids: tuple[str, ...] = ()


class Conversation(BaseModel, frozen=True):
    conversation_id: str
    messages: tuple[ConversationMessage, ...] = ()
    model: str | None = None
    created_at: float
    updated_at: float


class ConversationStore:
    """Persistent conversation storage following the ImageStore pattern.

    Stores conversations as JSON files on disk with an in-memory index.
    Loaded on startup, persisted on every mutation.
    """

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._conversations: dict[str, Conversation] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        """Load all conversation JSON files from disk on startup."""
        count = 0
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                conv = Conversation.model_validate(data)
                self._conversations[conv.conversation_id] = conv
                count += 1
            except Exception as exc:
                logger.warning(f"Failed to load conversation {path.name}: {exc}")
        if count > 0:
            logger.info(f"Loaded {count} conversations from disk")

    def _persist(self, conversation: Conversation) -> None:
        """Write a conversation to disk as JSON."""
        path = self._storage_dir / f"{conversation.conversation_id}.json"
        path.write_text(
            conversation.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def append_message(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        model: str | None = None,
        image_ids: tuple[str, ...] = (),
    ) -> Conversation:
        """Append a message to a conversation, creating it if it doesn't exist."""
        now = time.time()
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=now,
            image_ids=image_ids,
        )

        existing = self._conversations.get(conversation_id)
        if existing is not None:
            conv = existing.model_copy(
                update={
                    "messages": (*existing.messages, msg),
                    "model": model or existing.model,
                    "updated_at": now,
                }
            )
        else:
            conv = Conversation(
                conversation_id=conversation_id,
                messages=(msg,),
                model=model,
                created_at=now,
                updated_at=now,
            )

        self._conversations[conversation_id] = conv
        self._persist(conv)
        return conv

    def delete(self, conversation_id: str) -> bool:
        """Remove a conversation from memory and disk."""
        conv = self._conversations.pop(conversation_id, None)
        if conv is None:
            return False
        path = self._storage_dir / f"{conversation_id}.json"
        if path.exists():
            path.unlink()
        return True

    def list_conversations(self) -> list[Conversation]:
        """Return all conversations sorted by updated_at descending."""
        return sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )

    def pinned_image_ids(self) -> set[str]:
        """Collect all image IDs referenced by any conversation.

        Used by ImageStore cleanup to avoid deleting conversation-linked images.
        """
        ids: set[str] = set()
        for conv in self._conversations.values():
            for msg in conv.messages:
                ids.update(msg.image_ids)
        return ids
