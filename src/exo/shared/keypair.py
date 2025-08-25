from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import final

import base58
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from filelock import FileLock

from exo.shared.constants import EXO_NODE_ID_KEYPAIR


@final
class PeerId:
    """
    A libp2p peer identifier derived from a cryptographic public key.
    Compatible with py-libp2p's PeerID interface.
    """

    def __init__(self, peer_id_bytes: bytes) -> None:
        self._bytes = peer_id_bytes

    @staticmethod
    def from_bytes(data: bytes) -> "PeerId":
        """Create PeerId from raw bytes."""
        return PeerId(data)

    @staticmethod
    def from_public_key(public_key_bytes: bytes) -> "PeerId":
        """Create PeerId from a public key by hashing it."""
        # For Ed25519 keys, libp2p uses the identity hash (no hashing) for keys <= 42 bytes
        # Since Ed25519 public keys are 32 bytes, we use identity hash
        if len(public_key_bytes) <= 42:
            return PeerId(public_key_bytes)
        else:
            # For larger keys, use SHA-256
            hash_digest = hashlib.sha256(public_key_bytes).digest()
            return PeerId(hash_digest)

    def to_bytes(self) -> bytes:
        """Return the raw bytes of this PeerId."""
        return self._bytes

    def to_base58(self) -> str:
        """Return the base58-encoded string representation."""
        return base58.b58encode(self._bytes).decode("ascii")

    def __str__(self) -> str:
        """Return the base58-encoded string representation."""
        return self.to_base58()

    def __repr__(self) -> str:
        """Return debug representation."""
        return f"PeerId('{self.to_base58()}')"

    def __eq__(self, other: object) -> bool:
        """Check equality with another PeerId."""
        if not isinstance(other, PeerId):
            return False
        return self._bytes == other._bytes

    def __hash__(self) -> int:
        """Make PeerId hashable."""
        return hash(self._bytes)


@final
class Keypair:
    """
    A py-libp2p compatible keypair implementation.
    Provides the same interface as py-libp2p's KeyPair.
    """

    def __init__(self, private_key: ed25519.Ed25519PrivateKey) -> None:
        self._private_key = private_key
        self._public_key = private_key.public_key()

    @staticmethod
    def generate_ed25519() -> "Keypair":
        """Generate a new Ed25519 keypair."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        return Keypair(private_key)

    @staticmethod
    def from_protobuf_encoding(data: bytes) -> "Keypair":
        """
        Deserialize a keypair from libp2p protobuf encoding.
        Compatible with py-libp2p's serialization format.
        """
        if len(data) < 2:
            raise ValueError("Invalid protobuf data: too short")

        # Simple protobuf parsing for our specific use case
        # We expect: field 1 (type) as varint, field 2 (data) as bytes
        offset = 0

        # Parse type field (field tag 1, wire type 0 = varint)
        if data[offset] != 0x08:  # field 1, varint
            raise ValueError("Expected type field")
        offset += 1

        key_type = data[offset]
        offset += 1

        if key_type != 1:  # Ed25519
            raise ValueError(f"Unsupported key type: {key_type}")

        # Parse data field (field tag 2, wire type 2 = length-delimited)
        if offset >= len(data) or data[offset] != 0x12:  # field 2, bytes
            raise ValueError("Expected data field")
        offset += 1

        # Parse length
        data_length = data[offset]
        offset += 1

        if data_length not in (32, 64):
            raise ValueError(f"Invalid Ed25519 private key length: {data_length}")

        if offset + data_length > len(data):
            raise ValueError("Truncated private key data")

        key_data = data[offset : offset + data_length]

        try:
            if data_length == 64:
                # libp2p format: 32 bytes private key seed + 32 bytes public key
                private_key_seed = key_data[:32]
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                    private_key_seed
                )
            else:
                # Raw 32-byte private key
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_data)

            return Keypair(private_key)
        except Exception as e:
            raise ValueError(f"Invalid Ed25519 private key: {e}") from e

    def to_protobuf_encoding(self) -> bytes:
        """
        Serialize this keypair to libp2p protobuf encoding.
        Compatible with py-libp2p's serialization format.
        """
        private_key_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # libp2p Ed25519 format: private key seed (32) + public key (32)
        combined_key_data = private_key_bytes + public_key_bytes

        # Build protobuf manually for our simple case
        # Field 1 (type): tag=0x08, value=1 (Ed25519)
        # Field 2 (data): tag=0x12, length=64, data=combined_key_data
        result = bytearray()
        result.extend([0x08, 0x01])  # field 1: type = 1 (Ed25519)
        result.extend([0x12, 0x40])  # field 2: length = 64 bytes
        result.extend(combined_key_data)

        return bytes(result)

    def to_peer_id(self) -> PeerId:
        """Generate a PeerId from this keypair's public key."""
        public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        return PeerId.from_public_key(public_key_bytes)

    def sign(self, data: bytes) -> bytes:
        """Sign data with this keypair's private key."""
        return self._private_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature against data using this keypair's public key."""
        try:
            self._public_key.verify(signature, data)
            return True
        except Exception:
            return False

    @property
    def public_key_bytes(self) -> bytes:
        """Get the raw public key bytes."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

    @property
    def private_key_bytes(self) -> bytes:
        """Get the raw private key bytes."""
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    # py-libp2p compatibility properties
    @property
    def private_key(self) -> ed25519.Ed25519PrivateKey:
        """Access to the underlying private key for py-libp2p compatibility."""
        return self._private_key

    @property
    def public_key(self) -> ed25519.Ed25519PublicKey:
        """Access to the underlying public key for py-libp2p compatibility."""
        return self._public_key


def get_node_id_keypair(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes] = EXO_NODE_ID_KEYPAIR,
) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """

    def lock_path(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Path:
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        with open(path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_protobuf_encoding(protobuf_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logging.warning(
                        f"Encountered error when trying to get keypair: {e}"
                    )

        # if no valid credentials, create new ones and persist
        with open(path, "w+b") as f:
            keypair = Keypair.generate_ed25519()
            f.write(keypair.to_protobuf_encoding())
            return keypair
