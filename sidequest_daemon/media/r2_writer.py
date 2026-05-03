"""Direct daemon → R2 artifact uploader.

Per CLAUDE.md no-silent-fallback: any boto error propagates. The daemon's
caller is responsible for surfacing the failure to the server, which emits
an image_unavailable event. We never hand back a fake URL.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Final, Literal

import boto3
from botocore.client import BaseClient

ArtifactKind = Literal["portraits", "poi", "scenes", "music", "sfx"]
_VALID_KINDS: Final[frozenset[str]] = frozenset(
    {"portraits", "poi", "scenes", "music", "sfx"}
)

_EXT_FOR_CONTENT_TYPE: Final[dict[str, str]] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/flac": "flac",
}

CACHE_CONTROL_ARTIFACTS: Final[str] = "public, max-age=86400"
BUCKET: Final[str] = "sidequest"


@lru_cache(maxsize=1)
def _client() -> BaseClient:
    """Lazy boto3 client singleton; respects env mutation across tests
    via the patch.object(r2_writer, "_client", ...) idiom."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_artifact(
    *,
    world_slug: str,
    session_id: str,
    kind: ArtifactKind,
    content_bytes: bytes,
    content_type: str,
) -> str:
    """Upload `content_bytes` to R2 under
    ``artifacts/<world>/<session>/<kind>/<sha256>.<ext>``.

    Returns the relative key. Raises ValueError on invalid kind/content_type.
    Raises any boto3/HTTP error verbatim — caller must propagate, not swallow.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
        )
    if content_type not in _EXT_FOR_CONTENT_TYPE:
        raise ValueError(
            f"content_type must be one of {sorted(_EXT_FOR_CONTENT_TYPE)}, "
            f"got {content_type!r}"
        )

    ext = _EXT_FOR_CONTENT_TYPE[content_type]
    sha = hashlib.sha256(content_bytes).hexdigest()
    key = f"artifacts/{world_slug}/{session_id}/{kind}/{sha}.{ext}"

    _client().put_object(
        Bucket=BUCKET,
        Key=key,
        Body=content_bytes,
        ContentType=content_type,
        CacheControl=CACHE_CONTROL_ARTIFACTS,
    )
    return key
