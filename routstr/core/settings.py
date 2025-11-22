from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel.ext.asyncio.session import AsyncSession


class Settings(BaseSettings):
    # --- UPDATED CONFIGURATION ---
    model_config = SettingsConfigDict(
        case_sensitive=False, 
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='',
        extra='ignore',
    )

    # --- UPDATED FIELD DEFINITIONS ---
    # Core
    upstream_base_url: str = Field(default="")
    upstream_api_key: str = Field(default="")
    admin_password: str = Field(default="")

    # Node info
    name: str = Field(default="ARoutstrNode")
    description: str = Field(default="A Routstr Node")
    npub: str = Field(default="")
    http_url: str = Field(default="")
    onion_url: str = Field(default="")

    # Cashu
    cashu_mints: str = Field(default="")
    receive_ln_address: str = Field(default="")
    primary_mint: str = Field(default="")
    primary_mint_unit: str = Field(default="sat")

    # Pricing
    fixed_pricing: bool = Field(default=False)
    fixed_cost_per_request: int = Field(default=1)
    fixed_per_1k_input_tokens: int = Field(default=0)
    fixed_per_1k_output_tokens: int = Field(default=0)
    exchange_fee: float = Field(default=1.005)
    upstream_provider_fee: float = Field(default=1.05)
    tolerance_percentage: float = Field(default=1.0)
    min_request_msat: int = Field(default=1)

    # Network
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    tor_proxy_url: str = Field(default="socks5://127.0.0.1:9050")
    providers_refresh_interval_seconds: int = Field(default=300)
    pricing_refresh_interval_seconds: int = Field(default=120)
    models_refresh_interval_seconds: int = Field(default=360)
    enable_pricing_refresh: bool = Field(default=True)
    enable_models_refresh: bool = Field(default=True)
    refund_cache_ttl_seconds: int = Field(default=3600)

    # Logging
    log_level: str = Field(default="INFO")
    enable_console_logging: bool = Field(default=True)

    # Other
    chat_completions_api_version: str = Field(default="")
    models_path: str = Field(default="models.json")
    source: str = Field(default="")

    # Secrets / optional runtime controls
    provider_id: str = Field(default="")
    nsec: str = Field(default="")

    # Discovery
    relays: list[str] = Field(default_factory=list)

    @field_validator('cashu_mints', mode='before')
    @classmethod
    def parse_comma_separated_string(cls, v: Any) -> str:
        """
        Ensures the field is always a string.
        The actual list conversion will happen where it's used.
        """
        if isinstance(v, list):
            # If for some reason it's already a list, join it back into a string
            return ",".join(v)
        if isinstance(v, str):
            return v.strip()
        return ""
    
def _compute_primary_mint(cashu_mints: list[str]) -> str:
    return cashu_mints[0] if cashu_mints else "https://mint.minibits.cash/Bitcoin"


def resolve_bootstrap() -> Settings:
    base = Settings()  # Reads env with custom parse_env_var
    # Back-compat env mapping
    try:
        # Map MODEL_BASED_PRICING -> fixed_pricing (inverted)
        if "MODEL_BASED_PRICING" in os.environ and "FIXED_PRICING" not in os.environ:
            mbp_raw = os.environ.get("MODEL_BASED_PRICING", "").strip().lower()
            mbp = mbp_raw in {"1", "true", "yes", "on"}
            base.fixed_pricing = not mbp
        # Map COST_PER_REQUEST -> fixed_cost_per_request if new not provided
        if (
            "COST_PER_REQUEST" in os.environ
            and "FIXED_COST_PER_REQUEST" not in os.environ
        ):
            try:
                base.fixed_cost_per_request = int(
                    os.environ["COST_PER_REQUEST"].strip()
                )
            except Exception:
                pass
        # Map COST_PER_1K_* -> CUSTOM_PER_1K_*
        if (
            "COST_PER_1K_INPUT_TOKENS" in os.environ
            and "FIXED_PER_1K_INPUT_TOKENS" not in os.environ
        ):
            try:
                base.fixed_per_1k_input_tokens = int(
                    os.environ["COST_PER_1K_INPUT_TOKENS"].strip()
                )
            except Exception:
                pass
        if (
            "COST_PER_1K_OUTPUT_TOKENS" in os.environ
            and "FIXED_PER_1K_OUTPUT_TOKENS" not in os.environ
        ):
            try:
                base.fixed_per_1k_output_tokens = int(
                    os.environ["COST_PER_1K_OUTPUT_TOKENS"].strip()
                )
            except Exception:
                pass
    except Exception:
        pass
    if not base.onion_url:
        try:
            from ..nip91 import discover_onion_url_from_tor  # type: ignore

            discovered = discover_onion_url_from_tor()
            if discovered:
                base.onion_url = discovered
        except Exception:
            pass
    # Derive NPUB from NSEC if not provided
    if not base.npub and base.nsec:
        try:
            from nostr.key import PrivateKey  # type: ignore

            if base.nsec.startswith("nsec"):
                pk = PrivateKey.from_nsec(base.nsec)
            elif len(base.nsec) == 64:
                pk = PrivateKey(bytes.fromhex(base.nsec))
            else:
                pk = None
            if pk is not None:
                try:
                    base.npub = pk.public_key.bech32()
                except Exception:
                    # Fallback to hex if bech32 not available
                    base.npub = pk.public_key.hex()
        except Exception:
            pass
    if not base.cors_origins:
        base.cors_origins = ["*"]
    if not base.primary_mint:
        base.primary_mint = _compute_primary_mint(base.cashu_mints)
    return base


class SettingsRow(BaseModel):
    id: int
    data: dict[str, Any]
    updated_at: datetime | None = None


# Single, concrete settings instance that callers import directly
settings: Settings = resolve_bootstrap()


class SettingsService:
    _current: Settings | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    def get(cls) -> Settings:
        if cls._current is None:
            raise RuntimeError("SettingsService not initialized")
        return cls._current

    @classmethod
    async def initialize(cls, db_session: AsyncSession) -> Settings:
        async with cls._lock:
            from sqlmodel import text

            await db_session.exec(  # type: ignore
                text(
                    "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, data TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
            )

            row = await db_session.exec(  # type: ignore
                text("SELECT id, data, updated_at FROM settings WHERE id = 1")
            )
            row = row.first()
            env_resolved = resolve_bootstrap()

            if row is None:
                await db_session.exec(  # type: ignore
                    text(
                        "INSERT INTO settings (id, data, updated_at) VALUES (1, :data, :updated_at)"
                    ).bindparams(
                        data=json.dumps(env_resolved.dict()),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await db_session.commit()
                cls._current = settings
                # Update the existing instance in-place for all live importers
                for k, v in env_resolved.dict().items():
                    setattr(settings, k, v)
                return cls._current

            db_id, db_data, _updated_at = row
            try:
                db_json = (
                    json.loads(db_data) if isinstance(db_data, str) else dict(db_data)
                )
            except Exception:
                db_json = {}

            merged_dict: dict[str, Any] = dict(env_resolved.dict())
            merged_dict.update(
                {k: v for k, v in db_json.items() if v not in (None, "", [], {})}
            )

            # Ensure primary_mint is consistent with cashu_mints if not explicitly set
            if not merged_dict.get("primary_mint"):
                merged_dict["primary_mint"] = _compute_primary_mint(
                    merged_dict.get("cashu_mints", [])
                )

            if any(k not in db_json for k in merged_dict.keys()):
                await db_session.exec(  # type: ignore
                    text(
                        "UPDATE settings SET data = :data, updated_at = :updated_at WHERE id = 1"
                    ).bindparams(
                        data=json.dumps(merged_dict),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await db_session.commit()

            # Update the existing instance in-place for all live importers
            for k, v in merged_dict.items():
                setattr(settings, k, v)
            cls._current = settings
            return cls._current

    @classmethod
    async def update(
        cls, partial: dict[str, Any], db_session: AsyncSession
    ) -> Settings:
        async with cls._lock:
            current = cls.get()
            candidate_dict = {**current.dict(), **partial}
            candidate = Settings(**candidate_dict)
            from sqlmodel import text

            # Ensure primary_mint reflects candidate mints if missing
            if not candidate.primary_mint:
                candidate.primary_mint = _compute_primary_mint(candidate.cashu_mints)

            await db_session.exec(  # type: ignore
                text(
                    "UPDATE settings SET data = :data, updated_at = :updated_at WHERE id = 1"
                ).bindparams(
                    data=json.dumps(candidate.dict()),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await db_session.commit()
            # Update in-place
            for k, v in candidate.dict().items():
                setattr(settings, k, v)
            cls._current = settings
            return settings

    @classmethod
    async def reload_from_db(cls, db_session: AsyncSession) -> Settings:
        async with cls._lock:
            from sqlmodel import text

            row = await db_session.exec(text("SELECT data FROM settings WHERE id = 1"))  # type: ignore
            row = row.first()
            if row is None:
                raise RuntimeError("Settings row missing")
            (data_str,) = row
            data = json.loads(data_str) if isinstance(data_str, str) else dict(data_str)
            # Update in-place
            for k, v in data.items():
                setattr(settings, k, v)
            cls._current = settings
            return settings
