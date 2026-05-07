"""Environment-backed settings for local scripts and the future app."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_key: str | None
    hf_key: str | None
    default_openai_model: str
    default_hf_model: str

    def __repr__(self) -> str:
        return (
            "Settings("
            f"openai_key={_mask(self.openai_key)}, "
            f"hf_key={_mask(self.hf_key)}, "
            f"default_openai_model={self.default_openai_model!r}, "
            f"default_hf_model={self.default_hf_model!r}"
            ")"
        )


def _mask(value: str | None) -> str:
    if not value:
        return "None"
    return "'***'"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openai_key=os.getenv("OPENAI_KEY"),
        hf_key=os.getenv("HF_KEY"),
        default_openai_model=os.getenv(
            "DEFAULT_OPENAI_MODEL",
            "text-embedding-3-large",
        ),
        default_hf_model=os.getenv(
            "DEFAULT_HF_MODEL",
            "ibm-granite/granite-embedding-97m-multilingual-r2",
        ),
    )


settings = load_settings()
