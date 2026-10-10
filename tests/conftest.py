"""Shared fixtures. Tests run fully offline: no model downloads, no external APIs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Must be set before backend settings are first read.
os.environ.setdefault("APP_ENV", "test")
os.environ["GEMINI_API_KEY"] = ""
os.environ["MLFLOW_ENABLED"] = "false"
os.environ["HF_OFFLINE"] = "true"
os.environ["LOG_FORMAT"] = "text"
os.environ["LOG_LEVEL"] = "WARNING"

TOY_MODEL_NAME = "toy/tiny-gpt2"
TOY_VOCAB = [f"w{i}" for i in range(400)]


def build_toy_lm(name: str = TOY_MODEL_NAME, seed: int = 0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from backend.providers.hf_local import HFCausalLM

    vocab = {"<unk>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(TOY_VOCAB)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers import decoders

    tk.decoder = decoders.WordPiece(prefix="##")  # joins tokens with spaces
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")
    hf_tok.bos_token = "<eos>"
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=len(vocab), n_positions=512, n_embd=32, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=1
    )
    model = GPT2LMHeadModel(cfg)
    return HFCausalLM(model, hf_tok, name)


@pytest.fixture(scope="session")
def toy_lm():
    from backend.providers.hf_local import register_lm

    lm = build_toy_lm()
    register_lm(lm)
    return lm


def random_words(n: int, seed: int) -> str:
    import random

    rng = random.Random(seed)
    return " ".join(rng.choice(TOY_VOCAB) for _ in range(n))


@pytest.fixture()
def settings_env(tmp_path: Path, monkeypatch):
    """Isolated settings: fresh SQLite DB and artifact dirs per test."""
    from backend.core.config import reset_settings_cache
    from backend.db.session import get_engine

    db = tmp_path / "test.db"
    # TEST_DATABASE_URL runs the suite against e.g. PostgreSQL (schema is reset per test).
    url = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{db.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATASET_DIR", str(tmp_path / "datasets"))
    reset_settings_cache()
    get_engine.cache_clear()
    yield tmp_path
    reset_settings_cache()
    get_engine.cache_clear()


@pytest.fixture()
def db_session(settings_env):
    from backend.db.migrate import downgrade, upgrade
    from backend.db.session import get_sessionmaker

    if os.getenv("TEST_DATABASE_URL"):
        downgrade()
    upgrade()
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
