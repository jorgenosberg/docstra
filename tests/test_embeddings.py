from __future__ import annotations

import sys
import types

from docstra.core.ingestion.embeddings import (
    HuggingFaceEmbeddingGenerator,
    OllamaEmbeddingGenerator,
    OpenAIEmbeddingGenerator,
)


def test_huggingface_embedding_generator_uses_sentence_transformers_directly(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, trust_remote_code: bool = False) -> None:
            calls["model_name"] = model_name
            calls["trust_remote_code"] = trust_remote_code

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
            show_progress_bar: bool,
        ) -> list[list[float]]:
            calls["texts"] = texts
            calls["convert_to_numpy"] = convert_to_numpy
            calls["normalize_embeddings"] = normalize_embeddings
            calls["show_progress_bar"] = show_progress_bar
            return [[1.0, 2.0], [3.0, 4.0]]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    generator = HuggingFaceEmbeddingGenerator(model_name="demo-model")

    assert generator.generate_embeddings(["alpha", "beta"]) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert calls == {
        "model_name": "demo-model",
        "trust_remote_code": True,
        "texts": ["alpha", "beta"],
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }


def test_openai_embedding_generator_uses_openai_sdk_directly(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.embeddings = self

        def create(self, *, model: str, input: list[str]) -> types.SimpleNamespace:
            calls["model"] = model
            calls["input"] = input
            data = [types.SimpleNamespace(embedding=[0.5, 0.25, 0.125])]
            return types.SimpleNamespace(data=data)

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    generator = OpenAIEmbeddingGenerator(
        model_name="text-embedding-3-small",
        api_key="test-key",
        api_base="https://example.invalid/v1",
    )

    assert generator.generate_embedding("hello") == [0.5, 0.25, 0.125]
    assert calls == {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "model": "text-embedding-3-small",
        "input": ["hello"],
    }


def test_ollama_embedding_generator_uses_embed_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        del timeout
        calls.append((url, json))
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    generator = OllamaEmbeddingGenerator(
        model_name="nomic-embed-text",
        api_base="http://localhost:11434",
    )

    assert generator.generate_embeddings(["alpha", "beta"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert calls == [
        (
            "http://localhost:11434/api/embed",
            {
                "model": "nomic-embed-text",
                "input": ["alpha", "beta"],
                "truncate": True,
            },
        )
    ]


def test_ollama_embedding_generator_truncates_long_inputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embeddings": [[1.0, 0.0]]}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        del url, timeout
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    generator = OllamaEmbeddingGenerator(model_name="nomic-embed-text", max_chars=100)
    generator.generate_embedding("x" * 5000)

    assert len(captured["input"]) == 100


def test_ollama_embedding_generator_falls_back_to_legacy_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Fake404Response:
        status_code = 404

        def raise_for_status(self) -> None:
            return None

    class FakeLegacyResponse:
        status_code = 200

        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embedding": self.embedding}

    def fake_post(url: str, *, json: dict[str, object], timeout: float):
        del timeout
        calls.append((url, json))
        if url.endswith("/api/embed"):
            return Fake404Response()
        prompt = json["prompt"]
        if prompt == "alpha":
            return FakeLegacyResponse([1.0, 0.0])
        return FakeLegacyResponse([0.0, 1.0])

    monkeypatch.setattr("requests.post", fake_post)

    generator = OllamaEmbeddingGenerator(model_name="legacy-model")

    assert generator.generate_embeddings(["alpha", "beta"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert calls == [
        (
            "http://localhost:11434/api/embed",
            {"model": "legacy-model", "input": ["alpha", "beta"], "truncate": True},
        ),
        (
            "http://localhost:11434/api/embeddings",
            {"model": "legacy-model", "prompt": "alpha"},
        ),
        (
            "http://localhost:11434/api/embeddings",
            {"model": "legacy-model", "prompt": "beta"},
        ),
    ]
