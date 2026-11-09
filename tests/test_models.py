"""Tests for lazy model loading and dependency injection in embed and reranker."""

from unittest.mock import MagicMock, patch

from rag import embed, reranker
from rag.retrieve import Hit


def test_embed_module_does_not_instantiate_model_until_called() -> None:
    embed.set_embedder(None)
    assert embed._model is None

    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.1, 0.2, 0.3]]

    with patch("rag.embed.SentenceTransformer", return_value=fake_model) as mock_cls:
        out = embed.embed_batch(["hello world"])
        mock_cls.assert_called_once()
        assert out == [[0.1, 0.2, 0.3]]

        # Subsequent call reuses cached singleton
        embed.embed_batch(["second call"])
        mock_cls.assert_called_once()

    embed.set_embedder(None)


def test_embed_batch_accepts_injected_embedder() -> None:
    class DummyEmbedder:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return [[float(len(t))] * 3 for t in texts]

    out = embed.embed_batch(["abc", "de"], embedder=DummyEmbedder())
    assert out == [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]


def test_embed_batch_empty_input_returns_empty_list() -> None:
    embed.set_embedder(None)
    with patch("rag.embed.SentenceTransformer") as mock_cls:
        assert embed.embed_batch([]) == []
        mock_cls.assert_not_called()


def test_reranker_module_does_not_instantiate_model_until_called() -> None:
    reranker.set_reranker(None)
    assert reranker._reranker is None

    fake_ce = MagicMock()
    fake_ce.predict.return_value = [0.2, 0.9]
    hits = [
        Hit(chunk_id=1, source="a.md", text="first", score=0.5),
        Hit(chunk_id=2, source="b.md", text="second", score=0.4),
    ]

    with patch("rag.reranker.CrossEncoder", return_value=fake_ce) as mock_cls:
        ranked = reranker.rerank("query", hits, k=1)
        mock_cls.assert_called_once()
        assert len(ranked) == 1
        assert ranked[0].chunk_id == 2
        assert ranked[0].score == 0.9

        # Subsequent call reuses cached singleton
        reranker.rerank("query", hits, k=2)
        mock_cls.assert_called_once()

    reranker.set_reranker(None)


def test_rerank_accepts_injected_reranker() -> None:
    class DummyReranker:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(i) for i in range(len(pairs))]

    hits = [
        Hit(chunk_id=10, source="a.md", text="low", score=0.9),
        Hit(chunk_id=20, source="b.md", text="high", score=0.1),
    ]
    ranked = reranker.rerank("q", hits, k=2, reranker=DummyReranker())
    assert [h.chunk_id for h in ranked] == [20, 10]
    assert ranked[0].score == 1.0


def test_rerank_empty_hits_returns_empty_list() -> None:
    reranker.set_reranker(None)
    with patch("rag.reranker.CrossEncoder") as mock_cls:
        assert reranker.rerank("q", [], k=5) == []
        mock_cls.assert_not_called()
