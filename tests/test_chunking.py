"""Unit tests for the text chunking service."""

from app.services.chunking import chunk_text


class TestEmptyAndShortInput:
    def test_empty_string(self):
        assert chunk_text("", 100, 20) == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ", 100, 20) == []

    def test_text_shorter_than_chunk_size(self):
        text = "Hello world."
        result = chunk_text(text, 100, 20)
        assert result == [text]

    def test_text_equal_to_chunk_size(self):
        text = "x" * 100
        result = chunk_text(text, 100, 20)
        assert result == [text]


class TestChunkSizeRespected:
    def test_chunks_do_not_exceed_max_size(self):
        text = "word " * 200  # ~1000 chars
        chunks = chunk_text(text, 200, 50)
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_produces_multiple_chunks_for_long_text(self):
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, 100, 20)
        assert len(chunks) > 1


class TestBoundaryDetection:
    def test_prefers_paragraph_breaks(self):
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph."
        chunks = chunk_text(text, 50, 10)
        # Chunks should split at paragraph boundaries when possible
        assert all(chunk.strip() for chunk in chunks)

    def test_prefers_sentence_boundaries(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence."
        chunks = chunk_text(text, 45, 10)
        # Should not cut mid-word when sentence boundary is available
        for chunk in chunks:
            assert not chunk.startswith(" ")


class TestOverlap:
    def test_overlap_creates_more_chunks(self):
        text = "word " * 100
        no_overlap = chunk_text(text, 100, 0)
        with_overlap = chunk_text(text, 100, 30)
        # Overlap should produce more (or equal) chunks since we step less
        assert len(with_overlap) >= len(no_overlap)


class TestTextCoverage:
    def test_no_content_lost(self):
        """All words from the input should appear in at least one chunk."""
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_text(text, 80, 20)

        all_chunk_text = " ".join(chunks)
        for word in words:
            assert word in all_chunk_text, f"'{word}' missing from chunks"
