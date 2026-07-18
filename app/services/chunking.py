"""Text chunking with configurable size and overlap.

Splits text into overlapping chunks while respecting natural text boundaries
(paragraphs, lines, sentences) to preserve semantic coherence within chunks.
"""


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks, respecting natural boundaries.

    Uses a sliding-window approach that looks backward from each cut point
    to find the nearest paragraph, line, sentence, or word boundary.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks. Empty list if input is empty.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If the remaining text fits in one chunk, take it all
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find a natural break point near the end of this chunk
        break_at = _find_break_point(text, start, end)
        chunk = text[start:break_at].strip()
        if chunk:
            chunks.append(chunk)

        # Advance the window, stepping back by overlap amount
        start = max(start + 1, break_at - chunk_overlap)

    return chunks


def _find_break_point(text: str, start: int, end: int) -> int:
    """Find the best position to split text near ``end``.

    Searches backward from ``end`` within the last 20% of the chunk
    (minimum 50 characters) for the nearest natural boundary.

    Priority order: paragraph break > line break > sentence end > word boundary.
    Falls back to a hard cut at ``end`` if no boundary is found.
    """
    search_start = end - max(50, (end - start) // 5)
    window = text[search_start:end]

    # Paragraph break (double newline)
    idx = window.rfind("\n\n")
    if idx != -1:
        return search_start + idx + 2

    # Line break
    idx = window.rfind("\n")
    if idx != -1:
        return search_start + idx + 1

    # Sentence boundary
    for delimiter in (". ", "! ", "? "):
        idx = window.rfind(delimiter)
        if idx != -1:
            return search_start + idx + len(delimiter)

    # Word boundary
    idx = window.rfind(" ")
    if idx != -1:
        return search_start + idx + 1

    # No good boundary found — hard cut
    return end
