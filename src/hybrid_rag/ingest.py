from pathlib import Path


def split_text(text, size=160, overlap=30):
    """Finestre di parole: la sovrapposizione conserva parte del contesto."""
    if not 0 <= overlap < size:
        raise ValueError("Servono size > 0 e 0 <= overlap < size.")
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def load_chunks(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Cartella documenti non trovata: {folder}")
    chunks = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            for number, content in enumerate(split_text(text), start=1):
                chunks.append({
                    "source": path.relative_to(folder).as_posix(),
                    "part": number,
                    "text": content,
                })
    if not chunks:
        raise ValueError("Non ci sono documenti .md o .txt con testo da indicizzare.")
    return chunks

