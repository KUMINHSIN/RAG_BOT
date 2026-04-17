from app.ingest import build_or_refresh_index


if __name__ == "__main__":
    stats = build_or_refresh_index()
    print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks.")
