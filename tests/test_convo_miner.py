import chromadb

from mempalace.convo_miner import mine_convos


def test_convo_mining(tmp_path):
    (tmp_path / "chat.txt").write_text(
        "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
    )

    palace_path = tmp_path / "palace"
    mine_convos(str(tmp_path), str(palace_path), wing="test_convos")

    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    assert col.count() >= 2

    # Verify search works
    results = col.query(query_texts=["memory persistence"], n_results=1)
    assert len(results["documents"][0]) > 0

    # Verify the trie was populated during ingestion.
    from mempalace.trie_index import TrieIndex, trie_db_path

    trie_db = trie_db_path(str(palace_path))
    assert (tmp_path / "palace" / "trie_index.lmdb").is_dir()
    trie = TrieIndex(db_path=trie_db)
    assert trie.stats()["postings"] > 0
    # "persistence" appears in the first exchange and isn't a stopword.
    assert len(trie.lookup("persistence")) >= 1
    trie.close()
