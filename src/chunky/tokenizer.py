import duckdb as db
import pandas as pd
from pathlib import Path
from pprint import pprint as print

temp_dir = Path("db/temp")
token_dir = temp_dir / "tokens"
count_dir = temp_dir / "counts"


def _load_corpus():
    conn = db.connect(str(Path("db/temp/corpus_cleaning.db").resolve()))
    return conn


def _tokenize_sentences(conn):
    # TODO: This can be batched
    n_parts = conn.execute("SELECT count(DISTINCT part_id) FROM part_ids").fetchone()[0]
    for this_part in range(1, n_parts + 1):
        # for this_part in [1, 2, 3, 4]:
        tokens_path = token_dir / (str(this_part) + "_tokens.parquet")
        count_path = count_dir / (str(this_part) + "_counts.parquet")
        conn.execute(
            """
            CREATE OR REPLACE TEMPORARY TABLE part_tokens AS
            SELECT part_id, sentence_id, tokens.text, tokens.pos
            FROM(
                SELECT part_id,
                    sentence_id,
                    list_filter(
                        string_split(text, ' '),
                        lambda x: x <> ''
                    ) AS token,
                FROM part_sentences
                WHERE part_id = ?
            ), unnest(token) WITH ORDINALITY AS tokens(text, pos)
        """,
            [this_part],
        )
        conn.execute(
            f"""
            COPY(
                SELECT * FROM part_tokens
            ) TO '{tokens_path}' (FORMAT PARQUET)
            """
        )
        conn.execute(
            f"""
            COPY(
                SELECT part_id, text, COUNT(*) AS frequency
                FROM part_tokens
                GROUP BY part_id, text
                ORDER BY frequency DESC
            ) TO '{count_path}' (FORMAT PARQUET)
            """
        )
    # TODO: Can probably delete the db now


def _build_vocabulary(conn):
    # This doesn't need to be the same conn
    all_counts = count_dir / "*.parquet"
    vocab_path = temp_dir / "vocabulary.parquet"
    conn.execute(f"""
        COPY(
        SELECT row_number() OVER (ORDER BY SUM(frequency) DESC) AS token_id,
            text,
            SUM(frequency) AS frequency
        FROM read_parquet('{all_counts}')
        GROUP BY text
        ORDER BY frequency DESC
        ) TO '{vocab_path}' (FORMAT PARQUET)
    """)


# TODO: Join with vocabulary and delete text, use only token_id
# Follow parquet file per part strategy: corpus folder with vocabulary.parquet and folders: ngram/ [1 / 2 / 3 / 4] / npart_1grams.parquet, etc.


# TODO: Store frequency, type frequency, and sum freq log2freq for each slot
def _unigram_stats(conn):
    # TODO: this should be built last to add the other n-gram stats to it
    # Including _, token; token, _; _, _, token; _, _, _, token.
    pass


def _bigram_stats(conn):
    # TODO: any programmatic way to get bigram, trigram, etc with a parameter?
    # Structure: Token_id, token_id
    pass


def _trigram_stats(conn):
    # TODO: Structure: bigram_id, token,
    pass


def fourgram_stats(conn):
    # TODO: Structure: trigram_id, token
    pass


if __name__ == "__main__":
    conn = _load_corpus()
    _tokenize_sentences(conn)
    _build_vocabulary(conn)
