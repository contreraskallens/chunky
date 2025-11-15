"""Module for allocating a corpus."""

import os
import re
from functools import reduce
from pathlib import Path

import duckdb
import pandas as pd

allowed_corpora = ["coca", "coca_sample", "bnc", "test"]
TEMP_DIR = Path("chunky/db/temp")
CORPUS_DIR = Path("chunky/db")
DEFAULT_CONFIG = {"memory_limit": 20, "cpu_cores": None}


def validate_corpus_name(name: str) -> bool:
    """Validate corpus name.

    - Must start with a letter (a-z, A-Z)
    - Can contain letters, underscores
    - Total length: 1-64 characters
    """
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", name))


def register_corpus(name: str) -> None:
    if validate_corpus_name(name):
        allowed_corpora.append(name)


def _quote_identifier(name: str) -> str:
    """Safely quote DuckDB identifier."""
    return f'"{name.replace('"', '""')}"'


def _is_valid_identifier(name: str) -> bool:
    """Validate identifier is safe alphanumeric format."""
    # Allow letters, numbers, underscores - must start with letter
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", name))


def _config_env(
    conn: duckdb.DuckDBPyConnection,
    env_config: dict = DEFAULT_CONFIG,
) -> None:
    memory_limit = env_config.get("memory_limit")
    cores = env_config.get("cpu_cores")
    cpu_count = os.cpu_count()
    if cores is not None:
        cpu_count = cores
    elif cpu_count is not None:
        cpu_count = cpu_count - 1
    else:
        cpu_count = 1
    conn.execute(f"SET threads TO {cpu_count}")
    conn.execute(f"SET memory_limit='{memory_limit}GB'")


def init_corpus(path: Path) -> None:
    """Initialize the corpus for first-time use."""
    with duckdb.connect(path) as conn:
        conn.execute("""
            CREATE TABLE ngram_db_temp
            (
                corpus TEXT,
                ug_1 UINT64,
                ug_2 UINT64,
                ug_3 UINT64,
                ug_4 UINT64,
                big_1 UINT64,
                trig_1 UINT64,
                freq INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE unigram_db_temp
            (
                corpus TEXT,
                ug TEXT,
                ug_hash UINT64,
                freq INTEGER
            )
        """)


def _get_chunk_dfs(ngrams: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_unigrams = ngrams["unigrams"]
    chunk_ngrams = ngrams["fourgrams"]
    chunk_unigrams = pd.DataFrame(
        chunk_unigrams,
        columns=[
            "corpus",
            "ug",
            "freq",
        ],
    )
    chunk_unigrams["corpus"] = chunk_unigrams["corpus"].astype(str)
    chunk_ngrams = pd.DataFrame(
        chunk_ngrams,
        columns=[
            "corpus",
            "ug_1",
            "ug_2",
            "ug_3",
            "ug_4",
            "freq",
        ],
    )
    chunk_ngrams["corpus"] = chunk_ngrams["corpus"].astype(str)
    chunk_ngrams["big_1"] = chunk_ngrams["ug_1"] + " " + chunk_ngrams["ug_2"]
    chunk_ngrams["trig_1"] = chunk_ngrams["big_1"] + " " + chunk_ngrams["ug_3"]
    return chunk_unigrams, chunk_ngrams


def add_chunk(path: Path, ngrams: dict) -> None:
    """Add ngram to the corpus.

    For use during allocation of ngram table. Takes unigram and
    ngram counts and adds them to the SQL database and the parquet files.
    Meant to be accessed by the processing_corpus methods.

    Args:
        ngram_lists (tuple): Tuple of unigram, fourgram frequency counts.

    """
    with duckdb.connect(path) as conn:
        chunk_unigrams, chunk_ngrams = _get_chunk_dfs(ngrams)

        conn.register("unigram_df", chunk_unigrams)
        conn.register("ngram_df", chunk_ngrams)
        conn.execute("""
            CREATE OR REPLACE TEMPORARY TABLE chunk_unigrams AS
            (
                SELECT
                    corpus,
                    ug,
                    HASH(ug) as ug_hash,
                    freq
                FROM
                    unigram_df
            )
        """)
        conn.execute("""
            CREATE OR REPLACE TEMPORARY TABLE chunk_ngrams AS
            (
                SELECT
                    corpus,
                    HASH(ug_1) AS ug_1,
                    HASH(ug_2) AS ug_2,
                    HASH(ug_3) AS ug_3,
                    HASH(ug_4) AS ug_4,
                    HASH(big_1) AS big_1,
                    HASH(trig_1) AS trig_1,
                    freq
                FROM
                    ngram_df
            )
        """)
        conn.execute("""
            INSERT INTO
                ngram_db_temp
            SELECT
                *
            FROM
                chunk_ngrams
        """)
        conn.execute("""
            INSERT INTO
                unigram_db_temp
            SELECT
                *
            FROM
                chunk_unigrams
        """)


def _get_valid_corpora(conn: duckdb.DuckDBPyConnection) -> list[str]:
    all_corpora = conn.execute(
        """
        SELECT DISTINCT
            corpus
        FROM
            ngram_db_temp
            """,
    ).fetchall()
    all_corpora = [str(corpus_list[0]) for corpus_list in all_corpora]
    all_corpora.sort()
    # Validate corpora names to avoid SQL injection
    valid_corpora = []
    for corpus in all_corpora:
        if not _is_valid_identifier(corpus):
            msg = f"{corpus} is not a valid corpus name"
            raise ValueError(msg)
        valid_corpora.append(_quote_identifier(corpus))
    return valid_corpora


def _pivot_tables(conn: duckdb.DuckDBPyConnection, valid_corpora: list[str]) -> None:
    corpora_query = reduce(lambda x, y: x + ", " + y, valid_corpora)
    temp_file = TEMP_DIR / "ngram_db_raw.parquet"
    temp_file = temp_file.resolve()
    pivot_ngram_q = f"""
    COPY (
        SELECT
            *
        FROM
            ngram_db_temp
        PIVOT(SUM(freq) FOR corpus IN ({corpora_query}))
    ) TO '{temp_file}' (FORMAT PARQUET)
    """  # noqa: S608 Insert was validated
    pivot_unigram_q = f"""
    CREATE OR REPLACE TABLE unigram_db AS (
        SELECT
            *
        FROM
            unigram_db_temp
        PIVOT(SUM(freq) FOR corpus IN ({corpora_query}))
    )
    """  # noqa: S608 Insert was validated
    conn.execute(pivot_unigram_q)
    conn.execute(pivot_ngram_q)
    conn.execute("DROP TABLE unigram_db_temp")
    conn.execute("DROP TABLE ngram_db_temp")
    conn.execute("VACUUM ANALYZE")


def _coalesce_corpus(conn: duckdb.DuckDBPyConnection) -> None:
    temp_read = TEMP_DIR / "ngram_db_raw.parquet"
    temp_read = temp_read.resolve()
    temp_write = TEMP_DIR / "ngram_db_coalesced.parquet"
    temp_write = temp_write.resolve()
    coalesce_ngrams = f"""
    COPY (
        SELECT
            ug_1,
            ug_2,
            ug_3,
            ug_4,
            big_1,
            trig_1,
            COALESCE(
                COLUMNS(* EXCLUDE(ug_1, ug_2, ug_3, ug_4, big_1, trig_1)),
                0
            )
        FROM
            READ_PARQUET('{temp_read}')
    ) TO '{temp_write}' (FORMAT PARQUET)
    """  # noqa: S608 Function internal strings
    coalesce_unigrams = """
            CREATE OR REPLACE TABLE unigram_db AS (
                SELECT
                    ug,
                    ug_hash,
                    COALESCE(COLUMNS(* EXCLUDE(ug, ug_hash)), 0)
            FROM
                unigram_db
            )
        """
    conn.execute(coalesce_ngrams)
    conn.execute(coalesce_unigrams)
    temp_read.unlink()


def _make_freqs(conn: duckdb.DuckDBPyConnection) -> None:
    temp_read = TEMP_DIR / "ngram_db_coalesced.parquet"
    temp_read = temp_read.resolve()
    temp_write = TEMP_DIR / "ngram_db_freq.parquet"
    temp_write = temp_write.resolve()
    ngram_freq = f"""
            COPY (
                SELECT
                    *,
                    LIST_SUM(
                        LIST_VALUE(
                            * COLUMNS(
                                * EXCLUDE (ug_1, ug_2, ug_3, ug_4, big_1, trig_1)
                                )
                            )
                        ) AS freq
                FROM
                    READ_PARQUET('{temp_read}')
            ) TO '{temp_write}' (FORMAT PARQUET)
            """  # noqa: S608 Function internal string
    unigram_freq = """
            CREATE OR REPLACE TABLE unigram_db AS (
                SELECT
                    *,
                    LIST_SUM(
                        LIST_VALUE(
                            *COLUMNS(
                                * EXCLUDE (ug, ug_hash)
                                )
                            )
                        ) AS freq
                FROM unigram_db
            )
        """
    conn.execute(ngram_freq)
    conn.execute(unigram_freq)
    temp_read.unlink()


def _sum_freqs(conn: duckdb.DuckDBPyConnection, valid_corpora: list) -> None:
    temp_read = TEMP_DIR / "ngram_db_freq.parquet"
    temp_read = temp_read.resolve()
    temp_write = TEMP_DIR / "ngram_db_summed.parquet"
    temp_write = temp_write.resolve()

    corpus_sum_query = [f"SUM({corpus}) AS {corpus}" for corpus in valid_corpora]
    corpus_sum_query = ",\n".join(corpus_sum_query)
    ngram_sum = f"""
        COPY (
            SELECT
                ug_1,
                ug_2,
                ug_3,
                ug_4,
                big_1,
                trig_1,
                {corpus_sum_query},
                SUM(freq) as freq
            FROM
                READ_PARQUET('{temp_read}')
            GROUP BY
                ug_1,
                ug_2,
                ug_3,
                ug_4,
                big_1,
                trig_1
        ) TO '{temp_write}' (FORMAT PARQUET)
            """  # noqa: S608 Corpus names are validated, temp files are internal
    unigram_sum = f"""
        CREATE OR REPLACE TABLE unigram_db AS(
            SELECT
                ug,
                ug_hash,
                {corpus_sum_query},
                SUM(freq) as freq
            FROM
                unigram_db
            GROUP BY
                ug,
                ug_hash
        )
            """  # noqa: S608 Corpus names are validated
    conn.execute(ngram_sum)
    conn.execute(unigram_sum)
    temp_read.unlink()


def _finalize_corpus(
    conn: duckdb.DuckDBPyConnection,
    corpus_name: str,
    threshold: int = 2,
) -> None:
    # Finalize unigrams first
    unigram_finalize = """
    CREATE OR REPLACE TABLE unigram_db AS (
        SELECT
            *
        FROM
            unigram_db
        WHERE
            freq > ?
    )
    """

    # Now ngrams
    temp_read = TEMP_DIR / "ngram_db_summed.parquet"
    temp_read = temp_read.resolve()
    if corpus_name not in allowed_corpora or not validate_corpus_name(corpus_name):
        msg = f"{corpus_name} is not an allowed corpus."
        raise ValueError(msg)

    corpus_dir = Path("chunky/db")
    ngram_file = corpus_dir / f"{corpus_name}_ngrams.parquet"
    ngram_file = ngram_file.resolve()

    # Check for relative to avoid directory traversal
    if not ngram_file.is_relative_to(corpus_dir.resolve()):
        msg = "Corpus path resolved outside corpus directory."
        raise ValueError(msg)

    # Filter on threshold and clean dummy trigrams
    ngram_finalize = f"""COPY (
            SELECT
                *
            FROM
                READ_PARQUET('{temp_read}')
            WHERE
                ug_1 != HASH('END')
                AND ug_2 != HASH('END')
                AND freq > ?
            ORDER BY
                ug_1,
                ug_2,
                ug_3,
                ug_4
        ) TO '{ngram_file}' (
            FORMAT PARQUET,
            CODEC 'zstd',
            COMPRESSION_LEVEL 10,
            ROW_GROUP_SIZE 200000,
            DICTIONARY_SIZE_LIMIT 100000,
            BLOOM_FILTER_FALSE_POSITIVE_RATIO 0.01
        )
        """  # noqa: S608 Already validated
    conn.execute(unigram_finalize, [threshold])
    conn.execute(ngram_finalize, [threshold])
    conn.execute("VACUUM ANALYZE")
    temp_read.unlink()


def _create_totals(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a table with the proportions of each corpus from total counts."""
    conn.execute(
        """
        CREATE OR REPLACE TABLE corpus_proportions AS (
            SELECT
                row_number() OVER () AS id,
                SUM(COLUMNS(* EXCLUDE(ug, ug_hash, freq))) / SUM(freq)
            FROM
            unigram_db
        )
    """,
    )
    conn.execute("ALTER TABLE corpus_proportions ADD PRIMARY KEY (id)")


def consolidate_corpus(
    path: Path,
    corpus_name: str = "test",
    threshold: int = 2,
    env_config: dict = DEFAULT_CONFIG,
) -> None:
    """Consolidate the temporary tables into the total ones.

    Ngram and unigram counts are added to the tables vertically.
    This method consolidates that table by summing frequency counts
    within each sub-corpus and tranposing the table to a horizontal format
    with one column per sub-corpus and a total frequency column.

    Args:
        threshold (int, optional): Minimum total token frequency
        for an ngram to be retained in the corpus. Defaults to 2.

    """
    if not TEMP_DIR.resolve().exists():
        TEMP_DIR.resolve().mkdir(parents=True)
    with duckdb.connect(path) as conn:
        _config_env(conn=conn, env_config=env_config)
        valid_corpora = _get_valid_corpora(conn)
        _pivot_tables(conn=conn, valid_corpora=valid_corpora)
        _coalesce_corpus(conn)
        _make_freqs(conn)
        _sum_freqs(conn=conn, valid_corpora=valid_corpora)
        _finalize_corpus(conn=conn, corpus_name=corpus_name, threshold=threshold)
        _create_totals(conn)
