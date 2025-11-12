"""Module for the Corpus class."""

# TODO(omfgzell): Refactor functions to make them simpler #05
# TODO(omfgzell): Exception logic #06
# TODO(omfgzell): Type hint for kwargs #08
# TODO(omfgzell): Test individual measure methods #11

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd
import sqlalchemy as db
from psycopg import sql
from sqlalchemy import (
    MetaData,
    PrimaryKeyConstraint,
    Subquery,
    Table,
    case,
    func,
    inspect,
    literal,
    text,
)
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from tqdm import tqdm

from chunky.processing_corpus import make_processed_corpus

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta

logger = logging.getLogger(__name__)
TEMP_PATH = Path("chunky/db/temp/")


@dataclass
class NgramQuery:
    # ? Maybe I can add the batches here?
    """Class that gathers parameters for querying the corpus.

    Must be ngrams of the same length.

    Attributes:
        ngrams list[list[str]]: A list of ngrams in list form.
        e.g. [["this", "ngram"], ["that", "ngram"]]
        source (str): Identifier for the first half of the ngram.
        "ug_1" for bigrams, "big_1" for trigrams, "trig_1" for fourgrams.
        target (str): Identifier for the second half of the ngram.
        "ug_2" for bigrams, "ug_3" for trigrams, "ug_4" for fourgrams.
        length (int): The length of the ngrams.

    """

    query: Any
    freq_table: Any
    total_proportions: Any
    full_query: Any
    results: Any
    query_ref: Any
    ngrams: list
    source: str
    target: str
    length: int

    def __init__(
        self,
        ngrams: list,
        source: str,
        target: str,
        length: int,
    ) -> None:
        self.ngrams = ngrams
        self.source = source
        self.target = target
        self.length = length
        self.full_query = None
        self.total_proportions = None
        self.freq_table = None
        self.query = None
        self.results = None
        self.query_ref = None

    def update_results(
        self,
        new_results: Any,
    ) -> None:
        self.results = new_results
        if not isinstance(self.results, Subquery):
            self.results = self.results.subquery()


def get_column(table, column):
    if isinstance(table, Subquery) or isinstance(table, Table):
        return getattr(table.c, column)
    else:
        return getattr(table, column)


class Corpus:
    """Object containing a corpus database and methods to manipulate it.

    Implements methods that query the underlying DuckDB database.
    This class is not meant to be norma interfaced with. Instead, use a
    Helper.

    Attributes:
        corpus_name (str): The name of the corpus. This determines
        the input files of the processing, the processing method, and the database
        name.
        _path (Path): Path to the DuckDB .db file. Determined as {corpus_name}.db
        in the /db directory.
        _ngram_db (Path): Path to the parquet file containing ngram counts. Determined
        as {corpus_name}_ngrams.parquet in the /db directory.
        _temp (Path): Directory to store temporary files used in the processing and
        querying of the database.

    """

    def __init__(
        self,
        corpus_name: str,
        *,
        make: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize an instance of a Corpus.

            Sets the paths to the database, parquet file, and temp directory.
            Optionally can initialize a corpus that hasn't been allocated yet
            if make = True.

        Args:
            corpus_name (str): A string with the name of the corpus.
            make (bool, optional): Whether to process and allocate the corpus.
            Defaults to False.
            **kwargs: Arguments for the creation of the corpus if make=True.
            These are:
            corpus_dir (str): A string pointing to the corpus files.
            For BNC, the provided directory of the corpus_dir must be bnc_tokenized.txt.
            For CoCA, the provided directory of the corpus_dir must be the CoCa folder
            corpus files containing the corpus files e.g. text_acad_1990.txt.
            No corpus_dir is needed for the test corpus.
            chunk_size (int, optional): Size of the text chunk to be processed.
            Defaults to 1000000 lines for the BNC and 5 texts for the CoCA.
            threshold (int, optional): Minimum token frequency of each ngram to be
            stored in the corpus file. Defaults to 2.

        """
        self.corpus_name = corpus_name
        self._path = Path(f"chunky/db/{corpus_name}.db")
        self._engine = db.create_engine(
            f"duckdb:///{self._path}",
            echo=False,
        )
        session_maker = sessionmaker(bind=self._engine)
        self._session = session_maker()

        # print(repr(self._session))
        # metadata = db.MetaData()
        # self._db = db.Table(
        #     "unigram_db",
        #     metadata,
        # )
        # print(repr(self._db))
        # # print(self._engine.connect().execute(self._db.select()).fetchall())
        # self._session = Session(self._engine)
        # print(self._engine)
        # with self._session as session:
        #     result = session.execute(text("SELECT * FROM unigram_db LIMIT 100"))
        #     for row in result:
        #         print(row)

        self._temp = TEMP_PATH
        if not self._temp.exists():
            self._temp.mkdir(parents=True)

        self._ngram_db = Path(f"chunky/db/{self.corpus_name}_ngrams.parquet")
        if not self._path.is_file() and not make:
            error_message = "No corpus found. Make first or run with make = True"
            raise RuntimeError(error_message)
        if make:
            logger.info("Making corpus %s", corpus_name)
            corpus_make_dir = kwargs.get("corpus_dir")
            if corpus_make_dir is None and corpus_name != "test":
                exception_msg = """To make a corpus, provide the corresponding \
corpus_dir at initialization."""
                raise RuntimeError(exception_msg)

            self._init_corpus()
            make_processed_corpus(self, **kwargs)

        elif self._path.is_file():
            logger.info("Using preexisting corpus")

        else:
            error_message = "Something happened!!"
            raise RuntimeError

    def __call__(self, query: str) -> list:
        """Query the underlying database.

        Args:
            query (str): An SQL query.

        Returns:
            list: A list of tuples with the results of the query.

        """
        with duckdb.connect(self._path) as conn:
            query_result = conn.execute(query)
            return query_result.fetchall()

    def _init_corpus(self) -> None:
        """Initialize the corpus for first-time use."""
        with duckdb.connect(self._path) as conn:
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

    def query_parquet(self, ug: int | None = None) -> list:
        """Dev class. Eliminate."""
        with duckdb.connect(self._path) as conn:
            if ug is None:
                this_query = conn.execute(
                    f"EXPLAIN ANALYZE SELECT * FROM '{self._ngram_db}'"
                )
            else:
                this_query = conn.execute(
                    f"EXPLAIN ANALYZE SELECT * FROM '{self._ngram_db}' WHERE ug_1 = {ug}"
                )
            return this_query.fetchall()

    def show_ngrams(self, limit: int = 100) -> pd.DataFrame:
        """Show a sample of the ngram frequency table.

        Queries the ngram parquet file and shows a sample of rows from it.

        Args:
            limit (int, optional): Number of rows to show. Defaults to 100.

        Returns:
            pd.DataFrame: pandas Dataframe containing the rows of the ngram
            frequency table.

        """
        with duckdb.connect(self._path) as conn:
            ngram_db_query = """
                            SELECT
                                *
                            FROM
                                {}
                            LIMIT
                                {}
                            """
            ngram_db_query = (
                sql.SQL(ngram_db_query)
                .format(
                    f"{self._ngram_db}",
                    limit,
                )
                .as_string()
            )
            return conn.execute(ngram_db_query).df()

    def df(self, query: str, params: list | dict | None = None) -> pd.DataFrame:
        """Query the database and return as dataframe.

        Args:
            query (str): An SQL query.
            params (list | dict | None, optional): Optional parameters to include in
            the query.
            See DuckDB documentation on Prepared Statements for behavior.
            https://duckdb.org/docs/stable/clients/python/dbapi#prepared-statements.
            Defaults to None.

        Returns:
            pd.DataFrame: A pandas Dataframe containing the results of the query.
        """
        with duckdb.connect(self._path) as conn:
            if not params:
                this_query = conn.execute(query)
            else:
                logger.debug(query)
                logger.debug(params)
                this_query = conn.execute(query, params)
            return this_query.df()

    def add_chunk(self, ngram_lists: tuple) -> None:
        """Add ngram to the corpus.

        For use during allocation of ngram table. Takes unigram and
        ngram counts and adds them to the SQL database and the parquet files.
        Meant to be accessed by the processing_corpus methods.

        Args:
            ngram_lists (tuple): Tuple of unigram, fourgram frequency counts.

        """
        with duckdb.connect(self._path) as conn:
            chunk_unigrams, chunk_ngrams = ngram_lists
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

    def consolidate_corpus(self, threshold: int = 2) -> None:
        """Consolidate the temporary tables into the total ones.

        Ngram and unigram counts are added to the tables vertically.
        This method consolidates that table by summing frequency counts
        within each sub-corpus and tranposing the table to a horizontal format
        with one column per sub-corpus and a total frequency column.

        Args:
            threshold (int, optional): Minimum total token frequency
            for an ngram to be retained in the corpus. Defaults to 2.

        """
        # TODO(omfgzell): very long function #09
        with duckdb.connect(self._path) as conn:
            cpu_count = os.cpu_count()
            if cpu_count:
                conn.execute(f"SET threads TO {cpu_count - 1}")
            else:
                conn.execute("SET threads TO 1")
            conn.execute("SET memory_limit='20GB'")
            all_corpora_names = conn.execute(
                """
                SELECT DISTINCT
                    corpus
                FROM
                    ngram_db_temp
                    """,
            ).fetchall()
            all_corpora_names = [
                str(corpus_list[0]) for corpus_list in all_corpora_names
            ]
            all_corpora_names.sort()

            temp_fname = sql.Identifier(f"{self._temp}/ngram_db_raw.parquet")
            all_corpora = sql.SQL(", ").join(
                sql.Identifier(corpus) for corpus in all_corpora_names
            )
            create_temp_query = """
            COPY (
                SELECT
                    *
                FROM
                    ngram_db_temp
                PIVOT(SUM(freq) FOR corpus IN ({}))
            ) TO {} (FORMAT PARQUET)
            """
            create_temp_query = (
                sql.SQL(create_temp_query).format(all_corpora, temp_fname).as_string()
            )
            conn.execute(create_temp_query)

            conn.execute("DROP TABLE ngram_db_temp")
            conn.execute("VACUUM ANALYZE")
            pivot_query = """
            CREATE OR REPLACE TABLE unigram_db AS (
                SELECT
                    *
                FROM
                    unigram_db_temp
                PIVOT(SUM(freq) FOR corpus IN ({}))
            )
            """
            pivot_query = sql.SQL(pivot_query).format(all_corpora).as_string()
            conn.execute(pivot_query)
            conn.execute("DROP TABLE unigram_db_temp")
            conn.execute("VACUUM ANALYZE")
            # Replace NA with 0
            coalesce_query = """
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
                        {}
                ) TO {} (FORMAT PARQUET)
                """
            coalesce_query = (
                sql.SQL(coalesce_query)
                .format(
                    f"{self._temp}/ngram_db_raw.parquet",
                    f"{self._temp}/ngram_db_coalesced.parquet",
                )
                .as_string()
            )
            conn.execute(coalesce_query)
            Path(f"{self._temp}/ngram_db_raw.parquet").unlink()
            freq_query = """
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
                        {}
                ) TO {} (FORMAT PARQUET)
                """
            freq_query = (
                sql.SQL(freq_query)
                .format(
                    f"{self._temp}/ngram_db_coalesced.parquet",
                    f"{self._temp}/ngram_db_freq.parquet",
                )
                .as_string()
            )
            conn.execute(freq_query)
            Path(f"{self._temp}/ngram_db_coalesced.parquet").unlink()

            corpus_sum_query = [
                sql.SQL("SUM({corpus}) AS {corpus}").format(
                    corpus=sql.Identifier(corpus_name),
                )
                for corpus_name in all_corpora_names
            ]
            corpus_sum_query = sql.SQL(",\n").join(corpus_sum_query)
            sum_query = """
            COPY (
                SELECT
                    ug_1,
                    ug_2,
                    ug_3,
                    ug_4,
                    big_1,
                    trig_1,
                    {},
                    SUM(freq) as freq
                FROM
                    {}
                GROUP BY
                    ug_1,
                    ug_2,
                    ug_3,
                    ug_4,
                    big_1,
                    trig_1
            ) TO {} (FORMAT PARQUET)
                """
            sum_query = (
                sql.SQL(sum_query)
                .format(
                    corpus_sum_query,
                    f"{self._temp}/ngram_db_freq.parquet",
                    f"{self._temp}/ngram_db_summed.parquet",
                )
                .as_string()
            )
            conn.execute(sum_query)

            Path(f"{self._temp}/ngram_db_freq.parquet").unlink()

            conn.execute("""
                CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT
                        ug,
                        ug_hash,
                        COALESCE(COLUMNS(* EXCLUDE(ug, ug_hash)), 0)
                FROM
                    unigram_db
                )
            """)
            conn.execute("""
                CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT
                        *,
                        LIST_SUM(LIST_VALUE(*COLUMNS(* EXCLUDE (ug, ug_hash)))) AS freq
                    FROM unigram_db
                )
            """)
            conn.execute(
                """
                CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT
                        *
                    FROM
                        unigram_db
                    WHERE
                        freq > ?
                )
                """,
                [threshold],
            )
            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE unigram_drop AS (
                    SELECT
                        *
                    FROM
                        unigram_db
                    WHERE
                        freq <= ?
                )""",
                [threshold],
            )
            # Filter on threshold and clean dummy trigrams
            conn.execute("SET preserve_insertion_order=false")
            ngram_db_query = """
                COPY (
                    SELECT
                        *
                    FROM
                        {}
                    WHERE
                        ug_1 != HASH('END')
                        AND ug_2 != HASH('END')
                    ORDER BY
                        ug_1,
                        ug_2,
                        ug_3,
                        ug_4
                ) TO {} (
                    FORMAT PARQUET,
                    CODEC 'zstd',
                    COMPRESSION_LEVEL 10,
                    ROW_GROUP_SIZE 200000,
                    DICTIONARY_SIZE_LIMIT 100000,
                    BLOOM_FILTER_FALSE_POSITIVE_RATIO 0.01
                )
                """
            ngram_db_query = (
                sql.SQL(ngram_db_query)
                .format(
                    sql.Identifier(f"{self._temp}/ngram_db_summed.parquet"),
                    sql.Identifier(
                        f"chunky/db/{self.corpus_name}_ngrams.parquet",
                    ),
                )
                .as_string()
            )
            conn.execute(ngram_db_query)
            Path(f"{self._temp}/ngram_db_summed.parquet").unlink()
            conn.execute("VACUUM ANALYZE")

    def create_totals(self) -> None:
        """Create a table with the proportions of each corpus from total counts."""
        with duckdb.connect(self._path) as conn:
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

    # Below: methods related to MWU extraction directly.

    def _create_query(self, ngram_query: NgramQuery) -> None:
        """Allocate a list of ngrams as a query for filtering DB.

        Args:
            ngram_query (ngram_query): An ngram_query object containing ngrams and
            source/target information.

        """
        query_df = pd.DataFrame(
            ngram_query.ngrams,
            columns=[
                ngram_query.source,
                ngram_query.target,
            ],
        )
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        source_hash_identifier = sql.Identifier(f"{ngram_query.source}_hash")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        target_hash_identifier = sql.Identifier(f"{ngram_query.target}_hash")
        with duckdb.connect(self._path) as conn:
            conn.register("query_df", query_df)
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE query_ref (
                    id INT,
                    {source_id} TEXT,
                    {target_id} TEXT,
                    {source_hash} UINT64,
                    {target_hash} UINT64
                )
                """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                    source_hash=source_hash_identifier,
                    target_hash=target_hash_identifier,
                )
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                INSERT INTO
                    query_ref
                SELECT
                    row_number() OVER () as id,
                    {source_id},
                    {target_id},
                    HASH({source_id}) AS {source_hash},
                    HASH({target_id}) AS {target_hash}
                FROM
                    query_df
            """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                    source_hash=source_hash_identifier,
                    target_hash=target_hash_identifier,
                )
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE this_query AS (
                    SELECT
                        {source_hash} AS {source_id},
                        {target_hash} AS {target_id}
                FROM
                    query_ref
            )""")
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                    source_hash=source_hash_identifier,
                    target_hash=target_hash_identifier,
                )
                .as_string(),
            )

    def _make_token_freq(self, ngram_query: NgramQuery) -> None:
        """Make a table with token frequencies for the queried ngrams.

        This is required by all other supported measures.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams
            and source/target information.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        ngram_db = sql.Identifier(f"{self._ngram_db}")
        token_freq_query = """
        CREATE OR REPLACE TABLE token_freq AS
        SELECT
            {source_id},
            {target_id},
            SUM(freq) AS token_freq
        FROM
            this_query
        INNER JOIN {ngram_db} USING({source_id}, {target_id})
        GROUP BY
            {source_id}, {target_id}
                """  # noqa: S105
        token_freq_query = (
            sql.SQL(token_freq_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
                ngram_db=ngram_db,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(token_freq_query)
        # After getting token freq, we can filter non-occurring ngrams from the query

    def get_token_freq(self, ngram_query: NgramQuery) -> pd.DataFrame:
        """Obtain a table with ngram token frequency.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.

        Returns:
            pd.DataFrame: A pandas DataFrame containing token frequency information
            for all queried ngrams.

        """
        self._create_query(ngram_query)
        self._make_token_freq(ngram_query)
        return self.df("SELECT * FROM token_freq")

    def _reduce_query(self, ngram_query: NgramQuery) -> NgramQuery:
        """Make a reduced query table that includes only the occurring ngrams.

        This is a substantial memory and runtime save for later operations. It
        eliminates from the query all ngrams that do not occur in the corpus.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        ngram_db = sql.Identifier(f"{self._ngram_db}")

        reduce_query = (
            sql.SQL("""
            CREATE OR REPLACE TABLE reduced_query AS
            SELECT
                row_number() OVER () as id,
                {source_id} AS comp_1,
                {target_id} AS comp_2
            FROM
                this_query
                SEMI JOIN token_freq USING({source_id}, {target_id})
            ORDER BY comp_1, comp_2
            """)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )
        filter_query = (
            sql.SQL("""
                CREATE OR REPLACE TABLE filtered_db AS
                SELECT
                    {source_id} AS comp_1,
                    {target_id} AS comp_2,
                    SUM(
                        COLUMNS(* EXCLUDE(ug_1, ug_2, ug_3, ug_4, big_1, trig_1))
                    )
                FROM
                    READ_PARQUET({ngram_db})
                WHERE
                    {source_id} IN (
                        SELECT
                            comp_1
                        FROM
                            reduced_query
                    )
                    OR {target_id} IN (
                        SELECT
                            comp_2
                        FROM
                            reduced_query
                    )
                GROUP BY
                    comp_1,
                    comp_2
                """)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
                ngram_db=ngram_db,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(reduce_query)
            conn.execute("ANALYZE reduced_query")
            conn.execute(filter_query)
            conn.execute("ALTER TABLE filtered_db ADD PRIMARY KEY (comp_1, comp_2)")
            conn.execute("ALTER TABLE reduced_query ADD PRIMARY KEY (id)")
            conn.execute("ALTER TABLE query_ref ADD PRIMARY KEY (id)")

        metadata = MetaData()
        metadata.reflect(bind=self._engine)

        filtered_table = metadata.tables["filtered_db"]

        if not filtered_table.primary_key or not filtered_table.primary_key.columns:
            filtered_table.append_constraint(
                PrimaryKeyConstraint(filtered_table.c.comp_1, filtered_table.c.comp_2),
            )

        reduced_query = metadata.tables["reduced_query"]

        if not reduced_query.primary_key or not reduced_query.primary_key.columns:
            reduced_query.append_constraint(
                PrimaryKeyConstraint(reduced_query.c.id),
            )
        query_ref = metadata.tables["query_ref"]

        if not query_ref.primary_key or not query_ref.primary_key.columns:
            query_ref.append_constraint(
                PrimaryKeyConstraint(query_ref.c.id),
            )
        corpus_proportions = metadata.tables["corpus_proportions"]
        if (
            not corpus_proportions.primary_key
            or not corpus_proportions.primary_key.columns
        ):
            corpus_proportions.append_constraint(
                PrimaryKeyConstraint(corpus_proportions.c.id),
            )

        base = automap_base(metadata=metadata)
        base.prepare()

        ngram_query.query = base.classes.reduced_query
        ngram_query.results = ngram_query.query
        ngram_query.freq_table = base.classes.filtered_db
        ngram_query.total_proportions = corpus_proportions
        ngram_query.query_ref = query_ref
        return ngram_query

    def _join_with_query(
        self,
        query_table,
        result_table,
        result_name: str,
        alt_name: str | None = None,
    ):
        if alt_name is None:
            select_statement = select(
                query_table,
                get_column(result_table, result_name),
            )
        else:
            select_statement = select(
                query_table,
                get_column(result_table, result_name).label(alt_name),
            )
        return select_statement.join(
            result_table,
            (get_column(query_table, "comp_1") == get_column(result_table, "comp_1"))
            & (get_column(query_table, "comp_2") == get_column(result_table, "comp_2")),
        )

    def _get_type_freq_sa(self, ngram_query: NgramQuery) -> None:
        """Make a table with type frequencies for the queried ngrams.

        Args:
            ngram_query: An NgramQuery object containing ngrams and
            source/target information.

        """
        reduced_query = ngram_query.results
        db = ngram_query.freq_table
        type_1_query = select(
            get_column(db, "comp_2"),
            func.count().label("typef_1"),
        )
        type_1_query = type_1_query.group_by(get_column(db, "comp_2"))
        type_1_query = type_1_query.subquery()
        type_2_query = select(
            get_column(db, "comp_1"),
            func.count().label("typef_2"),
        )
        type_2_query = type_2_query.group_by(get_column(db, "comp_1"))
        type_2_query = type_2_query.subquery()
        results = select(
            reduced_query,
            get_column(type_1_query, "typef_1"),
            get_column(type_2_query, "typef_2"),
        )
        results = results.join(
            type_1_query,
            (get_column(reduced_query, "comp_2") == get_column(type_1_query, "comp_2")),
        ).join(
            type_2_query,
            (get_column(reduced_query, "comp_1") == get_column(type_2_query, "comp_1")),
        )
        ngram_query.update_results(results)

    def _get_prop_columns(
        self,
        corpus_columns,
        freq_column,
    ):
        return [(column / freq_column).label(column.name) for column in corpus_columns]

    def _get_kld(
        self,
        column_1,
        column_2,
    ):
        return case(
            ((column_1 == 0) | (column_2 == 0), 0),
            else_=column_1 * func.log2(column_1 / column_2),
        )

    def _get_distances(
        self,
        prop_columns: list,
        all_corpus_props,
    ):
        # distance to corpus proportion
        mapper_all = inspect(all_corpus_props)
        # Use scalar_subquery because corpus_proportion.X always has length=1
        return [
            self._get_kld(
                column,
                select(mapper_all.columns[column.name]).scalar_subquery(),
            ).label(column.name)
            for column in prop_columns
        ]

    def _sum_rows(self, columns):
        return reduce(lambda x, y: x + y, columns)

    def _normalize_kld(self, column):
        return 1 - func.pow(func.exp(1), -column)

    def _get_dispersion_column(
        self,
        corpus_columns,
        freqs,
        corpus_proportions,
    ):
        prop_columns = self._get_prop_columns(corpus_columns, freqs)
        distance_columns = self._get_distances(prop_columns, corpus_proportions)
        kld_column = self._sum_rows(distance_columns)
        return self._normalize_kld(kld_column)

    def _get_dispersion_sa(self, ngram_query: NgramQuery) -> None:
        """Make a table with a dispersion measure for the queried ngrams.

        Args:
            source (str): Identifier of the first half of the ngram.
            target (str): Identifier of the second half of the ngram.

        """
        # Should I make this reduced_table before and pass it down instead?
        reduced_query = ngram_query.results
        corpus_proportions = ngram_query.total_proportions
        db = ngram_query.freq_table

        reduced_table = select(
            db,
        ).join(
            reduced_query,
            (get_column(reduced_query, "comp_1") == get_column(db, "comp_1"))
            & (get_column(reduced_query, "comp_2") == get_column(db, "comp_2")),
        )

        reduced_table = reduced_table.subquery()

        dispersion_table = select(
            get_column(reduced_table, "comp_1"),
            get_column(reduced_table, "comp_2"),
            self._get_dispersion_column(
                [
                    column
                    for column in reduced_table.c
                    if column.name not in ["comp_1", "comp_2", "id", "freq"]
                ],
                get_column(reduced_table, "freq"),
                corpus_proportions,
            ).label("dispersion"),
        )
        dispersion_table = dispersion_table.subquery()
        results = self._join_with_query(
            reduced_query,
            dispersion_table,
            "dispersion",
        )
        ngram_query.update_results(results)

    def _get_rel_freqs(
        self,
        db,
        reduced_query,
    ):
        source_freq = select(
            get_column(db, "comp_1"),
            func.sum(get_column(db, "freq")).label("source_freq"),
        ).group_by(get_column(db, "comp_1"))
        source_freq = source_freq.subquery()
        target_freq = select(
            get_column(db, "comp_2"),
            func.sum(get_column(db, "freq")).label("target_freq"),
        ).group_by(get_column(db, "comp_2"))
        target_freq = target_freq.subquery()
        total_freq = self._session.execute(
            text("SELECT SUM(freq) AS total_freq FROM unigram_db"),
        ).fetchone()
        if total_freq is not None:
            total_freq = total_freq[0]
        else:
            exception_msg = "Oops something happened"
            raise RuntimeError(exception_msg)
        rel_freqs = select(
            reduced_query,
            literal(total_freq).label("total_freq"),
            get_column(source_freq, "source_freq"),
            get_column(target_freq, "target_freq"),
        )
        rel_freqs = rel_freqs.join(
            source_freq,
            (get_column(source_freq, "comp_1") == get_column(reduced_query, "comp_1")),
        )
        return rel_freqs.join(
            target_freq,
            (get_column(target_freq, "comp_2") == get_column(reduced_query, "comp_2")),
        )

    def _get_probs(
        self,
        rel_freq,
        token_freq,
    ):
        token_freq = token_freq.subquery()
        rel_freq = select(rel_freq, get_column(token_freq, "freq")).join(
            token_freq,
            (get_column(rel_freq, "comp_1") == get_column(token_freq, "comp_1"))
            & (get_column(rel_freq, "comp_2") == get_column(token_freq, "comp_2")),
        )
        rel_freq = rel_freq.subquery()
        probs = select(
            get_column(rel_freq, "comp_1"),
            get_column(rel_freq, "comp_2"),
            (get_column(rel_freq, "freq") / get_column(rel_freq, "source_freq")).label(
                "prob_2_1",
            ),
            (get_column(rel_freq, "freq") / get_column(rel_freq, "target_freq")).label(
                "prob_1_2",
            ),
            (
                get_column(rel_freq, "source_freq") / get_column(rel_freq, "total_freq")
            ).label("prob_1"),
            (
                get_column(rel_freq, "target_freq") / get_column(rel_freq, "total_freq")
            ).label("prob_2"),
        ).subquery()
        return select(
            probs,
            (1 - get_column(probs, "prob_2_1")).label("prob_no_2_1"),
            (1 - get_column(probs, "prob_1_2")).label("prob_no_1_2"),
            (1 - get_column(probs, "prob_1")).label("prob_no_1"),
            (1 - get_column(probs, "prob_2")).label("prob_no_2"),
        )

    def _get_normalized_kld(
        self,
        pair_1: tuple,
        pair_2: tuple,
    ):
        kld_1 = self._get_kld(*pair_1)
        kld_2 = self._get_kld(*pair_2)
        return self._normalize_kld(kld_1 + kld_2)

    def _get_associations_sa(self, ngram_query: NgramQuery) -> None:
        reduced_query = ngram_query.results
        db = ngram_query.freq_table
        rel_freq = self._get_rel_freqs(db, reduced_query).subquery()
        token_freq = select(reduced_query, db.freq).join(
            db,
            (get_column(reduced_query, "comp_1") == get_column(db, "comp_1"))
            & (get_column(reduced_query, "comp_2") == get_column(db, "comp_2")),
        )
        probs = self._get_probs(rel_freq, token_freq).subquery()
        fw_assoc = self._get_normalized_kld(
            (get_column(probs, "prob_2_1"), get_column(probs, "prob_2")),
            (get_column(probs, "prob_no_2_1"), get_column(probs, "prob_no_2")),
        )
        bw_assoc = self._get_normalized_kld(
            (get_column(probs, "prob_1_2"), get_column(probs, "prob_1")),
            (get_column(probs, "prob_no_1_2"), get_column(probs, "prob_no_1")),
        )

        assoc_table = select(
            get_column(probs, "comp_1"),
            get_column(probs, "comp_2"),
            fw_assoc.label("fw_assoc"),
            bw_assoc.label("bw_assoc"),
        )
        assoc_table = assoc_table.subquery()
        results = self._join_with_query(
            reduced_query,
            assoc_table,
            "fw_assoc",
        )
        results = results.subquery()
        results = self._join_with_query(
            results,
            assoc_table,
            "bw_assoc",
        )
        ngram_query.update_results(results)

    def _get_total_freq(
        self,
        reduced_query,
        db,
        column,
        *,
        cf: bool = False,
    ):
        id_columns = [get_column(db, "comp_1"), get_column(db, "comp_2")]
        if cf:
            id_columns.append(get_column(db, "target"))
        token_freq = select(
            *id_columns,
            get_column(db, "freq"),
        ).where(get_column(db, column).in_(select(get_column(reduced_query, column))))
        token_freq = token_freq.subquery()
        id_columns = [get_column(token_freq, column)]
        if cf:
            id_columns.append(get_column(token_freq, "target"))
        total_freq = select(
            *id_columns,
            func.sum(get_column(token_freq, "freq")).label("total_freq"),
        ).group_by(*id_columns)
        total_freq = total_freq.subquery()
        if cf:
            return select(
                token_freq,
                total_freq.c.total_freq,
            ).join(
                total_freq,
                (
                    (get_column(total_freq, column) == get_column(token_freq, column))
                    & (
                        get_column(total_freq, "target")
                        == get_column(total_freq, "target")
                    )
                ),
            )
        return select(
            token_freq,
            total_freq.c.total_freq,
        ).join(
            total_freq,
            (getattr(total_freq.c, column) == getattr(token_freq.c, column)),
        )

    def _get_info(
        self,
        freqs,
        total_freqs,
    ):
        prob = freqs / total_freqs
        info = func.log2(prob)
        return prob * info

    def _get_entropy(
        self,
        reduced_query,
        db,
        source_column: str,
        *,
        cf: bool = False,
    ):
        if cf:
            total_freq = self._get_total_freq(reduced_query, db, source_column, cf=True)
        else:
            total_freq = self._get_total_freq(reduced_query, db, source_column)
        total_freq = total_freq.subquery()

        weighted_info = select(
            total_freq,
            self._get_info(
                total_freq.c.freq,
                total_freq.c.total_freq,
            ).label("weighted_info"),
        )
        weighted_info = weighted_info.subquery()
        wi_id_columns = [get_column(weighted_info, source_column)]
        if cf:
            wi_id_columns.append(get_column(weighted_info, "target"))
        entropy = select(
            *wi_id_columns,
            (-func.sum(weighted_info.c.weighted_info)).label("raw_entropy"),
            func.count(weighted_info.c.weighted_info).label("n"),
        ).group_by(*wi_id_columns)
        entropy = entropy.subquery()

        ent_id_columns = [get_column(entropy, source_column)]
        if cf:
            ent_id_columns.append(get_column(entropy, "target"))
        return select(
            *ent_id_columns,
            (entropy.c.raw_entropy / func.log2(entropy.c.n)).label("entropy"),
        )

    def _get_mult_table(
        self,
        reduced_query,
        db,
        source_column,
        target_column,
    ):
        mult_table = select(
            get_column(reduced_query, source_column).label(source_column),
            get_column(reduced_query, target_column).label("target"),
            get_column(db, target_column).label(target_column),
            db.freq,
        ).join(
            db,
            (get_column(reduced_query, source_column) == get_column(db, source_column)),
        )
        mult_table = mult_table.subquery()
        return select(mult_table).where(
            mult_table.c.target != get_column(mult_table, target_column),
        )

    def _get_entropy_diff(
        self,
        entropy_real,
        entropy_cf,
        source_column: str,
        target_column: str,
    ):
        entropy_real = entropy_real.subquery()
        entropy_cf = entropy_cf.subquery()

        both_entropy = select(
            get_column(entropy_cf, source_column),
            get_column(entropy_cf, "target").label(target_column),
            entropy_real.c.entropy.label("entropy_real"),
            entropy_cf.c.entropy.label("entropy_cf"),
        ).join(
            entropy_real,
            (
                get_column(entropy_cf, source_column)
                == get_column(entropy_real, source_column)
            ),
        )
        both_entropy = both_entropy.subquery()
        return select(
            get_column(both_entropy, source_column),
            get_column(both_entropy, target_column),
            (both_entropy.c.entropy_cf - both_entropy.c.entropy_real).label(
                "entropy_diff",
            ),
        )

    def _get_entropy_sa(
        self,
        reduced_query,
        db,
        source_column,
        target_column,
    ):
        mult_table = self._get_mult_table(
            reduced_query,
            db,
            source_column,
            target_column,
        )
        mult_table = mult_table.subquery()
        entropy_real = self._get_entropy(reduced_query, db, source_column)
        entropy_cf = self._get_entropy(
            reduced_query,
            mult_table,
            source_column,
            cf=True,
        )
        return self._get_entropy_diff(
            entropy_real,
            entropy_cf,
            source_column,
            target_column,
        )

    def _get_entropies_sa(self, ngram_query: NgramQuery) -> None:
        reduced_query = ngram_query.results
        db = ngram_query.freq_table
        entropy_1 = self._get_entropy_sa(reduced_query, db, "comp_2", "comp_1")
        entropy_1 = entropy_1.subquery()
        entropy_2 = self._get_entropy_sa(reduced_query, db, "comp_1", "comp_2")
        entropy_2 = entropy_2.subquery()

        results = self._join_with_query(
            reduced_query,
            entropy_1,
            "entropy_diff",
            alt_name="entropy_1",
        )
        results = results.subquery()
        results = self._join_with_query(
            results,
            entropy_2,
            "entropy_diff",
            alt_name="entropy_2",
        )
        ngram_query.update_results(results)

    def _join_measures(self, ngram_query: NgramQuery) -> None:
        """Join all pre-allocated measures into a single table.

        Does it for a specified length to add an ngram_length field.

        Args:
            source (str): Identifier for the first half of the ngram.
            target (str): Identifier for the second half of the ngram.
            length (int): The length of the queried ngrams.

        """
        query_ref = ngram_query.query_ref
        results = ngram_query.results

        results = select(
            get_column(query_ref, ngram_query.source).label("comp_1"),
            get_column(query_ref, ngram_query.target).label("comp_2"),
            *[
                get_column(results, column.name)
                for column in results.c
                if column.name not in ["id", "comp_1", "comp_2"]
            ],
            literal(ngram_query.length).label("ngram_length"),
        ).join(
            results,
            (
                get_column(query_ref, ngram_query.source + "_hash")
                == get_column(results, "comp_1")
            )
            & (
                get_column(query_ref, ngram_query.target + "_hash")
                == get_column(results, "comp_2")
            ),
        )
        ngram_query.update_results(results)

    def _get_all_scores(self, ngram_query: NgramQuery) -> NgramQuery:
        """Allocate all measures for the ngrams in the query table.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        with tqdm(total=8, unit="step", leave=True) as pbar:
            pbar.update(1)
            logger.debug("Computing token frequencies...")
            self._make_token_freq(ngram_query)
            pbar.update(1)
            logger.debug("Making reduced table...")
            ngram_query = self._reduce_query(ngram_query)
            pbar.update(1)
            logger.debug("Computing type frequencies...")
            self._get_type_freq_sa(ngram_query)
            pbar.update(1)
            logger.debug("Computing dispersion...")
            self._get_dispersion_sa(ngram_query)
            pbar.update(1)
            logger.debug("Computing association...")
            self._get_associations_sa(ngram_query)
            pbar.update(1)
            logger.debug("Computing entropy...")
            self._get_entropies_sa(ngram_query)
            pbar.update(1)
            logger.debug("Joining results...")
            self._join_measures(ngram_query)
            print(pd.read_sql(select(ngram_query.results), self._engine))
            pbar.update(1)
            return ngram_query

    def get_scores(self, ngram_query: NgramQuery) -> pd.DataFrame:
        """Compute all ngram measures for a given set of ngrams.

        Given a list of ngrams, compute and obtain all ngram measures on the list.
        This includes token frequency, dispersion, type frequencies, associations,
        and entropy differences.
        All ngrams must be of the same length.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.

        Returns:
            pd.DataFrame: A pandas DataFrame containing all raw MWU measures
            for the queried ngrams.

        """
        self._create_query(ngram_query)
        ngram_query = self._get_all_scores(ngram_query)
        return ngram_query.results
