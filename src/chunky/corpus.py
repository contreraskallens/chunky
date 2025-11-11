"""Module for the Corpus class."""

# TODO(omfgzell): Refactor functions to make them simpler #05
# TODO(omfgzell): Exception logic #06
# TODO(omfgzell): Type hint for kwargs #08
# TODO(omfgzell): Test individual measure methods #11

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql import select
from sqlalchemy import (
    create_engine,
    MetaData,
    BigInteger,
    CHAR,
    Column,
    DateTime,
    Float,
    Integer,
    SmallInteger,
    String,
    Table,
    Unicode,
    text,
    inspect,
    PrimaryKeyConstraint,
    func,
)
import duckdb
import pandas as pd
from psycopg import sql
from tqdm import tqdm

from chunky.processing_corpus import make_processed_corpus

if TYPE_CHECKING:
    from chunky.corpus_helper import NgramQuery

logger = logging.getLogger(__name__)
TEMP_PATH = Path("chunky/db/temp/")


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
                        SUM(COLUMNS(* EXCLUDE(ug, ug_hash, freq))) / SUM(freq)
                    FROM
                    unigram_db
                )
            """,
            )

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

    def _reduce_query(self, ngram_query: NgramQuery) -> None:
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

        print(self.df("SELECT * FROM reduced_query ORDER BY comp_1"))
        print(self.df("SELECT * FROM filtered_db ORDER BY comp_1"))
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

        base = automap_base(metadata=metadata)
        base.prepare()
        return base.classes.reduced_query, base.classes.filtered_db

    def _make_type_freq_sa(
        self, reduced_query: DeclarativeMeta, db: DeclarativeMeta
    ) -> None:
        """Make a table with type frequencies for the queried ngrams.

        Args:
            ngram_query: An NgramQuery object containing ngrams and
            source/target information.

        """

        type_1_query = select(db.comp_2, func.count().label("typef_1"))
        type_1_query = type_1_query.group_by(db.comp_2)
        type_1_query = type_1_query.subquery()
        type_2_query = select(db.comp_1, func.count().label("typef_2"))
        type_2_query = type_2_query.group_by(db.comp_1)
        type_2_query = type_2_query.subquery()
        type_freq_query = (
            select(
                reduced_query,
                type_1_query.c.typef_1,
                type_2_query.c.typef_2,
            )
            .join(
                type_1_query,
                reduced_query.comp_2 == type_1_query.c.comp_2,
            )
            .join(
                type_2_query,
                reduced_query.comp_1 == type_2_query.c.comp_1,
            )
        )

        return type_freq_query

    def _make_type_freq(self, ngram_query: NgramQuery) -> None:
        """Make a table with type frequencies for the queried ngrams.

        Args:
            ngram_query: An NgramQuery object containing ngrams and
            source/target information.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        type_1_query = (
            sql.SQL("""
            CREATE OR REPLACE TEMPORARY TABLE type_1 AS
            SELECT
                {target_id},
                COUNT( * ) AS typef_1
            FROM
                filtered_db
            GROUP BY
                {target_id}
        """)
            .format(target_id=target_identifier)
            .as_string()
        )
        type_2_query = (
            sql.SQL("""
            CREATE OR REPLACE TEMPORARY TABLE type_2 AS
            SELECT
                {source_id},
                COUNT( * ) AS typef_2
            FROM
                filtered_db
            GROUP BY
                {source_id}
        """)
            .format(source_id=source_identifier)
            .as_string()
        )

        type_freq_query = """
            CREATE OR REPLACE TABLE type_freq AS
            WITH
                type_1 AS (
                    SELECT
                        {target_id},
                        COUNT( * ) AS typef_1
                    FROM
                        filtered_db
                    GROUP BY
                        {target_id}
                ),
                type_2 AS (
                    SELECT
                        {source_id},
                        COUNT( * ) AS typef_2
                    FROM
                        filtered_db
                    GROUP BY
                        {source_id}
                ),
                freq_1_reduced AS (
                    SELECT
                        *
                    FROM
                        reduced_query
                        LEFT JOIN type_1 USING ({target_id})
            ),
            type_freq_temp AS (
                SELECT
                    *
                FROM
                    freq_1_reduced
                    LEFT JOIN type_2 USING ({source_id})
            )
            SELECT
                *
            FROM
                reduced_query
                LEFT JOIN type_freq_temp USING({source_id}, {target_id})
        """
        type_freq_query = (
            sql.SQL(type_freq_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(type_1_query)
            conn.execute(type_2_query)
            conn.execute(type_freq_query)

    def get_type_freq(
        self,
        ngram_query: NgramQuery,
        *,
        standalone: bool = True,
    ) -> pd.DataFrame:
        """Obtain a table with ngram type frequencies.

        Type frequency is calculated separately for the first ("typef_1") and the second
        ("typef_2") components of the ngram.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.

            standalone (bool, optional): Whether the method is being used only to obtain
            type frequencies or as a part of a sequence of operations. Set to False if
            you have already obtained token frequencies for this set of ngrams.
            Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame containing type frequency information
            for all queried ngrams.

        """
        if standalone:
            self._create_query(ngram_query)
            self._make_token_freq(ngram_query)
            self._reduce_query(ngram_query)
        self._make_type_freq(ngram_query)
        return self.df("SELECT * FROM type_freq")

    def _make_dispersion(self, ngram_query: NgramQuery) -> None:
        """Make a table with a dispersion measure for the queried ngrams.

        Args:
            source (str): Identifier of the first half of the ngram.
            target (str): Identifier of the second half of the ngram.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        prop_table_query = (
            sql.SQL("""
            CREATE OR REPLACE TABLE prop_table AS
            SELECT
                {source_id},
                {target_id},
                COLUMNS(* EXCLUDE({source_id}, {target_id}, freq)) / freq
            FROM
                filtered_db
                RIGHT JOIN reduced_query USING({source_id}, {target_id})
        """)
            .format(source_id=source_identifier, target_id=target_identifier)
            .as_string()
        )

        with duckdb.connect(self._path) as conn:
            conn.execute(prop_table_query)
            corpus_names = conn.execute(
                """
                SELECT
                    column_name
                FROM
                    information_schema.columns
                WHERE
                    table_name = 'filtered_db'
                    AND column_name NOT IN (
                        'ug_1',
                        'ug_2',
                        'ug_3',
                        'ug_4',
                        'big_1',
                        'trig_1',
                        'freq'
                    )
                """,
            ).fetchall()

        corpus_names = [name[0] for name in corpus_names]
        corpus_names.sort()

        corpus_query = [
            sql.SQL("""
                    CASE
                        WHEN
                            prop_table.{corpus} > 0
                        THEN
                            prop_table.{corpus} * log2(prop_table.{corpus} / corpus_proportions.{corpus})
                        ELSE 0
                        END AS
                            {corpus}
                    """).format(  # noqa: E501
                corpus=sql.Identifier(corpus_name),
            )
            for corpus_name in corpus_names
        ]
        corpus_query = sql.SQL(", ").join(corpus_query)

        dispersion_query = """
        CREATE OR REPLACE TABLE dispersion AS
        WITH
            dist_table AS (
                SELECT
                    {source_id},
                    {target_id},
                    {corpus_query}
                FROM
                    prop_table,
                    corpus_proportions
        ),
        kld_table AS (
            SELECT
                {source_id},
                {target_id},
                LIST_SUM(
                    LIST_VALUE(*COLUMNS(* EXCLUDE ({source_id}, {target_id})))
                ) AS kld
            FROM
                dist_table
        ),
        dispersion_temp AS (
            SELECT
                {source_id},
                {target_id},
                1 - POW(EXP(1), -(kld)) AS dispersion
            FROM
                kld_table
            )
        SELECT
            *
        FROM
            reduced_query
            LEFT JOIN dispersion_temp USING({source_id}, {target_id})
    """
        dispersion_query = (
            sql.SQL(dispersion_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
                corpus_query=corpus_query,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(dispersion_query)
            conn.execute("DROP TABLE prop_table")

    def get_dispersion(
        self,
        ngram_query: NgramQuery,
        *,
        standalone: bool = True,
    ) -> pd.DataFrame:
        """Obtain a table with ngram disoersion.

        Obtain a measure of how well distributed ngrams are across all
        subcorpora of the corpus. Computed as specified by Gries.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.

            standalone (bool, optional): Whether the method is being used only to obtain
            type frequencies or as a part of a sequence of operations. Set to False if
            you have already obtained token frequencies for this set of ngrams.
            Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame containing dispersion information
            for all queried ngrams.

        """
        if standalone:
            self._create_query(ngram_query)
            self._make_token_freq(ngram_query)
            self._reduce_query(ngram_query)
        self._make_dispersion(ngram_query)
        return self.df("SELECT * FROM dispersion")

    def _make_associations(self, ngram_query: NgramQuery) -> None:
        """Make a table with association measures for the queried ngrams.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        ngram_db = sql.Identifier(f"{self._ngram_db}")
        rel_freq_query = """
            CREATE OR REPLACE TEMPORARY TABLE rel_freqs AS
            WITH
                source_freq AS (
                    SELECT DISTINCT
                        {source_id},
                        SUM(freq) AS source_freq
                    FROM
                        filtered_db
                    GROUP BY
                        {source_id}
            ),
            target_freq AS (
                SELECT DISTINCT
                    {target_id},
                    SUM(freq) AS target_freq
                FROM
                    filtered_db
                GROUP BY
                    {target_id}
            )
            SELECT
                *,
                token_freq,
                (
                    SELECT
                        SUM(freq) AS total_freq
                    FROM
                        {ngram_db}
                ) AS total_freq
            FROM
                token_freq
                LEFT JOIN source_freq USING ({source_id})
                LEFT JOIN target_freq USING ({target_id})
            """
        rel_freq_query = (
            sql.SQL(rel_freq_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
                ngram_db=ngram_db,
            )
            .as_string()
        )

        association_query = """
            CREATE OR REPLACE TABLE associations AS
            WITH
            probs_db AS (
                SELECT
                    {source_id},
                    {target_id},
                    token_freq / source_freq AS prob_2_1,
                    token_freq / target_freq AS prob_1_2,
                    source_freq / total_freq AS prob_1,
                    target_freq / total_freq AS prob_2
                FROM
                    rel_freqs
            ),
            all_probs AS (
                SELECT
                    {source_id},
                    {target_id},
                    prob_2_1,
                    prob_1_2,
                    prob_1,
                    prob_2,
                    1 - prob_2_1 AS prob_no_2_1,
                    1 - prob_1_2 AS prob_no_1_2,
                    1 - prob_1 AS prob_no_1,
                    1 - prob_2 AS prob_no_2
                FROM
                    probs_db
            ),
            forward_kld AS (
                SELECT
                    {source_id},
                    {target_id},
                    prob_2_1 * log2(prob_2_1 / prob_2) AS kld_1,
                    CASE
                        WHEN prob_no_2_1 = 0 THEN 0
                        ELSE (1 - prob_2_1) * LOG2((1 - prob_2_1) / (1 - prob_2))
                    END AS kld_2
                FROM
                    all_probs
            ),
            forward_assoc AS (
                SELECT
                    {source_id},
                    {target_id},
                    1 - POW(EXP(1), - (kld_1 + kld_2)) AS fw_assoc
                FROM
                    forward_kld
            ),
            backward_kld AS (
                SELECT
                    {source_id},
                    {target_id},
                    prob_1_2 * LOG2(prob_1_2 / prob_1) AS kld_1,
                    CASE
                        WHEN prob_no_1_2 = 0 THEN 0
                        ELSE (1 - prob_1_2) * LOG2((1 - prob_1_2) / (1 - prob_1))
                    END AS kld_2
                FROM
                    all_probs
            ),
            backward_assoc AS (
                SELECT
                    {source_id},
                    {target_id},
                    1 - POW(EXP(1), - (kld_1 + kld_2)) AS bw_assoc
                FROM
                    backward_kld
            ),
            associations_temp AS (
                SELECT
                    *
                FROM
                    forward_assoc
                    LEFT JOIN backward_assoc USING ({source_id}, {target_id})
            )
            SELECT
                *
            FROM
                reduced_query
                LEFT JOIN associations_temp USING ({source_id}, {target_id})
            """
        association_query = (
            sql.SQL(association_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(rel_freq_query)
            conn.execute(association_query)

    def get_associations(
        self,
        ngram_query: NgramQuery,
        *,
        standalone: bool = True,
    ) -> pd.DataFrame:
        """Obtain a table with ngram association measures.

        Association is calculated separately for component 1 to component 2
        ("fw_assoc") and component 2 to component 1 ("bw_assoc"). Computed
        as specified in Gries.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.
            standalone (bool, optional): Whether the method is being used only to obtain
            type frequencies or as a part of a sequence of operations. Set to False if
            you have already obtained token frequencies for this set of ngrams.
            Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame containing association information
            for all queried ngrams.

        """
        if standalone:
            self._create_query(ngram_query)
            self._make_token_freq(ngram_query)
            self._reduce_query(ngram_query)
        self._make_entropy_diffs(ngram_query)
        return self.df("SELECT * FROM associations")

    def _make_entropy_diff(self, ngram_query: NgramQuery, slot: str) -> None:
        """Make a table with entropy difference measures for the queried ngrams.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.
            slot (str): The slot of the difference being computed. "1" for entropy_2,
            "2" for entropy_1.

        """
        # ? There has to be a better way of doing all this CTEing for entropies. Functions?
        # Real entropy
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        entropy_real_query = """
            CREATE OR REPLACE TEMPORARY TABLE entropy_real AS
            WITH
            all_freqs AS (
                SELECT
                    {source_id},
                    {target_id},
                    freq
                FROM
                    filtered_db
                    SEMI JOIN reduced_query USING ({source_id})
            ),
            total_freqs AS (
                SELECT
                    {source_id},
                    SUM(freq) AS total_freq
                FROM
                    all_freqs
                GROUP BY
                    {source_id}
            ),
            all_probs AS (
                SELECT
                    *
                FROM
                    all_freqs
                    LEFT JOIN total_freqs USING ({source_id})
            ),
            all_infos AS (
                SELECT
                    *,
                    freq / total_freq AS prob,
                    LOG2(freq / total_freq) AS info
                FROM
                    all_probs
            )
            SELECT
                {source_id},
                -1 * (SUM(prob * info) / LOG2(COUNT(*))) AS entropy_real
            FROM
                all_infos
            GROUP BY
                {source_id}
        """
        entropy_real_query = (
            sql.SQL(entropy_real_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )
        entropy_cf_query = """
        CREATE OR REPLACE TEMPORARY TABLE entropy_cf AS
        WITH
        all_freqs AS (
            SELECT
                {source_id},
                reduced_query.{target_id} AS target,
                filtered_db.{target_id} AS {target_id},
                freq
            FROM
                reduced_query
                LEFT JOIN filtered_db USING({source_id})
        ),
        filtered_freqs AS (
            SELECT
                *
            FROM
                all_freqs
            WHERE
                target <> {target_id}
        ),
        total_freqs AS (
            SELECT
                {source_id},
                SUM(freq) AS total_freq
            FROM
                filtered_freqs
            GROUP BY
                {source_id}
        ),
        all_probs AS (
            SELECT
                *
            FROM
                filtered_freqs
                LEFT JOIN total_freqs USING({source_id})
        ),
        all_infos AS (
            SELECT
                *,
                freq / total_freq AS prob,
                LOG2(freq / total_freq) AS info
            FROM
                all_probs
        )
        SELECT
            {source_id},
            target AS {target_id},
            -1 * (SUM(prob * info) / LOG2(COUNT(*))) AS entropy_cf
        FROM
            all_infos
        GROUP BY
            {source_id}, target
            """
        entropy_cf_query = (
            sql.SQL(entropy_cf_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )

        slot_identifier = sql.Identifier(f"entropy_{slot}")
        entropy_query = """
            CREATE OR REPLACE TABLE {slot_id} AS
            WITH
                both_entropies AS (
                    SELECT
                        *
                    FROM
                        entropy_cf
                        LEFT JOIN entropy_real USING({source_id})
                )
                SELECT
                    {source_id},
                    {target_id},
                    entropy_cf - entropy_real AS {slot_id}
                FROM
                    both_entropies
            """
        entropy_query = (
            sql.SQL(entropy_query)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
                slot_id=slot_identifier,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(entropy_real_query)
            conn.execute(entropy_cf_query)
            conn.execute(entropy_query)

    def _make_entropy_diffs(self, ngram_query: NgramQuery) -> None:
        """Make and join tables for entropy diffs for the two slots of the ngram.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        self._make_entropy_diff(ngram_query, "2")
        self._make_entropy_diff(ngram_query, "1")
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        entropy_query = (
            sql.SQL("""
            CREATE OR REPLACE TABLE entropy_diffs AS
            WITH
                entropy_diffs_temp AS (
                    SELECT
                        *
                    FROM
                        entropy_1
                    INNER JOIN entropy_2 USING({source_id}, {target_id})
            )
            SELECT
                {source_id},
                {target_id},
                COALESCE(entropy_1, 1) AS entropy_1,
                COALESCE(entropy_2, 1) AS entropy_2
            FROM
                reduced_query
                LEFT JOIN entropy_diffs_temp USING({source_id}, {target_id})
        """)
            .format(
                source_id=source_identifier,
                target_id=target_identifier,
            )
            .as_string()
        )
        with duckdb.connect(self._path) as conn:
            conn.execute(entropy_query)

            conn.execute("DROP TABLE entropy_1")
            conn.execute("DROP TABLE entropy_2")
            conn.execute("VACUUM ANALYZE")

    def get_entropy(
        self,
        ngram_query: NgramQuery,
        *,
        standalone: bool = True,
    ) -> pd.DataFrame:
        """Obtain a table with ngram entropy difference information.

        Entropy difference is the difference in the entropy of the distribution of
        each slot between the actual distribution and a distribution that doesn't
        include the queried component (e.g. for "this ngram", the entropy difference
        between the probability distribution of "this X" with and without "ngram").
        Computed following Gries.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information. See NgramQuery documentation for details.
            standalone (bool, optional): Whether the method is being used only to obtain
            type frequencies or as a part of a sequence of operations. Set to False if
            you have already obtained token frequencies for this set of ngrams.
            Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame containing entropy difference information
            for all queried ngrams.

        """
        if standalone:
            self._create_query(ngram_query)
            self._make_token_freq(ngram_query)
            self._reduce_query(ngram_query)
        self._make_entropy_diffs(ngram_query)
        return self.df("SELECT * FROM entropy_diffs")

    def _join_measures(self, ngram_query: NgramQuery) -> None:
        """Join all pre-allocated measures into a single table.

        Does it for a specified length to add an ngram_length field.

        Args:
            source (str): Identifier for the first half of the ngram.
            target (str): Identifier for the second half of the ngram.
            length (int): The length of the queried ngrams.

        """
        source_identifier = sql.Identifier(f"{ngram_query.source}")
        target_identifier = sql.Identifier(f"{ngram_query.target}")
        with duckdb.connect(self._path) as conn:
            conn.execute("ATTACH ':memory:' AS results")
            conn.execute(
                sql.SQL("""
                CREATE TABLE results.raw_measures_temp AS
                SELECT
                    *,
                    {length} ngram_length
                FROM
                    reduced_query
            """)
                .format(length=ngram_query.length)
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT
                    *
                FROM
                    results.raw_measures_temp
                    LEFT JOIN token_freq USING({source_id}, {target_id})
            """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                )
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT
                    *
                FROM
                    results.raw_measures_temp
                    LEFT JOIN dispersion USING({source_id}, {target_id})
                """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                )
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT
                    *
                FROM
                    results.raw_measures_temp
                    LEFT JOIN type_freq USING({source_id}, {target_id})
                """)
                .format(source_id=source_identifier, target_id=target_identifier)
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT
                    *
                FROM
                    results.raw_measures_temp
                    LEFT JOIN entropy_diffs USING({source_id}, {target_id})
            """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                )
                .as_string(),
            )
            conn.execute(
                sql.SQL("""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT
                    *
                FROM
                    results.raw_measures_temp
                    LEFT JOIN associations USING({source_id}, {target_id})
            """)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                )
                .as_string(),
            )
            conn.execute("DROP TABLE token_freq")
            conn.execute("DROP TABLE dispersion")
            conn.execute("DROP TABLE type_freq")
            conn.execute("DROP TABLE entropy_diffs")
            conn.execute("DROP TABLE associations")
            conn.execute("DROP TABLE reduced_query")
            conn.execute("DROP TABLE filtered_db")
            source_hash = sql.Identifier(f"{ngram_query.source}_hash")
            target_hash = sql.Identifier(f"{ngram_query.target}_hash")
            raw_measures_query = """
                CREATE OR REPLACE TABLE raw_measures AS
                SELECT
                    query_ref.{source_id},
                    query_ref.{target_id},
                    token_freq,
                    dispersion,
                    typef_1,
                    typef_2,
                    entropy_1,
                    entropy_2,
                    fw_assoc,
                    bw_assoc,
                    {length} ngram_length
                FROM
                    query_ref
                    LEFT JOIN results.raw_measures_temp
                        ON query_ref.{source_hash} = raw_measures_temp.{source_id}
                        AND query_ref.{target_hash} = raw_measures_temp.{target_id}
            """
            raw_measures_query = (
                sql.SQL(raw_measures_query)
                .format(
                    source_id=source_identifier,
                    target_id=target_identifier,
                    source_hash=source_hash,
                    target_hash=target_hash,
                    length=ngram_query.length,
                )
                .as_string()
            )
            conn.execute(raw_measures_query)
            conn.execute("DROP TABLE query_ref")
            conn.execute("VACUUM ANALYZE")

    def _make_all_scores(self, ngram_query: NgramQuery) -> None:
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
            reduced_query, reduced_table = self._reduce_query(ngram_query)
            pbar.update(1)
            logger.debug("Computing type frequencies...")
            if reduced_query is not None and reduced_table is not None:
                self._make_type_freq_sa(reduced_query, reduced_table)
            # self._make_type_freq(ngram_query)
            pbar.update(1)
            logger.debug("Computing dispersion...")
            self._make_dispersion(ngram_query)
            pbar.update(1)
            logger.debug("Computing association...")
            self._make_associations(ngram_query)
            pbar.update(1)
            logger.debug("Computing entropy...")
            self._make_entropy_diffs(ngram_query)
            pbar.update(1)
            logger.debug("Joining results...")
            self._join_measures(ngram_query)
            pbar.update(1)
            self("DROP TABLE this_query")

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
        self._make_all_scores(ngram_query)
        raw_measures = self.df("SELECT * FROM raw_measures", None)
        raw_measures = raw_measures.rename(
            columns={
                "ug_1": "comp_1",
                "ug_2": "comp_2",
                "big_1": "comp_1",
                "ug_3": "comp_2",
                "trig_1": "comp_1",
                "ug_4": "comp_2",
            },
        )

        logger.debug("Cleaning up...")
        self("DROP TABLE raw_measures")
        return raw_measures
