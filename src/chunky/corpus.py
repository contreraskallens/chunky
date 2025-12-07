"""Module for the Corpus class."""

# TODO: Exception logic #06
# TODO: Test individual measure methods #11

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import reduce
from typing import cast, TYPE_CHECKING

import duckdb
import pandas as pd
from rich.console import Console
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.sql import select

from .create_corpus import (
    ALLOWED_CORPORA,
    CORPUS_DIR,
    is_valid_identifier,
    quote_identifier,
    validate_corpus_name,
)

if TYPE_CHECKING:
    from pathlib import Path

VALID_COLUMNS = ["ug_1", "ug_2", "ug_3", "ug_4", "big_1", "trig_1"]

logger = logging.getLogger(__name__)

class Base(orm.DeclarativeBase):
    pass

def _validate_query(ngram_query: NgramQuery) -> bool:
    return (
        is_valid_identifier(ngram_query.source)
        and is_valid_identifier(ngram_query.target)
        and (ngram_query.source in VALID_COLUMNS)
        and (ngram_query.target) in VALID_COLUMNS
    )


def _validate_corpus(corpus: Corpus) -> bool:
    corpus_file = corpus.get_path()
    ngram_file = corpus.get_ngram_file()
    corpus_check = corpus_file.is_relative_to(CORPUS_DIR.resolve())
    ngram_check = ngram_file.is_relative_to(CORPUS_DIR.resolve())

    return corpus_check and ngram_check


class ReducedQuery(Base):
    __tablename__: str = "reduced_query"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    comp_1: orm.Mapped[int] = orm.mapped_column(sa.BigInteger)
    comp_2: orm.Mapped[int] = orm.mapped_column(sa.BigInteger)


class TokenFreq(Base):
    __tablename__: str = "token_freq"
    comp_1: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, primary_key=True)
    comp_2: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, primary_key=True)
    token_freq: orm.Mapped[float] = orm.mapped_column(sa.Double)


class QueryRef(Base):
    __tablename__: str = "query_ref"
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True)
    comp_1: orm.Mapped[str] = orm.mapped_column()
    comp_2: orm.Mapped[str] = orm.mapped_column()
    comp_1_hash: orm.Mapped[int] = orm.mapped_column(sa.BigInteger)
    comp_2_hash: orm.Mapped[int] = orm.mapped_column(sa.BigInteger)


def create_corpus_proportions(corpus_cols: list[str]) -> type:
    """Create ORM class with dynamic columns."""

    # Build class attributes
    attrs = {
        "__tablename__": "corpus_proportions",
        "id": sa.Column(sa.Integer, primary_key=True),
    }
    # Add dynamic sum columns
    for col_name in corpus_cols:
        attrs[col_name] = sa.Column(sa.Double)

    # Create class dynamically
    corpus_proportions = type("corpus_proportions", (Base,), attrs)

    return corpus_proportions


def create_filtered_db(corpus_cols: list[str]) -> type:
    """Create ORM class with dynamic columns."""

    # Build class attributes
    attrs: dict[str, str | dict[str, bool] | sa.Column[int] | sa.Column[float]] = {
        "__tablename__": "filtered_db",
        "__table_args__": {"extend_existing": True},
        "comp_1": sa.Column(sa.Integer, primary_key=True),
        "comp_2": sa.Column(sa.Integer, primary_key=True),
        "freq": sa.Column(sa.Double),
    }
    # Add dynamic sum columns
    for col_name in corpus_cols:
        attrs[f"{col_name}"] = sa.Column(sa.Double)

    # Create class dynamically
    filtered_db = type("FilteredDB", (Base,), attrs)

    return filtered_db


def benchmark_query(query_name, query_sql, con):
    start = time.time()
    table = con.execute(query_sql).fetchall()
    table = pd.DataFrame(table)
    elapsed = time.time() - start
    print(f"{query_name}: {elapsed:.2f} seconds")
    print(table)
    return elapsed

@dataclass
class NgramQuery:
    # TODO: Maybe I can add the batches here?
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

    results: list[
        sa.Select[tuple[TokenFreq]] |
        sa.Select[tuple[ReducedQuery, object]] |
        sa.Select[tuple[ReducedQuery, object, object ]]
    ]
    ngrams: list[str]
    source: str
    target: str
    length: int

    def __init__(
        self,
        ngrams: list[str],
        source: str,
        target: str,
        length: int,
    ) -> None:
        self.ngrams = ngrams
        self.source = source
        self.target = target
        self.length = length
        self.results = []

    def _validate_columns(self, name: str) -> str:
        if is_valid_identifier(name) and name in VALID_COLUMNS:
            return quote_identifier(name)
        msg = "Not a valid column name"
        raise ValueError(msg)


def get_column(
    table: sa.Selectable | type[orm.DeclarativeBase], column: str
) -> sa.Column[object]:
    if isinstance(table, (sa.Subquery, sa.Table, sa.CTE)):
        return cast(sa.Column[object], getattr(table.c, column))
    return cast(sa.Column[object], getattr(table, column))


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

    """

    corpus_name: str
    _path: Path
    _ngram_db: Path
    _engine: sa.Engine
    _corpus_proportions: type[orm.DeclarativeBase] 
    _filtered_db: type[orm.DeclarativeBase]

    def __init__(self, corpus_name: str) -> None:
        """Initialize an instance of a Corpus.

            Sets the paths to the database, parquet file, and temp directory.
            Do a bunch of checks to avoid SQL injection.

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
        if corpus_name not in ALLOWED_CORPORA or not validate_corpus_name(corpus_name):
            msg = f"{corpus_name} is not an allowed corpus."
            raise ValueError(msg)
        self.corpus_name = corpus_name

        corpus_file = CORPUS_DIR / f"{corpus_name}.db"
        corpus_file = corpus_file.resolve()
        ngram_file = CORPUS_DIR / f"{corpus_name}_ngrams.parquet"
        ngram_file = ngram_file.resolve()

        if not corpus_file.is_file() or not ngram_file.is_file():
            error_message = "No corpus found. Make first with make_processed_corpus()."
            raise RuntimeError(error_message)

        self._path = corpus_file
        self._ngram_db = ngram_file
        self._engine = sa.create_engine(f"duckdb:///{self._path}", echo=False)
        with self._engine.connect() as conn:
            parquet_columns = conn.execute(
                sa.text(f"DESCRIBE SELECT * FROM PARQUET_SCAN('{self._ngram_db}')"),  # noqa: S608 Validated before
            ).fetchall()
        corpus_columns = [
            column[0]
            for column in parquet_columns
            if column[0] not in ["ug_1", "ug_2", "ug_3", "ug_4", "big_1", "trig_1"]
        ]
        self._corpus_proportions = create_corpus_proportions(corpus_columns)
        self._filtered_db = create_filtered_db(corpus_columns)

    def __call__(self, query: str) -> list[tuple[object]]:
        """Query the underlying database.

        Args:
            query (str): An SQL query.

        Returns:
            list: A list of tuples with the results of the query.

        """
        with duckdb.connect(self._path) as conn:
            query_result = conn.execute(query)
            return query_result.fetchall()

    def get_path(self) -> Path:
        return self._path

    def get_ngram_file(self) -> Path:
        return self._ngram_db

    def query_parquet(self, ug: int | None = None) -> list:
        """Dev class. Eliminate."""
        with duckdb.connect(self._path) as conn:
            if not _validate_corpus(self):
                msg = "Problem in corpus object. Please initialize again correctly."
                raise ValueError(msg)
            if ug is None:
                this_query = conn.execute(
                    f"""EXPLAIN ANALYZE
                        SELECT *
                        FROM '{self._ngram_db}'
                        """,  # noqa: S608 Validated during initialization
                )
            else:
                this_query = conn.execute(
                    f"""
                    EXPLAIN ANALYZE
                    SELECT *
                    FROM '{self._ngram_db}'
                    WHERE ug_1 = {ug}
                    """,  # noqa: S608 Validated during initialization
                )
            return this_query.fetchall()

    def _show_ngrams(self, limit: int = 100) -> pd.DataFrame:
        """Show a sample of the ngram frequency table.

        Queries the ngram parquet file and shows a sample of rows from it.

        Args:
            limit (int, optional): Number of rows to show. Defaults to 100.

        Returns:
            pd.DataFrame: pandas Dataframe containing the rows of the ngram
            frequency table.

        """
        if not _validate_corpus(self):
            msg = "Problem with corpus information. Please initialize again"
            raise ValueError(msg)
        ngram_db_query = f"SELECT * FROM '{self._ngram_db}' LIMIT {limit}"  # noqa: S608
        ngram_data =  pd.read_sql_query(
            ngram_db_query,
            self._engine,
        )
        ngram_data: pd.DataFrame
        return ngram_data

    def df(self, query: str, params: list[str | int | float] | dict[str, str | int | float] | None = None) -> pd.DataFrame:
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

    def _create_query(self, ngram_query: NgramQuery) -> None:
        """Allocate a list of ngrams as a query for filtering DB.

        Args:
            ngram_query (ngram_query): An ngram_query object containing ngrams and
            source/target information.

        """
        query_df = pd.DataFrame(
            ngram_query.ngrams,
            columns=pd.Index([ngram_query.source, ngram_query.target]),
        )

        with self._engine.connect() as conn:
            _ = conn.execute(
                sa.text("register(:name, :df)"), {"name": "query_df", "df": query_df}
            )
            query_ref_create = """
                CREATE OR REPLACE TABLE query_ref (
                    id INT,
                    comp_1 TEXT,
                    comp_2 TEXT,
                    comp_1_hash UINT64,
                    comp_2_hash UINT64
                )
                """
            _ = conn.execute(sa.text(query_ref_create))

            if _validate_query(ngram_query):
                query_ref_insert = f"""
                    INSERT INTO
                        query_ref
                    SELECT
                        row_number() OVER () AS id,
                        {quote_identifier(ngram_query.source)} AS comp_1,
                        {quote_identifier(ngram_query.target)} AS comp_2,
                        HASH({quote_identifier(ngram_query.source)}) AS comp_1_hash,
                        HASH({quote_identifier(ngram_query.target)}) AS comp_2_hash
                    FROM
                        query_df
                """  # noqa: S608 Identifiers validated before
            else:
                msg = "Not valid column"
                raise ValueError(msg)
            _ = conn.execute(sa.text(query_ref_insert))
            conn.commit()

    def _get_token_freq(self, ngram_query: NgramQuery) -> sa.Select[tuple[TokenFreq]]:
        """Make a table with token frequencies for the queried ngrams.

        This is required by all other supported measures.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams
            and source/target information.

        """
        if not _validate_query(ngram_query):
            msg = "Not valid column names in query. Initialize again."
            raise ValueError(msg)
        if not _validate_corpus(self):
            msg = "Problem with corpus information. Please initialize again"
            raise ValueError(msg)
        token_freq_query = f"""
        CREATE OR REPLACE TABLE token_freq AS
        SELECT
            query_ref.comp_1_hash AS comp_1,
            query_ref.comp_2_hash AS comp_2,
            SUM(freq) AS token_freq
        FROM
            query_ref
        INNER JOIN READ_PARQUET('{self._ngram_db}')
            ON query_ref.comp_1_hash = {quote_identifier(ngram_query.source)}
            AND query_ref.comp_2_hash = {quote_identifier(ngram_query.target)}
        GROUP BY
            query_ref.comp_1_hash, query_ref.comp_2_hash
                """  # noqa: S608 Validated before
        with self._engine.connect() as conn:
            _ = conn.execute(sa.text(token_freq_query))
            conn.commit()
        return select(TokenFreq)

    def _reduce_query(self, ngram_query: NgramQuery) -> None:
        """Make a reduced query table that includes only ngrams with token_freq > 0.

        This is a substantial memory and runtime save for later operations. It
        eliminates from the query all ngrams that do not occur in the corpus.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        # token_freq = TokenFreq
        # query = QueryRef

        reduced_query = select(
            QueryRef.id.label("id"),
            QueryRef.comp_1_hash.label("comp_1"),
            QueryRef.comp_2_hash.label("comp_2"),
        ).where(
            sa.exists().where(
                (TokenFreq.comp_1 == QueryRef.comp_1_hash)
                & (TokenFreq.comp_2 == QueryRef.comp_2_hash),
            ),
        )
        if not _validate_query(ngram_query):
            msg = "Not valid column names in query. Initialize again."
            raise ValueError(msg)
        if not _validate_corpus(self):
            msg = "Problem with corpus information. Please initialize again"
            raise ValueError(msg)
        with self._engine.connect() as conn:
            parquet_columns = conn.execute(
                sa.text(f"DESCRIBE SELECT * FROM PARQUET_SCAN('{self._ngram_db}')"),  # noqa: S608 Validated before
            ).fetchall()
        corpus_columns: list[str] = [
            column[0]
            for column in parquet_columns
            if column[0] not in ["ug_1", "ug_2", "ug_3", "ug_4", "big_1", "trig_1"]
        ]
        corpus_columns.sort() 
        corpus_sums = ", ".join([f"SUM({column}) AS {column}" for column in corpus_columns])
        filter_query = f"""
                CREATE OR REPLACE TABLE filtered_db AS
                SELECT
                    {quote_identifier(ngram_query.source)} AS comp_1,
                    {quote_identifier(ngram_query.target)} AS comp_2,
                    {corpus_sums}
                FROM
                    READ_PARQUET('{self._ngram_db}')
                WHERE
                    {ngram_query.source} IN (
                        SELECT
                            comp_1
                        FROM
                            reduced_query
                    )
                    OR {ngram_query.target} IN (
                        SELECT
                            comp_2
                        FROM
                            reduced_query
                    )
                GROUP BY
                    comp_1,
                    comp_2
                """

        with self._engine.connect() as conn:
            _ = conn.execute(sa.text(f"CREATE OR REPLACE TABLE reduced_query AS {reduced_query}"))
            _ = conn.execute(sa.text(filter_query))
            conn.commit()

    def _join_with_query(
        self,
        query_table: sa.CTE,
        result_table: sa.CTE,
        result_name: str,
        alt_name: str | None = None,
    ) -> sa.Select[tuple[int, int, float]]:
        if alt_name is None:
            select_statement = select(
                query_table, get_column(result_table, result_name)
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

    def _get_type_freq(self) -> sa.Select[tuple[ReducedQuery, object, object]]:
        """Make a table with type frequencies for the queried ngrams.

        Args:
            ngram_query: An NgramQuery object containing ngrams and
            source/target information.

        """
        db = self._filtered_db
        type_1_query = select(
            get_column(db, "comp_2"),
            sa.func.count().label("typef_1"),
        )
        type_1_query = type_1_query.group_by(get_column(db, "comp_2"))
        type_1_query = type_1_query.cte()
        type_2_query = select(
            get_column(db, "comp_1"),
            sa.func.count().label("typef_2"),
        )
        type_2_query = type_2_query.group_by(get_column(db, "comp_1"))
        type_2_query = type_2_query.cte()
        results = select(
            ReducedQuery,
            get_column(type_1_query, "typef_1"),
            get_column(type_2_query, "typef_2"),
        )
        results = results.join(
            type_1_query,
            (get_column(ReducedQuery, "comp_2") == get_column(type_1_query, "comp_2")),
        ).join(
            type_2_query,
            (get_column(ReducedQuery, "comp_1") == get_column(type_2_query, "comp_1")),
        )
        return results

    def _get_prop_columns(
        self,
        corpus_columns: list[sa.ColumnElement[float]],
        freq_column: sa.ColumnElement[float],
    ) -> list[sa.ColumnElement[float]]:
        prop_columns: list[sa.ColumnElement[float]] = []
        for column in corpus_columns:
            column_name = cast(str, column.name)
            prop_column = column / freq_column
            prop_columns.append(prop_column.label(column_name))
        return prop_columns

    def _get_kld(
        self,
        column_1: sa.ColumnElement[float],
        column_2: sa.ColumnElement[object] | sa.ScalarSelect[object],
    ) -> sa.Case[float]:
        return sa.case(
            ((column_1 == 0) | (column_2 == 0), 0),
            else_=column_1 * sa.func.log2(column_1 / column_2),
        )

    def _get_distances(
        self,
        prop_columns: list[sa.ColumnElement[float]],
        all_corpus_props: type[orm.DeclarativeBase],
    ) -> list[sa.ColumnElement[float]]:
        # distance to corpus proportion
        mapper_all = sa.inspect(all_corpus_props)
        # Use scalar_subquery because corpus_proportion.X always has length=1
        kld_columns: list[sa.ColumnElement[float]] = []
        for column in prop_columns:
            column_name: str = cast(str, column.name)
            this_column: sa.ColumnElement[object] = cast(sa.ColumnElement[object], mapper_all.columns[column_name])
            this_kld: sa.ColumnElement[float] = self._get_kld(column,
                                     sa.select(this_column).scalar_subquery(),
                                     ).label(column_name)
            kld_columns.append(this_kld)
        return kld_columns

    def _sum_rows(self, columns: list[sa.ColumnElement[float]]) -> sa.ColumnElement[float]:
        return reduce(lambda x, y: x + y, columns)

    def _normalize_kld(self, column: sa.Column[float] | sa.ColumnElement[float]) -> sa.ColumnElement[float]:
        return 1 - sa.func.pow(sa.func.exp(1), -column)

    def _get_dispersion_column(
        self,
        corpus_columns: list[sa.ColumnElement[float]],
        freqs: sa.ColumnElement[float],
    ) -> sa.ColumnElement[float]:
        prop_columns = self._get_prop_columns(corpus_columns, freqs)
        distance_columns = self._get_distances(prop_columns, self._corpus_proportions)
        kld_column = self._sum_rows(distance_columns)
        return self._normalize_kld(kld_column)

    def _get_dispersion(self) -> sa.Select[tuple[ReducedQuery, object]]:
        """Make a table with a dispersion measure for the queried ngrams.

        Args:
            source (str): Identifier of the first half of the ngram.
            target (str): Identifier of the second half of the ngram.

        """
        # Should I make this reduced_table before and pass it down instead?
        db = self._filtered_db

        reduced_table = select(
            db,
        ).join(
            ReducedQuery,
            (get_column(ReducedQuery, "comp_1") == get_column(db, "comp_1"))
            & (get_column(ReducedQuery, "comp_2") == get_column(db, "comp_2")),
        )

        reduced_table = reduced_table.cte()
        
        corpus_columns: list[sa.ColumnElement[float]] = []
        for column in reduced_table.c:
            column_name: str = cast(str, column.name)
            if column_name not in ["comp_1", "comp_2", "id", "freq"]:
                corpus_columns.append(column)

        dispersion = select(
            get_column(reduced_table, "comp_1"),
            get_column(reduced_table, "comp_2"),
            self._get_dispersion_column(
                corpus_columns,
                cast(sa.ColumnElement[float], get_column(reduced_table, "freq"))
            ).label("dispersion")
        )
        dispersion = dispersion.cte()
        return select(
            ReducedQuery,
            get_column(dispersion, "dispersion")
        ).join(dispersion,
               (get_column(dispersion, "comp_1") == get_column(ReducedQuery, "comp_1"))
                & (get_column(dispersion, "comp_2") == get_column(ReducedQuery, "comp_2"))
               )

    def _get_rel_freqs(
        self,
        db: type[orm.DeclarativeBase],
    ) -> sa.Select[tuple[ReducedQuery, float, object, object]]:
        source_freq = select(
            get_column(db, "comp_1"),
            sa.func.sum(get_column(db, "freq")).label("source_freq"),
        ).group_by(get_column(db, "comp_1"))
        source_freq = source_freq.cte()
        target_freq = select(
            get_column(db, "comp_2"),
            sa.func.sum(get_column(db, "freq")).label("target_freq"),
        ).group_by(get_column(db, "comp_2"))
        target_freq = target_freq.cte()
        with self._engine.connect() as conn:
            total_freq_query: sa.Row[tuple[float]] = cast(sa.Row[tuple[float]], conn.execute(
                sa.text("SELECT SUM(freq) AS total_freq FROM unigram_db"),
            ).fetchone())
        total_freq: float = cast(float, total_freq_query[0])
        rel_freqs = select(
            ReducedQuery,
            sa.literal(total_freq).label("total_freq"),
            get_column(source_freq, "source_freq"),
            get_column(target_freq, "target_freq"),
        )
        rel_freqs = rel_freqs.join(
            source_freq,
            (get_column(source_freq, "comp_1") == get_column(ReducedQuery, "comp_1")),
        )
        return rel_freqs.join(
            target_freq,
            (get_column(target_freq, "comp_2") == get_column(ReducedQuery, "comp_2")),
        )

    def _get_probs(
        self,
        rel_freq: sa.Select[tuple[ReducedQuery, float, object, object]],
    ) -> sa.Select[tuple[object, object, object, object, object, object]]:
        rel_freq_cte = rel_freq.cte()
        filtered_rel_freq = select(
            rel_freq_cte, get_column(TokenFreq, "token_freq")
        ).join(
            TokenFreq,
            (get_column(rel_freq_cte, "comp_1") == get_column(TokenFreq, "comp_1"))
            & (get_column(rel_freq_cte, "comp_2") == get_column(TokenFreq, "comp_2")),
        )
        filtered_rel_freq = filtered_rel_freq.cte()
        probs = select(
            get_column(filtered_rel_freq, "comp_1"),
            get_column(filtered_rel_freq, "comp_2"),
            (
                get_column(filtered_rel_freq, "token_freq")
                / get_column(filtered_rel_freq, "source_freq")
            ).label(
                "prob_2_1",
            ),
            (
                get_column(filtered_rel_freq, "token_freq")
                / get_column(filtered_rel_freq, "target_freq")
            ).label(
                "prob_1_2",
            ),
            (
                get_column(filtered_rel_freq, "source_freq")
                / get_column(filtered_rel_freq, "total_freq")
            ).label("prob_1"),
            (
                get_column(filtered_rel_freq, "target_freq")
                / get_column(filtered_rel_freq, "total_freq")
            ).label("prob_2"),
        ).cte()
        return select(
            probs,
            (1 - get_column(probs, "prob_2_1")).label("prob_no_2_1"),
            (1 - get_column(probs, "prob_1_2")).label("prob_no_1_2"),
            (1 - get_column(probs, "prob_1")).label("prob_no_1"),
            (1 - get_column(probs, "prob_2")).label("prob_no_2"),
        )

    def _get_normalized_kld(
        self,
        pair_1: tuple[sa.ColumnElement[object], sa.ColumnElement[object]],
        pair_2: tuple[sa.ColumnElement[object], sa.ColumnElement[object]],
    ) -> sa.ColumnElement[float]:
        pair_1_float = cast(tuple[sa.ColumnElement[float], sa.ColumnElement[object]], pair_1)
        pair_2_float = cast(tuple[sa.ColumnElement[float], sa.ColumnElement[object]], pair_2)
        kld_1 = self._get_kld(*pair_1_float)
        kld_2 = self._get_kld(*pair_2_float)
        return self._normalize_kld(kld_1 + kld_2)

    def _get_associations(self) -> sa.Select[tuple[ReducedQuery, object, object]]:
        db = self._filtered_db
        rel_freq = self._get_rel_freqs(db)
        probs = self._get_probs(rel_freq).cte()
        fw_assoc = select(
            get_column(probs, "comp_1"),
            get_column(probs, "comp_2"),
            self._get_normalized_kld(
                (get_column(probs, "prob_2_1"), get_column(probs, "prob_2")),
                (get_column(probs, "prob_no_2_1"), get_column(probs, "prob_no_2")),
            ).label("fw_assoc")
        )
        fw_assoc = fw_assoc.cte()
        bw_assoc = select(
            get_column(probs, "comp_1"),
            get_column(probs, "comp_2"),
            self._get_normalized_kld(
                (get_column(probs, "prob_1_2"), get_column(probs, "prob_1")),
                (get_column(probs, "prob_no_1_2"), get_column(probs, "prob_no_1")),
            ).label("bw_assoc")
        )
        bw_assoc = bw_assoc.cte()

        return select(
            ReducedQuery,
            get_column(fw_assoc, "fw_assoc"),
            get_column(bw_assoc, "bw_assoc")
        ).join(
            fw_assoc,
            (get_column(fw_assoc, "comp_1") == get_column(ReducedQuery, "comp_1")) 
                & (get_column(fw_assoc, "comp_2") == get_column(ReducedQuery, "comp_2"))
        ).join(
            bw_assoc,
            (get_column(bw_assoc, "comp_1") == get_column(ReducedQuery, "comp_1")) 
                & (get_column(bw_assoc, "comp_2") == get_column(ReducedQuery, "comp_2"))
        )


    def _get_total_freq(
        self,
        db: type[orm.DeclarativeBase] | sa.CTE,
        column: str,
        *,
        cf: bool = False,
    ) -> sa.Select[tuple[int, int, float, float]]:
        id_columns = [get_column(db, "comp_1"), get_column(db, "comp_2")]
        if cf:
            id_columns.append(get_column(db, "target"))
        token_freq = select(
            *id_columns,
            get_column(db, "freq"),
        ).where(get_column(db, column).in_(select(get_column(ReducedQuery, column))))
        token_freq = token_freq.cte()
        id_columns = [get_column(token_freq, column)]
        if cf:
            id_columns.append(get_column(token_freq, "target"))
        total_freq = select(
            *id_columns,
            sa.func.sum(get_column(token_freq, "freq")).label("total_freq"),
        ).group_by(*id_columns)
        total_freq = total_freq.cte()
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
            (get_column(total_freq, column) == get_column(token_freq, column)),
        )

    def _get_info(
        self,
        freqs: sa.ColumnElement[object],
        total_freqs: sa.ColumnElement[object],
    ) -> sa.ColumnElement[float]:
        prob = freqs / total_freqs
        info = sa.func.log2(prob)
        return prob * info

    def _compute_entropy(
        self,
        db: type[orm.DeclarativeBase] | sa.CTE,
        source_column: str,
        *,
        cf: bool = False,
    ) -> sa.Select[tuple[int, float]] | sa.Select[tuple[int, int, float]]:
        if cf:
            total_freq = self._get_total_freq(db, source_column, cf=True)
        else:
            total_freq = self._get_total_freq(db, source_column)
        total_freq = total_freq.cte()

        weighted_info = select(
            total_freq,
            self._get_info(
                get_column(total_freq, "freq"),
                get_column(total_freq, "total_freq"),
            ).label("weighted_info"),
        )
        weighted_info = weighted_info.cte()
        wi_id_columns = [get_column(weighted_info, source_column)]
        if cf:
            wi_id_columns.append(get_column(weighted_info, "target"))
        entropy = select(
            *wi_id_columns,
            (-sa.func.sum(weighted_info.c.weighted_info)).label("raw_entropy"),
            sa.func.count(weighted_info.c.weighted_info).label("n"),
        ).group_by(*wi_id_columns)
        entropy = entropy.cte()

        ent_id_columns = [get_column(entropy, source_column)]
        if cf:
            ent_id_columns.append(get_column(entropy, "target"))
        return select(
            *ent_id_columns,
            (entropy.c.raw_entropy / sa.func.log2(entropy.c.n)).label("entropy"),
        )

    def _get_mult_table(self, db: type[orm.DeclarativeBase] | sa.CTE,
                        source_column: str,
                        target_column: str
                        ) -> sa.Select[tuple[object, object, object, object]]:
        mult_table = select(
            get_column(ReducedQuery, source_column).label(source_column),
            get_column(ReducedQuery, target_column).label("target"),
            get_column(db, target_column).label(target_column),
            get_column(db, "freq"),
        ).join(
            db,
            (get_column(ReducedQuery, source_column) == get_column(db, source_column)),
        )
        mult_table = mult_table.cte()
        return select(mult_table).where(
            get_column(mult_table, "target") != get_column(mult_table, target_column),
        )

    def _get_entropy_diff(
        self,
        entropy_real: sa.Select[tuple[int, float]],
        entropy_cf: sa.Select[tuple[int, int, float]],
        source_column: str,
        target_column: str,
    ) -> sa.Select[tuple[object, object, float]]:
        entropy_real_cte = entropy_real.cte()
        entropy_cf_cte = entropy_cf.cte()

        both_entropy = select(
            get_column(entropy_cf_cte, source_column),
            get_column(entropy_cf_cte, "target").label(target_column),
            entropy_real_cte.c.entropy.label("entropy_real"),
            entropy_cf_cte.c.entropy.label("entropy_cf"),
        ).join(
            entropy_real_cte,
            (
                get_column(entropy_cf_cte, source_column)
                == get_column(entropy_real_cte, source_column)
            ),
        )
        both_entropy = both_entropy.cte()
        diff_column = get_column(both_entropy, "entropy_cf") - get_column(both_entropy, "entropy_real")
        return select(
            get_column(both_entropy, source_column),
            get_column(both_entropy, target_column),
            diff_column.label("entropy_diff"),
        )

    def _get_entropy(
        self,
        db: type[orm.DeclarativeBase],
        source_column: str,
        target_column: str,
    ) -> sa.Select[tuple[object, object, float]]:
        mult_table = self._get_mult_table(
            db,
            source_column,
            target_column,
        )
        mult_table = mult_table.cte()
        entropy_real = self._compute_entropy(db, source_column)
        entropy_real = cast(sa.Select[tuple[int, float]], entropy_real)
        entropy_cf = self._compute_entropy(
            mult_table,
            source_column,
            cf=True,
        )
        entropy_cf = cast(sa.Select[tuple[int, int, float]], entropy_cf)
        return self._get_entropy_diff(
            entropy_real,
            entropy_cf,
            source_column,
            target_column
        )

    def _get_entropies(self) -> sa.Select[tuple[ReducedQuery, object, object]]:
        db = self._filtered_db
        entropy_1 = self._get_entropy(db, "comp_2", "comp_1")
        entropy_1 = entropy_1.cte()
        entropy_2 = self._get_entropy(db, "comp_1", "comp_2")
        entropy_2 = entropy_2.cte()
        return (
            select(
                ReducedQuery,
                get_column(entropy_1, "entropy_diff").label("entropy_1"),
                get_column(entropy_2, "entropy_diff").label("entropy_2"),
            )
            .join(
                entropy_1,
                (ReducedQuery.comp_1 == get_column(entropy_1, "comp_1"))
                    & (ReducedQuery.comp_2 == get_column(entropy_1, "comp_2")),
            )
            .join(
                entropy_2,
                (ReducedQuery.comp_1 == get_column(entropy_2, "comp_1"))
                    & (ReducedQuery.comp_2 == get_column(entropy_2, "comp_2")),
            )
        )

    def _join_measures(self, ngram_query: NgramQuery) -> sa.CTE:
        """Join all pre-allocated measures into a single table.
        Does it for a specified length to add an ngram_length field.

        Args:
            source (str): Identifier for the first half of the ngram.
            target (str): Identifier for the second half of the ngram.
            length (int): The length of the queried ngrams.

        """
        results_cte: list[sa.CTE] = [result.cte() for result in ngram_query.results]

        all_results = select(ReducedQuery).cte()


        for result in results_cte:
            measure_columns = [
                column
                for column in result.c
                if cast(str, column.name) not in ["comp_1", "comp_2", "id"]
            ]
            all_results = select(all_results, *measure_columns).join(
                result,
                (get_column(all_results, "comp_1") == get_column(result, "comp_1"))
                & (get_column(all_results, "comp_2") == get_column(result, "comp_2")),
            )
            all_results = all_results.cte()

        return all_results

    def _get_all_scores(self, ngram_query: NgramQuery) -> sa.CTE:
        """Allocate all measures for the ngrams in the query table"""
        ngram_query.results.append(self._get_token_freq(ngram_query))
        self._reduce_query(ngram_query)
        ngram_query.results.append(self._get_type_freq())
        ngram_query.results.append(self._get_dispersion())
        ngram_query.results.append(self._get_associations())
        ngram_query.results.append(self._get_entropies())
        return self._join_measures(ngram_query)

    def get_scores(self, ngram_query: NgramQuery, *, verbose: bool=False) -> pd.DataFrame:
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
        with self._engine.connect() as conn:
            if verbose:
                console = Console()
                # TODO: Better status info
                with console.status("[bold red]Executing...", spinner='pong') as status:
                    self._create_query(ngram_query)
                    all_scores = self._get_all_scores(ngram_query)
                    results = conn.execute(select(all_scores)).fetchall()
                    status.update("[bold green]Done!")
            else:
                self._create_query(ngram_query)
                all_scores = self._get_all_scores(ngram_query)
                results = conn.execute(select(all_scores)).fetchall()
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            return results_df.astype({
                "id": "int64",
                "comp_1": "str",
                "comp_2": "str", 
                "token_freq": "float64",
                "typef_1": "float64",
                "typef_2": "float64",
                "dispersion": "float64",
                "fw_assoc": "float64",
                "bw_assoc": "float64",
                "entropy_1": "float64",
                "entropy_2": "float64"
            })
        else: 
            return pd.DataFrame(results)
