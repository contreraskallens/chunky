"""Module for the Corpus class."""

# TODO(omfgzell): Exception logic #06
# TODO(omfgzell): Test individual measure methods #11

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.sql import select
from tqdm import tqdm

logger = logging.getLogger(__name__)
Base = orm.declarative_base()


class ReducedQuery(Base):
    __tablename__ = "reduced_query"
    id = sa.Column(sa.Integer, primary_key=True)
    comp_1 = sa.Column(sa.BigInteger)
    comp_2 = sa.Column(sa.BigInteger)


class TokenFreq(Base):
    __tablename__ = "token_freq"
    comp_1 = sa.Column(sa.BigInteger, primary_key=True)
    comp_2 = sa.Column(sa.BigInteger, primary_key=True)
    token_freq = sa.Column(sa.Double)


class QueryRef(Base):
    __tablename__ = "query_ref"
    id = sa.Column(sa.Integer, primary_key=True)
    comp_1 = sa.Column(sa.String)
    comp_2 = sa.Column(sa.String)
    comp_1_hash = sa.Column(sa.BigInteger)
    comp_2_hash = sa.Column(sa.BigInteger)


def create_corpus_proportions(corpus_cols: list) -> tuple:
    """Create ORM class with dynamic columns."""
    base = orm.declarative_base()

    # Build class attributes
    attrs = {
        "__tablename__": "corpus_proportions",
        "id": sa.Column(sa.Integer, primary_key=True),
    }
    # Add dynamic sum columns
    for col_name in corpus_cols:
        attrs[f"{col_name}"] = sa.Column(sa.Double)

    # Create class dynamically
    corpus_proportions = type("corpus_proportions", (base,), attrs)

    return corpus_proportions, base


def create_filtered_db(corpus_cols: list) -> tuple:
    """Create ORM class with dynamic columns."""
    base = orm.declarative_base()

    # Build class attributes
    attrs = {
        "__tablename__": "filtered_db",
        "comp_1": sa.Column(sa.Integer, primary_key=True),
        "comp_2": sa.Column(sa.Integer, primary_key=True),
        "freq": sa.Column(sa.Double),
    }
    # Add dynamic sum columns
    for col_name in corpus_cols:
        attrs[f"{col_name}"] = sa.Column(sa.Double)

    # Create class dynamically
    filtered_db = type("FilteredDB", (base,), attrs)

    return filtered_db, base


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

    query: type[orm.DeclarativeBase]
    freq_table: type[orm.DeclarativeBase]
    total_proportions: type[orm.DeclarativeBase]
    results: Any  # Messy with the CTE stuff. Leave it as Any.
    query_ref: type[orm.DeclarativeBase]
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

    def update_results(self, new_results: sa.Selectable) -> None:
        self.results = new_results
        if isinstance(self.results, sa.HasCTE):
            self.results = self.results.cte()


def get_column(
    table: sa.Selectable | type[orm.DeclarativeBase],
    column: str,
) -> sa.Column:
    if isinstance(table, (sa.Subquery, sa.Table, sa.CTE)):
        return getattr(table.c, column)
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

    """

    def __init__(
        self,
        corpus_name: str,
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
        self._engine = sa.create_engine(
            f"duckdb:///{self._path}",
            echo=False,
        )
        self._ngram_db = Path(f"chunky/db/{self.corpus_name}_ngrams.parquet")
        if not self._path.is_file():
            error_message = "No corpus found. Make first with make_processed_corpus()."
            raise RuntimeError(error_message)
        logger.info("Using preexisting corpus")

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

    def _show_ngrams(self, limit: int = 100) -> pd.DataFrame:
        """Show a sample of the ngram frequency table.

        Queries the ngram parquet file and shows a sample of rows from it.

        Args:
            limit (int, optional): Number of rows to show. Defaults to 100.

        Returns:
            pd.DataFrame: pandas Dataframe containing the rows of the ngram
            frequency table.

        """
        ngram_db_query = f"SELECT * FROM {self._ngram_db} LIMIT {limit}"  # noqa: S608
        return pd.read_sql(ngram_db_query, self._engine)

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

        with self._engine.connect() as conn:
            conn.execute(
                sa.text("register(:name, :df)"),
                {"name": "query_df", "df": query_df},
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
            conn.execute(sa.text(query_ref_create))
            query_ref_insert = f"""
                INSERT INTO
                    query_ref
                SELECT
                    row_number() OVER () AS id,
                    {ngram_query.source} AS comp_1,
                    {ngram_query.target} AS comp_2,
                    HASH({ngram_query.source}) AS comp_1_hash,
                    HASH({ngram_query.target}) AS comp_2_hash
                FROM
                    query_df
            """
            conn.execute(sa.text(query_ref_insert))
            conn.commit()

    def _get_token_freq(self, ngram_query: NgramQuery) -> sa.Select:
        """Make a table with token frequencies for the queried ngrams.

        This is required by all other supported measures.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams
            and source/target information.

        """

        token_freq_query = f"""
        CREATE OR REPLACE TABLE token_freq AS
        SELECT
            query_ref.comp_1_hash AS comp_1,
            query_ref.comp_2_hash AS comp_2,
            SUM(freq) AS token_freq
        FROM
            query_ref
        INNER JOIN READ_PARQUET('{self._ngram_db}')
            ON query_ref.comp_1_hash = {ngram_query.source}
            AND query_ref.comp_2_hash = {ngram_query.target}
        GROUP BY
            query_ref.comp_1_hash, query_ref.comp_2_hash
                """
        with self._engine.connect() as conn:
            conn.execute(sa.text(token_freq_query))
            conn.commit()
        return select(TokenFreq)

    def _reduce_query(self, ngram_query: NgramQuery) -> NgramQuery:
        """Make a reduced query table that includes only ngrams with token_freq > 0.

        This is a substantial memory and runtime save for later operations. It
        eliminates from the query all ngrams that do not occur in the corpus.

        Args:
            ngram_query (NgramQuery): An NgramQuery object containing ngrams and
            source/target information.

        """
        token_freq = TokenFreq
        query = QueryRef

        reduced_query = select(
            query.comp_1_hash.label("comp_1"),
            query.comp_2_hash.label("comp_2"),
        ).where(
            sa.exists().where(
                (token_freq.comp_1 == query.comp_1_hash)
                & (token_freq.comp_2 == query.comp_2_hash),
            ),
        )

        filter_query = f"""
                CREATE OR REPLACE TABLE filtered_db AS
                SELECT
                    {ngram_query.source} AS comp_1,
                    {ngram_query.target} AS comp_2,
                    SUM(
                        COLUMNS(* EXCLUDE(ug_1, ug_2, ug_3, ug_4, big_1, trig_1))
                    )
                FROM
                    READ_PARQUET('{self._ngram_db}')
                WHERE
                    {ngram_query.source} IN (
                        SELECT
                            comp_1
                        FROM
                            token_freq
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
            conn.execute(sa.text(filter_query))
            conn.commit()
            parquet_columns = conn.execute(
                sa.text(f"DESCRIBE SELECT * FROM PARQUET_SCAN('{self._ngram_db}')")
            ).fetchall()
        corpus_columns = [
            column[0]
            for column in parquet_columns
            if column[0] not in ["ug_1", "ug_2", "ug_3", "ug_4", "big_1", "trig_1"]
        ]
        filtered_db, _ = create_filtered_db(corpus_columns)
        corpus_proportions, _ = create_corpus_proportions(corpus_columns)
        ngram_query.query = reduced_query
        ngram_query.results = reduced_query
        ngram_query.freq_table = filtered_db
        ngram_query.total_proportions = corpus_proportions
        ngram_query.query_ref = query
        return ngram_query

    def _join_with_query(
        self,
        query_table: sa.CTE,
        result_table: sa.CTE,
        result_name: str,
        alt_name: str | None = None,
    ) -> sa.Select:
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
        corpus_columns: list,
        freq_column: sa.Column,
    ) -> list[sa.Column]:
        return [(column / freq_column).label(column.name) for column in corpus_columns]

    def _get_kld(
        self,
        column_1: sa.Column,
        column_2: sa.Column | sa.ScalarSelect,
    ) -> sa.Case:
        return sa.case(
            ((column_1 == 0) | (column_2 == 0), 0),
            else_=column_1 * sa.func.log2(column_1 / column_2),
        )

    def _get_distances(
        self,
        prop_columns: list,
        all_corpus_props: type[orm.DeclarativeBase],
    ) -> list[sa.ColumnElement]:
        # distance to corpus proportion
        mapper_all = sa.inspect(all_corpus_props)
        # Use scalar_subquery because corpus_proportion.X always has length=1
        return [
            self._get_kld(
                column,
                select(mapper_all.columns[column.name]).scalar_subquery(),
            ).label(column.name)
            for column in prop_columns
        ]

    def _sum_rows(self, columns: list) -> sa.Column:
        return reduce(lambda x, y: x + y, columns)

    def _normalize_kld(self, column: sa.Column | sa.ColumnElement) -> sa.ColumnElement:
        return 1 - sa.func.pow(sa.func.exp(1), -column)

    def _get_dispersion_column(
        self,
        corpus_columns: list,
        freqs: sa.Column,
        corpus_proportions: type[orm.DeclarativeBase],
    ) -> sa.ColumnElement:
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

        reduced_table = reduced_table.cte()

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
        dispersion_table = dispersion_table.cte()
        results = self._join_with_query(
            reduced_query,
            dispersion_table,
            "dispersion",
        )
        ngram_query.update_results(results)

    def _get_rel_freqs(
        self,
        db: type[orm.DeclarativeBase],
        reduced_query: ReducedQuery | sa.Selectable,
    ) -> sa.Selectable:
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
            total_freq = conn.execute(
                sa.text("SELECT SUM(freq) AS total_freq FROM unigram_db"),
            ).fetchone()
        if total_freq is not None:
            total_freq = total_freq[0]
        else:
            exception_msg = "Oops something happened"
            raise RuntimeError(exception_msg)
        rel_freqs = select(
            reduced_query,
            sa.literal(total_freq).label("total_freq"),
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
        rel_freq: sa.CTE,
        token_freq: sa.Select,
    ) -> sa.Select:
        token_freq_cte = token_freq.cte()
        filtered_rel_freq = select(rel_freq, get_column(token_freq_cte, "freq")).join(
            token_freq_cte,
            (get_column(rel_freq, "comp_1") == get_column(token_freq_cte, "comp_1"))
            & (get_column(rel_freq, "comp_2") == get_column(token_freq_cte, "comp_2")),
        )
        filtered_rel_freq = filtered_rel_freq.cte()
        probs = select(
            get_column(filtered_rel_freq, "comp_1"),
            get_column(filtered_rel_freq, "comp_2"),
            (
                get_column(filtered_rel_freq, "freq")
                / get_column(filtered_rel_freq, "source_freq")
            ).label(
                "prob_2_1",
            ),
            (
                get_column(filtered_rel_freq, "freq")
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
        pair_1: tuple,
        pair_2: tuple,
    ) -> sa.ColumnElement:
        kld_1 = self._get_kld(*pair_1)
        kld_2 = self._get_kld(*pair_2)
        return self._normalize_kld(kld_1 + kld_2)

    def _get_associations_sa(self, ngram_query: NgramQuery) -> None:
        reduced_query = ngram_query.results
        db = ngram_query.freq_table
        rel_freq = self._get_rel_freqs(db, reduced_query).cte()  # pyright: ignore[reportAttributeAccessIssue]
        token_freq = select(reduced_query, get_column(db, "freq")).join(
            db,
            (get_column(reduced_query, "comp_1") == get_column(db, "comp_1"))
            & (get_column(reduced_query, "comp_2") == get_column(db, "comp_2")),
        )
        probs = self._get_probs(rel_freq, token_freq).cte()
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
        assoc_table = assoc_table.cte()
        results = self._join_with_query(
            reduced_query,
            assoc_table,
            "fw_assoc",
        )
        results = results.cte()
        results = self._join_with_query(
            results,
            assoc_table,
            "bw_assoc",
        )
        ngram_query.update_results(results)

    def _get_total_freq(
        self,
        reduced_query: sa.Select | sa.Selectable,
        db: type[orm.DeclarativeBase] | sa.CTE,
        column: str,
        *,
        cf: bool = False,
    ) -> sa.Select:
        id_columns = [get_column(db, "comp_1"), get_column(db, "comp_2")]
        if cf:
            id_columns.append(get_column(db, "target"))
        token_freq = select(
            *id_columns,
            get_column(db, "freq"),
        ).where(get_column(db, column).in_(select(get_column(reduced_query, column))))
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
            (getattr(total_freq.c, column) == getattr(token_freq.c, column)),
        )

    def _get_info(
        self,
        freqs: sa.Column,
        total_freqs: sa.ColumnElement,
    ) -> sa.ColumnElement:
        prob = freqs / total_freqs
        info = sa.func.log2(prob)
        return prob * info

    def _get_entropy(
        self,
        reduced_query: sa.Select | sa.Selectable,
        db: type[orm.DeclarativeBase] | sa.CTE,
        source_column: str,
        *,
        cf: bool = False,
    ) -> sa.Select:
        if cf:
            total_freq = self._get_total_freq(reduced_query, db, source_column, cf=True)
        else:
            total_freq = self._get_total_freq(reduced_query, db, source_column)
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

    def _get_mult_table(
        self,
        reduced_query: sa.Selectable,
        db: type[orm.DeclarativeBase]
        | sa.CTE,  # DeclarativeBase for original CTE for CF
        source_column: str,
        target_column: str,
    ) -> sa.Select:
        mult_table = select(
            get_column(reduced_query, source_column).label(source_column),
            get_column(reduced_query, target_column).label("target"),
            get_column(db, target_column).label(target_column),
            get_column(db, "freq"),
        ).join(
            db,
            (get_column(reduced_query, source_column) == get_column(db, source_column)),
        )
        mult_table = mult_table.cte()
        return select(mult_table).where(
            mult_table.c.target != get_column(mult_table, target_column),
        )

    def _get_entropy_diff(
        self,
        entropy_real: sa.Select,
        entropy_cf: sa.Select,
        source_column: str,
        target_column: str,
    ) -> sa.Select:
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
        return select(
            get_column(both_entropy, source_column),
            get_column(both_entropy, target_column),
            (both_entropy.c.entropy_cf - both_entropy.c.entropy_real).label(
                "entropy_diff",
            ),
        )

    def _get_entropy_sa(
        self,
        reduced_query: sa.Selectable,
        db: type[orm.DeclarativeBase],
        source_column: str,
        target_column: str,
    ) -> sa.Select:
        mult_table = self._get_mult_table(
            reduced_query,
            db,
            source_column,
            target_column,
        )
        mult_table = mult_table.cte()
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
        entropy_1 = entropy_1.cte()
        entropy_2 = self._get_entropy_sa(reduced_query, db, "comp_1", "comp_2")
        entropy_2 = entropy_2.cte()

        results = self._join_with_query(
            reduced_query,
            entropy_1,
            "entropy_diff",
            alt_name="entropy_1",
        )
        results = results.cte()
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
            get_column(query_ref, "comp_1"),
            get_column(query_ref, "comp_2"),
            *[
                get_column(results, column.name)
                for column in results.c
                if column.name not in ["id", "comp_1", "comp_2"]
            ],
            sa.literal(ngram_query.length).label("ngram_length"),
        ).join(
            results,
            (get_column(query_ref, "comp_1_hash") == get_column(results, "comp_1"))
            & (get_column(query_ref, "comp_2_hash") == get_column(results, "comp_2")),
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
            token_freq = self._get_token_freq(ngram_query)
            pbar.update(1)
            logger.debug("Making reduced table...")
            ngram_query = self._reduce_query(ngram_query)
            ngram_query.update_results(token_freq)
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
        with self._engine.connect() as conn:
            results = conn.execute(select(ngram_query.results)).fetchall()
        return pd.DataFrame(results)
