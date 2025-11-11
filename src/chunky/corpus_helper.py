"""Module for the Fetcher class."""

# TODO(omfgzell): Variable max length of ngram #08
# TODO(omfgzell): Exception logic #10
# TODO(omfgzell): make method for large batches and for specific measures. #12
# TODO(omfgzell): Check that weights sum 1 #13
# TODO(omfgzell): test for corpora other than CoCA. #14
# TODO(omfgzell): Return dataclass instead of tuples and dictionaries #15

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nltk import everygrams

from chunky.corpus import Corpus

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = [
    1 / 8,
    1 / 8,
    1 / 8,
    1 / 8,
    1 / 8,
    1 / 8,
    1 / 8,
    1 / 8,
]
VARIABLE_NAMES = [
    "token_freq",
    "dispersion",
    "typef_1",
    "typef_2",
    "entropy_1",
    "entropy_2",
    "fw_assoc",
    "bw_assoc",
]
BIGRAM_LEN = 2
TRIGRAM_LEN = 3
FOURGRAM_LEN = 4


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

    ngrams: list
    source: str
    target: str
    length: int


class Fetcher:
    """Interface for manipulating a Corpus class.

    Implements methods so that the user doesn't have to query the corpus directly.

    Attributes:
        corpus (corpus.Corpus): The Corpus object with which this interfaces.
        _bigram_scores, _trigram_scores, fourgram_scores (DataFrame):
        Dataframes containing the raw MWU measures for a set of ngrams.
        Not meant to be interfaced with.

    """

    def __init__(
        self,
        corpus: Corpus | str,
        *,
        make: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the Corpus Helper class.

        Initializes an instance of the Corpus Helper Class.
        Links the instance with a specific corpus.
        Optionally, can be instructed to make a corpus before linking with it.

        Args:
            corpus (Corpus | str): A corpus to link with and query.
            If argument is a string with a corpus name, Helper makes the corpus before.
            make (bool, optional): Whether to make the corpus before linking with it.
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
        logger.debug("Initializing helper for corpus %s", corpus)
        if isinstance(corpus, str):
            self.corpus = Corpus(str(corpus), make=make, **kwargs)
        else:
            self.corpus = corpus

    def __call__(
        self,
        query: str,
    ) -> list:
        """Query the corpus.

        Directly queries the corpus with the specified string.

        Args:
            query (str): String with a query to the corpus. Must be valid SQL.

        Returns:
            list: The results of the query as a list of tuples, from .fetchall()

        """
        return self.corpus(query)

    def df(
        self,
        query: str,
        params: list | dict | None,
    ) -> pd.DataFrame:
        """Query the corpus for a dataframe.

        Directly query the corpus with the specified string
        and obtain a dataframe of the result.

        Args:
            query (str): String with a query to the corpus. Must be valid SQL.
            params (list | dict | None): Optional parameters to include in the query.
            See DuckDB documentation on Prepared Statements for behavior.
            https://duckdb.org/docs/stable/clients/python/dbapi#prepared-statements

        Returns:
            pd.DataFrame: The result of the query as a dataframe.

        """
        return self.corpus.df(query, params)

    def get_ngram_table(self, limit: int = 100) -> pd.DataFrame:
        """Obtain the table of ngrams from the corpus.

        Obtains the data contained in the linked corpus representing each ngram and
        their frequency per sub-corpus.
        Note that all components of ngrams are hashed as integers.

        Args:
            limit (int, optional): The maximum number of rows to obtain.
            Defaults to 100.

        Returns:
            pd.DataFrame: Dataframe containing the ngram table from the corpus.

        """
        return self.corpus.show_ngrams(limit=limit)

    def _split_ngrams(self, all_ngrams: list[str]) -> tuple:
        """Split and sort ngrams according to length.

        Take a list of ngrams represented as strings "take the"
        and transform them into lists of bigrams, ["take", "the"].
        Then, sort them into lists according to the length.
        When the provided string contains a longer sequence such as
        "take the other way", generate all subcomponents: ["take", "the"],
        ["take the", "other"], and ["take the other", "way"].

        # !Currently and for the foreseeable future supports lengths of up to n=4.

        Args:
            all_ngrams (list[str]): List with all ngrams to be split and sorted.

        Returns:
            tuple: A tuple containing the lists of bigrams, trigrams, and fourgrams.

        """
        bigrams = []
        trigrams = []
        fourgrams = []
        for ngram in all_ngrams:
            split_ngram = ngram.split()
            len_ngram = len(split_ngram)
            if len_ngram >= BIGRAM_LEN:
                bigrams.append((split_ngram[0], split_ngram[1]))
            if len_ngram >= TRIGRAM_LEN:
                trigrams.append(
                    (
                        f"{split_ngram[0]} {split_ngram[1]}",
                        split_ngram[2],
                    ),
                )
            if len_ngram >= FOURGRAM_LEN:
                fourgrams.append(
                    (
                        f"{split_ngram[0]} {split_ngram[1]} {split_ngram[2]}",
                        split_ngram[3],
                    ),
                )
            if len_ngram not in [BIGRAM_LEN, TRIGRAM_LEN, FOURGRAM_LEN]:
                except_msg = "Length of ngram not supported"
                raise NotImplementedError(except_msg)
        bigrams = [list(bigram) for bigram in set(bigrams)]
        trigrams = [list(trigram) for trigram in set(trigrams)]
        fourgrams = [list(fourgram) for fourgram in set(fourgrams)]
        return bigrams, trigrams, fourgrams

    def _make_scores_ngrams(
        self,
        ngrams: list[str],
    ) -> None:
        """Allocate scores for all provided ngrams.

        Queries the corpus for the provided ngrams and obtains the
        scores for all available measures of MWUness.
        The scores are allocated as attributes in this instance of Helper.

        Args:
            ngrams (list[str]): List of ngrams for which scores will be allocated.

        """
        ngrams = list(set(ngrams))
        bigrams, trigrams, fourgrams = self._split_ngrams(ngrams)
        bigram_query = NgramQuery(bigrams, "ug_1", "ug_2", 2)
        trigram_query = NgramQuery(trigrams, "big_1", "ug_3", 3)
        fourgram_query = NgramQuery(fourgrams, "trig_1", "ug_4", 4)
        self._bigram_scores = self.corpus.get_scores(bigram_query)
        self._trigram_scores = self.corpus.get_scores(trigram_query)
        self._fourgram_scores = self.corpus.get_scores(fourgram_query)

    def _process_text(
        self,
        text: str,
        line_sep: str = "\n",
    ) -> list[str]:
        r"""Prepare a chunk of text to obtain measures.

        Take a string containing text and prepare clean it to match the format
        of the corpus. Then, split into a list of all ngrams of supported length
        contained in the text.

        Args:
            text (str): A string containing a text to be cleaned and split.
            line_sep (str, optional): The character separating lines in the text.
            Defaults to "\n".

        Returns:
            list[str]: List of all ngrams of supported length contained in the text.

        """
        this_text = text.split(line_sep)
        this_text = pd.Series(this_text)
        this_text = this_text.str.lower()
        this_text = this_text.str.replace("\n", "")
        this_text = this_text.str.replace("-", " ")
        this_text = this_text.str.replace(r"\d+", " NUMBER ", regex=True)
        this_text = this_text.str.strip()
        this_text = this_text.str.replace(r" +\W+", " ", regex=True)
        this_text = this_text.str.replace(r"\W+ +", " ", regex=True)
        this_text = this_text.str.replace(r"^\W+", " ", regex=True)
        this_text = this_text.str.replace(r"\W+$", " ", regex=True)
        this_text = this_text.str.replace(r"(\w)[\.,]+(\w)", r"\1 \2", regex=True)
        this_text = this_text.str.replace(r"\s+", " ", regex=True)
        all_ngrams = this_text.apply(
            lambda line: [
                " ".join(ngram)  # type: ignore[attr-defined]
                for ngram in everygrams(line.split(), 2, 4)
            ],
        )
        return [ngram for line in all_ngrams.to_list() for ngram in line]

    def _make_scores_text(
        self,
        text: str,
        **kwargs: str,
    ) -> None:
        """Allocate scores for a chunk of text.

        Makes and allocates as attributes of this Helper instance the
        scores for the text provided as input. First cleans and splits it,
        then queries the corpus through the method for lists of ngrams.

        Args:
            text (str): A chunk of text to be processed.
            **kwargs (str): Arguments to be passed on to _process_text.

        """
        ngrams = self._process_text(text=text, **kwargs)
        self._make_scores_ngrams(
            ngrams=ngrams,
        )

    def _min_max(self, column: pd.Series) -> pd.Series:
        """Min-max normalize a column of a dataframe.

        Take the column of a pandas DataFrame and normalize it
        by substracting the minimum value from each value and dividing them
        by the difference between the maximum and the minimum.

        Args:
            column (pd.Series): The column of a dataframe to normalize.

        Returns:
            pd.Series: A pandas Series containing the min-max normalized
            values of the original column.

        """
        min_value = column.min()
        max_value = column.max()
        column_norm = column - min_value
        return column_norm.div(max_value - min_value)

    def _normalize_results(
        self,
        results: pd.DataFrame,
        entropy_limits: list | None = None,
    ) -> pd.DataFrame:
        """Apply the respective normalization to each measure.

        Normalize each MWU measure of the raw results using the formulas
        presented by Gries.
        Token and type frequencies are logged.
        Dispersion and type frequencies are substraced from 1.
        Entropies are limited to specified boundaries.
        Token frequency, type frequencies, and entropies are min-max normalized.

        Args:
            results (pd.DataFrame): Dataframe containing raw measures to be normalized.
            entropy_limits (list | None, optional): A list specifying the boundaries of
            the entropy measure. Defaults to [-0.1, 0.1] when provided None.

        Returns:
            pd.DataFrame: A dataframe containing normalized measures.

        """
        if entropy_limits is None:
            entropy_limits = [-0.1, 0.1]
        normalized_results = results.copy()
        norm_no_data = normalized_results[results.isna().any(axis=1)]
        norm_no_data = norm_no_data[["comp_1", "comp_2", "ngram_length"]]
        norm_no_data = norm_no_data.replace(np.nan, pd.NA, inplace=False)
        normalized_results = normalized_results[normalized_results.notna().all(axis=1)]
        normalized_results[["token_freq", "typef_1", "typef_2"]] = normalized_results[
            ["token_freq", "typef_1", "typef_2"]
        ].apply(lambda x: np.log(x))

        normalized_results["entropy_1"] = normalized_results["entropy_1"].apply(
            lambda x: max(entropy_limits[0], x),
        )

        normalized_results["entropy_2"] = normalized_results["entropy_2"].apply(
            lambda x: max(entropy_limits[0], x),
        )
        normalized_results["entropy_1"] = normalized_results["entropy_1"].apply(
            lambda x: min(entropy_limits[1], x),
        )
        normalized_results["entropy_2"] = normalized_results["entropy_2"].apply(
            lambda x: min(entropy_limits[1], x),
        )
        normalized_results[
            ["token_freq", "typef_1", "typef_2", "entropy_1", "entropy_2"]
        ] = normalized_results[
            ["token_freq", "typef_1", "typef_2", "entropy_1", "entropy_2"]
        ].apply(lambda x: self._min_max(x))

        normalized_results[["dispersion", "typef_1", "typef_2"]] = normalized_results[
            ["dispersion", "typef_1", "typef_2"]
        ].apply(lambda x: 1 - x)

        return (
            pd.concat([normalized_results, norm_no_data])
            if len(norm_no_data) > 0
            else normalized_results
        )

    def _weight_results(
        self,
        mwu_measures: pd.DataFrame,
        weights: list[float] | dict = DEFAULT_WEIGHTS,
    ) -> pd.DataFrame:
        """Apply weighting scheme to normalized measures.

        To compute the final MWU score, weights are assigned to each measure.
        Take a frame with normalized results and apply weights to each column.
        Weights are multiplied to each column to sum and obtain a weighted average.

        Args:
            mwu_measures (pd.DataFrame): pandas DataFrame containing normalized
            MWU measures.
            weights (list[float] | dict, optional): Weights for each column. Can be
            a list of floats in the following order: token frequency, dispersion,
            type frequency 1, type frequency 2, entropy 1, entropy 2, forward
            association, and backward association. Can also be provided as a
            dictionary with the keys "token_freq", "dispersion", "typef_1",
            "typef_2", "entropy_1", "entropy_2", "fw_assoc", and "bw_assoc".
            Weights must sum to 1. Defaults to uniform weights.

        Returns:
            pd.DataFrame: A dataframe with normalized MWU measures weighted by
            their respective weights.

        """
        if isinstance(weights, list):
            weights = dict(zip(VARIABLE_NAMES, weights, strict=True))
        for col in mwu_measures.columns:
            if col in weights:
                mwu_measures[col] = mwu_measures[col] * weights[col]
        return mwu_measures

    def _compute_mwu(
        self,
        mwu_measures: pd.DataFrame,
        weights: list | dict = DEFAULT_WEIGHTS,
    ) -> pd.Series:
        """Obtain the MWU score given normalized measures.

        Args:
            mwu_measures (pd.DataFrame): A dataframe containing normalized MWU measures.
            weights (list | dict, optional): The weight of each measure. See
            documentation of _weight_results() for more details.
            Defaults to equal weights.

        Returns:
            pd.Series: A pandas Series containing the MWU scores for each ngram of the
            results frame.

        """
        weighted_results = self._weight_results(mwu_measures, weights)
        return weighted_results[VARIABLE_NAMES].sum(axis=1, skipna=False)

    def _normalize_and_compute(
        self,
        results: pd.DataFrame,
        weights: list | dict = DEFAULT_WEIGHTS,
    ) -> pd.DataFrame:
        """Normalize raw measures and compute the MWU score.

        Args:
            results (pd.DataFrame): pandas DataFrame containing raw measures for each
            variable.
            weights (list | dict, optional): The weight of each measure. See
            documentation of _weight_results() for more details.
            Defaults to equal weights.

        Returns:
            pd.DataFrame: A dataframe containing normalized measures and MWU scores.

        """
        normalized = self._normalize_results(results)
        normalized["mwu_score"] = self._compute_mwu(normalized, weights=weights)
        return normalized

    def get_mwu_scores(
        self,
        ngrams: str | list,
        weights: list | dict = DEFAULT_WEIGHTS,
        mode: str = "normalized",
        **kwargs: str,
    ) -> dict:
        r"""Obtain MWU measures and score for the provided input.

        Take a chunk of text or ngrams and obtain all MWU measures for them, including
        the final MWU score. Can obtain either only the raw measures or the raw measures
        and the normalized scores. MWU score is computed using the provided scores.
        Input can be either list of ngrams  or a string containing a longer text.


        Args:
            ngrams (str | list): Input to compute MWU measures. Can be provided as a
            list of ngrams (e.g. ["hello there", "good friend of", ...]) or as a single
            string containing a text.
            weights (list | dict, optional): Weights for each column. Can be
            a list of floats in the following order: token frequency, dispersion,
            type frequency 1, type frequency 2, entropy 1, entropy 2, forward
            association, and backward association. Can also be provided as a
            dictionary with the keys "token_freq", "dispersion", "typef_1",
            "typef_2", "entropy_1", "entropy_2", "fw_assoc", and "bw_assoc".
            Weights must sum to 1. Defaults to uniform weights.
            mode (str, optional): Provide "raw" to compute only the raw measures
            or "normalized" to compute the raw measures, normalized measures, and
            MWU score.
            **kwargs (str): Arguments for the text processing function. Currently
            can only be line_sep, which defaults to "\n".

        Returns:
            dict: Dictionary containing raw and normalized scores. If mode="raw",
            the "normalized" entry will be empty.

        """
        if mode not in ["raw", "normalized"]:
            except_msg = "Specify the mode as either raw or normalized"
            raise NotImplementedError(except_msg)
        if isinstance(ngrams, str):
            ngrams = ngrams.lower()
            self._make_scores_text(str(ngrams), **kwargs)
        elif isinstance(ngrams, list):
            ngrams = [ngram.lower() for ngram in ngrams]
            self._make_scores_ngrams(list(ngrams))
        else:
            exception_error = "Input must be text as a string or a list of ngrams."
            raise TypeError(exception_error)
        all_raw = pd.concat(
            [
                self._bigram_scores,
                self._trigram_scores,
                self._fourgram_scores,
            ],
        )
        if mode == "raw":
            return {"raw": all_raw, "normalized": None}
        if mode == "normalized":
            normalized_bigram = self._normalize_and_compute(
                self._bigram_scores,
                weights,
            )
            normalized_trigram = self._normalize_and_compute(
                self._trigram_scores,
                weights,
            )
            normalized_fourgram = self._normalize_and_compute(
                self._fourgram_scores,
                weights,
            )
            all_normalized = pd.concat(
                [
                    normalized_bigram,
                    normalized_trigram,
                    normalized_fourgram,
                ],
            )
            return {"raw": all_raw, "normalized": all_normalized}
        error_message = "Specify either 'raw' or 'normalized' as a return mode"
        raise RuntimeError(error_message)
