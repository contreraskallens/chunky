"""Module for cleaning lines of corpora."""

# TODO: provide own corpus #06
# TODO: make it so that corpus key can contain multiple subcorpora.  Currently only supports one as a string #10

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from itertools import islice, tee
from pathlib import Path
from typing import Callable

import pandas as pd
import regex

logger: logging.Logger = logging.getLogger(__name__)


def _clean_bnc(
    raw_lines: str | list[str],
    **kwargs: str | None,
) -> list[tuple[str, str]]:
    """Clean a chunk of lines from the BNC corpus.

    Extracts the corpus ids from the text, joins contractions,
    replaces newline symbols, repeated spaces, and changes standalone
    numbers to the placeholder NUMBERS.

    Args:
        raw_lines (str | list[str]): Lines from the corpus file. Can be either
        a single string or a list of strings for more efficient processing.
        **kwargs: Just a placeholder.

    Returns:
        list: A list of tuples of corpus IDs, clean text for corpus.

    """
    placeholder = kwargs.get("nothing")
    logger.debug(placeholder)  # TODO: fix this #1X
    all_lines = pd.Series(raw_lines)
    corpus_list: pd.Series[str] = all_lines.str.extract(
        r"(^.)",
        expand=False,
    )
    processed_lines: pd.Series[str] = all_lines.str.replace(
        r"^.+\t",
        "",
        regex=True,
    )
    processed_lines = processed_lines.str.lower()
    processed_lines = processed_lines.str.replace(
        r" (n't|'s|'ll|'d|'re|'ve|'m)",
        r"\1",
        regex=True,
    )
    processed_lines = processed_lines.str.replace(
        "wan na",
        "wanna",
        regex=False,
    )
    processed_lines = processed_lines.str.replace("\n", "")
    processed_lines = processed_lines.str.replace("-", "")
    processed_lines = processed_lines.str.replace(
        r"\s\d+\s|^\d+\s|\s\d+$",
        " NUMBER ",
        regex=True,
    )
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(
        r"\s*\W+\s*",
        " ",
        regex=True,
    )
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    processed_lines = "START START " + processed_lines + " END END"
    return list(zip(corpus_list, processed_lines.to_list(), strict=True))


def _clean_coca(raw_line: str, corpus_ids: str) -> list[tuple[str, str]]:
    """Clean a chunk of lines from the CoCA corpus.

    Deals with some of the quirks of the CoCA formats, such as markers,
    tokenization of contractions, numbering, and spacing.

    Args:
        raw_line (str): Lines from the CoCA corpus to be cleaned.
        corpus_ids (str | None): IDs of the sub-corpus (e.g., acad) to which
        the raw_line belong.

    Returns:
        list[tuple]: A list of tuples of corpus id, clean text for corpus.

    """

    logger.debug("Cleaning text of length %s characters", len(raw_line))
    processed_lines: str = regex.sub(
        r" [\.\?\!] |\n|(@ )+|</*[ph]>|<br>",
        " splitmehere ",
        raw_line.lower(),
    )
    processed_lines = regex.sub(
        r" (n't|'s|'ll|'d|'re|'ve|'m)",
        r"\1",
        processed_lines,
    )
    processed_lines = regex.sub(
        r"@@\d+\s*",
        r"",
        processed_lines,
    )
    processed_lines = processed_lines.replace(
        "wan na",
        "wanna",
    )
    processed_lines = processed_lines.replace(
        "-",
        " ",
    )
    processed_lines = regex.sub(
        r"\d+",
        " NUMBER ",
        processed_lines,
    )
    processed_lines = regex.sub(
        r" \W|\W ",
        " ",
        processed_lines,
    )
    processed_lines = regex.sub(
        r"\s+",
        " ",
        processed_lines,
    )
    line_list: list[str] = regex.split(
        r"\s*splitmehere\s*",
        processed_lines,
    )
    line_list = [line for line in line_list if len(line) > 0]
    # Get rid of double spaces and trailing spaces, add header and footer in line
    line_list = [
        "START START " + " ".join(line.split()).strip() + " END END"
        for line in line_list
        if len(line) > 0
    ]
    logger.debug("Resulted in %s clean lines", len(line_list))
    clean_lines = (
        corpus_ids,
        " ".join(line_list),
    )
    return [clean_lines]


clean_functions: dict[str, Callable[..., list[tuple[str, str]]]] = {
    "bnc": _clean_bnc,
    "coca": _clean_coca,
}


def _ngram_tuple(unigram_list: list[str], n: int = 2) -> zip[tuple[str, ...]]:
    """Turn a list of unigrams into an iterable of ngrams.

    Do this by making n copies of the iterable, transposing them
    by a step size of 0 to n, and zipping them together.

    Args:
        unigram_list (list): List of unigrams
        n (int, optional): Length of ngram to extract. Defaults to 2.

    Returns:
        An iterable of tuples containing all ngrams of length n in the unigram list.

    """
    repeated_unigrams: tuple[Iterator[str], ...] = tee(unigram_list, n)
    index_and_unigram: enumerate[Iterator[str]] = enumerate(repeated_unigrams)
    # islice returns the whole iterator but skipping pos (n) starting elements
    # e.g. (a, b, c, d, e), (a, b, c, d),  (a, b, c), (a, b)
    transposed_unigrams = (
        islice(unigrams, transpose_n, None)
        for transpose_n, unigrams in index_and_unigram
    )
    # zip *(transposed_unigrams) makes tuples of successive members of the original
    # iterable by combining the elements at index [i] of each transposed list
    #  e.g. (a, b, c, d), # (b, c, d, e), etc
    return zip(
        *transposed_unigrams,
        strict=False,
    )


def _line_to_ngram(text: str, n: int = 2) -> zip[tuple[str, ...]]:
    """Split text and turn it into ngrams of length n.

    Args:
        text (str): A line of text.
        n (int, optional): Length of ngram to extract. Defaults to 2.

    Returns:
        An iterable of tuples containing all ngrams of length n in the provided text.

    """
    words = text.split()
    return _ngram_tuple(words, n)


def _preprocess_test() -> dict[str, list[tuple[str | int, ...]]]:
    """Extract unigrams and ngrams from the test corpus.

    Complete process of generation for the test corpus. Obtains
    the text file, reads the lines, extracts unigrams and ngrams,
    and counts their frequency of occurrence in each chunk.

    Returns:
        tuple: A tuple of frequency counts for unigrams and fourgrams.

    """
    with Path("chunky/corpora/test_corpus.txt").open(
        encoding="utf-8",
    ) as corpus_file:
        raw_lines: list[str] = corpus_file.read().splitlines()
    split_lines: list[list[str]] = [line.split() for line in raw_lines]
    fourgrams_counts: list[Counter[tuple[str, ...]]] = [
        Counter(_line_to_ngram(line, 4)) for line in raw_lines
    ]
    corpora = ["A", "B", "C"]
    fourgrams: list[tuple[str, str, str, str, str, int]] = [
        (corpus, trigram[0], trigram[1], trigram[2], trigram[3], freq)
        for corpus, corpus_dict in zip(corpora, fourgrams_counts, strict=True)
        for trigram, freq in corpus_dict.items()
    ]
    unigrams_counts: list[Counter[str]] = [
        Counter(unigrams) for unigrams in split_lines
    ]
    unigrams: list[tuple[str, str, int]] = [
        (corpus, unigram, freq)
        for corpus, corpus_dict in zip(corpora, unigrams_counts, strict=True)
        for unigram, freq in corpus_dict.items()
    ]
    return {"unigrams": unigrams, "fourgrams": fourgrams}


def _extract_ngrams(
    clean_lines: list[tuple[str, str]],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, *tuple[str, ...], int]]]:
    """Extract unigrams and ngrams from a corpus chunk.

    Take cleaned lines from a corpus, extract unigrams and fourgrams from them,
    and obtain their frequencies.

    Args:
        clean_lines (list): List of lines cleaned from a supported corpus. Each line
        of the cleaned corpus must be presented as a tuple of corpus key, line,
        e.g. [(A, "a b c d"), (B, "x y z")].

    Returns:
        tuple: A tuple containing dictionaries of unigram and fourgram counts for
        each sub corpus provided.

    """
    logger.info("Extracting ngrams...")
    all_fourgrams: dict[str, zip[tuple[str, ...]]] = {
        key: _line_to_ngram(corpus, 4) for key, corpus in clean_lines
    }
    fourgrams_counts: dict[str, Counter[tuple[str, ...]]] = {
        key: Counter(list(fourgrams)) for key, fourgrams in all_fourgrams.items()
    }
    for key, this_fourgrams in fourgrams_counts.items():
        logger.debug(
            """Corpus %(corpus)s contains %(n_ngrams)s total fourgrams, \
%(n_unique)s unique.""",
            {
                "corpus": key,
                "n_ngrams": sum(this_fourgrams.values()),
                "n_unique": len(this_fourgrams),
            },
        )
    fourgrams: list[tuple[str, *tuple[str, ...], int]] = [
        (corpus, *ngram, freq)
        for corpus, corpus_dict in fourgrams_counts.items()
        for ngram, freq in corpus_dict.items()
    ]
    unigrams_counts: dict[str, Counter[str]] = {
        corpus: Counter(corpus_lines.split()) for corpus, corpus_lines in clean_lines
    }
    unigrams: list[tuple[str, str, int]] = [
        (corpus, ngram, freq)
        for corpus, corpus_dict in unigrams_counts.items()
        for ngram, freq in corpus_dict.items()
    ]
    return unigrams, fourgrams


def preprocess_corpus(
    corpus: str,
    raw_lines: str | list[str] | None = None,
    corpus_id: str | None = None,
) -> dict[str, list[tuple[str | int, ...]]]:
    """Clean corpus and extract ngram frequencies from it.

    Args:
        corpus (str): The name of the corpus as a string.
        raw_lines (str | list[str] | None, optional): Raw line or lines from the corpus
        as either a string or a list of strings. Defaults to None for test corpus.
        corpus_id (str | None, optional): Corpus id of the chunk of text for the CoCA
        corpus. Currently it's the only supported one.

    Returns:
        tuple: A tuple containing dictionaries of unigram and fourgram counts. Each
        member of the tuple is a dictionary with corpus key and counts. If processing
        CoCA, only one corpus key is supported.
        e.g. {"A": Counter((a, b, c): 3, (b, c, d): 4), "B": Counter((x, y, z): 2)}

    """
    logger.debug('Using preprocessing method for corpus "%s"', corpus)
    if corpus == "test":
        return _preprocess_test()
    clean_function: Callable[..., list[tuple[str, str]]] | None = clean_functions.get(
        corpus
    )
    if clean_function is None:
        error_message = "Corpus not supported"
        raise NotImplementedError(error_message)
    this_lines: list[tuple[str, str]] = clean_function(raw_lines, corpus_id)
    unigrams, fourgrams = _extract_ngrams(this_lines)
    return {"unigrams": unigrams, "fourgrams": fourgrams}
