"""Module for cleaning lines of corpora."""

# TODO(omfgzell): provide own corpus #06
# TODO(omfgzell): make it so that corpus key can contain multiple subcorpora.  Currently only supports one as a string #10
from __future__ import annotations

import logging
from collections import Counter
from itertools import islice, tee
from pathlib import Path

import pandas as pd
import regex

logger = logging.getLogger(__name__)


def _clean_bnc(
    raw_lines: str | list[str],
    **kwargs: str | None,
) -> list[tuple]:
    """Clean a chunk of lines from the BNC corpus.

    Extracts the corpus ids from the text, joins contractions,
    replaces newline symbols, repeated spaces, and changes standalone
    numbers to the placeholder NUMBERS.

    Args:
        raw_lines (str | list[str]): Lines from the corpus file. Can be either
        a single string or a list of strings for more efficient processing.
        **kwargs: Just a placeholder.
    #!TODO Fix

    Returns:
        list: A list of tuples of corpus IDs, list of clean lines.

    """
    placeholder = kwargs.get("nothing")
    logger.debug(placeholder)  # TODO(omfgzell): fix this #1X
    all_lines = pd.Series(raw_lines)
    corpus_list = all_lines.str.extract(
        r"(^.)",
        expand=False,
    )
    processed_lines = all_lines.str.replace(
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


def _clean_coca(raw_line: str, corpus_ids: str | None) -> list[tuple]:
    """Clean a chunk of lines from the CoCA corpus.

    Deals with some of the quirks of the CoCA formats, such as markers,
    tokenization of contractions, numbering, and spacing.

    Args:
        raw_line (str): Lines from the CoCA corpus to be cleaned.
        corpus_ids (str | None): IDs of the sub-corpus (e.g., acad) to which
        the raw_line belong.

    Returns:
        list[tuple]: A list of tuples of corpus id, list of clean lines.

    """
    if isinstance(raw_line, list):
        exception_message = "Oops, something happened"
        raise TypeError(exception_message)
    logger.debug("Cleaning text of length %s characters", len(raw_line))
    processed_lines = regex.sub(
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
    processed_lines = regex.split(
        r"\s*splitmehere\s*",
        processed_lines,
    )
    # Get rid of empty lines
    processed_lines = [line for line in processed_lines if len(line) > 0]
    # Get rid of double spaces and trailing spaces, add header and footer in line
    processed_lines = [
        "START START " + " ".join(line.split()).strip() + " END END"
        for line in processed_lines
        if len(line) > 0
    ]
    logger.debug("Resulted in %s clean lines", len(processed_lines))
    processed_lines = (
        corpus_ids,
        " ".join(processed_lines),
    )
    return [processed_lines]


clean_functions = {
    "bnc": _clean_bnc,
    "coca": _clean_coca,
}


def _ngram_tuple(unigram_list: list, n: int = 2) -> zip:
    """Turn a list of unigrams into an iterable of ngrams.

    Do this by making n copies of the iterable, transposing them
    by a step size of 0 to n, and zipping them together.

    Args:
        unigram_list (list): List of unigrams
        n (int, optional): Length of ngram to extract. Defaults to 2.

    Returns:
        An iterable of tuples containing all ngrams of length n in the unigram list.

    """
    repeated_unigrams = tee(unigram_list, n)
    index_and_unigram = enumerate(repeated_unigrams)
    # islice returns the whole iterator but skipping pos (n) starting elements
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


def _line_to_ngram(text: str, n: int = 2) -> zip:
    """Split text and turn it into ngrams of length n.

    Args:
        text (str): A line of text.
        n (int, optional): Length of ngram to extract. Defaults to 2.

    Returns:
        An iterable of tuples containing all ngrams of length n in the provided text.

    """
    words = text.split()
    return _ngram_tuple(words, n)


def _preprocess_test() -> dict:
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
        raw_lines = corpus_file.read().splitlines()
    split_lines = [line.split() for line in raw_lines]
    fourgrams = [Counter(_line_to_ngram(line, 4)) for line in raw_lines]
    corpora = ["A", "B", "C"]
    fourgrams = [
        (corpus, trigram[0], trigram[1], trigram[2], trigram[3], freq)
        for corpus, corpus_dict in zip(corpora, fourgrams, strict=True)
        for trigram, freq in corpus_dict.items()
    ]
    unigrams = [Counter(unigrams) for unigrams in split_lines]
    unigrams = [
        (corpus, unigram, freq)
        for corpus, corpus_dict in zip(corpora, unigrams, strict=True)
        for unigram, freq in corpus_dict.items()
    ]
    return {"unigrams": unigrams, "fourgrams": fourgrams}


def _extract_ngrams(clean_lines: list) -> tuple:
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
    all_fourgrams = {key: _line_to_ngram(corpus, 4) for key, corpus in clean_lines}
    fourgrams = {
        key: Counter(list(fourgrams)) for key, fourgrams in all_fourgrams.items()
    }
    for key, this_fourgrams in fourgrams.items():
        logger.debug(
            """Corpus %(corpus)s contains %(n_ngrams)s total fourgrams, \
%(n_unique)s unique.""",
            {
                "corpus": key,
                "n_ngrams": sum(this_fourgrams.values()),
                "n_unique": len(this_fourgrams),
            },
        )
    fourgrams = [
        (corpus, *ngram, freq)
        for corpus, corpus_dict in fourgrams.items()
        for ngram, freq in corpus_dict.items()
    ]
    unigrams = {
        corpus: Counter(corpus_lines.split()) for corpus, corpus_lines in clean_lines
    }
    unigrams = [
        (corpus, ngram, freq)
        for corpus, corpus_dict in unigrams.items()
        for ngram, freq in corpus_dict.items()
    ]
    return unigrams, fourgrams


def preprocess_corpus(
    corpus: str,
    raw_lines: str | list[str] | None = None,
    corpus_id: str | None = None,
) -> dict:
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
    clean_function = clean_functions.get(corpus)
    if clean_function is None:
        error_message = "Corpus not supported"
        raise NotImplementedError(error_message)
    this_lines = clean_function(raw_lines, corpus_id)
    unigrams, fourgrams = _extract_ngrams(this_lines)
    return {"unigrams": unigrams, "fourgrams": fourgrams}
