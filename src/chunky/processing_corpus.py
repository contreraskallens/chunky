"""Module with functions for building data structures for MWU extraction."""

# TODO(omfgzell): Registry instead of if functions for custom corpus #03
# TODO(omfgzell): Clean cat_name #04

from __future__ import annotations

import logging
import re
import string
from itertools import groupby
from pathlib import Path

from . import create_corpus as create
from . import preprocessing_corpus as preprocess

logger = logging.getLogger(__name__)


def _process_test() -> dict:
    """Build the test corpus.

    Processes the test corpus from Gries' original publication and
    adds them to a Corpus object.

    Args:
        corpus (corpus.Corpus): A Corpus object that's been already initialized.

    """
    return preprocess.preprocess_corpus(corpus="test")


def _process_bnc(
    corpus_path: Path,
    corpus_dir: Path,
    chunk_size: int = 1000000,
) -> None:
    """Build the BNC corpus.

    Processes the BNC corpus line by line and adds the obtained
    ngrams to a Corpus object. Can do this in chunks to speed
    up the process.

    Args:
        corpus_path (Path): The path to the database.
        corpus_dir (Path): The directory where the BNC corpus file is located.
        The function is prepared to work with bnc_tokenized.txt.
        chunk_size (int, optional): The number of lines to process at once.
        Can improve speed of processing but adds memory load. Defaults to 1000000.

    """
    with corpus_dir.open(encoding="utf-8") as corpus_file:
        i = 0
        while True:
            raw_lines = corpus_file.readlines(chunk_size)
            if not raw_lines:
                break
            ngram_dicts = preprocess.preprocess_corpus(
                corpus="bnc",
                raw_lines=raw_lines,
            )
            create.add_chunk(path=corpus_path, ngrams=ngram_dicts)
            i += len(raw_lines)
            logger.debug("%s lines processed", i)


def _prepare_coca(corpus_dir: Path) -> tuple:
    """Prepare the files in the CoCA directory.

    Take the files int he CoCA directory and build a catalog with them
    to process them into the ngram corpus. Extract the name of each of
    sub-corpus and associate them with the text file.

    Args:
        corpus_dir (Path): Folder with CoCA texts, usually the coca_texts folder.

    Returns:
        tuple: A tuple containing the ids of each corpus as a dictionary and the
        category of the corresponding files.

    """
    coca_texts = sorted(corpus_dir.iterdir())
    coca_cats = [
        re.search(r"_.+_", str(text_name), re.IGNORECASE) for text_name in coca_texts
    ]
    coca_cats = [text_name.group(0) for text_name in coca_cats if text_name is not None]
    coca_cats = list(set(coca_cats))
    corpus_ids = dict(
        zip(
            sorted(coca_cats),
            [string.ascii_uppercase[i] for i in range(len(coca_cats))],
            strict=True,
        ),
    )
    coca_text_cats = groupby(
        coca_texts,
        lambda x: re.search(r"_.+_", str(x), re.IGNORECASE).group(0),  # type: ignore[attr-defined]
    )
    coca_text_cats = [
        (cat_name, list(cat_chunk)) for cat_name, cat_chunk in coca_text_cats
    ]
    return corpus_ids, coca_text_cats


def _process_coca(
    corpus_path: Path,
    cat_chunk: list,
    cat_name: str,
    corpus_ids: dict,
    chunk_size: int = 5,
) -> None:
    """Build the CoCA corpus.

    Take a prepared CoCA corpus, divide into chunks of texts, obtain the ngrams,
    and then add them to a Corpus object.

    Args:
        corpus_path (Path): The path to the database.
        cat_chunk (list): Chunk of texts for a CoCA subcorpus
        cat_name (str): Name of the subcorpus as a file path.
        corpus_ids (dict): Ids of the corpora, e.g. {"academic": "A"}
        chunk_size (int, optional): Number of texts to process at once.
        Defaults to 5. Larger numbers might improve speed at the cost of memory.

    """
    short_name = re.search("acad|blog|fic|mag|news|spok|tvm|web", cat_name)
    if short_name:
        logger.info("Adding subcorpus '%s'.", short_name.group())
    # ? Could maybe flatten this loop
    text_chunks = [
        cat_chunk[i : i + chunk_size] for i in range(0, len(cat_chunk), chunk_size)
    ]
    for chunk in text_chunks:
        chunk_text = ""
        chunk_cat = corpus_ids[cat_name]
        for coca_text in chunk:
            with coca_text.open() as corpus_file:
                raw_lines = corpus_file.read()
            chunk_text = chunk_text + " \n " + raw_lines
        ngram_dicts = preprocess.preprocess_corpus(
            raw_lines=chunk_text,
            corpus="coca",
            corpus_id=chunk_cat,
        )
        create.add_chunk(path=corpus_path, ngrams=ngram_dicts)


def make_processed_corpus(
    corpus_name: str = "test",
    corpus_dir: str | Path = "test",
    chunk_size: int = 1000000,
    threshold: int = 2,
    env_config: dict = create.DEFAULT_CONFIG,
) -> None:
    """Construct and allocate the data of a corpus object.

    Given a corpus object that was created with the make=True argument,
    process and allocate its data as for computing MWU scores. Data is allocated
    in the disk under the /db directory. Files needed for each supported corpus
    can be located in the /corpora directory.
    For BNC, the provided directory of the corpus must be the bnc_tokenized.txt file.
    For CoCA, the provided directory of the corpus must be a folder containing the CoCA
    corpus files, each one using the standard file names of e.g. text_acad_1990.txt.
    No corpus_dir is needed for the test corpus.

    Args:
        corpus (str): The name of the corpus to process.
        corpus_dir (str | Path | None, optional): The directory of the corpus files.
        See the description for more details. Defaults to "test".
        chunk_size (int, optional): Size of the text chunk to be processed.
        Defaults to 1000000 lines for the BNC and 5 texts for the CoCA.
        threshold (int, optional): Minimum token frequency of each ngram to be stored
        in the corpus file. Defaults to 2.

    Returns:
        corpus.Corpus: An allocated Corpus object.

    """
    db_path = Path(f"chunky/db/{corpus_name}.db")
    create.init_corpus(db_path)
    if not create.validate_corpus_name(corpus_name):
        msg = "Not a valid corpus name."
        raise ValueError(msg)
    if corpus_name == "test":
        ngrams = _process_test()
        create.add_chunk(db_path, ngrams)
    elif corpus_dir is None:
        exception_msg = "Corpus file not provided"
        raise RuntimeError(exception_msg)
    # ? Turn into registry of functions?

    elif corpus_name == "bnc":
        corpus_dir = Path(corpus_dir)
        _process_bnc(corpus_path=db_path, corpus_dir=corpus_dir, chunk_size=chunk_size)
    elif corpus_name in {"coca", "coca_sample", "coca_fourgrams"}:
        corpus_dir = Path(corpus_dir)
        corpus_ids, coca_text_cats = _prepare_coca(corpus_dir)
        for cat_name, cat_chunk in coca_text_cats:
            _process_coca(
                corpus_path=db_path,
                cat_name=cat_name,
                cat_chunk=cat_chunk,
                corpus_ids=corpus_ids,
            )
    else:
        exception_msg = "Corpus not supported."
        raise NotImplementedError(exception_msg)
    logger.info("Done adding to DB. Consolidating...")
    create.consolidate_corpus(
        path=db_path,
        corpus_name=corpus_name,
        threshold=threshold,
        env_config=env_config,
    )
    logger.info("Done creating totals. Corpus allocated and ready for use.")
