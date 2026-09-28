import logging
from pathlib import Path
from pprint import pprint as print

import duckdb as db
import regex

logger: logging.Logger = logging.getLogger(__name__)

import pandas as pd


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


def _clean_coca_old(raw_line: str, corpus_ids: str) -> list[tuple[str, str]]:
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


# clean_functions: dict[str, Callable[..., list[tuple[str, str]]]] = {
#     "bnc": _clean_bnc,
#     "coca": _clean_coca,
# }


def _read_coca(corpus_path: Path, con: db.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE corpus_parts AS
        SELECT row_number() OVER (ORDER BY filename) AS part_id,
            filename,
            content
        FROM read_text(?)
        """,
        [str(corpus_path.resolve())],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE part_ids AS
        SELECT part_id,
            parse_filename(filename, true) AS filename
        FROM corpus_parts
        """
    )


def _clean_coca(con):
    con.execute(r"""
        CREATE OR REPLACE TEMP TABLE clean_text AS
        SELECT part_id,
            content.
                nfc_normalize().
                lower().
                regexp_replace(
                    '[‘’]',
                    '''',
                    'g'
                ).
                regexp_replace(
                    ' [.?!] |\n|(@ )+|</*[ph]>|<br>',
                    ' splitmehere ',
                    'g'
                ).
                regexp_replace(
                    ' (n''t|''s|''ll|''d|''re|''ve|''m)\b',
                    '\1',
                    'g'
                ).
                regexp_replace(
                    '@@\d+\s*',
                    '',
                    'g'
                ).
                regexp_replace(
                    '\bwan na\b',
                    'wanna',
                    'g'
                ).
                regexp_replace(
                    '\bgon na\b',
                    'gonna',
                    'g'
                ).
                regexp_replace(
                    '\bgot ta\b',
                    'gotta',
                    'g'
                ).
                regexp_replace(
                    '-',
                    ' ',
                    'g'
                ).
                regexp_replace(
                    '\d+([.,]\d+)*',
                    ' NUMBER ',
                    'g'
                ).
                regexp_replace(
                    '[^\p{L}\p{N}'' ]',
                    ' ',
                    'g'
                ).
                regexp_replace(
                    '[^\p{L}]''|''[^\p{L}]',
                    ' ',
                    'g'
                ).
                regexp_replace(
                    '\s+',
                    ' ',
                    'g'
                )
            AS text

        FROM corpus_parts
    """)
    con.execute("DROP TABLE corpus_parts")


def _split_coca(con):
    con.execute(r"""
        CREATE OR REPLACE TABLE part_sentences AS
        SELECT part_id, sentences.text, sentences.sentence_id
        FROM(
            SELECT part_id,
                list_filter(
                    string_split_regex(text, '(\s*splitmehere\s*)+'),
                    lambda x: x <> ''
                ) AS text,
            FROM clean_text
        ), unnest(text) WITH ORDINALITY AS sentences(text, sentence_id)
    """)
    con.execute("DROP TABLE clean_text")


def _process_coca(
    # corpus_path: Path,
    # cat_chunk: list[Path],
    # cat_name: str,
    # corpus_ids: dict[str, str],
    # chunk_size: int = 5,
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
    # logger.info("Adding subcorpus '%s'.", cat_name)
    # text_batches = batched(cat_chunk, chunk_size)
    # text_chunks: list[str] = []

    con = db.connect(str(Path("db/temp/corpus_cleaning.db").resolve()))
    # This reads the corpus into a a table, one line per corpus part
    _read_coca(Path("corpora/coca_sample/*.txt"), con)
    _clean_coca(con)
    _split_coca(con)


if __name__ == "__main__":
    _process_coca()
