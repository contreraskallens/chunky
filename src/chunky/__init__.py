"""Main loading."""

import logging

from .corpus import Corpus
from .corpus_helper import Fetcher
from .processing_corpus import make_processed_corpus

__all__ = ["Corpus", "Fetcher", "make_processed_corpus"]
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
