"""Main loading."""

import logging

from .corpus import Corpus
from .corpus_helper import Fetcher

__all__ = ["Corpus", "Fetcher"]
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
