# Multiword unit measures
Package of functions to obtain MWU scores for chunks.
Based on the proposal by Stefan Gries, "Multi-word units (and tokenization more generally): a multi-dimensional and largely information-theoretic approach", https://journals.openedition.org/lexis/6231.

# How to use this package
This package leverages the `duckdb` SQL-like engine to provide fast access with an easy interface.

The package is based around two main types of object: a `Corpus` and a `Helper`. `Corpus` objects store frequency information about a named preprocessed corpus of text. `Helper` objects have simple functions to query the `Corpus` to obtain statistics about provided _n_-grams or lists of _n_-grams.
* `make_corpus.py` illustrates how to make your local copy of a named corpus. Currently, the only supported corpus is `COCA`. Soon, support for the `BNC` and `Brown` will be fixed from previous versions.
* The main workhorse of the package is the `Helper` method `get_mwu_scores()`. This takes a list of _n_-grams and returns a dataframe with MWU measures. For now, I recommend passing `normalized = False` and normalizing and averaging the scores yourself, as I'm still working on the part of the script that normalizes all measures at once. You can find instructions for normalization in Gries' original paper.