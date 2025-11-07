import chunky

my_corpus = chunky.Corpus(
    "coca",
    make=True,
    corpus_dir="chunky/corpora/coca_texts/",
    chunk_size=10,
    threshold=1,
)
