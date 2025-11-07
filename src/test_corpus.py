from chunky.corpus_helper import Fetcher

helper = Fetcher(corpus="test", make=True)

print(helper.get_ngram_table())


bigrams = ["b d", "c b", "a c"]

test_corpus_results = helper.get_mwu_scores(bigrams, mode="raw")

test_corpus_results["raw"]["dispersion"] = 1 - test_corpus_results["raw"]["dispersion"]
print(test_corpus_results["raw"])
