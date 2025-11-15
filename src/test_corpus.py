import chunky

chunky.make_processed_corpus(corpus_name="test", threshold=0)
helper = chunky.Fetcher(corpus="test")

bigrams = ["b d", "c b", "a c"]

test_corpus_results = helper.get_mwu_scores(bigrams, mode="raw")
# print(test_corpus_results)
test_corpus_results["raw"]["dispersion"] = 1 - test_corpus_results["raw"]["dispersion"]
print(test_corpus_results["raw"])
