from chunky.corpus_helper import Fetcher

# mwu_examples = pd.read_csv("../MultiwordExpression_Concreteness_Ratings.csv")
# mwu_examples["length"] = mwu_examples["Expression"].apply(lambda x: len(x.split()))
# mwu_examples = mwu_examples.loc[
#     (mwu_examples["length"] == 2) | (mwu_examples["length"] == 3)
# ]
# mwu_examples["Expression"] = mwu_examples["Expression"].apply(lambda x: x.lower())


helper = Fetcher(
    "coca_sample",
    make=False,
    # make=True,
    # corpus_dir="chunky/corpora/coca_sample/",
    # threshold=1,
    # chunk_size=5,
)

# helper = Fetcher("coca_sample")


sentence = (
    "Hello! Come in and eat boiled potatoes they are some of the most but it was never"
)
x = helper.get_mwu_scores(sentence)
print(x)
print(
    helper.get_mwu_scores(
        ["Come in and eat", "I don't want to", "What do you know"],
        mode="normalized",
    )
)


# print(x["normalized"])


# pp(helper.corpus.query_parquet())

# pp(helper.corpus.query_parquet(ug=4756275072626810161))
