import re
import pandas as pd

res = pd.read_csv("data/explainer_check.csv")


def count_sentences(text):
    t = re.sub(r"\b(?:[A-Z]\.){2,}", lambda m: m.group().replace(".", ""), text)
    parts = [p for p in re.split(r"(?<=[.!?])\s+", t.strip()) if p.strip()]
    return len(parts)


res["real_sentences"] = res["text"].apply(count_sentences)
print("Exactly two sentences (initials handled):", int((res["real_sentences"] == 2).sum()), "of", len(res))
print(res.loc[res["real_sentences"] != 2, ["player", "real_sentences"]].to_string(index=False))

print()
print("Wording scan: how many of the 50 contain each phrase")
WATCH = ["league", "than usual", "more minutes", "fewer minutes", "since ", "recent", "lately"]
low = res["text"].str.lower()
for w in WATCH:
    print(f"  {w!r}: {int(low.str.contains(w, regex=False).sum())}")

print()
print("Examples of the phrases that matter most:")
for w in ["league", "than usual", "since "]:
    hits = res[low.str.contains(w, regex=False)]
    for _, r in hits.head(3).iterrows():
        print("-" * 50)
        print(f"[{w!r}] {r['player']}: {r['text']}")