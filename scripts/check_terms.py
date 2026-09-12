import json

terms = [
    "Artificial Intelligence",
    "Machine Learning",
    "Neural Network",
    "Reinforcement Learning",
    "Autonomous Agent",
]

vocab = json.load(open("data/vocab.json"))
glosses = list(vocab.keys()) if isinstance(vocab, dict) else vocab
gloss_set = set(g.lower() for g in glosses)

print(f"Total glosses in vocab: {len(glosses)}\n")

for term in terms:
    words = term.lower().split()
    exact_matches = [w for w in words if w in gloss_set]

    print(f"--- {term} ---")
    if exact_matches:
        print(f"  Exact word matches in vocab: {exact_matches}")
    else:
        print(f"  No exact word matches")
    missing = [w for w in words if w not in gloss_set]
    if missing:
        print(f"  Missing: {missing}")
    print()