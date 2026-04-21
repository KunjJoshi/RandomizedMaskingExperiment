import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def preprocess_text(text):
    return set(text.lower().split())

def create_minhash(tokens, num_perm = 128):
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode('utf-8'))
    return m

df = pd.read_csv('../datasets/enron/train_split.csv')

lsh = MinHashLSH(threshold = 0.8, num_perm=128)
minhashes = {}

for idx, email in tqdm(enumerate(df['message']), desc = "Creating Minhashes"):
    tokens = preprocess_text(email)
    m = create_minhash(tokens)
    minhashes[idx] = m
    lsh.insert(idx, m)

duplicates = set()
for idx in tqdm(range(len(df)), desc="Gathering Results"):
    result = lsh.query(minhashes[idx])
    if len(result) > 1:
        duplicates.update(result[1:])

df_dedup_docs = df.drop(index=list(duplicates)).reset_index(drop=True)

print(f"Original: {len(df)} emails")
print(f"After document dedup: {len(df_dedup_docs)} emails")
print(f"Removed: {len(duplicates)} duplicate emails")

df_dedup_docs.to_csv('../datasets/enron/deduplicated_docs_enron_training_split.csv', index=False)


