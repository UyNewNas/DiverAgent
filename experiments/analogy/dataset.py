import os, sys, random, zipfile, io, urllib.request, gzip, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import EMBED_DIM, BATCH_SIZE, RELATION_TYPES, NUM_RELATIONS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
GLOVE_URL = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip'
GLOVE_ZIP = os.path.join(DATA_DIR, 'glove.6B.zip')
GLOVE_CACHE = os.path.join(DATA_DIR, 'glove_embeddings.pt')
VOCAB_CACHE = os.path.join(DATA_DIR, 'word_vocab.pkl')
TRIPLE_CACHE = os.path.join(DATA_DIR, 'analogy_triples.pkl')


def download_glove():
    if os.path.exists(GLOVE_ZIP):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print('Downloading GloVe 6B embeddings (~822MB)...')
    urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)
    print('Download complete.')


def load_glove():
    if os.path.exists(GLOVE_CACHE) and os.path.exists(VOCAB_CACHE):
        embeddings = torch.load(GLOVE_CACHE, weights_only=True)
        with open(VOCAB_CACHE, 'rb') as f:
            vocab = pickle.load(f)
        return embeddings, vocab

    download_glove()
    print('Parsing GloVe embeddings...')
    word_vec = {}
    with zipfile.ZipFile(GLOVE_ZIP, 'r') as zf:
        with zf.open('glove.6B.300d.txt') as f:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                parts = line.strip().split()
                if len(parts) != 301:
                    continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                word_vec[word] = vec

    words = sorted(word_vec.keys())
    vocab = {w: i for i, w in enumerate(words)}
    embed_mat = np.zeros((len(words), EMBED_DIM), dtype=np.float32)
    for w, i in vocab.items():
        embed_mat[i] = word_vec[w]

    embeddings = torch.from_numpy(embed_mat)
    torch.save(embeddings, GLOVE_CACHE)
    with open(VOCAB_CACHE, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Loaded {len(vocab)} words, embedding shape: {embeddings.shape}')
    return embeddings, vocab


def extract_relation_triples(vocab):
    if os.path.exists(TRIPLE_CACHE):
        with open(TRIPLE_CACHE, 'rb') as f:
            return pickle.load(f)

    from nltk.corpus import wordnet as wn

    def lemma_name(synset):
        name = synset.lemmas()[0].name().lower().replace('_', ' ')
        return name

    triples = []
    relation_funcs = {
        'hypernym': lambda s: s.hypernyms(),
        'hyponym': lambda s: s.hyponyms(),
        'meronym': lambda s: s.part_meronyms() + s.substance_meronyms() + s.member_meronyms(),
        'holonym': lambda s: s.part_holonyms() + s.substance_holonyms() + s.member_holonyms(),
        'attribute': lambda s: s.attributes(),
        'entailment': lambda s: s.entailments(),
        'cause': lambda s: s.causes(),
    }

    synsets = list(wn.all_synsets())
    for i, syn in enumerate(synsets):
        if i % 10000 == 0:
            print(f'  Processing synset {i}/{len(synsets)}...')
        head_word = lemma_name(syn)
        if head_word not in vocab:
            continue
        for rel_name in RELATION_TYPES:
            related = relation_funcs[rel_name](syn)
            for rel_syn in related:
                tail_word = lemma_name(rel_syn)
                if tail_word in vocab and tail_word != head_word:
                    triples.append((head_word, rel_name, tail_word))

    with open(TRIPLE_CACHE, 'wb') as f:
        pickle.dump(triples, f)
    print(f'Extracted {len(triples)} analogy triples')
    return triples


def relation_to_idx(rel_name):
    return RELATION_TYPES.index(rel_name)


def split_triples(triples, test_ratio=0.2, seed=42):
    rng = random.Random(seed)

    rel_triples = {rel: [] for rel in RELATION_TYPES}
    for h, r, t in triples:
        rel_triples[r].append((h, t))

    train_pairs = set()
    test_pairs = set()
    train_triples = []
    test_triples = []

    for rel_name in RELATION_TYPES:
        pairs = rel_triples[rel_name]
        rng.shuffle(pairs)
        if len(pairs) < 10:
            train_triples.extend([(h, rel_name, t) for h, t in pairs])
            train_pairs.update(pairs)
            continue
        n_test = max(1, int(len(pairs) * test_ratio))
        test_batch = pairs[:n_test]
        train_batch = pairs[n_test:]
        train_triples.extend([(h, rel_name, t) for h, t in train_batch])
        test_triples.extend([(h, rel_name, t) for h, t in test_batch])
        train_pairs.update(train_batch)
        test_pairs.update(test_batch)

    overlap = train_pairs & test_pairs
    if overlap:
        print(f'  WARNING: {len(overlap)} pairs overlap between train/test!')

    print(f'Train triples: {len(train_triples)}, Test triples: {len(test_triples)}')
    return train_triples, test_triples


class AnalogyDataset(Dataset):
    def __init__(self, triples, vocab):
        self.triples = triples
        self.vocab = vocab

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head_word, rel_name, tail_word = self.triples[idx]
        head_idx = self.vocab[head_word]
        tail_idx = self.vocab[tail_word]
        rel_idx = relation_to_idx(rel_name)
        return head_idx, rel_idx, tail_idx, head_word, tail_word


def collate_analogy(batch):
    head_idxs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    rel_idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    tail_idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    head_words = [b[3] for b in batch]
    tail_words = [b[4] for b in batch]
    return head_idxs, rel_idxs, tail_idxs, head_words, tail_words


def get_dataloaders(batch_size=BATCH_SIZE):
    embeddings, vocab = load_glove()
    print('Extracting relation triples from WordNet (may take a few minutes)...')
    triples = extract_relation_triples(vocab)
    print('Splitting triples...')
    train_triples, test_triples = split_triples(triples)

    train_ds = AnalogyDataset(train_triples, vocab)
    test_ds = AnalogyDataset(test_triples, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_analogy, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_analogy, drop_last=False,
    )

    return train_loader, test_loader, embeddings, vocab, train_ds, test_ds
