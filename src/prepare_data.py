import argparse, random
from pathlib import Path
import torch
from src.config import load_config
from src.tokenizer import TikTokenTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/development.yaml")
    a = p.parse_args(); cfg = load_config(a.config)
    random.seed(cfg.get("seed", 42))
    path = Path(cfg["data"]["input_path"])
    if not path.exists(): raise FileNotFoundError(f"Input text not found: {path}")
    docs = [s.strip() for s in path.read_text(encoding="utf-8").split("\n\n") if s.strip()]
    if not docs: raise ValueError("Add text to data/input.txt first")
    random.shuffle(docs)
    nv = max(1, int(len(docs)*float(cfg["data"].get("validation_fraction", .1)))) if len(docs)>1 else 0
    val_docs, train_docs = docs[:nv], docs[nv:] if nv else docs
    tok = TikTokenTokenizer(cfg["tokenizer"].get("encoding_name", "cl100k_base"))
    def encode(items):
        ids=[]
        for doc in items:
            ids.extend(tok.encode(doc)); ids.append(tok.eot_id)
        return torch.tensor(ids, dtype=torch.long)
    train, val = encode(train_docs), encode(val_docs) if val_docs else encode(train_docs)
    for key, data in (("train_path", train), ("val_path", val)):
        out=Path(cfg["data"][key]); out.parent.mkdir(parents=True, exist_ok=True); torch.save(data, out)
    print(f"Encoding: {tok.encoding_name}; vocab size: {tok.vocab_size:,}; EOT: {tok.eot_id}")
    print(f"Documents: {len(train_docs)} train, {len(val_docs)} validation")
    print(f"Tokens: {len(train):,} train, {len(val):,} validation")
if __name__ == "__main__": main()
