# SLM with tiktoken

Educational decoder-only Transformer using the fixed OpenAI `cl100k_base` tokenizer. This project does not train a tokenizer.

## Windows setup
```powershell
cd slm_tiktoken_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, use `\.venv\Scripts\python.exe` in place of `python`.

1. Add text to `data/input.txt`; separate paragraphs with blank lines.
2. Prepare tokens: `python -m src.prepare_data --config configs/quick_test.yaml`
3. Train: `python -m src.train --config configs/quick_test.yaml`
4. Generate: `python -m src.generate --checkpoint checkpoints/best.pt --config configs/quick_test.yaml --prompt "Hello"`

Use `configs/development.yaml` for a longer run after the quick test works.

Important: the tokenizer is fixed and pretrained as an encoding, but the language model weights are not pretrained. A small dataset and only a few steps will not produce a useful chatbot. Old custom-tokenizer checkpoints and token files are incompatible and must be regenerated. `cl100k_base` has about 100K token IDs, so the embedding/output layers use more memory than a tiny custom vocabulary.
