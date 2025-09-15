# local_Codex_LLM

Local LLM I named codex, I had this named before OpenAI created theirs, this is not related, I just like the name.

## Codex AI Bridge (Modular)

- Entry: `main.py`
- Config: `config.py` (backward-compatible with CODEX_* envs)
- Routes: `routes/` (Flask blueprints)
- Core: `core/` (metrics, sessions, MCP client, security)
- Templates: `templates/` (UI extracted from monolith)

### Dev

- `pip install -r requirements.txt`
- `python main.py`

### Deploy

- `rsync -av . user@server:~/codex-ai/`
- `ssh user@server "cd ~/codex-ai && chmod +x run.sh && ./run.sh"`
