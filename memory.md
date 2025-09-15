# OAC-Chat — Phase 5.2 Working Memory

Purpose: Keep a concise, actionable memory of context, decisions, and next steps while fixing the remaining issues on the server (IDE context).

## Context
- App: Codex MCP Bridge and Chat UI in `docs/files-from-pm-claude`.
- Phase 5.2 remaining bugs addressed:
  1) Smart auto-scroll logic in the chat UI.
  2) AI identity awareness (use current sender, not conversation history).
  3) Human-readable timestamps in the UI.
- Primary UI served at: `/ui/chat` from `mcp_bridge.py` (embedded HTML/JS served by Flask at port 8080).

## Files Touched (Key)
- `docs/files-from-pm-claude/mcp_bridge.py`
  - Adds/updates UI script for chat page: `isAtBottom`, `shouldAutoScroll()`, `formatTimestamp()`; preserves scroll; identity tagging.
  - Chat endpoints (`/chat-with-codex` and `/chat-with-codex/stream`) updated to:
    - Annotate historical user messages with `[from:{ai_id}]`.
    - Insert a strong system message emphasizing current sender.
    - Tag current user message with `[from:{requester_ai_id}]`.
- `docs/files-from-pm-claude/codex-ai/templates/chat.html`
  - Synchronized timestamp formatting and auto-scroll logic for the template-based UI.

## Current Server Symptom
- On server UI: timestamps show raw ISO, e.g., `codex · 2025-09-14T22:42:03.361982+00:00` for session `d39b2d5f-54c2-41be-8a4c-e903773e25ab` at `http://192.168.1.139:8080/ui/chat`.

## Hypothesis
1) The running server is serving an older UI (cache/stale process) that still writes raw ISO to `meta.textContent`.
2) JS `Date` parsing failed for microsecond timestamps, causing fallback to raw text. Fix added: trim fractional seconds to 3 digits.

## What To Check On Server
1) Confirm the route in use:
   - `mcp_bridge.py` has `@app.route('/ui/chat')` -> `_render_chat_ui()`.
2) View page source of `/ui/chat` and verify the presence of:
   - `function formatTimestamp(isoString)` with regex: `/\.(\d{3})\d+(?=[Z+-])/` (microseconds → milliseconds)
   - `isAtBottom` and `shouldAutoScroll()` functions.
3) In `renderMessages(...)`, confirm `meta.textContent` is set using `formatTimestamp(m.ts || '')` (no raw ISO assignment left).
4) Hard-refresh the browser (Ctrl+F5 / Cmd+Shift+R) to bypass cache.

## UI — Expected Behaviors
- Timestamps: Today → `h:mm AM/PM`; older → `Mon dd, h:mm AM/PM`.
- Auto-scroll: Only scroll to bottom if user is already at/near bottom; otherwise preserve scroll offset across refreshes and streaming.

## Identity Awareness — Implementation Notes
- Historical user messages: prepend `[from:{ai_id}]` before content when building model messages.
- Add a strong system message before the current user message:
  - “CRITICAL: The current message is being sent by AI_ID: '{requester_ai_id}'. Respond based on WHO IS CURRENTLY SPEAKING …”
- Current user message: prepend `[from:{requester_ai_id}]`.
- Applied to both non-streaming and streaming endpoints.

## Quick Debug Snippets (Browser Console)
- Check function exists:
  - `typeof formatTimestamp` → should be `"function"`.
- Validate parsing fix:
  ```js
  (() => {
    const s = "2025-09-14T22:42:03.361982+00:00";
    const t = s.replace(/\.(\d{3})\d+(?=[Z+-])/, '.$1');
    console.log('Parsed:', new Date(t).toString());
  })();
  ```

## Next Actions Checklist
1) Ensure the 8080 process is running the updated `mcp_bridge.py` (restart if needed).
2) Verify `/ui/chat` source contains the updated `formatTimestamp` and scroll logic.
3) Confirm timestamps render human-readable (no raw ISO).
4) Test identity: send “Who am I?” with `X-AI-Id: claude` after other messages; expect “You are Claude.”
5) Test auto-scroll: scroll up, wait for refresh/stream; position should be preserved; at bottom should auto-scroll.

## Useful Paths & Commands
- UI route: `/ui/chat` from `docs/files-from-pm-claude/mcp_bridge.py`.
- Tests: `python tests/test_endpoints.py` (passed previously).
- Deploy helper: `docs/files-from-pm-claude/deploy-to-server.ps1` (if applicable).
- PM copy script (not using for live server debugging now): `docs/files-from-pm-claude/send-to-pm-claude.ps1`.

## Notes
- If the page still shows raw ISO timestamps, likely the process is serving an older embedded UI or a proxy is caching. Restart the bridge and hard-refresh the browser.
- If desired, remove any initial raw `meta.textContent` assignment and set only the formatted value to avoid any flicker.

