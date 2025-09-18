# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.0](https://github.com/Wazaq/local_Codex_LLM/compare/v0.1.0...v0.2.0) (2025-09-18)


### Features

* add friendly session names and ui controls ([12e7df9](https://github.com/Wazaq/local_Codex_LLM/commit/12e7df901edcc47be99164d3fba1e2096d507060))
* add health dashboard and control center ([55d1797](https://github.com/Wazaq/local_Codex_LLM/commit/55d17978fe09c1b718cdf9b055e14a0a682e7b48))
* integrate MCP server and session updates ([01722d3](https://github.com/Wazaq/local_Codex_LLM/commit/01722d31ad5df5dcf312f66983909bd5cc1e4bce))
* polish chat ui error handling ([8cba22c](https://github.com/Wazaq/local_Codex_LLM/commit/8cba22c6fff92bf82a9359f60982eeb59b11c486))
* **session:** add file-backed session persistence ([3ec47ed](https://github.com/Wazaq/local_Codex_LLM/commit/3ec47edff2d1855189e324d68e62c4b1efda34cc))
* standardize error handling and add tests ([299d2f5](https://github.com/Wazaq/local_Codex_LLM/commit/299d2f58da9d90bac3133792a98b692ad3635b48))
* **ui:** add "Copy SysID" button to copy current session ID ([ae45351](https://github.com/Wazaq/local_Codex_LLM/commit/ae453518b84d06adfb59b92d5b6af1cc821d6a3a))

## [Unreleased]

## [0.1.0] - 2025-09-15

### Added
- Human-readable timestamps in chat UI with ISO microsecond normalization.
- Smart auto-scroll logic that preserves position when user is not at bottom.
- Strong current-speaker identity handling in backend (system emphasis + message tags).

### Changed
- README consolidated with project overview and quick-start commands.

### Fixed
- UI auto-refresh no longer forces scroll to bottom.
- Identity confusion during multi-speaker sessions (responds based on current sender).

### Removed
- Duplicate `codex-ai/*` subfolder to avoid confusion with active app files.

[Unreleased]: https://github.com/Wazaq/local_Codex_LLM/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Wazaq/local_Codex_LLM/releases/tag/v0.1.0
