# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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
