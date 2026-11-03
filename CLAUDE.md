# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AstrBot plugin for a "Sea Turtle Soup" (海龟汤) reasoning game. It's a Chinese puzzle game where players ask yes/no questions to deduce a story.

## Architecture

- **Main plugin**: `main.py` holds the plugin class, commands and game flow
- **`core/` package**: Jev decision backend split into two modules
  - `core/jev_transport.py` — HTTP layer: local request pre-validation, idempotency
    keys, error-code mapping, remaining-credit tracking
  - `core/jev_judge.py` — pure decision logic: structured state construction,
    language-aware 4-way choice mapping, confidence gating. **Does not import
    astrbot**, so it is unit-testable on its own. Bump `PROMPT_VERSION` whenever
    the instructions / criteria / state shape change
- **AstrBot framework**: Built on AstrBot >= v3.4.36
- **Thread-safe storage**: `ThreadSafeStoryStorage` class for managing puzzle usage
- **Game state management**: `GameState` class tracks active games per group
- **LLM integration**: Uses AstrBot's Provider interface for AI functionality
  (puzzle generation, hints, verification); the question-judging path can be
  delegated to Jev instead — see `judge_question()`
- **Session management**: Uses `session_waiter` for conversation control
- **Relative imports**: AstrBot loads plugins as `data.plugins.<dir>.main`, so
  `from .core.x import y` is the correct import style here

## Key Components

- **Network puzzle database**: `network_soupai.json` with ~300 pre-scraped puzzles
- **Configuration**: `_conf_schema.json` defines plugin settings
  (`judge_backend` = llm/jev/hybrid, plus the `jev_judge` group)
- **Metadata**: `metadata.yaml` for plugin registration
- **Tests**: `tests/` — pytest, with a minimal astrbot stub in `tests/conftest.py`
- **Eval**: `scripts/eval_judgment.py` — offline judgment regression against the
  real Jev API (needs `JEVMODEL_API_KEY`); run it after any prompt change

## Development Commands

This is an AstrBot plugin, so development involves:
1. **Testing**: `pytest tests/ -v` (needs `pytest` + `pytest-asyncio`; the plugin
   itself does not depend on them at runtime)
2. **Linting**: No specific linting configuration found
3. **Building**: No build process - it's a Python plugin directory
4. **Installation**: Copy to `AstrBot/data/plugins/` directory

## Plugin Structure

- **Main class**: Registered with `@register` decorator
- **Command handlers**: Decorated with `@filter.command`
- **Session handlers**: Use `@session_waiter` for conversation flow
- **LLM integration**: Uses `self.context.get_using_provider()` and `self.context.get_provider_by_id()`

## Important Patterns

- **Thread safety**: Uses `threading.Lock` for shared state
- **Persistence**: JSON files for usage tracking
- **Error handling**: Comprehensive try-catch blocks with logging
- **Configuration**: Managed through AstrBot's plugin configuration system

## Development Workflow

1. Modify the code
2. Run `pytest tests/ -v`
3. Reload plugin in AstrBot WebUI
4. Test commands in chat
5. Check logs for errors

## Performance Notes

The plugin's latency is dominated by network calls, not local computation: a
profiled game (20 questions + 3 hints + 1 verification) burns ~2 ms of CPU
total. When changing the hot path, measure first — micro-optimisations of the
Python side are irrelevant next to a 2-15 second model call.

## Key Files

- `main.py` - Core plugin implementation
- `core/jev_transport.py` - Jev HTTP transport
- `core/jev_judge.py` - Jev decision logic (pure, no astrbot import)
- `network_soupai.json` - Puzzle database
- `_conf_schema.json` - Configuration schema
- `metadata.yaml` - Plugin metadata
- `docs/superpowers/specs/` - Design documents