"""TURBO runtime patches for the Duck harness (applied from a notebook cell AFTER the AGENTFIX cell and
BEFORE the benchmark runs; ToolAgents and game sessions are built per game inside bm.run, so patching the
module globals / classes here reaches every game).

Every patch is fail-open: an installer that cannot find its anchor, or a wrapper that raises, leaves the
stock (AGENTFIX) behaviour untouched. `install()` never raises; it returns a report the notebook prints (and
asserts on in a non-submission commit run, so a broken anchor is caught before a submission).

T1  TURBO_HISTORY_KEEP (64)   The runtime-state file handed to the python sandbox keeps the start frame of the
                               current level plus the newest N transitions, as compact JSON. Stock rewrites the
                               WHOLE history with indent=2 after every action and every `action()` call reloads and
                               re-serializes it into the sandbox: ~1.7 s per action at 300 entries on one core,
                               shared by all game threads through the GIL, so long games hit the 30 s tool timeout
                               and lose the whole model call (Tufa example-run: 26-38 % tool timeouts past 200
                               actions). Cut entries get an empty action, so `transitions` never has a None
                               `before_frame`, and `history[0]` is the start of the current level. The solver's own
                               in-memory history (scoring, level tracking) is untouched.
T2  TURBO_SANDBOX_BUILTINS    `class` statements, `object`, `super` and the common exception names (KeyError,
                               IndexError, AttributeError, ...) were missing from the sandbox, so any snippet that
                               defined a class or caught KeyError failed with NameError. Also allows the pure-stdlib
                               dataclasses / typing / enum modules.
T3  TURBO_ANALYZE_GUARD (6)   An analyzer exception (or a None result) used to mark the game `crashed` for the
                               rest of the run. Now it is retried (backoff 5 s, 10 s, ... max 30 s) up to N
                               consecutive times before falling back to the stock behaviour.
T4  TURBO_TEMPERATURE (1.0)   Sampling temperature when the serving profile has MTP off (Qwen3.8-Flash-Next model
                               card thinking setting; every Son Pham Flash-Next config that scored used 1.0). With
                               the MTP-3 fallback profile the stock 0.6 is kept (MTP acceptance drops at 1.0).
    TURBO_YIELD_SECONDS (120) Analyzer yield window (stock setup persists 60 s; each yield re-sends the full user
                               prompt + board image), with TURBO_TOOL_STEPS (3) so a long turn can never evict its own
                               user prompt from the context window.
T5  TURBO_PROMPT              System prompt: Son Pham & Mark Barney's Kaggle-measured arm B (delete four false or
                               distracting claims) + arm C (six "mechanics may be" bullets), measured on this exact
                               keithtyser bundle on Kaggle hardware (control 0.864, B 1.216, B+C 1.550 on the 7 hardest
                               public games, 4 passes); and the action-minimising line is deleted (cleared levels are
                               already at 0.7-0.8x the human baseline; reach, not thrift, is what is missing).
                               ACTION7 is shown as `UNDO` (executable; it is undo per the ARC docs) instead of being
                               advertised as ACTION7 and rejected. The stale "The game is over." line (the harness has
                               already reset the level) is corrected. A neutral game-clock line is added to the
                               outgoing copy of the current turn prompt only (never stored in history).
T6  per-game budget           Each game's time budget is computed when the game starts: the time left before the
                               soft deadline divided by the number of waves still needed for the games not yet
                               started (configure_budget() is called by the run cell). A wave can no longer be cut
                               by the deadline after a slow setup, a serving fallback or a watchdog restart, and
                               games that start after an early finish get the freed time.
T7  TURBO_HALF_SWAP (auto)    Only when the serving profile has prefix caching on: when a new turn overflows the
                               context budget, trim history down to 60 % of the budget instead of dropping one block
                               per turn, so the next few turns extend a stable, cache-hit prefix.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

_ON = lambda k, d="1": os.environ.get(k, d).strip().lower() not in ("0", "", "false", "no", "off")


def _int_env(key: str, default: int) -> int:
    try:
        return int(str(os.environ.get(key, default)).strip() or default)
    except Exception:
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(str(os.environ.get(key, default)).strip() or default)
    except Exception:
        return default


# --------------------------------------------------------------------------------------------- T1 history cap
HISTORY_NOTE_OLD = "- `history` is a chronological list of action/frame snapshots.\n"
HISTORY_NOTE_NEW = ("- `history` is a chronological list of action/frame snapshots: the start of the current level "
                    "followed by the most recent {keep} actions.\n")


def capped_entries(history: list, keep: int) -> list:
    """(entry, blank_action) pairs: the current level's start frame (if it falls before the tail) and the tail of
    `keep` transitions; the first entry of each kept run gets an empty action, so the sandbox builds no transition
    whose before-frame is missing or belongs to a cut part of the history."""
    n = len(history)
    if keep <= 0 or n <= keep + 1:
        return [(entry, False) for entry in history]
    tail_start = n - (keep + 1)
    level = getattr(history[-1].frame, "level", None)
    start = n - 1
    while start > 0 and getattr(history[start - 1].frame, "level", None) == level:
        start -= 1
    pairs = []
    if start < tail_start:
        pairs.append((history[start], True))
    pairs.append((history[tail_start], True))
    pairs.extend((entry, False) for entry in history[tail_start + 1:])
    return pairs


def _install_history_cap(rs_mod, solv_mod, ta_mod, rep: dict) -> None:
    keep = _int_env("TURBO_HISTORY_KEEP", 64)
    rep["T1_history_keep"] = keep
    if keep <= 0:
        return
    orig = solv_mod.write_runtime_state  # the name _HarnessGameSession.write_runtime_state calls

    def write_runtime_state(path, *, current_frame, history):
        try:
            items = []
            for entry, blank in capped_entries(list(history), keep):
                # looked up at call time: AGENTFIX F13 wraps history_entry_to_payload (keyed on the entry object)
                payload = rs_mod.history_entry_to_payload(entry)
                if blank:
                    payload["action"] = ""
                items.append(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            document = {"current_frame": rs_mod.frame_to_payload(current_frame), "history": items}
            tmp_path = path.with_suffix(f"{path.suffix}.tmp")
            tmp_path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            return orig(path, current_frame=current_frame, history=history)

    solv_mod.write_runtime_state = write_runtime_state
    try:
        if HISTORY_NOTE_OLD in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
            ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM.replace(
                HISTORY_NOTE_OLD, HISTORY_NOTE_NEW.format(keep=keep))
            rep["T1_history_note"] = True
    except Exception as exc:
        rep["T1_note_error"] = repr(exc)
    rep["T1_history_cap"] = True


# --------------------------------------------------------------------------------------------- T2 sandbox builtins
EXTRA_BUILTINS = (
    "__build_class__", "object", "super", "property", "staticmethod", "classmethod", "id", "setattr", "delattr",
    "vars", "KeyError", "IndexError", "AttributeError", "NameError", "ZeroDivisionError", "StopIteration",
    "LookupError", "ArithmeticError", "AssertionError", "NotImplementedError", "RecursionError",
    "OverflowError", "BaseException", "ImportError", "UnboundLocalError",
)
EXTRA_MODULES = ("dataclasses", "typing", "enum")  # pure-stdlib helpers models often import
BUILTINS_ANCHOR = "SAFE_BUILTINS = {\n"
MODULES_ANCHOR = "SAFE_MODULES = {\n"
GLOBALS_ANCHOR = '        "result": None,\n    }\n'


def _extend_set_literal(src: str, anchor: str, names) -> tuple[str, int]:
    head, tail = src.split(anchor, 1)
    body, rest = tail.split("}", 1)
    existing = {item.strip().strip('"') for item in body.split(",") if item.strip()}
    missing = [name for name in names if name not in existing]
    return head + anchor + "".join(f'    "{name}",\n' for name in missing) + body + "}" + rest, len(missing)


def _install_sandbox_builtins(sb_mod, rep: dict) -> None:
    src = sb_mod._SANDBOX_BOOTSTRAP
    if any(src.count(anchor) != 1 for anchor in (BUILTINS_ANCHOR, MODULES_ANCHOR, GLOBALS_ANCHOR)):
        rep["T2_error"] = "anchor not found"
        return
    patched, n_builtins = _extend_set_literal(src, BUILTINS_ANCHOR, EXTRA_BUILTINS)
    patched, n_modules = _extend_set_literal(patched, MODULES_ANCHOR, EXTRA_MODULES)
    patched = patched.replace(GLOBALS_ANCHOR,
                              '        "result": None,\n        "__name__": "__python_tool__",\n    }\n', 1)
    compile(patched, "<python_tool_sandbox_bootstrap>", "exec")
    sb_mod._SANDBOX_BOOTSTRAP = patched
    rep["T2_sandbox_builtins"] = n_builtins
    rep["T2_sandbox_modules"] = n_modules


# --------------------------------------------------------------------------------------------- T3 analyze guard
def _install_analyze_guard(ta_mod, rep: dict) -> None:
    max_failures = _int_env("TURBO_ANALYZE_RETRIES", 6)
    if max_failures <= 0:
        return
    TA = ta_mod.ToolAgent
    orig = TA.analyze
    Result = ta_mod.AnalyzerTurnResult

    def analyze(self, *args, **kwargs):
        error = None
        try:
            result = orig(self, *args, **kwargs)
        except Exception as exc:  # stock analyze lets these escape from its prompt building
            result, error = None, exc
        if result is not None:
            self._turbo_failures = 0
            return result
        failures = int(getattr(self, "_turbo_failures", 0) or 0) + 1
        self._turbo_failures = failures
        try:
            print(f"TURBO_ANALYZE_GUARD failure={failures}/{max_failures} error={error!r}"[:400], flush=True)
        except Exception:
            pass
        if failures > max_failures:
            if error is not None:
                raise error
            return None
        should_stop = kwargs.get("should_stop")
        deadline = time.monotonic() + min(30.0, 5.0 * failures)
        while time.monotonic() < deadline:
            try:
                if should_stop is not None and should_stop():
                    break
            except Exception:
                break
            time.sleep(1.0)
        return Result(step_executed=False, retryable_failure=True)

    TA.analyze = analyze
    rep["T3_analyze_guard"] = max_failures


# --------------------------------------------------------------------------------------------- T4 sampling / turns
def _install_sampling(ta_mod, rep: dict) -> None:
    mtp_tokens = _int_env("TAAF_VLLM_MTP_TOKENS", 3)  # the winning serving profile's env stays in os.environ
    temperature = _float_env("TURBO_TEMPERATURE", 1.0) if mtp_tokens == 0 else _float_env("TURBO_TEMPERATURE_MTP", 0.6)
    if 0.0 < temperature <= 2.0:
        ta_mod._LOCAL_ANALYZER_TEMPERATURE = temperature  # read per request
        rep["T4_temperature"] = temperature
    yield_seconds = _float_env("TURBO_YIELD_SECONDS", 120.0)
    if yield_seconds > 0:
        ta_mod._LOCAL_ANALYZER_YIELD_SECONDS = yield_seconds  # read in ToolAgent.__init__ (per game)
        rep["T4_yield_seconds"] = yield_seconds
    tool_steps = _int_env("TURBO_TOOL_STEPS", 3)
    if tool_steps > 0:
        ta_mod._LOCAL_ANALYZER_TOOL_STEPS = tool_steps  # read in ToolAgent.__init__ (per game)
        rep["T4_tool_steps"] = tool_steps


# --------------------------------------------------------------------------------------------- T5 prompt
MECHANICS_BLOCK = (  # Son Pham & Mark Barney arm C, verbatim (kaggle/experiments/sparse-deletion/build_bundles.py)
    "- The visible board may be a window onto a larger world, and that window can move. If the whole scene shifts "
    "at once, the frame moved, not the contents; the same row/col may not mean the same place as it did last turn.\n"
    "- What you control may change. An action may hand control to a different object instead of acting on the "
    "board, so do not assume there is exactly one controllable thing for the whole game.\n"
    "- State that decides the outcome may not be drawn on the board: what is being carried, what was collected, or "
    "what happened on an earlier attempt can change what the same action does now.\n"
    "- Order can matter as much as position. A set of things in the right places may still be wrong if they are "
    "read in a sequence.\n"
    "- Something that looks solved can come un-solved by a later action, and parts of the board may act on their "
    "own between your actions, with or against you.\n"
    "- Control may be indirect: you may move something that drags what you care about, or set values that are read "
    "together as a code rather than acting one at a time.\n"
)
HUD_PARAGRAPH = (  # arm B deletes this two-line visual-guidance paragraph
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often "
    "shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core "
    "gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of "
    "small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while "
    "the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment "
    "by segment. DON'T DO THIS!\n"
)


def prompt_edits(color_legend: str) -> list[tuple[str, str]]:
    legend_line = f"- Color legend: {color_legend}.\n"
    return [
        ("You are a coding agent solving a grid-based puzzle game.", "You are a coding agent solving a grid-based game."),
        ("- You are solving a multi-level grid puzzle game.", "- You are solving a multi-level grid game."),
        ("- Optimize for as few in-game actions as possible while still being reliable.\n", ""),
        ("boards are presented as 64 x 64 color grids", "boards are presented as color grids"),
        (legend_line, legend_line + MECHANICS_BLOCK),
        ("- Some games are logic or layout puzzles with no explicit player avatar",
         "- Some games are logic or layout games with no explicit player avatar"),
        (HUD_PARAGRAPH, ""),
    ]


GAME_OVER_OLD = "The game is over."
GAME_OVER_NEW = "That sequence ended in GAME_OVER; the harness has already reset this level to its starting board."
CLOCK_PREFIX = "Game clock:"


def _fmt_min(seconds: float) -> str:
    minutes = max(0.0, float(seconds)) / 60.0
    return f"{minutes:.1f}" if minutes < 10 else f"{minutes:.0f}"


def clock_line(session) -> str | None:
    timing = session.timing_payload()
    elapsed = float(timing.get("run_elapsed_seconds") or 0.0)
    remaining = timing.get("time_remaining_seconds")
    soft = session.solver.soft_time_remaining_seconds()
    if soft is not None:
        remaining = soft if remaining is None else min(float(remaining), float(soft))
    if remaining is None:
        return None
    line = f"{CLOCK_PREFIX} {_fmt_min(elapsed)} min used, {_fmt_min(remaining)} min left for this game."
    try:
        cleared = int(session.game.current_state.levels_completed)
        try:
            line += f" Levels cleared: {cleared} of {int(session.game.number_of_levels)}."
        except Exception:
            line += f" Levels cleared: {cleared}."
    except Exception:
        pass
    return line


def _with_clock(messages: list, clock: str) -> list:
    """Outgoing copy with the clock line after 'Current state:' in the newest turn prompt (never stored)."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts, text_index = None, None
            text = content
        elif isinstance(content, list):
            text_index = next((i for i, part in enumerate(content) if isinstance(part, dict)
                               and part.get("type") == "text" and "Current state:" in str(part.get("text", ""))), None)
            if text_index is None:
                continue
            parts, text = content, str(content[text_index].get("text", ""))
        else:
            continue
        if "Current state:" not in text or CLOCK_PREFIX in text:
            continue
        lines = text.split("\n")
        at = next(i for i, line in enumerate(lines) if line.startswith("Current state:"))
        lines.insert(at + 1, clock)
        new_text = "\n".join(lines)
        if parts is None:
            new_message = {**message, "content": new_text}
        else:
            new_parts = list(parts)
            new_parts[text_index] = {**parts[text_index], "text": new_text}
            new_message = {**message, "content": new_parts}
        copy = list(messages)
        copy[index] = new_message
        return copy
    return messages


def _install_prompt(ta_mod, an_mod, solv_mod, rep: dict) -> None:
    # system prompt (asserted on the real assembled prompt, AGENTFIX notes included; the wrapper fails open)
    try:
        import inference.utils.grid_utils as grid_utils
        legend = grid_utils.ARC_COLOR_LEGEND
    except Exception:
        legend = ""
    edits = prompt_edits(legend) if legend else []
    orig_build = ta_mod._build_system_prompt
    probe = orig_build(tool_output_tokens=1024)
    usable = [(old, new) for old, new in edits if probe.count(old) == 1]
    missing = [old[:60] for old, new in edits if probe.count(old) != 1]
    if missing:
        rep["T5_prompt_missing"] = missing

    def _build_system_prompt(*, tool_output_tokens):
        prompt = orig_build(tool_output_tokens=tool_output_tokens)
        try:
            edited = prompt
            for old, new in usable:
                if edited.count(old) == 1:
                    edited = edited.replace(old, new)
            return edited
        except Exception:
            return prompt

    if usable:
        ta_mod._build_system_prompt = _build_system_prompt
        edited = _build_system_prompt(tool_output_tokens=1024)
        rep["T5_system_edits"] = len(usable)
        rep["T5_system_chars"] = [len(probe), len(edited)]
        leftovers = [p for p in ("puzzle", "64 x 64", "remaining-steps bar", "DON'T DO THIS", "as few in-game actions")
                     if p in edited]
        if leftovers:
            rep["T5_prompt_missing"] = rep.get("T5_prompt_missing", []) + leftovers

    # ACTION7 -> UNDO (in-place dict edits keep every alias the solver/agent imported valid)
    if _ON("TURBO_UNDO") and not _ON("AGENTFIX_ACTION7", "0"):
        an_mod.ENGINE_TO_MODEL_ACTION["ACTION7"] = "UNDO"
        an_mod.MODEL_TO_ENGINE_ACTION.clear()
        an_mod.MODEL_TO_ENGINE_ACTION.update({v: k for k, v in an_mod.ENGINE_TO_MODEL_ACTION.items()})
        assert an_mod.to_engine_action("UNDO") == "ACTION7" and an_mod.to_model_action("ACTION7") == "UNDO"
        rep["T5_undo"] = True

    TA = ta_mod.ToolAgent

    # stale GAME_OVER line
    orig_bup = TA._build_user_prompt

    def _build_user_prompt(self, action_num, **kwargs):
        text = orig_bup(self, action_num, **kwargs)
        try:
            if GAME_OVER_OLD in text:
                text = "\n".join(GAME_OVER_NEW if line == GAME_OVER_OLD else line for line in text.split("\n"))
        except Exception:
            pass
        return text

    TA._build_user_prompt = _build_user_prompt
    rep["T5_game_over_line"] = True

    if not _ON("TURBO_CLOCK"):
        return
    # game clock: computed once per analyze() call (stable for every request of the turn), injected into the
    # outgoing copy of the newest turn prompt only
    orig_analyze = TA.analyze

    def analyze(self, *args, **kwargs):
        try:
            session = getattr(kwargs.get("step_env"), "__self__", None)
            self._turbo_clock = clock_line(session) if session is not None else None
        except Exception:
            self._turbo_clock = None
        return orig_analyze(self, *args, **kwargs)

    TA.analyze = analyze
    orig_chat = TA._chat_completion

    def _chat_completion(self, messages, *args, **kwargs):
        try:
            clock = getattr(self, "_turbo_clock", None)
            if clock:
                messages = _with_clock(messages, clock)
        except Exception:
            pass
        return orig_chat(self, messages, *args, **kwargs)

    TA._chat_completion = _chat_completion
    rep["T5_clock"] = True


# --------------------------------------------------------------------------------------------- T6 per-game budget
_BUDGET = {"total": None, "lanes": 1, "reserve": 180.0, "floor": 600.0, "cap": None, "started": 0}
_BUDGET_LOCK = threading.Lock()


def configure_budget(*, total_games: int, lanes: int, reserve_s: float = 180.0, floor_s: float = 600.0,
                     cap_s: float | None = None) -> dict:
    with _BUDGET_LOCK:
        _BUDGET.update(total=max(1, int(total_games)), lanes=max(1, int(lanes)), reserve=float(reserve_s),
                       floor=float(floor_s), cap=None if cap_s is None else float(cap_s), started=0)
        return dict(_BUDGET)


def session_budget(session) -> float | None:
    budget = getattr(session, "_turbo_budget", None)
    if budget is not None:
        return budget
    base = session.solver.max_runtime_s_per_game
    budget = base
    try:
        with _BUDGET_LOCK:
            if _BUDGET["total"] is not None:
                remaining = max(1, int(_BUDGET["total"]) - int(_BUDGET["started"]))
                _BUDGET["started"] += 1
                soft = session.solver.soft_time_remaining_seconds()
                if soft is not None:
                    waves = max(1, math.ceil(remaining / int(_BUDGET["lanes"])))
                    budget = (float(soft) - float(_BUDGET["reserve"])) / waves
                    cap = _BUDGET["cap"] if _BUDGET["cap"] is not None else base
                    if cap is not None:
                        budget = min(budget, float(cap))
                    budget = max(float(_BUDGET["floor"]), budget)
                    try:
                        run = session.game.game_run
                        game_id = run.game_id if run is not None else "?"
                        print(f"TURBO_BUDGET game={game_id} remaining_games={remaining} waves={waves} "
                              f"soft_left_s={soft:.0f} budget_s={budget:.0f}", flush=True)
                    except Exception:
                        pass
    except Exception:
        budget = base
    try:
        session._turbo_budget = budget
    except Exception:
        pass
    return budget


def _install_budget(solv_mod, rep: dict) -> None:
    Session = solv_mod._HarnessGameSession
    orig_limit, orig_timing = Session.runtime_limit_reached, Session.timing_payload

    def runtime_limit_reached(self):
        try:
            budget = session_budget(self)
            if budget is None:
                return False
            return (time.monotonic() - self.started_at) >= budget
        except Exception:
            return orig_limit(self)

    def timing_payload(self):
        try:
            budget = session_budget(self)
            elapsed = max(0.0, time.monotonic() - self.started_at)
            remaining = None if budget is None else max(0.0, budget - elapsed)
            return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}
        except Exception:
            return orig_timing(self)

    Session.runtime_limit_reached = runtime_limit_reached
    Session.timing_payload = timing_payload
    rep["T6_budget"] = True


# --------------------------------------------------------------------------------------------- T7 half swap
def _message_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _install_half_swap(ta_mod, rep: dict) -> None:
    mode = os.environ.get("TURBO_HALF_SWAP", "auto").strip().lower()
    prefix_on = os.environ.get("TAAF_VLLM_ENABLE_PREFIX_CACHING", "0").strip() == "1"
    if mode in ("0", "off", "false", "no") or (mode == "auto" and not prefix_on):
        rep["T7_half_swap"] = False
        return
    low_water = min(0.95, max(0.3, _float_env("TURBO_HALF_SWAP_FRACTION", 0.6)))
    TA = ta_mod.ToolAgent
    orig = TA._trim_messages_for_context

    def _trim_messages_for_context(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        try:
            # only at a turn start (newest message is the fresh user prompt): a turn in flight keeps the stock
            # one-block trimming, so it can never evict its own prompt
            if (messages and len(messages) > 2 and str(messages[-1].get("role", "")) == "user"
                    and "Current state:" in _message_text(messages[-1])):
                budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
                if self._estimate_request_input_tokens(messages, tools=tools) > budget:
                    extra = max(0, extra_safety_tokens) + int((1.0 - low_water) * budget)
                    trimmed = orig(self, messages, tools=tools, preserve_recent=preserve_recent,
                                   extra_safety_tokens=extra)
                    if len(trimmed) >= 2 and str(trimmed[-1].get("role", "")) == "user":
                        return trimmed
        except Exception:
            pass
        return orig(self, messages, tools=tools, preserve_recent=preserve_recent,
                    extra_safety_tokens=extra_safety_tokens)

    TA._trim_messages_for_context = _trim_messages_for_context
    rep["T7_half_swap"] = low_water


def install(ta_mod, an_mod, solv_mod, rs_mod, sb_mod) -> dict:
    rep: dict = {"installed": True}
    steps = (
        ("T1", lambda: _install_history_cap(rs_mod, solv_mod, ta_mod, rep), "TURBO_HISTORY_CAP"),
        ("T2", lambda: _install_sandbox_builtins(sb_mod, rep), "TURBO_SANDBOX_BUILTINS"),
        ("T3", lambda: _install_analyze_guard(ta_mod, rep), "TURBO_ANALYZE_GUARD"),
        ("T4", lambda: _install_sampling(ta_mod, rep), "TURBO_SAMPLING"),
        ("T5", lambda: _install_prompt(ta_mod, an_mod, solv_mod, rep), "TURBO_PROMPT"),
        ("T6", lambda: _install_budget(solv_mod, rep), "TURBO_BUDGET"),
        ("T7", lambda: _install_half_swap(ta_mod, rep), "TURBO_HALF_SWAP_INSTALL"),
    )
    for tag, step, switch in steps:
        if not _ON(switch):
            rep[f"{tag}_disabled"] = True
            continue
        try:
            step()
        except Exception as exc:  # fail-open: keep the stock behaviour for this item
            rep[f"{tag}_error"] = repr(exc)
    return rep
