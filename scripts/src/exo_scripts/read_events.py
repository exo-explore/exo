import asyncio
import json
import argparse
import sys
import time
from dataclasses import is_dataclass, asdict
from logging import getLogger
from typing import List, Optional, Any, Sequence, Tuple

# Your existing imports — unchanged
from exo.shared.types.state import State
from exo.shared.apply import apply
from exo.shared.db.sqlite.event_log_manager import EventLogManager, EventLogConfig
from exo.shared.types.events.components import EventFromEventLog
from exo.shared.types.events import Event

# --- Third-party UI (new) ---
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel
from rich.console import RenderableType

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, ListView, ListItem, Input, Footer, Label
from textual.reactive import reactive
from textual import on
from textual.binding import Binding
from textual.message import Message

logger = getLogger("helper_log")

# Worker-related event types (same set)
WORKER_EVENT_TYPES = {
    'TaskCreated', 'TaskStateUpdated', 'TaskFailed', 'TaskDeleted',
    'ChunkGenerated',
    'InstanceCreated', 'InstanceDeleted', 'InstanceActivated', 'InstanceDeactivated', 'InstanceReplacedAtomically',
    'RunnerStatusUpdated', 'RunnerDeleted'
}


# ---------- Data / DB helpers (mostly your original logic) ----------

event_log_manager: Optional[EventLogManager] = None

async def init_db() -> None:
    global event_log_manager
    event_log_manager = EventLogManager(EventLogConfig())
    await event_log_manager.initialize()

async def get_events_since(since: int) -> Sequence[EventFromEventLog[Event]]:
    # type: ignore[attr-defined, return-value]
    return await event_log_manager.global_events.get_events_since(since)

async def load_all_events() -> List[EventFromEventLog[Event]]:
    events: List[EventFromEventLog[Event]] = []
    since = 0
    while True:
        new_events = await get_events_since(since)
        if not new_events:
            break
        events.extend(new_events)
        since += len(new_events)
    return events

def compute_states(events: List[EventFromEventLog[Event]]) -> List[State]:
    states: List[State] = [State()]
    state = states[0]
    for event in events:
        state = apply(state, event)
        states.append(state)
    return states

def filter_worker_state(state: State) -> dict:
    state_dict = json.loads(state.model_dump_json())
    return {
        'node_status': state_dict.get('node_status', {}),
        'instances': state_dict.get('instances', {}),
        'runners': state_dict.get('runners', {}),
        'tasks': state_dict.get('tasks', {}),
        'last_event_applied_idx': state_dict.get('last_event_applied_idx', 0)
    }

def event_type_name(e: EventFromEventLog[Event]) -> str:
    return type(e.event).__name__

def is_worker_event(e: EventFromEventLog[Event]) -> bool:
    return event_type_name(e) in WORKER_EVENT_TYPES

def safe_json(obj: Any) -> str:
    """Serialize unknown objects to JSON-ish string safely."""
    def to_serializable(x: Any):
        try:
            if is_dataclass(x):
                return asdict(x)
        except Exception:
            pass
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [to_serializable(v) for v in x]
        try:
            json.dumps(x)  # type: ignore
            return x
        except Exception:
            return repr(x)
    try:
        return json.dumps(to_serializable(obj), indent=2, ensure_ascii=False)
    except Exception:
        # Last resort
        return repr(obj)

def summarize_event_line(e: EventFromEventLog[Event], max_len: int = 160) -> Text:
    etype = event_type_name(e)
    attrs = vars(e.event)
    prefix = Text(f"[{e.idx_in_log}] ", style="bold dim")
    t = Text(etype, style="bold cyan")
    t = prefix + t + Text(": ", style="dim")
    first = True
    for k, v in attrs.items():
        if not first:
            t.append(", ", style="dim")
        first = False
        t.append(str(k), style="magenta")
        t.append("=")
        # Coarse coloring by type
        if isinstance(v, str):
            t.append(repr(v), style="green")
        elif isinstance(v, (int, float)):
            t.append(repr(v), style="yellow")
        elif isinstance(v, bool):
            t.append(repr(v), style="cyan")
        else:
            t.append(repr(v), style="")
    if len(t.plain) > max_len:
        t.truncate(max_len - 1)
        t.append("…", style="dim")
    return t

def event_detail_renderable(e: EventFromEventLog[Event]) -> RenderableType:
    payload = {
        "idx_in_log": e.idx_in_log,
        "event_type": event_type_name(e),
        "attributes": vars(e.event)
    }
    return Syntax(safe_json(payload), "json", word_wrap=True)


# ---------- Non-TUI (stdout) mode, like your current script ----------

async def run_non_tui(worker_mode: bool) -> None:
    await init_db()
    events = await load_all_events()
    states = compute_states(events)
    final_state = states[-1]

    if worker_mode:
        filtered_events = [e for e in events if is_worker_event(e)]
        events = filtered_events
        filtered_state = filter_worker_state(final_state)
        print("Final State (filtered):")
        print(json.dumps(filtered_state, indent=2))
    else:
        print("Final State:")
        print(final_state.model_dump_json(indent=2))

    print("\nEvents:")
    for e in events:
        etype = event_type_name(e)
        attrs = ', '.join(f"{k}={value!r}" for k, value in vars(e.event).items())
        print(f"[{e.idx_in_log}] {etype}: {attrs}")


# ---------- Textual TUI ----------

class StateView(Static):
    """Left pane: shows state JSON, with optional worker filter."""
    def update_state(self, state: State, worker_mode: bool, index_in_log_for_status: Optional[int]) -> None:
        if worker_mode:
            data = filter_worker_state(state)
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            json_str = state.model_dump_json(indent=2)
        syntax = Syntax(json_str, "json", word_wrap=True)
        title = f"State after event #{index_in_log_for_status}" if index_in_log_for_status is not None else "Initial State"
        self.update(Panel(syntax, title=title, border_style="cyan"))

class EventListItem(ListItem):
    def __init__(self, e: EventFromEventLog[Event]) -> None:
        super().__init__(Static(summarize_event_line(e)))
        self._event = e

    @property
    def wrapped_event(self) -> EventFromEventLog[Event]:
        return self._event

class EventDetail(Static):
    """Right-bottom: details of the selected event."""
    def show_event(self, e: Optional[EventFromEventLog[Event]]) -> None:
        if e is None:
            self.update(Panel(Text("No event selected.", style="dim"), title="Event Details"))
        else:
            self.update(Panel(event_detail_renderable(e), title=f"Event #{e.idx_in_log} • {event_type_name(e)}", border_style="magenta"))

class StatusBar(Static):
    def set_status(self, realtime: bool, total_events: int, current_idx_in_log: Optional[int]) -> None:
        mode = "Realtime" if realtime else "Timetravel"
        parts = [
            f"[{mode}]",
            f"Events: {total_events}",
        ]
        if current_idx_in_log is not None:
            parts.append(f"Current: #{current_idx_in_log}")
        parts.append("Keys: ↑/↓ Select • PgUp/PgDn Scroll • Ctrl+↑/↓ ±5 • [/] State PgUp/PgDn • g Goto • r Realtime • q Quit")
        self.update(Text("  ".join(parts), style="dim"))


class GotoPrompt(Static):
    """Simple inline goto prompt (appears above Footer)."""
    class Submitted(Message):
        def __init__(self, value: Optional[int]) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        yield Label("Go to event id (idx_in_log):", id="goto-label")
        yield Input(placeholder="e.g., 123", id="goto-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        try:
            value = int(text)
        except ValueError:
            value = None
        self.post_message(self.Submitted(value))


class EventLogApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #left {
        width: 60%;
    }
    #right {
        width: 40%;
    }
    #events {
        height: 3fr;
    }
    #detail {
        height: 2fr;
        border: tall;
    }
    #status {
        height: 1;
        padding: 0 1;
    }
    #goto {
        dock: bottom;
        height: 3;
        padding: 1 2;
        background: $panel;
        border: round $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "toggle_realtime", "Realtime"),
        Binding("[", "state_page_up", "State PgUp"),
        Binding("]", "state_page_down", "State PgDn"),
        Binding("g", "prompt_goto", "Goto"),
        Binding("ctrl+up", "jump_up", "Jump Up"),
        Binding("ctrl+down", "jump_down", "Jump Down"),
    ]

    # Reactive state
    realtime: reactive[bool] = reactive(False)
    worker_mode: bool

    # Data
    wrapped_events: List[EventFromEventLog[Event]]
    states: List[State]
    filtered_indices: Optional[List[int]]  # maps filtered idx -> original idx
    update_interval: float = 1.0
    _poll_timer = None

    def __init__(self, worker_mode: bool) -> None:
        super().__init__()
        self.worker_mode = worker_mode
        self.wrapped_events = []
        self.states = [State()]
        self.filtered_indices = None

    async def on_mount(self) -> None:
        await init_db()
        await self._initial_load()
        # periodic polling for new events
        self._poll_timer = self.set_interval(self.update_interval, self._tick_poll)
        # Put list selection at end (last event) by default
        self._select_last()

    async def _initial_load(self) -> None:
        self.wrapped_events = await load_all_events()
        self.states = compute_states(self.wrapped_events)

        # Build filtered view if needed
        if self.worker_mode:
            self.filtered_indices = [i for i, e in enumerate(self.wrapped_events) if is_worker_event(e)]
        else:
            self.filtered_indices = None

        # Populate the ListView
        lv = self.query_one("#events", ListView)
        lv.clear()
        events_to_show = self._view_events()
        for e in events_to_show:
            lv.append(EventListItem(e))

        # Update left state & details
        self._refresh_views()

    def compose(self) -> ComposeResult:
        # Layout: [Header optional] -> main Horizontal -> Status bar + Footer
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield StateView(id="state")
            with Vertical(id="right"):
                yield ListView(id="events")
                yield EventDetail(id="detail")
        yield StatusBar(id="status")
        yield Footer()

    def _current_original_index(self) -> int:
        lv = self.query_one("#events", ListView)
        idx = lv.index
        if idx is None or idx < 0:
            return -1
        if self.filtered_indices is not None:
            if idx >= len(self.filtered_indices):
                return -1
            return self.filtered_indices[idx]
        return idx

    def _view_events(self) -> List[EventFromEventLog[Event]]:
        if self.filtered_indices is not None:
            return [self.wrapped_events[i] for i in self.filtered_indices]
        return self.wrapped_events

    def _select_last(self) -> None:
        lv = self.query_one("#events", ListView)
        n = len(lv.children)
        if n:
            lv.index = n - 1

    def _refresh_views(self) -> None:
        # Update State pane and Detail pane and Status bar
        original_idx = self._current_original_index()
        state_idx = (original_idx + 1) if original_idx >= 0 else 0
        state = self.states[state_idx]
        state_view = self.query_one("#state", StateView)
        idx_in_log = None
        if original_idx >= 0:
            idx_in_log = self.wrapped_events[original_idx].idx_in_log
        state_view.update_state(state, self.worker_mode, idx_in_log)

        # Detail pane
        detail = self.query_one("#detail", EventDetail)
        current_event = self.wrapped_events[original_idx] if original_idx >= 0 else None
        detail.show_event(current_event)

        # Status bar
        status = self.query_one("#status", StatusBar)
        total_events = len(self.wrapped_events)
        status.set_status(self.realtime, total_events, current_event.idx_in_log if current_event else None)

    async def _poll_once(self) -> bool:
        """Fetch and append new events; return True if updated."""
        last_since = len(self.wrapped_events)
        new_wrapped = await get_events_since(last_since)
        if not new_wrapped:
            return False

        # Extend states incrementally (avoid recomputing all)
        for nw in new_wrapped:
            state = self.states[-1]
            self.states.append(apply(state, nw))

        start_len = len(self.wrapped_events)
        self.wrapped_events.extend(new_wrapped)

        # Update filtered mapping and UI list
        lv = self.query_one("#events", ListView)
        if self.worker_mode:
            if self.filtered_indices is None:
                self.filtered_indices = []
            for k in range(start_len, len(self.wrapped_events)):
                if is_worker_event(self.wrapped_events[k]):
                    self.filtered_indices.append(k)
                    lv.append(EventListItem(self.wrapped_events[k]))
        else:
            for k in range(start_len, len(self.wrapped_events)):
                lv.append(EventListItem(self.wrapped_events[k]))

        # Auto-follow the tail in realtime mode
        if self.realtime:
            self._select_last()

        # Refresh panes
        self._refresh_views()
        return True

    def _tick_poll(self) -> None:
        # called by timer; schedule the async poll
        asyncio.create_task(self._poll_once())

    # ------ Actions / key handlers ------
    def action_quit(self) -> None:
        self.exit()

    def action_toggle_realtime(self) -> None:
        self.realtime = not self.realtime
        if self.realtime:
            self._select_last()
        self._refresh_views()

    def action_state_page_up(self) -> None:
        state_view = self.query_one("#state", StateView)
        state_view.scroll_page_up()

    def action_state_page_down(self) -> None:
        state_view = self.query_one("#state", StateView)
        state_view.scroll_page_down()

    def action_jump_up(self) -> None:
        lv = self.query_one("#events", ListView)
        if lv.children:
            lv.index = max(0, (lv.index or 0) - 5)
            self._refresh_views()

    def action_jump_down(self) -> None:
        lv = self.query_one("#events", ListView)
        if lv.children:
            lv.index = min(len(lv.children) - 1, (lv.index or 0) + 5)
            self._refresh_views()

    def action_prompt_goto(self) -> None:
        # mount a small prompt near bottom
        if self.query("#goto"):
            return
        prompt = GotoPrompt(id="goto")
        self.mount(prompt)

    @on(GotoPrompt.Submitted)
    def _on_goto_submitted(self, msg: GotoPrompt.Submitted) -> None:
        # Remove prompt
        for node in self.query("#goto"):
            node.remove()

        if msg.value is None:
            return

        target = msg.value
        # find in current view's idx_in_log
        events_to_show = self._view_events()
        lv = self.query_one("#events", ListView)
        for i, e in enumerate(events_to_show):
            if e.idx_in_log == target:
                lv.index = i
                self._refresh_views()
                break

    @on(ListView.Highlighted, "#events")
    @on(ListView.Selected, "#events")
    def _on_event_selected(self, *_: Any) -> None:
        # Update panes when selection changes
        self._refresh_views()


# ---------- Entrypoint ----------

def main() -> None:
    parser = argparse.ArgumentParser(description='Read and display events from the event log (Textual UI)')
    parser.add_argument('--worker', action='store_true',
                        help='Only show worker-related events (task, streaming, instance, runner status)')
    parser.add_argument('--no-ui', action='store_true',
                        help='Print to stdout (non-interactive), like the original non-TUI mode')
    args = parser.parse_args()

    # Non-interactive fallback if no TTY or user requests it
    if args.no_ui or not sys.stdout.isatty():
        asyncio.run(run_non_tui(worker_mode=args.worker))
        return

    # TUI mode
    app = EventLogApp(worker_mode=args.worker)
    app.run()

if __name__ == "__main__":
    main()
