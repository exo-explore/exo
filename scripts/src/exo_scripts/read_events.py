# pyright: reportAny=false

import asyncio
import curses
import time
import json
import argparse
import sys
from logging import Logger
from typing import List, Optional, Any, Sequence, Tuple

from exo.shared.types.state import State
from exo.shared.apply import apply
from exo.shared.db.sqlite.event_log_manager import EventLogManager, EventLogConfig
from exo.shared.types.events.components import EventFromEventLog
from exo.shared.types.events import Event

# Globals
logger: Logger = Logger('helper_log')
event_log_manager: Optional[EventLogManager] = None
worker_mode: bool = False

# Worker-related event types
WORKER_EVENT_TYPES = {
    'TaskCreated', 'TaskStateUpdated', 'TaskFailed', 'TaskDeleted',
    'ChunkGenerated',
    'InstanceCreated', 'InstanceDeleted', 'InstanceActivated', 'InstanceDeactivated', 'InstanceReplacedAtomically',
    'RunnerStatusUpdated', 'RunnerDeleted'
}


async def init_db() -> None:
    global event_log_manager
    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()


async def get_events_since(since: int) -> Sequence[EventFromEventLog[Event]]:
    assert event_log_manager is not None
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


def print_event(event: EventFromEventLog[Event]) -> None:
    event_type_name = type(event.event).__name__
    event_type = event_type_name.replace('_', ' ').title()
    attributes = ', '.join(f"{key}={value!r}" for key,
                           value in vars(event.event).items())
    print(f"[{event.idx_in_log}] {event_type}: {attributes}")


async def non_tui_mode() -> None:
    await init_db()
    events = await load_all_events()
    states = compute_states(events)
    final_state = states[-1]

    if worker_mode:
        filtered_events = [e for e in events if type(
            e.event).__name__ in WORKER_EVENT_TYPES]
        events = filtered_events
        # Recompute states? But states are cumulative, so perhaps just print filtered events and full state, or filter state too.
        state_dict = json.loads(final_state.model_dump_json())
        filtered_state = {
            'node_status': state_dict.get('node_status', {}),
            'instances': state_dict.get('instances', {}),
            'runners': state_dict.get('runners', {}),
            'tasks': state_dict.get('tasks', {}),
            'last_event_applied_idx': state_dict.get('last_event_applied_idx', 0)
        }
        print("Final State (filtered):")
        print(json.dumps(filtered_state, indent=2))
    else:
        print("Final State:")
        print(final_state.model_dump_json(indent=2))

    print("\nEvents:")
    for event in events:
        print_event(event)


async def update_events(wrapped_events: List[EventFromEventLog[Event]], states: List[State],
                        filtered_indices: Optional[List[int]] = None) -> bool:
    last_since = len(wrapped_events)
    new_wrapped = await get_events_since(last_since)
    if new_wrapped:
        last_len = len(wrapped_events)
        for nw in new_wrapped:
            state = states[-1]
            new_state = apply(state, nw)
            states.append(new_state)
        wrapped_events.extend(new_wrapped)
        if filtered_indices is not None:
            for k in range(last_len, len(wrapped_events)):
                if type(wrapped_events[k].event).__name__ in WORKER_EVENT_TYPES:
                    filtered_indices.append(k)
        return True
    return False


def draw_state(win: Any, state: State, height: int, width: int, worker_mode: bool, state_scroll: int) -> int:
    win.clear()
    state_dict = json.loads(state.model_dump_json())
    if worker_mode:
        filtered_state = {
            'node_status': state_dict.get('node_status', {}),
            'instances': state_dict.get('instances', {}),
            'runners': state_dict.get('runners', {}),
            'tasks': state_dict.get('tasks', {}),
            'last_event_applied_idx': state_dict.get('last_event_applied_idx', 0)
        }
        state_pretty = json.dumps(filtered_state, indent=2)
    else:
        state_pretty = json.dumps(state_dict, indent=2)
    lines = state_pretty.split('\n')
    max_scroll = max(0, len(lines) - height)
    current_scroll = min(state_scroll, max_scroll)
    for i in range(height):
        line_idx = current_scroll + i
        if line_idx >= len(lines):
            break
        line = lines[line_idx]
        y = i
        x = 0
        leading_spaces = len(line) - len(line.lstrip())
        win.addstr(y, x, ' ' * leading_spaces)
        x += leading_spaces
        stripped = line.lstrip()
        if stripped.startswith('"'):
            end_key = stripped.find('": ')
            if end_key != -1:
                key_str = stripped[:end_key + 3]  # include ":
                win.addstr(y, x, key_str, curses.color_pair(3))
                x += len(key_str)
                value_str = stripped[end_key + 3:]
                if value_str.startswith('"'):
                    color = 2
                elif value_str.replace('.', '', 1).isdigit() or (
                        value_str.startswith('-') and value_str[1:].replace('.', '', 1).isdigit()):
                    color = 4
                elif value_str in ['true', 'false', 'null']:
                    color = 5
                elif value_str.startswith('{') or value_str.startswith('[') or value_str.startswith(
                        '}') or value_str.startswith(']'):
                    color = 0
                else:
                    color = 0
                win.addstr(y, x, value_str, curses.color_pair(color))
            else:
                win.addstr(y, x, stripped)
        else:
            win.addstr(y, x, stripped)
    win.refresh()
    return current_scroll


def get_event_pairs(event: EventFromEventLog[Event]) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    idx_str = f"[{event.idx_in_log}] "
    pairs.append((idx_str, 5))
    event_type_name = type(event.event).__name__
    event_type = event_type_name.replace('_', ' ').title()
    pairs.append((event_type, 1))
    pairs.append((": ", 0))
    attrs = vars(event.event)
    first = True
    for key, value in attrs.items():
        if not first:
            pairs.append((", ", 0))
        first = False
        pairs.append((key, 3))
        pairs.append(("=", 0))
        v_str = repr(value)
        if isinstance(value, str):
            color = 2
        elif isinstance(value, (int, float)):
            color = 4
        elif isinstance(value, bool):
            color = 5
        else:
            color = 6
        pairs.append((v_str, color))
    return pairs


def calculate_event_lines(pairs: List[Tuple[str, int]], win_width: int, subsequent_indent: int) -> int:
    lines = 1
    x = 0
    for text, _ in pairs:
        i = 0
        while i < len(text):
            remaining = win_width - x
            part_len = min(len(text) - i, remaining)
            i += part_len
            x += part_len
            if i < len(text):
                lines += 1
                x = subsequent_indent
    return lines


def render_event(win: Any, start_y: int, pairs: List[Tuple[str, int]], is_bold: bool, win_width: int,
                 subsequent_indent: int) -> int:
    y = start_y
    x = 0
    for text, color in pairs:
        attr = curses.color_pair(color) | (curses.A_BOLD if is_bold else 0)
        i = 0
        while i < len(text):
            remaining = win_width - x
            part_len = min(len(text) - i, remaining)
            part = text[i:i + part_len]
            try:
                win.addstr(y, x, part, attr)
            except curses.error:
                pass
            i += part_len
            x += part_len
            if i < len(text):
                y += 1
                if y >= win.getmaxyx()[0]:
                    return y
                x = subsequent_indent
    if x > 0:
        y += 1
    return y


def draw_events(win: Any, events_list: List[EventFromEventLog[Event]], current_events: int, height: int) -> None:
    win.clear()
    if len(events_list) == 0:
        win.addstr(0, 0, "No events")
        win.refresh()
        return
    win_width = win.getmaxyx()[1]
    current_event = events_list[current_events]
    current_pairs = get_event_pairs(current_event)
    subsequent_indent = len(f"[{current_event.idx_in_log}] ")
    lines_current = calculate_event_lines(
        current_pairs, win_width, subsequent_indent)
    if lines_current > height:
        render_event(win, 0, current_pairs, True, win_width, subsequent_indent)
        win.refresh()
        return

    target_above = (height - lines_current) // 2
    target_below = height - lines_current - target_above

    # Collect previous events
    prev_events: List[int] = []
    remaining = target_above
    i = current_events - 1
    while i >= 0 and remaining > 0:
        event = events_list[i]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        lines = calculate_event_lines(pairs, win_width, indent)
        if lines <= remaining:
            remaining -= lines
            prev_events.append(i)
            i -= 1
        else:
            break
    prev_events.reverse()

    # Collect next events
    next_events: List[int] = []
    remaining = target_below
    j = current_events + 1
    while j < len(events_list) and remaining > 0:
        event = events_list[j]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        lines = calculate_event_lines(pairs, win_width, indent)
        if lines <= remaining:
            remaining -= lines
            next_events.append(j)
            j += 1
        else:
            break

    # Calculate total lines
    total_lines = lines_current
    for idx in prev_events:
        event = events_list[idx]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        total_lines += calculate_event_lines(pairs, win_width, indent)
    for idx in next_events:
        event = events_list[idx]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        total_lines += calculate_event_lines(pairs, win_width, indent)

    padding = (height - total_lines) // 2 if total_lines < height else 0

    y = padding
    # Draw prev
    for idx in prev_events:
        event = events_list[idx]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        y = render_event(win, y, pairs, False, win_width, indent)

    # Draw current
    y = render_event(win, y, current_pairs, True, win_width, subsequent_indent)

    # Draw next
    for idx in next_events:
        event = events_list[idx]
        pairs = get_event_pairs(event)
        indent = len(f"[{event.idx_in_log}] ")
        y = render_event(win, y, pairs, False, win_width, indent)

    win.refresh()


def draw_status(win: Any, realtime: bool, current: int, total_events: int) -> None:
    win.clear()
    mode = "Realtime" if realtime else "Timetravel"
    win.addstr(0, 0,
               f"Mode: {mode} | Current event: {current} / {total_events} | Arrows: navigate events, [/]: scroll state, g: goto, r: toggle realtime, q: quit")
    win.refresh()


def get_input(stdscr: Any, prompt: str) -> str:
    curses.echo()
    stdscr.addstr(0, 0, prompt)
    stdscr.refresh()
    input_str = stdscr.getstr(0, len(prompt), 20).decode('utf-8')
    curses.noecho()
    return input_str


def get_key(win: Any) -> Any:
    ch = win.getch()
    if ch == -1:
        return -1
    if ch == 27:
        ch2 = win.getch()
        if ch2 == -1:
            return 27
        if ch2 == 91:
            ch3 = win.getch()
            if ch3 == -1:
                return -1
            if ch3 == 65:
                return curses.KEY_UP
            if ch3 == 66:
                return curses.KEY_DOWN
            if ch3 == 53:
                ch4 = win.getch()
                if ch4 == 126:
                    return curses.KEY_PPAGE
            if ch3 == 54:
                ch4 = win.getch()
                if ch4 == 126:
                    return curses.KEY_NPAGE
            if ch3 == 49:
                ch4 = win.getch()
                if ch4 == -1:
                    return -1
                if ch4 == 59:
                    ch5 = win.getch()
                    if ch5 == -1:
                        return -1
                    if ch5 == 53:
                        ch6 = win.getch()
                        if ch6 == -1:
                            return -1
                        if ch6 == 65:
                            return 'CTRL_UP'
                        if ch6 == 66:
                            return 'CTRL_DOWN'
    return ch


def tui(stdscr: Any) -> None:
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.use_default_colors()
    stdscr.timeout(100)
    curses.curs_set(0)

    wrapped_events: List[EventFromEventLog[Event]] = []
    states: List[State] = [State()]
    asyncio.run(init_db())
    asyncio.run(update_events(wrapped_events, states))  # Initial load

    filtered_indices: Optional[List[int]] = None
    current_filtered: int = -1
    current: int = -1
    if worker_mode:
        filtered_indices = [i for i in range(len(wrapped_events)) if
                            type(wrapped_events[i].event).__name__ in WORKER_EVENT_TYPES]
        current_filtered = len(filtered_indices) - \
            1 if filtered_indices else -1
    else:
        current = len(wrapped_events) - 1 if wrapped_events else -1

    realtime: bool = False
    last_update: float = time.time()
    update_interval: float = 1.0
    state_scroll: int = 0

    while True:
        height, width = stdscr.getmaxyx()
        status_height = 1
        pane_height = height - status_height
        pane_width = width // 2

        state_win = curses.newwin(pane_height, pane_width, 0, 0)
        events_win = curses.newwin(
            pane_height, width - pane_width, 0, pane_width)
        status_win = curses.newwin(status_height, width, pane_height, 0)

        if worker_mode:
            assert filtered_indices is not None
            current_original = filtered_indices[current_filtered] if current_filtered >= 0 else -1
            events_list = [wrapped_events[i] for i in filtered_indices]
            current_events = current_filtered
        else:
            current_original = current
            events_list = wrapped_events
            current_events = current

        state_idx = current_original + 1 if current_original >= 0 else 0
        state_scroll = draw_state(
            state_win, states[state_idx], pane_height, pane_width, worker_mode, state_scroll)
        draw_events(events_win, events_list, current_events, pane_height)
        total_events = len(wrapped_events) - 1 if wrapped_events else -1
        draw_status(status_win, realtime,
                    current_original if worker_mode else current, total_events)

        key = get_key(stdscr)
        if key != -1:
            if key == curses.KEY_UP:
                if worker_mode and current_filtered > 0:
                    current_filtered -= 1
                elif not worker_mode and current > 0:
                    current -= 1
            elif key == 'CTRL_UP':
                if worker_mode:
                    current_filtered = max(0, current_filtered - 5)
                else:
                    current = max(0, current - 5)
            elif key == curses.KEY_DOWN:
                assert filtered_indices is not None
                if worker_mode and current_filtered < len(filtered_indices) - 1:
                    current_filtered += 1
                elif not worker_mode and current < len(wrapped_events) - 1:
                    current += 1
            elif key == 'CTRL_DOWN':
                assert filtered_indices is not None
                if worker_mode:
                    current_filtered = min(
                        len(filtered_indices) - 1, current_filtered + 5)
                else:
                    current = min(len(wrapped_events) - 1, current + 5)
            elif key == ord('['):
                state_scroll = max(0, state_scroll - pane_height // 2)
            elif key == ord(']'):
                state_scroll += pane_height // 2  # clamped in draw_state
            elif key == ord('q'):
                break
            elif key == ord('r'):
                realtime = not realtime
                if realtime:
                    assert filtered_indices is not None
                    if worker_mode:
                        current_filtered = len(
                            filtered_indices) - 1 if filtered_indices else -1
                    else:
                        current = len(wrapped_events) - \
                            1 if wrapped_events else -1
                    state_scroll = 0
            elif key == ord('g'):
                stdscr.timeout(-1)  # block for input
                input_str = get_input(status_win, "Go to event: ")
                try:
                    goto = int(input_str)
                    if worker_mode:
                        assert filtered_indices is not None
                        for i, orig in enumerate(filtered_indices):
                            if wrapped_events[orig].idx_in_log == goto:
                                current_filtered = i
                                state_scroll = 0
                                break
                    else:
                        for i in range(len(wrapped_events)):
                            if wrapped_events[i].idx_in_log == goto:
                                current = i
                                state_scroll = 0
                                break
                except ValueError:
                    pass
                stdscr.timeout(100)
                status_win.clear()
                status_win.refresh()

        if realtime and time.time() - last_update > update_interval:
            updated = asyncio.run(update_events(
                wrapped_events, states, filtered_indices if worker_mode else None))
            if updated:
                assert filtered_indices is not None
                if worker_mode:
                    current_filtered = len(filtered_indices) - 1
                else:
                    current = len(wrapped_events) - 1
                state_scroll = 0
            last_update = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Read and display events from the event log')
    parser.add_argument('--worker', action='store_true',
                        help='Only show worker-related events (task, streaming, instance, runner status)')
    args = parser.parse_args()

    worker_mode = args.worker

    if not sys.stdout.isatty():
        asyncio.run(non_tui_mode())
    else:
        try:
            curses.wrapper(tui)
        except curses.error as e:
            if "could not find terminal" in str(e):
                print("Error: Could not find terminal. Falling back to non-TUI mode.")
                asyncio.run(non_tui_mode())
            else:
                raise
