"""Seat Belt Warning System — FSM simulation with real-time timers.

This module models an automotive seat belt warning system as two cooperating
finite state machines (FSMs):

1. SeatBeltWarningSystem (main FSM)
   ALARM_OFF  -> ignition ON           -> WAITING
   WAITING    -> belt fastened / ign OFF -> ALARM_OFF
   WAITING    -> 10s expires, belt open  -> ALARM_ON
   ALARM_ON   -> belt fastened / ign OFF -> ALARM_OFF
   ALARM_ON   -> 5s alarm expires        -> ALARM_OFF

2. TimerFSM (nested timer FSM, reused for both delays)
   IDLE -> load() -> READY -> start() -> RUNNING -> expire/stop -> IDLE

Inputs are ignition and seat belt status; time-driven transitions are handled
by TimerFSM callbacks running on a background thread.

OOP structure in this file:
  - Encapsulation: each class hides its fields behind methods/properties (_state
    is private; outsiders call set_ignition(), not direct assignment).
  - Composition: SeatBeltWarningSystem *has-a* TimerFSM rather than inheriting
    from it — each class owns one clear responsibility.
  - Separation of concerns: SeatBeltWarningSystem (logic), ConsoleDisplay (view),
    Simulator (orchestration) are independent and can be tested separately.
  - Dependency injection: TimerFSM accepts an on_expire callback; Simulator
    accepts optional system/display instances for flexibility.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


# ---------------------------------------------------------------------------
# Timer FSM — counts down real wall-clock delays (10s wait, 5s alarm)
# ---------------------------------------------------------------------------

class TimerState(Enum):
  """Enum: type-safe named constants for timer states (avoids magic strings)."""
  IDLE = auto()     # No countdown active
  READY = auto()    # Duration loaded, not yet counting
  RUNNING = auto()  # Counting down toward expiration


class TimerFSM:
  """Nested timer FSM: load a delay, start it, fire on_expire when done.

  OOP notes:
    - Encapsulation: _state, _duration, etc. are private; the public API is
      load(), start(), stop(), and read-only properties.
    - Single responsibility: this class only manages countdown timing, not
      ignition or seat belt logic.
    - Callback (observer-style): on_expire lets the owner react without
      TimerFSM knowing about SeatBeltWarningSystem (loose coupling).

  Lifecycle: load(seconds) -> start() -> [RUNNING] -> expire or stop() -> IDLE
  Uses time.monotonic() for accurate remaining time and threading.Timer to
  notify the main warning FSM when a delay finishes.
  """

  def __init__(self, on_expire: Callable[[], None] | None = None) -> None:
    self._state = TimerState.IDLE
    self._duration = 0.0          # Seconds loaded in READY / used when RUNNING starts
    self._deadline: float | None = None  # monotonic() timestamp when RUNNING ends
    self._handle: threading.Timer | None = None
    self._on_expire = on_expire  # Called on background thread when timer expires
    self._lock = threading.Lock()

  @property
  def state(self) -> TimerState:
    # Read-only property: external code can inspect state but not assign to it.
    with self._lock:
      if self._state == TimerState.RUNNING and self._deadline is not None:
        if time.monotonic() >= self._deadline:
          return TimerState.IDLE
      return self._state

  @property
  def remaining(self) -> float:
    with self._lock:
      if self._state == TimerState.READY:
        return self._duration
      if self._state == TimerState.RUNNING and self._deadline is not None:
        return max(0.0, self._deadline - time.monotonic())
      return 0.0

  def is_running(self) -> bool:
    return self.state == TimerState.RUNNING

  def load(self, seconds: float) -> None:
    """IDLE -> READY: arm the timer with a delay but do not start yet."""
    with self._lock:
      self._cancel_handle()
      self._duration = seconds
      self._deadline = None
      self._state = TimerState.READY

  def start(self) -> None:
    """READY -> RUNNING: begin the real-time countdown."""
    with self._lock:
      if self._state != TimerState.READY:
        return
      self._state = TimerState.RUNNING
      self._deadline = time.monotonic() + self._duration
      self._schedule_handle(self._duration)

  def stop(self) -> None:
    """Force -> IDLE: cancel countdown (e.g. belt fastened or ignition OFF)."""
    with self._lock:
      self._cancel_handle()
      self._duration = 0.0
      self._deadline = None
      self._state = TimerState.IDLE

  def _schedule_handle(self, delay: float) -> None:
    # Private helper: implementation detail, not part of the public interface.
    self._cancel_handle()
    self._handle = threading.Timer(delay, self._handle_expire)
    self._handle.daemon = True
    self._handle.start()

  def _cancel_handle(self) -> None:
    if self._handle is not None:
      self._handle.cancel()
      self._handle = None

  def _handle_expire(self) -> None:
    # Invoke callback outside the lock to avoid deadlocks with the main FSM.
    callback = None
    with self._lock:
      if self._state != TimerState.RUNNING:
        return
      self._state = TimerState.IDLE
      self._deadline = None
      self._duration = 0.0
      self._handle = None
      callback = self._on_expire

    if callback is not None:
      callback()


# ---------------------------------------------------------------------------
# Main warning FSM — monitors ignition + seat belt, drives the alarm
# ---------------------------------------------------------------------------

class WarningState(Enum):
  """Enum: the three states of the main warning finite state machine."""
  ALARM_OFF = auto()  # Normal: no warning
  WAITING = auto()    # Ignition ON: 10s waiting time before alarm can sound
  ALARM_ON = auto()   # Belt still open after waiting time: alarm for up to 5s


class SeatBeltWarningSystem:
  """Main seat belt warning FSM driven by ignition, belt status, and timers.

  OOP notes:
    - Composition: owns a TimerFSM instance (_timer) instead of subclassing it.
    - Class constants: WAIT_DURATION / ALARM_DURATION belong to the type, not
      any single instance (shared configuration).
    - Public interface: set_ignition() and set_seat_belt_fastened() are the
      inputs; properties expose read-only state for monitoring.
    - Private transitions (_enter_waiting, etc.) keep state-change logic in one
      place so every path updates alarm and timer consistently.

  One TimerFSM instance is shared: first for the 10s waiting period, then
  (if needed) reloaded for the 5s alarm duration. The two delays never overlap.
  """

  # Class attributes — same value for every instance of this class.
  WAIT_DURATION = 10   # Seconds to buckle up after ignition ON
  ALARM_DURATION = 5   # Seconds the audible alarm stays on (max)

  def __init__(self) -> None:
    self._state = WarningState.ALARM_OFF
    self._ignition_on = False
    self._seat_belt_fastened = False
    self._alarm_active = False
    self._lock = threading.Lock()
    # Pass a bound method as callback — TimerFSM will call back into this object.
    self._timer = TimerFSM(on_expire=self._handle_timer_expired)

  @property
  def state(self) -> WarningState:
    with self._lock:
      return self._state

  @property
  def alarm_active(self) -> bool:
    with self._lock:
      return self._alarm_active

  @property
  def timer(self) -> TimerFSM:
    return self._timer

  @property
  def ignition_on(self) -> bool:
    with self._lock:
      return self._ignition_on

  @property
  def seat_belt_fastened(self) -> bool:
    with self._lock:
      return self._seat_belt_fastened

  def set_ignition(self, on: bool) -> None:
    with self._lock:
      previous = self._ignition_on
      self._ignition_on = on

      # Rising edge while idle: start the 10s waiting time.
      if on and not previous and self._state == WarningState.ALARM_OFF:
        self._enter_waiting()
      # Ignition OFF always silences the system immediately.
      elif not on:
        self._enter_alarm_off()

  def set_seat_belt_fastened(self, fastened: bool) -> None:
    with self._lock:
      self._seat_belt_fastened = fastened
      # Buckling up during waiting time or alarm returns to idle with no warning.
      if fastened and self._state in (WarningState.WAITING, WarningState.ALARM_ON):
        self._enter_alarm_off()

  def _handle_timer_expired(self) -> None:
    """Time-driven transitions (runs on the timer's background thread)."""
    with self._lock:
      if self._state == WarningState.WAITING and not self._seat_belt_fastened:
        self._enter_alarm_on()
      elif self._state == WarningState.ALARM_ON:
        self._enter_alarm_off()

  def _enter_waiting(self) -> None:
    # Private transition methods: centralize how each state is entered.
    self._state = WarningState.WAITING
    self._alarm_active = False
    self._timer.load(self.WAIT_DURATION)
    self._timer.start()

  def _enter_alarm_on(self) -> None:
    self._state = WarningState.ALARM_ON
    self._alarm_active = True
    self._timer.load(self.ALARM_DURATION)
    self._timer.start()

  def _enter_alarm_off(self) -> None:
    self._state = WarningState.ALARM_OFF
    self._alarm_active = False
    self._timer.stop()


# ---------------------------------------------------------------------------
# Simulation & console display — demo driver, not part of the embedded FSM
# ---------------------------------------------------------------------------

@dataclass
class SimulationStep:
  """Data class: a simple immutable-style record holding step inputs (no behavior).

  @dataclass auto-generates __init__ and __repr__ so we avoid boilerplate for
  a plain data container used by the demo scenario.
  """
  ignition_on: bool
  seat_belt_fastened: bool
  wait_seconds: float = 0.0


class ConsoleDisplay:
  """View layer: formats system state for the console (no FSM logic here).

  OOP notes:
    - Single responsibility: only handles presentation, not business rules.
    - Class-level dicts (STATE_LABELS): shared lookup tables on the class itself.
    - Depends on SeatBeltWarningSystem via method parameters, not inheritance.
  """

  # Class attributes — mapping tables reused by every ConsoleDisplay instance.
  STATE_LABELS = {
    WarningState.ALARM_OFF: "All clear",
    WarningState.WAITING: "Waiting time",
    WarningState.ALARM_ON: "** WARNING **",
  }

  TIMER_LABELS = {
    WarningState.WAITING: "Wait",
    WarningState.ALARM_ON: "Alarm",
  }

  def __init__(self) -> None:
    self._live_mode = False

  def print_banner(self) -> None:
    print()
    print("=" * 62)
    print("  SEAT BELT WARNING SYSTEM")
    print("  Real-time FSM simulation")
    print("=" * 62)
    print("  Waiting time   : 10 seconds after ignition ON")
    print("  Alarm duration : 5 seconds if belt stays unfastened")
    print("=" * 62)
    print()

  def print_event(self, message: str) -> None:
    self._end_live()
    print(f"  >> {message}")

  def print_step_header(self, step_number: int, label: str) -> None:
    self._end_live()
    title = label or f"Step {step_number}"
    print()
    print(f"-- Step {step_number}: {title} " + "-" * max(0, 40 - len(title)))

  def _end_live(self) -> None:
    if self._live_mode:
      sys.stdout.write("\r" + " " * 78 + "\r\n")
      sys.stdout.flush()
      self._live_mode = False

  def _progress_bar(self, remaining: float, total: float, width: int = 16) -> str:
    if total <= 0:
      return "[" + ("-" * width) + "]"
    filled = int(width * (total - remaining) / total)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

  def _compact_line(self, system: SeatBeltWarningSystem, elapsed: float) -> str:
    # Accepts a SeatBeltWarningSystem object — uses its public API only.
    state_label = self.STATE_LABELS[system.state]
    ignition = "IGN ON " if system.ignition_on else "IGN OFF"
    belt = "BELT OK " if system.seat_belt_fastened else "BELT OPEN"
    alarm = "BEEP!" if system.alarm_active else "quiet"

    timer = system.timer
    if timer.state == TimerState.RUNNING:
      total = (
        SeatBeltWarningSystem.WAIT_DURATION
        if system.state == WarningState.WAITING
        else SeatBeltWarningSystem.ALARM_DURATION
      )
      label = self.TIMER_LABELS.get(system.state, "Timer")
      bar = self._progress_bar(timer.remaining, total)
      timer_part = f"{label} {bar} {timer.remaining:4.1f}s"
    else:
      timer_part = "timer idle"

    return (
      f"  [{elapsed:5.1f}s] {state_label:14} | {ignition} | {belt} | "
      f"{timer_part} | {alarm}"
    )

  def render_live_line(self, system: SeatBeltWarningSystem, elapsed: float) -> None:
    """Update a single in-place status line (Windows-friendly)."""
    if not system.timer.is_running() and not system.alarm_active:
      return

    line = self._compact_line(system, elapsed)
    self._live_mode = True
    sys.stdout.write("\r" + line)
    sys.stdout.flush()

  def render_snapshot(self, system: SeatBeltWarningSystem, elapsed: float) -> None:
    """Print a full status block after input changes."""
    self._end_live()
    print(self._compact_line(system, elapsed))


class Simulator:
  """Controller: wires inputs, real-time waits, and display together.

  OOP notes:
    - Dependency injection: optional system/display args allow swapping mocks
      in tests or alternate UIs without changing this class.
    - Composition: has-a SeatBeltWarningSystem and has-a ConsoleDisplay.
    - Does not inherit from either — it coordinates them (composition over
      inheritance).
  """

  def __init__(
    self,
    system: SeatBeltWarningSystem | None = None,
    display: ConsoleDisplay | None = None,
  ) -> None:
    # Default construction if caller does not inject dependencies.
    self.system = system or SeatBeltWarningSystem()
    self.display = display or ConsoleDisplay()
    self.start_time = time.monotonic()
    # Snapshot previous values to detect changes and print transition events.
    self._prev_state = self.system.state
    self._prev_alarm = self.system.alarm_active
    self._prev_ignition = self.system.ignition_on
    self._prev_belt = self.system.seat_belt_fastened

  @property
  def elapsed(self) -> float:
    return time.monotonic() - self.start_time

  def _announce_transitions(self) -> None:
    system = self.system
    state = system.state

    if system.ignition_on != self._prev_ignition:
      if system.ignition_on:
        self.display.print_event("Ignition turned ON")
      else:
        self.display.print_event("Ignition turned OFF")

    if system.seat_belt_fastened != self._prev_belt:
      if system.seat_belt_fastened:
        self.display.print_event("Seat belt fastened")
      else:
        self.display.print_event("Seat belt unfastened")

    if state != self._prev_state:
      if state == WarningState.WAITING:
        self.display.print_event(
          f"Waiting time started - {SeatBeltWarningSystem.WAIT_DURATION}s to buckle up"
        )
      elif state == WarningState.ALARM_ON:
        self.display.print_event("ALARM ACTIVATED - buckle your seat belt!")
      elif state == WarningState.ALARM_OFF and self._prev_state == WarningState.ALARM_ON:
        self.display.print_event("Alarm cleared - system idle")

    self._prev_state = state
    self._prev_alarm = system.alarm_active
    self._prev_ignition = system.ignition_on
    self._prev_belt = system.seat_belt_fastened

  def apply_inputs(self, ignition_on: bool, seat_belt_fastened: bool) -> None:
    self.system.set_ignition(ignition_on)
    self.system.set_seat_belt_fastened(seat_belt_fastened)
    self._announce_transitions()

  def wait_real(self, seconds: float) -> None:
    """Sleep for real time, updating a single live status line when active."""
    if seconds <= 0:
      return

    end = time.monotonic() + seconds
    while True:
      self._poll_timer_transitions()
      self.display.render_live_line(self.system, self.elapsed)
      remaining = end - time.monotonic()
      if remaining <= 0:
        break
      time.sleep(min(0.5, remaining))

    self.display._end_live()

  def _poll_timer_transitions(self) -> None:
    """Timer expiry runs on a background thread; poll to print events promptly."""
    state = self.system.state
    alarm = self.system.alarm_active
    if state != self._prev_state or alarm != self._prev_alarm:
      self.display._end_live()
      self._announce_transitions()

  def apply_step(self, step: SimulationStep) -> None:
    self.apply_inputs(step.ignition_on, step.seat_belt_fastened)

  def run_scenario(
    self,
    steps: list[SimulationStep],
    labels: list[str] | None = None,
  ) -> None:
    self.display.print_banner()
    for index, step in enumerate(steps):
      label = labels[index] if labels and index < len(labels) else ""
      self.display.print_step_header(index + 1, label)
      self.apply_step(step)
      self.display.render_snapshot(self.system, self.elapsed)
      if step.wait_seconds > 0:
        self.wait_real(step.wait_seconds)

    self.display.print_event("Simulation complete")
    print()


def main() -> None:
  # Entry point: create objects and run the demo (script execution starts here).
  # Scripted walkthrough covering every FSM path from the system specification.
  simulator = Simulator()
  scenario = [
    SimulationStep(ignition_on=False, seat_belt_fastened=False),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=1),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=4),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=6),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=2),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=3),
    SimulationStep(ignition_on=False, seat_belt_fastened=False, wait_seconds=1),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=1),
    SimulationStep(ignition_on=True, seat_belt_fastened=True, wait_seconds=2),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=2),
    SimulationStep(ignition_on=False, seat_belt_fastened=False),
    SimulationStep(ignition_on=True, seat_belt_fastened=False, wait_seconds=1),
    SimulationStep(ignition_on=False, seat_belt_fastened=False, wait_seconds=1),
  ]

  labels = [
    "System at rest",
    "Driver starts the engine",
    "Waiting time counting down",
    "Waiting for timer to expire",
    "Alarm is sounding",
    "Waiting for alarm to finish",
    "Driver turns engine off",
    "Driver restarts the engine",
    "Passenger buckles up in time",
    "Passenger unbuckles (no new warning yet)",
    "Engine off, then on again",
    "New waiting time begins",
    "Engine off cancels warning",
  ]

  simulator.run_scenario(scenario, labels)


if __name__ == "__main__":
  main()
