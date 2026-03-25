"""Event table construction for dosing regimens and observation times.

Provides the Python equivalent of rxode2's ``et()`` for building
NONMEM-style event datasets with dosing, sampling, and repeat semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class _Event:
    """A single row in the event table."""

    time: float
    amt: float
    evid: int
    cmt: int
    dur: float
    ii: float
    addl: int
    rate: float = 0.0


def _sort_key(event: _Event) -> tuple[float, int]:
    """Sort events by time, with doses (evid>0) before observations (evid=0) at same time."""
    # Lower sort value = comes first. Doses (evid>0) get 0, obs (evid=0) get 1.
    obs_flag = 0 if event.evid > 0 else 1
    return (event.time, obs_flag)


class EventTable:
    """NONMEM-style event dataset builder.

    Supports method chaining: each mutating method returns a *new*
    ``EventTable`` so the original is never modified.

    Standard NONMEM column semantics:
        - evid=0: observation
        - evid=1: dose
        - evid=2: other type event
        - amt: dose amount (0 for observations)
        - cmt: compartment number
        - dur: infusion duration (0 = bolus)
        - ii: inter-dose interval for steady-state
        - addl: additional doses
    """

    def __init__(self, events: Sequence[_Event] | None = None) -> None:
        self._events: tuple[_Event, ...] = tuple(events) if events else ()

    # -- mutating (return new) methods --------------------------------------

    def add_dosing(
        self,
        amt: float,
        time: float,
        cmt: int = 1,
        evid: int = 1,
        dur: float = 0.0,
        ii: float = 0.0,
        addl: int = 0,
        rate: float = 0.0,
    ) -> EventTable:
        """Add a dose event and return a new EventTable."""
        ev = _Event(
            time=float(time),
            amt=float(amt),
            evid=int(evid),
            cmt=int(cmt),
            dur=float(dur),
            ii=float(ii),
            addl=int(addl),
            rate=float(rate),
        )
        return EventTable(self._events + (ev,))

    def add_sampling(self, times: Sequence[float]) -> EventTable:
        """Add observation time points (evid=0) and return a new EventTable."""
        new_events = tuple(
            _Event(time=float(t), amt=0.0, evid=0, cmt=0, dur=0.0, ii=0.0, addl=0)
            for t in times
        )
        return EventTable(self._events + new_events)

    def repeat(self, n: int, interval: float) -> EventTable:
        """Repeat the current event pattern *n* additional times with the given interval.

        Parameters
        ----------
        n
            Number of additional repetitions (0 means no change).
        interval
            Time offset between successive repetitions.

        Returns
        -------
        EventTable
            New table containing the original events plus *n* shifted copies.
        """
        if n <= 0:
            return EventTable(self._events)

        all_events = list(self._events)
        for rep in range(1, n + 1):
            offset = rep * interval
            for ev in self._events:
                all_events.append(
                    _Event(
                        time=ev.time + offset,
                        amt=ev.amt,
                        evid=ev.evid,
                        cmt=ev.cmt,
                        dur=ev.dur,
                        ii=ev.ii,
                        addl=ev.addl,
                        rate=ev.rate,
                    )
                )
        return EventTable(all_events)

    def expand(self) -> EventTable:
        """Return a new EventTable with ADDL/II expanded into individual dose events."""
        return expand_addl(self)

    # -- output methods -----------------------------------------------------

    def _sorted_events(self) -> list[_Event]:
        return sorted(self._events, key=_sort_key)

    def to_dict(self) -> dict[str, list]:
        """Return a dict of Python lists, sorted by time.

        Returns
        -------
        dict
            Keys: ``time``, ``amt``, ``evid``, ``cmt``, ``dur``, ``ii``, ``addl``, ``rate``.
        """
        events = self._sorted_events()
        return {
            "time": [e.time for e in events],
            "amt": [e.amt for e in events],
            "evid": [e.evid for e in events],
            "cmt": [e.cmt for e in events],
            "dur": [e.dur for e in events],
            "ii": [e.ii for e in events],
            "addl": [e.addl for e in events],
            "rate": [e.rate for e in events],
        }

    def to_arrays(self) -> dict[str, jnp.ndarray]:
        """Return a dict of JAX arrays, sorted by time.

        Returns
        -------
        dict
            Same keys as :meth:`to_dict` but values are 1-D ``jax.numpy`` arrays
            with ``float32`` dtype.
        """
        d = self.to_dict()
        return {k: jnp.array(v, dtype=jnp.float32) for k, v in d.items()}


def expand_addl(event_table: EventTable) -> EventTable:
    """Expand ADDL/II fields into individual dose events.

    For each dose event (evid > 0) with addl > 0 and ii > 0, create
    ``addl`` additional dose events spaced ``ii`` apart after the original.
    Each generated dose copies amt, cmt, dur, evid from the original but
    has addl=0 and ii=0.  Observation records pass through unchanged.

    Parameters
    ----------
    event_table
        Source EventTable (not modified).

    Returns
    -------
    EventTable
        New table with all ADDL/II records expanded.

    Raises
    ------
    ValueError
        If a dose has addl > 0 but ii <= 0.
    """
    expanded: list[_Event] = []
    for ev in event_table._events:
        if ev.evid > 0 and ev.addl > 0:
            if ev.ii <= 0:
                raise ValueError(
                    f"ii must be positive when addl > 0 (got ii={ev.ii}, addl={ev.addl})"
                )
            # Original dose with addl/ii zeroed out
            expanded.append(_Event(
                time=ev.time, amt=ev.amt, evid=ev.evid,
                cmt=ev.cmt, dur=ev.dur, ii=0.0, addl=0, rate=ev.rate,
            ))
            # Additional doses
            for k in range(1, ev.addl + 1):
                expanded.append(_Event(
                    time=ev.time + k * ev.ii,
                    amt=ev.amt, evid=ev.evid,
                    cmt=ev.cmt, dur=ev.dur, ii=0.0, addl=0, rate=ev.rate,
                ))
        else:
            expanded.append(ev)
    return EventTable(sorted(expanded, key=_sort_key))


def et() -> EventTable:
    """Create a new empty EventTable (factory function)."""
    return EventTable()
