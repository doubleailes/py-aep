"""Pause the cyclic garbage collector while building large object graphs."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator


@contextmanager
def gc_paused() -> Iterator[None]:
    """Disable the cyclic garbage collector for the duration of the block.

    Parsing a project and building a layer's property tree allocate tens of
    thousands of long-lived objects in one go. Each allocation burst keeps
    triggering collections that walk the whole growing heap without freeing
    anything - a third of the time spent building a large property tree.
    Reference counting still frees acyclic garbage meanwhile; the collector
    is re-enabled (only if it was enabled on entry) when the block exits, so
    nested blocks and callers that disabled it themselves are left alone.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
