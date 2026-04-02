"""Compaction subsystem — context management and file deduplication."""

from __future__ import annotations

from agentis.compaction.compactor import CompactionResult, ContextCompactor
from agentis.compaction.dedup import DedupResult, FileDeduplicator

__all__ = ["CompactionResult", "ContextCompactor", "DedupResult", "FileDeduplicator"]
