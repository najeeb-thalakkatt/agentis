"""Session — stateful conversation management.

AgentRuntime IS a session. Creating a runtime starts a session.
Calling run() adds turns. reset() clears conversation but keeps memory.
fork() creates a new runtime inheriting the current state.
"""

from __future__ import annotations

import copy
import uuid

from agentis.types import Message


class Session:
    """Manages conversation state across multiple turns.

    Holds messages, turn count, and provides reset/fork operations
    for session lifecycle management.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._messages: list[Message] = []
        self.turn_count: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        """Return all messages in the conversation."""
        return list(self._messages)

    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self.turn_count += 1

    def reset(self) -> None:
        """Clear conversation history and turn count.

        Keeps the session_id — this is a new chat within the same session.
        """
        self._messages.clear()
        self.turn_count = 0

    def fork(self) -> Session:
        """Create an independent copy of this session.

        The forked session gets a new ID but inherits the full
        conversation history and turn count. Changes to either
        session are isolated.
        """
        forked = Session(session_id=f"fork-{uuid.uuid4()}")
        forked._messages = copy.deepcopy(self._messages)
        forked.turn_count = self.turn_count
        return forked
