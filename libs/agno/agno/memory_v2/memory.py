from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass

from agno.memory_v2.db.memory.base import MemoryDb
from agno.memory_v2.db.schema import MemoryRow, SummaryRow
from agno.memory_v2.db.summary.base import SummaryDb
from agno.memory_v2.manager import MemoryManager
from agno.memory_v2.summarizer import SessionSummarizer, SessionSummary
from agno.models.base import Model
from agno.models.message import Message
from agno.run.response import RunResponse
from agno.utils.log import log_debug, log_info, logger



@dataclass
class UserMemory:
    """Model for User Memories"""

    memory: str
    topic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "memory": self.memory,
            "topic": self.topic,
        }
        return {k: v for k, v in response.items() if v is not None}


@dataclass
class SessionSummary:
    """Model for Session Summary."""

    summary: str 
    topics: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "summary": self.summary,
            "topics": self.topics,
        }
        return {k: v for k, v in response.items() if v is not None}



@dataclass
class Memory:
    # Model used for memories and summaries
    model: Optional[Model] = None

    # Memories per memory ID per user 
    memories: Optional[Dict[str, Dict[str, UserMemory]]] = None

    # Manager to manage memories
    memory_manager: Optional[MemoryManager] = None
    # Create and store personalized memories for this user
    create_user_memories: bool = False

    # Session summaries per session per user
    summaries: Optional[Dict[str, Dict[str, SessionSummary]]] = None
    # Summarizer to generate session summaries
    summarizer: Optional[SessionSummarizer] = None
    # Create and store session summaries
    create_session_summaries: bool = False

    memory_db: Optional[MemoryDb] = None
    summary_db: Optional[SummaryDb] = None

    retrieval: Literal["last_n", "first_n", "semantic"] = "last_n"
    retrieval_limit: Optional[int] = None

    # runs per session
    runs: Optional[Dict[str, list[RunResponse]]] = None


    def __init__(self, 
                 model: Optional[Model] = None, 
                 memory_manager: Optional[MemoryManager] = None,
                 create_user_memories: bool = False,
                 summarizer: Optional[SessionSummarizer] = None,
                 create_session_summaries: bool = False,
                 memory_db: Optional[MemoryDb] = None,
                 summary_db: Optional[SummaryDb] = None,
                 retrieval: Literal["last_n", "first_n", "semantic"] = "last_n",
                 retrieval_limit: Optional[int] = None):
        self.model = model

        if self.model is None:
            self.model = self.get_model()

        self.memory_manager = memory_manager
        self.create_user_memories = create_user_memories
                
        self.summarizer = summarizer
        self.create_session_summaries = create_session_summaries
        
        self.memory_db = memory_db
        self.summary_db = summary_db

        self.retrieval = retrieval
        self.retrieval_limit = retrieval_limit

        # We are making memories
        if self.create_user_memories:
            if not self.memory_db:
                raise ValueError("MemoryDb not provided")

            if self.memory_manager is None:
                self.memory_manager = MemoryManager(model=self.model, memory_db=self.memory_db)
            # Set the model on the memory manager if it is not set
            if self.memory_manager.model is None:
                self.memory_manager.model = self.model

        # We are making session summaries
        if self.create_session_summaries:
            if not self.summary_db:
                raise ValueError("SummaryDb not provided")

            if self.summarizer is None:
                self.summarizer = SessionSummarizer(self.model)
            # Set the model on the summarizer if it is not set
            elif self.summarizer.model is None:
                self.summarizer.model = self.model
        
        if self.memory_db or self.summary_db:
            self.initialize()


    def get_model(self) -> Model:
        if self.model is None:
            try:
                from agno.models.openai import OpenAIChat
            except ModuleNotFoundError as e:
                logger.exception(e)
                logger.error(
                    "Agno uses `openai` as the default model provider. Please provide a `model` or install `openai`."
                )
                exit(1)
            self.model = OpenAIChat(id="gpt-4o")
        return self.model
    
    def initialize(self):
        # TODO: Rehydrate memories from DB
        pass

    def to_dict(self) -> Dict[str, Any]:
        _memory_dict = self.model_dump(
            exclude_none=True,
            include={
                "create_session_summaries",
                "create_user_memories",
                "retrieval",
                "retrieval_limit",
            },
        )
        # Add summary if it exists
        if self.summaries is not None:
            _memory_dict["summaries"] = {user_id: {session_id: summary.to_dict() for session_id, summary in self.summaries.items()} for user_id in self.summaries}
        # Add memories if they exist
        if self.memories is not None:
            _memory_dict["memories"] = {user_id: {memory_id: memory.to_dict() for memory_id, memory in self.memories.items()} for user_id in self.memories}
        # Add runs if they exist
        if self.runs is not None:
            _memory_dict["runs"] = {session_id: [run.to_dict() for run in self.runs[session_id]] for session_id in self.runs}
        return _memory_dict

    # -*- Public Functions
    def get_user_memories(self, user_id: str) -> Dict[str, UserMemory]:
        """Get the user memories for a given user id"""
        return self.memories.get(user_id, {})

    def get_session_summaries(self, user_id: str) -> Dict[str, SessionSummary]:
        """Get the session summaries for a given user id"""
        return self.session_summaries.get(user_id, {})

    def get_user_memory(self, user_id: str, memory_id: str) -> UserMemory:
        """Get the user memory for a given user id"""
        return self.memories.get(user_id, {}).get(memory_id, None)

    def get_session_summary(self, user_id: str, session_id: str) -> SessionSummary:
        """Get the session summary for a given user id"""
        return self.session_summaries.get(user_id, {}).get(session_id, None)

    def add_user_memory(self, memory: UserMemory, user_id: Optional[str] = None, ) -> str:
        """Add a user memory for a given user id
        Args:
            memory (UserMemory): The memory to add
            user_id (Optional[str]): The user id to add the memory to. If not provided, the memory is added to the "default" user.
        Returns:
            str: The id of the memory
        """
        from uuid import uuid4
        memory_id = str(uuid4())
        if user_id is None:
            user_id = "default"
        self.memories.setdefault(user_id, {})[memory_id] = memory
        if self.memory_db:
            self.upsert_db_memory(memory=MemoryRow(id=memory_id, user_id=user_id, memory=UserMemory.to_dict()))
                
        # TODO: Log the addition
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()

        return memory_id
    
    def replace_user_memory(self, memory_id: str, memory: UserMemory, user_id: Optional[str] = None, ) -> str:
        """Replace a user memory for a given user id
        Args:
            memory (UserMemory): The memory to add
            user_id (Optional[str]): The user id to add the memory to. If not provided, the memory is added to the "default" user.
        Returns:
            str: The id of the memory
        """
        if user_id is None:
            user_id = "default"
        self.memories.setdefault(user_id, {})[memory_id] = memory
        if self.memory_db:
            self.upsert_db_memory(memory=MemoryRow(id=memory_id, user_id=user_id, memory=UserMemory.to_dict()))
        
        # TODO: Log the addition
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()

        return memory_id

    def delete_user_memory(self, user_id: str, memory_id: str) -> None:
        """Delete a user memory for a given user id"""
        del self.memories[user_id][memory_id]
        if self.memory_db:
            self.delete_db_memory(memory_id=memory_id)
        
        # TODO: Log the deletion
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()


    # -*- Agent Functions
    def create_session_summary(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        if user_id is None: 
            user_id = "default"
        
        summary = self.summarizer.run(conversation=self.get_messages_for_session(session_id=session_id))
        self.session_summaries.setdefault(user_id, {})[session_id] = summary
        
        if self.summary_db:
            self.upsert_db_summary(summary=SummaryRow(id=session_id, user_id=user_id, summary=summary.to_dict()))
        
        # TODO: Log the summary
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()
        return summary

    async def acreate_session_summary(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        if user_id is None: 
            user_id = "default"
        
        summary = await self.summarizer.arun(conversation=self.get_messages_for_session(session_id=session_id))
        self.session_summaries[user_id][session_id] = summary

        if self.summary_db:
            self.upsert_db_summary(summary=SummaryRow(id=session_id, user_id=user_id, summary=summary.to_dict()))
        
        # TODO: Log the summary
        # import asyncio
        # asyncio.create_task(_do_log_summary())

        return summary

    # TODO TODO TODO
    def update_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""
        from agno.memory.manager import MemoryManager

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            logger.warning("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or self.should_update_memory(input=input)
        log_debug(f"Update memory: {should_update_memory}")

        if not should_update_memory:
            log_debug("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = self.manager.run(input)
        self.load_user_memories()
        self.updating_memory = False
        return response

    async def aupdate_memory(self, input: str, force: bool = False) -> Optional[str]:
        """Creates a memory from a message and adds it to the memory db."""
        from agno.memory.manager import MemoryManager

        if input is None or not isinstance(input, str):
            return "Invalid message content"

        if self.db is None:
            logger.warning("MemoryDb not provided.")
            return "Please provide a db to store memories"

        self.updating_memory = True

        # Check if this user message should be added to long term memory
        should_update_memory = force or await self.ashould_update_memory(input=input)
        log_debug(f"Async update memory: {should_update_memory}")

        if not should_update_memory:
            log_debug("Memory update not required")
            return "Memory update not required"

        if self.manager is None:
            self.manager = MemoryManager(user_id=self.user_id, db=self.db)

        else:
            self.manager.db = self.db
            self.manager.user_id = self.user_id

        response = await self.manager.arun(input)
        self.load_user_memories()
        self.updating_memory = False
        return response



    # -*- DB Functions
    def upsert_db_memory(self, memory: MemoryRow) -> str:
        """Use this function to add a memory to the database."""
        try:
            self.memory_db.upsert_memory(memory)
            return "Memory added successfully"
        except Exception as e:
            logger.warning(f"Error storing memory in db: {e}")
            return f"Error adding memory: {e}"

    def delete_db_memory(self, memory_id: str) -> str:
        """Use this function to delete a memory from the database."""
        try:
            self.memory_db.delete_memory(id=memory_id)
            return "Memory deleted successfully"
        except Exception as e:
            logger.warning(f"Error deleting memory in db: {e}")
            return f"Error deleting memory: {e}"

    def upsert_db_summary(self, summary: SummaryRow) -> str:
        """Use this function to add a summary to the database."""
        try:
            self.summary_db.upsert_summary(summary)
            return "Summary added successfully"
        except Exception as e:
            logger.warning(f"Error storing summary in db: {e}")
            return f"Error adding summary: {e}"

    def delete_db_summary(self, summary_id: str) -> str:
        """Use this function to delete a summary from the database."""
        try:
            self.summary_db.delete_summary(id=summary_id)
            return "Summary deleted successfully"
        except Exception as e:
            logger.warning(f"Error deleting summary in db: {e}")
            return f"Error deleting summary: {e}"


    def get_messages_for_session(
        self, session_id: str, user_role: str = "user", assistant_role: Optional[List[str]] = None, skip_history_messages: bool = True
    ) -> List[Message]:
        """Returns a list of messages for the session that iterate through user message and assistant response."""

        if assistant_role is None:
            assistant_role = ["assistant", "model", "CHATBOT"]

        final_messages: List[Message] = []
        session_runs = self.runs.get(session_id, [])
        for run in session_runs:
            if run.response and run.response.messages:
                user_message_from_run = None
                assistant_message_from_run = None

                # Start from the beginning to look for the user message
                for message in run.response.messages:
                    if hasattr(message, "from_history") and message.from_history and skip_history_messages:
                        continue
                    if message.role == user_role:
                        user_message_from_run = message
                        break

                # Start from the end to look for the assistant response
                for message in run.response.messages[::-1]:
                    if hasattr(message, "from_history") and message.from_history and skip_history_messages:
                        continue
                    if message.role in assistant_role:
                        assistant_message_from_run = message
                        break

                if user_message_from_run and assistant_message_from_run:
                    final_messages.append(user_message_from_run)
                    final_messages.append(assistant_message_from_run)
        return final_messages


    def add_run(self, session_id: str, run: RunResponse) -> None:
        """Adds a RunResponse to the runs list."""
        self.runs.setdefault(session_id, []).append(run)
        log_debug("Added RunResponse to Memory")

    def get_messages_from_last_n_runs(
        self, last_n: Optional[int] = None, skip_role: Optional[str] = None
    ) -> List[Message]:
        """Returns the messages from the last_n runs, excluding previously tagged history messages.
        Args:
            last_n: The number of runs to return from the end of the conversation.
            skip_role: Skip messages with this role.
        Returns:
            A list of Messages from the specified runs, excluding history messages.
        """
        if not self.runs:
            return []

        runs_to_process = self.runs if last_n is None else self.runs[-last_n:]
        messages_from_history = []

        for run in runs_to_process:
            if not (run.response and run.response.messages):
                continue

            for message in run.response.messages:
                # Skip messages with specified role
                if skip_role and message.role == skip_role:
                    continue
                # Skip messages that were tagged as history in previous runs
                if hasattr(message, "from_history") and message.from_history:
                    continue

                messages_from_history.append(message)

        log_debug(f"Getting messages from previous runs: {len(messages_from_history)}")
        return messages_from_history


    def get_tool_calls(self, session_id: str, num_calls: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns a list of tool calls from the messages"""

        tool_calls = []
        session_runs = self.runs.get(session_id, [])
        for run in session_runs[::-1]:
            if run.response and run.response.messages:
                for message in run.response.messages:
                    if message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_calls.append(tool_call)
                            if num_calls and len(tool_calls) >= num_calls:
                                return tool_calls
        return tool_calls

    def load_user_memories(self) -> None:
        """Load memories from memory db for this user."""

        if self.db is None:
            return

        try:
            if self.retrieval in (MemoryRetrieval.last_n, MemoryRetrieval.first_n):
                memory_rows = self.db.read_memories(
                    user_id=self.user_id,
                    limit=self.num_memories,
                    sort="asc" if self.retrieval == MemoryRetrieval.first_n else "desc",
                )
            else:
                raise NotImplementedError("Semantic retrieval not yet supported.")
        except Exception as e:
            log_debug(f"Error reading memory: {e}")
            return

        # Clear the existing memories
        self.memories = []

        # No memories to load
        if memory_rows is None or len(memory_rows) == 0:
            return

        for row in memory_rows:
            try:
                self.memories.append(Memory.model_validate(row.memory))
            except Exception as e:
                logger.warning(f"Error loading memory: {e}")
                continue


    def clear(self) -> None:
        """Clear the AgentMemory"""

        self.runs = []
        self.messages = []
        self.summary = None
        self.memories = None

    def deep_copy(self) -> "AgentMemory":
        from copy import deepcopy

        # Create a shallow copy of the object
        copied_obj = self.__class__(**self.to_dict())

        # Manually deepcopy fields that are known to be safe
        for field_name, field_value in self.__dict__.items():
            if field_name not in ["db", "classifier", "manager", "summarizer"]:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    logger.warning(f"Failed to deepcopy field: {field_name} - {e}")
                    setattr(copied_obj, field_name, field_value)

        copied_obj.db = self.db
        copied_obj.classifier = self.classifier
        copied_obj.manager = self.manager
        copied_obj.summarizer = self.summarizer

        return copied_obj