from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from dataclasses import dataclass

from pydantic import BaseModel, Field

from agno.memory_v2.db.memory.base import MemoryDb
from agno.memory_v2.db.schema import MemoryRow, SummaryRow
from agno.memory_v2.db.summary.base import SummaryDb
from agno.memory_v2.manager import MemoryManager, MemoryUpdatesResponse
from agno.memory_v2.summarizer import SessionSummarizer
from agno.models.base import Model
from agno.models.message import Message
from agno.run.response import RunResponse
from agno.utils.log import log_debug, log_warning, logger
from agno.utils.prompts import get_json_output_prompt
from agno.utils.string import parse_response_model_str


class MemorySearchResponse(BaseModel):
    """Model for Memory Search Response."""

    memory_ids: List[str] = Field(..., description="The IDs of the memories that are most semantically similar to the query.")


@dataclass
class UserMemory:
    """Model for User Memories"""

    memory: str
    topics: Optional[str] = None
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "memory": self.memory,
            "topics": self.topics,
        }
        return {k: v for k, v in response.items() if v is not None}


@dataclass
class SessionSummary:
    """Model for Session Summary."""

    summary: str
    topics: Optional[List[str]] = None
    last_updated: Optional[datetime] = None

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
    make_user_memories: bool = False

    # Session summaries per session per user
    summaries: Optional[Dict[str, Dict[str, SessionSummary]]] = None
    # Summarizer to generate session summaries
    summarizer: Optional[SessionSummarizer] = None
    # Create and store session summaries
    make_session_summaries: bool = False

    memory_db: Optional[MemoryDb] = None
    summary_db: Optional[SummaryDb] = None

    retrieval: Literal["last_n", "first_n", "semantic"] = "last_n"
    retrieval_limit: Optional[int] = None

    # runs per session
    runs: Optional[Dict[str, list[RunResponse]]] = None


    def __init__(self,
                 model: Optional[Model] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 make_user_memories: bool = False,
                 summarizer: Optional[SessionSummarizer] = None,
                 make_session_summaries: bool = False,
                 memory_db: Optional[MemoryDb] = None,
                 summary_db: Optional[SummaryDb] = None,
                 retrieval: Literal["last_n", "first_n", "semantic"] = "last_n",
                 retrieval_limit: Optional[int] = None):
        self.model = model

        if self.model is None:
            self.model = self.get_model()

        self.memory_manager = memory_manager
        self.make_user_memories = make_user_memories

        self.summarizer = summarizer
        self.make_session_summaries = make_session_summaries

        self.memory_db = memory_db
        self.summary_db = summary_db

        self.retrieval = retrieval
        self.retrieval_limit = retrieval_limit

        # We are making memories
        if self.make_user_memories:
            if not self.memory_db:
                raise ValueError("MemoryDb not provided")

            if self.memory_manager is None:
                self.memory_manager = MemoryManager(model=self.model)
            # Set the model on the memory manager if it is not set
            if self.memory_manager.model is None:
                self.memory_manager.model = self.model

        # We are making session summaries
        if self.make_session_summaries:
            if not self.summary_db:
                raise ValueError("SummaryDb not provided")

            if self.summarizer is None:
                self.summarizer = SessionSummarizer(model=self.model)
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
        _memory_dict = {
            "create_session_summaries": self.make_session_summaries,
            "create_user_memories": self.make_user_memories,
            "retrieval": self.retrieval,
            "retrieval_limit": self.retrieval_limit,
        }
        # Add summary if it exists
        if self.summaries is not None:
            _memory_dict["summaries"] = {user_id: {session_id: summary.to_dict() for session_id, summary in session_summaries.items()} for user_id, session_summaries in self.summaries.items()}
        # Add memories if they exist
        if self.memories is not None:
            _memory_dict["memories"] = {user_id: {memory_id: memory.to_dict() for memory_id, memory in user_memories.items()} for user_id, user_memories in self.memories.items()}
        # Add runs if they exist
        if self.runs is not None:
            _memory_dict["runs"] = {session_id: [run.to_dict() for run in self.runs[session_id]] for session_id in self.runs}

        _memory_dict = {k: v for k, v in _memory_dict.items() if v is not None}
        return _memory_dict

    # -*- Public Functions
    def get_user_memories(self, user_id: str) -> Dict[str, UserMemory]:
        """Get the user memories for a given user id"""
        return self.memories.get(user_id, {})

    def get_session_summaries(self, user_id: str) -> Dict[str, SessionSummary]:
        """Get the session summaries for a given user id"""
        return self.summaries.get(user_id, {})

    def get_user_memory(self, user_id: str, memory_id: str) -> UserMemory:
        """Get the user memory for a given user id"""
        return self.memories.get(user_id, {}).get(memory_id, None)

    def get_session_summary(self, user_id: str, session_id: str) -> SessionSummary:
        """Get the session summary for a given user id"""
        return self.summaries.get(user_id, {}).get(session_id, None)

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
            self.upsert_db_memory(memory=MemoryRow(id=memory_id, user_id=user_id, memory=memory.to_dict(), last_updated=memory.last_updated or datetime.now()))

        # TODO: Log the addition
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()

        return memory_id

    def replace_user_memory(self, memory_id: str, memory: UserMemory, user_id: Optional[str] = None, ) -> str:
        """Replace a user memory for a given user id
        Args:
            memory_id (str): The id of the memory to replace
            memory (UserMemory): The memory to add
            user_id (Optional[str]): The user id to add the memory to. If not provided, the memory is added to the "default" user.
        Returns:
            str: The id of the memory
        """
        if user_id is None:
            user_id = "default"
        self.memories.setdefault(user_id, {})[memory_id] = memory
        if self.memory_db:
            self.upsert_db_memory(memory=MemoryRow(id=memory_id, user_id=user_id, memory=memory.to_dict(), last_updated=memory.last_updated or datetime.now()))

        # TODO: Log the addition
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()

        return memory_id

    def delete_user_memory(self, user_id: str, memory_id: str) -> None:
        """Delete a user memory for a given user id
        Args:
            user_id (str): The user id to delete the memory from
            memory_id (str): The id of the memory to delete
        """
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

        summary_response = self.summarizer.run(conversation=self.get_messages_for_session(session_id=session_id))
        session_summary = SessionSummary(
            summary=summary_response.summary,
            topics=summary_response.topics,
            last_updated=datetime.now()
        )
        self.summaries.setdefault(user_id, {})[session_id] = session_summary

        if self.summary_db:
            self.upsert_db_summary(summary=SummaryRow(id=session_id, user_id=user_id, summary=session_summary.to_dict(), last_updated=session_summary.last_updated))

        # TODO: Log the summary
        # thread = threading.Thread(target=_do_log_summary, daemon=True)
        # thread.start()
        return session_summary

    async def acreate_session_summary(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionSummary]:
        """Creates a summary of the session"""
        if user_id is None:
            user_id = "default"

        summary_response = await self.summarizer.arun(conversation=self.get_messages_for_session(session_id=session_id))
        session_summary = SessionSummary(
            summary=summary_response.summary,
            topics=summary_response.topics,
            last_updated=datetime.now()
        )
        self.summaries.setdefault(user_id, {})[session_id] = session_summary

        if self.summary_db:
            self.upsert_db_summary(summary=SummaryRow(id=session_id, user_id=user_id, summary=session_summary.to_dict(), last_updated=session_summary.last_updated))

        # TODO: Log the summary
        # import asyncio
        # asyncio.create_task(_do_log_summary())

        return session_summary

    def create_user_memory(self, message: Message, user_id: Optional[str] = None) -> Dict[str, UserMemory]:
        """Creates a memory from a message and adds it to the memory db."""
        return self.create_user_memories(messages=[message], user_id=user_id)

    def create_user_memories(self, messages: List[Message], user_id: Optional[str] = None) -> Dict[str, UserMemory]:
        """Creates memories from multiple messages and adds them to the memory db."""
        if not messages or not isinstance(messages, list):
            raise ValueError("Invalid messages list")

        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")

        if user_id is None:
            user_id = "default"

        existing_memories = self.memories.get(user_id, {})
        memory_updates: MemoryUpdatesResponse = self.memory_manager.run(messages=messages, existing_memories=list(existing_memories.values()) if existing_memories else None)

        response_memories = {}
        for update in memory_updates.updates:
            # We have an existing memory id, so we need to replace the memory
            if update.id is not None:
                user_memory = UserMemory(memory=update.memory, topics=update.topics, last_updated=datetime.now())
                memory_id = self.replace_user_memory(memory_id=update.id, memory=user_memory, user_id=user_id)
                response_memories[memory_id] = user_memory
            # We don't have an existing memory id, so we need to add a new memory
            else:
                user_memory = UserMemory(memory=update.memory, topics=update.topics, last_updated=datetime.now())
                memory_id = self.add_user_memory(memory=user_memory, user_id=user_id)
                response_memories[memory_id] = user_memory

        if not response_memories:
            log_warning("No memories created")

        return response_memories

    async def acreate_user_memory(self, message: Message, user_id: Optional[str] = None) -> Dict[str, UserMemory]:
        """Creates a memory from a message and adds it to the memory db."""
        return await self.acreate_user_memories(messages=[message], user_id=user_id)

    async def acreate_user_memories(self, messages: List[Message], user_id: Optional[str] = None) -> Dict[str, UserMemory]:
        """Creates memories from multiple messages and adds them to the memory db."""
        if not messages or not isinstance(messages, list):
            raise ValueError("Invalid messages list")

        if not self.memory_manager:
            raise ValueError("Memory manager not initialized")

        if user_id is None:
            user_id = "default"

        existing_memories = self.memories.get(user_id, {})
        memory_updates: MemoryUpdatesResponse = await self.memory_manager.arun(messages=messages, existing_memories=list(existing_memories.values()) if existing_memories else None)

        response_memories = {}
        for update in memory_updates.updates:
            # We have an existing memory id, so we need to replace the memory
            if update.id is not None:
                user_memory = UserMemory(memory=update.memory, topics=update.topics, last_updated=datetime.now())
                memory_id = self.replace_user_memory(memory_id=update.id, memory=user_memory, user_id=user_id)
                response_memories[memory_id] = user_memory
            # We don't have an existing memory id, so we need to add a new memory
            else:
                user_memory = UserMemory(memory=update.memory, topics=update.topics, last_updated=datetime.now())
                memory_id = self.add_user_memory(memory=user_memory, user_id=user_id)
                response_memories[memory_id] = user_memory

        if not response_memories:
            log_warning("No memories created")

        return response_memories


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
            self.memory_db.delete_memory(memory_id=memory_id)
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
            self.summary_db.delete_summary(summary_id=summary_id)
            return "Summary deleted successfully"
        except Exception as e:
            logger.warning(f"Error deleting summary in db: {e}")
            return f"Error deleting summary: {e}"


    # -*- Utility Functions
    def get_messages_for_session(
        self, session_id: str, user_role: str = "user", assistant_role: Optional[List[str]] = None, skip_history_messages: bool = True
    ) -> List[Message]:
        """Returns a list of messages for the session that iterate through user message and assistant response."""

        if assistant_role is None:
            assistant_role = ["assistant", "model", "CHATBOT"]

        final_messages: List[Message] = []
        session_runs = self.runs.get(session_id, [])
        for run_response in session_runs:
            if run_response and run_response.messages:
                user_message_from_run = None
                assistant_message_from_run = None

                # Start from the beginning to look for the user message
                for message in run_response.messages:
                    if hasattr(message, "from_history") and message.from_history and skip_history_messages:
                        continue
                    if message.role == user_role:
                        user_message_from_run = message
                        break

                # Start from the end to look for the assistant response
                for message in run_response.messages[::-1]:
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
        self, session_id: str, last_n: Optional[int] = None, skip_role: Optional[str] = None, skip_history_messages: bool = True
    ) -> List[Message]:
        """Returns the messages from the last_n runs, excluding previously tagged history messages.
        Args:
            session_id: The session id to get the messages from.
            last_n: The number of runs to return from the end of the conversation.
            skip_role: Skip messages with this role.
            skip_history_messages: Skip messages that were tagged as history in previous runs.
        Returns:
            A list of Messages from the specified runs, excluding history messages.
        """
        if not self.runs:
            return []

        session_runs = self.runs.get(session_id, [])
        runs_to_process = session_runs if last_n is None else session_runs[-last_n:]
        messages_from_history = []

        for run_response in runs_to_process:
            if not (run_response and run_response.messages):
                continue

            for message in run_response.messages:
                # Skip messages with specified role
                if skip_role and message.role == skip_role:
                    continue
                # Skip messages that were tagged as history in previous runs
                if hasattr(message, "from_history") and message.from_history and skip_history_messages:
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


    def search_user_memories(
        self, 
        query: Optional[str] = None, 
        limit: Optional[int] = None,
        retrieval_method: Optional[Literal["last_n", "first_n", "semantic"]] = None,
        user_id: Optional[str] = None
    ) -> List[UserMemory]:
        """Search through user memories using the specified retrieval method.
        
        Args:
            query: The search query for semantic search. Required if retrieval_method is "semantic".
            limit: Maximum number of memories to return. Defaults to self.retrieval_limit if not specified.
            retrieval_method: The method to use for retrieving memories. Defaults to self.retrieval if not specified.
                - "last_n": Return the most recent memories
                - "first_n": Return the oldest memories
                - "semantic": Return memories most semantically similar to the query
                
        Returns:
            A list of UserMemory objects matching the search criteria.
        """
        if not self.memories:
            return []
        
        if user_id is None:
            user_id = "default"
            
        # Use default retrieval method if not specified
        retrieval_method = retrieval_method or self.retrieval
        # Use default limit if not specified
        limit = limit or self.retrieval_limit
        
        # Handle different retrieval methods
        if retrieval_method == "semantic":
            if not query:
                raise ValueError("Query is required for semantic search")
                
            return self._search_user_memories_semantic(query=query, limit=limit)
            
        elif retrieval_method == "first_n":
            return self._get_first_n_memories(user_id=user_id, limit=limit)
            
        else:  # Default to last_n
            return self._get_last_n_memories(user_id=user_id, limit=limit)
    
    def _update_model_for_semantic_search(self) -> None:
        if self.model.supports_native_structured_outputs:
            self.model.response_format = MemorySearchResponse
            self.model.structured_outputs = True

        elif self.model.supports_json_schema_outputs:
            self.model.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": MemorySearchResponse.__name__,
                    "schema": MemorySearchResponse.model_json_schema(),
                },
            }
        else:
            self.model.response_format = {"type": "json_object"}
        
    def _search_user_memories_semantic(self, user_id: str, query: str, limit: Optional[int] = None) -> List[UserMemory]:
        """Search through user memories using semantic search."""
        if not self.memories:
            return []
        
        self._update_model_for_semantic_search()

        log_debug("Searching for memories", center=True)

        # Get all memories as a list
        user_memories: Dict[str, UserMemory] = self.memories[user_id]
        system_message_str = "You are a helpful assistant that can search through user memories."
        system_message_str += "<user_memories>\n"
        for memory in user_memories.values():
            system_message_str += f"ID: {memory.id}\n"
            system_message_str += f"Memory: {memory.memory}\n"
            if memory.topics:
                system_message_str += f"Topics: {memory.topics}\n"
            system_message_str += "\n"
        system_message_str += "</user_memories>\n"
        system_message_str += "Only return the IDs of the memories that are most semantically similar to the query."

        if self.model.response_format == {"type": "json_object"}:
            system_message_str += "\n" + get_json_output_prompt(MemorySearchResponse)

        messages_for_model = [
            Message(role="system", content=system_message_str),
            Message(role="user", content=f"Search for memories that are most semantically similar to the following query: {query}")
        ]
        
        # Generate a response from the Model (includes running function calls)
        response = self.model.response(messages=messages_for_model)
        log_debug("Search for memories complete", center=True)

        # If the model natively supports structured outputs, the parsed value is already in the structured format
        if self.model.supports_native_structured_outputs and response.parsed is not None and isinstance(response.parsed, MemorySearchResponse):
            memory_search = response.parsed

        # Otherwise convert the response to the structured format
        if isinstance(response.content, str):
            try:
                memory_search: Optional[MemorySearchResponse] = parse_response_model_str(response.content, MemorySearchResponse)

                # Update RunResponse
                if memory_search is None:
                    log_warning("Failed to convert memory_search response to MemorySearchResponse")
                    return []
            except Exception as e:
                log_warning(f"Failed to convert memory_search response to MemorySearchResponse: {e}")
                return []

        memories_to_return = []
        for memory_id in memory_search.memory_ids:
            memories_to_return.append(user_memories[memory_id])
        return memories_to_return
        
    
    def _get_last_n_memories(self, user_id: str, limit: Optional[int] = None) -> List[UserMemory]:
        """Get the most recent user memories.
        
        Args:
            limit: Maximum number of memories to return.
            
        Returns:
            A list of the most recent UserMemory objects.
        """
        if not self.memories:
            return []
            
        memories_to_return = self.memories.get(user_id, {})
        if limit is not None and limit > 0:
            memories_to_return = list(memories_to_return.values())[-limit:]
            
        return memories_to_return
    
    def _get_first_n_memories(self, user_id: str, limit: Optional[int] = None) -> List[UserMemory]:
        """Get the oldest user memories.
        
        Args:
            limit: Maximum number of memories to return.
            
        Returns:
            A list of the oldest UserMemory objects.
        """
        if not self.memories:
            return []
            
        memories_to_return = self.memories
        if limit is not None and limit > 0:
            memories_to_return = memories_to_return[:limit]
            
        return memories_to_return


    def deep_copy(self) -> "Memory":
        from copy import deepcopy

        # Create a shallow copy of the object
        copied_obj = self.__class__(**self.to_dict())

        # Manually deepcopy fields that are known to be safe
        for field_name, field_value in self.__dict__.items():
            if field_name not in ["memory_db", "summary_db", "memory_manager", "summarizer"]:
                try:
                    setattr(copied_obj, field_name, deepcopy(field_value))
                except Exception as e:
                    logger.warning(f"Failed to deepcopy field: {field_name} - {e}")
                    setattr(copied_obj, field_name, field_value)

        copied_obj.memory_db = self.memory_db
        copied_obj.summary_db = self.summary_db
        copied_obj.memory_manager = self.memory_manager
        copied_obj.summarizer = self.summarizer

        return copied_obj
