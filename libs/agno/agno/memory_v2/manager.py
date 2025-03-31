from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field

from agno.memory.db import MemoryDb
from agno.memory.memory import Memory
from agno.memory.row import MemoryRow
from agno.models.base import Model
from agno.models.message import Message
from agno.tools.function import Function
from agno.utils.log import log_debug, log_error, logger


class MemoryUpdate(BaseModel):
    """Model for updates to the user's memory."""

    memory: str = Field(
        ...,
        description="The user memory to be stored or updated.",
    )
    topic: Optional[str] = Field(None, description="The topic of the memory.")
    id: Optional[str] = Field(None, description="The id of the memory to update. ONLY use if you want to update an existing memory.")

class MemoryUpdates(BaseModel):
    """Model for updates to the user's memory."""

    updates: List[MemoryUpdate] = Field(
        ...,
        description="The updates to the user's memory.",
    )
    
@dataclass
class MemoryManager:
    """Model for Memory Manager"""

    model: Optional[Model] = None

    # Provide the system prompt for the manager as a string
    system_prompt: Optional[str] = None

    def update_model(self) -> None:

        if self.model.supports_native_structured_outputs:
            self.model.response_format = MemoryUpdates
            self.model.structured_outputs = True

        elif self.model.supports_json_schema_outputs:
            self.model.response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": MemoryUpdates.__name__,
                    "schema": MemoryUpdates.model_json_schema(),
                },
            }
        else:
            self.model.response_format = {"type": "json_object"}


    def get_system_message(self, messages: List[Message], existing_memories: Optional[List[Memory]] = None) -> Message:
        if self.system_prompt is not None:
            return Message(role="system", content=self.system_prompt)
        
        # -*- Return a system message for the memory manager
        system_prompt_lines = [
            "Your task is to generate concise memories for the user's messages. "
            "Create one or more memories that captures the key information provided by the user, as if you were storing it for future reference. "
            "Each memory should be a brief, third-person statement that encapsulates the most important aspect of the user's input, without adding any extraneous information. "
            "Memories should include details that could personalize ongoing interactions with the user, such as:\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - Significant life events or experiences shared by the user\n"
            "  - Important context about the user's current situation, challenges or goals\n"
            "  - What the user likes or dislikes, their opinions, beliefs, values, etc.\n"
            "  - Any other details that provide valuable insights into the user's personality, perspective or needs",
            "You will also be provided with a list of existing memories. You may:",
            "  1. Decide to make no changes to the existing memories.",
            "  2. Decide to add new memories.",
            "  3. Decide to update existing memories.",
        ]
        system_prompt_lines.append("<user_messages>")
        user_messages = []
        for message in messages:
            if message.role == "user":
                user_messages.append(message.content)
        system_prompt_lines.append("\n".join(user_messages))
        system_prompt_lines.append("</user_messages>")
        
        if existing_memories and len(existing_memories) > 0:
            system_prompt_lines.extend(
                [
                    "\nExisting memories:",
                    "<existing_memories>\n"
                    + "\n".join([f"  - {m.memory}" for m in self.existing_memories])
                    + "\n</existing_memories>",
                ]
            )
        return Message(role="system", content="\n".join(system_prompt_lines))

    def run(
        self,
        messages: Optional[List[Message]] = None,
        existing_memories: Optional[List[Memory]] = None,
    ) -> Optional[str]:
        if self.model is None:
            log_error("No model provided for memory manager")
            return
        
        log_debug("MemoryManager Start", center=True)

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [
            self.get_system_message(messages, existing_memories),
            # For models that require a non-system message
            Message(role="user", content="Create or update memories based on the user's messages."),
        ]

        # Generate a response from the Model (includes running function calls)
        response = self.model.response(messages=messages_for_model)
        log_debug("MemoryManager End", center=True)
        return response.content

    async def arun(
        self,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        if self.model is None:
            log_error("No model provided for memory manager")
            return
        log_debug("MemoryManager Start", center=True)

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [self.get_system_message()]
        # Add the user prompt message
        user_prompt_message = Message(role="user", content=message, **kwargs) if message else None
        if user_prompt_message is not None:
            messages_for_model += [user_prompt_message]

        # Set input message added with the memory
        self.input_message = message

        # Generate a response from the Model (includes running function calls)
        self.model = cast(Model, self.model)
        response = await self.model.aresponse(messages=messages_for_model)
        log_debug("MemoryManager End", center=True)
        return response.content
