from dataclasses import dataclass
from typing import Any, List, Optional, cast

from pydantic import BaseModel

from agno.memory.memory import Memory
from agno.models.base import Model
from agno.models.message import Message
from agno.utils.log import log_debug, log_error, logger


@dataclass
class MemoryClassifier:
    model: Optional[Model] = None

    # Provide the system prompt for the classifier as a string
    system_prompt: Optional[str] = None

    def __init__(self, model: Optional[Model] = None, system_prompt: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt

    def get_system_message(self, messages: List[Message], existing_memories: Optional[List[Memory]] = None) -> Message:
        if self.system_prompt is not None:
            return Message(role="system", content=self.system_prompt)
        
        # -*- Return a system message for classification
        system_prompt_lines = [
            "Your task is to identify if the user's messages contains information that is worth remembering for future conversations.",
            "This includes details that could personalize ongoing interactions with the user, such as:\n"
            "  - Personal facts: name, age, occupation, location, interests, preferences, etc.\n"
            "  - Significant life events or experiences shared by the user\n"
            "  - Important context about the user's current situation, challenges or goals\n"
            "  - What the user likes or dislikes, their opinions, beliefs, values, etc.\n"
            "  - Any other details that provide valuable insights into the user's personality, perspective or needs",
            "Your task is to decide whether the user input contains any of the above information worth remembering.",
            "If the user input contains any information worth remembering for future conversations, respond with 'yes'.",
            "If the input does not contain any important details worth saving, respond with 'no' to disregard it.",
            "You will also be provided with a list of existing memories to help you decide if the input is new or already known.",
            "If the memory already exists that matches the input, respond with 'no' to keep it as is.",
            "If a memory exists that needs to be updated or deleted, respond with 'yes' to update/delete it.",
            "You must only respond with 'yes' or 'no'. Nothing else will be considered as a valid response.",
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
            log_error("No model provided for classifier")
            return
        
        log_debug("MemoryClassifier Start", center=True)

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [
            self.get_system_message(messages, existing_memories),
            # For models that require a non-system message
            Message(role="user", content="Are any of these messages worth remembering? You must only respond with 'yes' or 'no'."),
        ]

        # Generate a response from the Model
        response = self.model.response(messages=messages_for_model)
        log_debug("MemoryClassifier End", center=True)
        return response.content

    async def arun(
        self,
        messages: Optional[List[Message]] = None,
        existing_memories: Optional[List[Memory]] = None,
    ) -> Optional[str]:
        if self.model is None:
            log_error("No model provided for classifier")
            return
        
        log_debug("MemoryClassifier Start", center=True)

        # Update the Model (set defaults, add logit etc.)
        self.update_model()

        # Prepare the List of messages to send to the Model
        messages_for_model: List[Message] = [
            self.get_system_message(messages, existing_memories),
            # For models that require a non-system message
            Message(role="user", content="Are any of these messages worth remembering? You must only respond with 'yes' or 'no'."),
        ]

        # Generate a response from the Model
        response = await self.model.aresponse(messages=messages_for_model)
        log_debug("MemoryClassifier End", center=True)
        return response.content
