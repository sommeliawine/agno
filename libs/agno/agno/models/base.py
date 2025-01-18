import collections.abc
from dataclasses import dataclass, field
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from agno.media import Audio, Image
from agno.models.message import Message
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.tools import Toolkit
from agno.tools.function import Function, FunctionCall, ToolCallException
from agno.utils.log import logger
from agno.utils.timer import Timer


@dataclass
class BaseMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[dict] = None
    completion_tokens_details: Optional[dict] = None

    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def _log(self, metric_lines: list[str]):
        logger.debug("**************** METRICS START ****************")
        for line in metric_lines:
            logger.debug(line)
        logger.debug("**************** METRICS END ******************")

    def log(self):
        metric_lines = []
        if self.time_to_first_token is not None:
            metric_lines.append(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        metric_lines.extend(
            [
                f"* Time to generate response:   {self.response_timer.elapsed:.4f}s",
                f"* Tokens per second:           {self.output_tokens / self.response_timer.elapsed:.4f} tokens/s",
                f"* Input tokens:                {self.input_tokens or self.prompt_tokens}",
                f"* Output tokens:               {self.output_tokens or self.completion_tokens}",
                f"* Total tokens:                {self.total_tokens}",
            ]
        )
        if self.prompt_tokens_details is not None:
            metric_lines.append(f"* Prompt tokens details:       {self.prompt_tokens_details}")
        if self.completion_tokens_details is not None:
            metric_lines.append(f"* Completion tokens details:   {self.completion_tokens_details}")
        self._log(metric_lines=metric_lines)

@dataclass
class Model:
    # ID of the model to use.
    id: str
    # Name for this Model. This is not sent to the Model API.
    name: Optional[str] = None
    # Provider for this Model. This is not sent to the Model API.
    provider: Optional[str] = None
    # Metrics collected for this Model. This is not sent to the Model API.
    metrics: Dict[str, Any] = field(default_factory=dict)
    response_format: Optional[Any] = None

    # A list of tools provided to the Model.
    # Tools are functions the model may generate JSON inputs for.
    # If you provide a dict, it is not called by the model.
    # Always add tools using the add_tool() method.
    tools: Optional[List[Dict]] = None
    # Controls which (if any) function is called by the model.
    # "none" means the model will not call a function and instead generates a message.
    # "auto" means the model can pick between generating a message or calling a function.
    # Specifying a particular function via {"type: "function", "function": {"name": "my_function"}}
    #   forces the model to call that function.
    # "none" is the default when no functions are present. "auto" is the default if functions are present.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # If True, runs the tool before sending back the response content.
    run_tools: bool = True
    # If True, shows function calls in the response.
    show_tool_calls: Optional[bool] = None
    # Maximum number of tool calls allowed.
    tool_call_limit: Optional[int] = None

    # -*- Functions available to the Model to call -*-
    # Functions extracted from the tools.
    # Note: These are not sent to the Model API and are only used for execution + deduplication.
    functions: Optional[Dict[str, Function]] = None
    # Function call stack.
    function_call_stack: Optional[List[FunctionCall]] = None

    # System prompt from the model added to the Agent.
    system_prompt: Optional[str] = None
    # Instructions from the model added to the Agent.
    instructions: Optional[List[str]] = None

    # Session ID of the calling Agent or Workflow.
    session_id: Optional[str] = None
    # Whether to use the structured outputs with this Model.
    structured_outputs: Optional[bool] = None
    # Whether the Model supports structured outputs.
    supports_structured_outputs: bool = False

    def __post_init__(self):
        if self.provider is None and self.name is not None:
            self.provider = f"{self.name} ({self.id})"

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        fields = {"name", "id", "provider", "metrics"}
        _dict = {field: getattr(self, field) for field in fields if getattr(self, field) is not None}
        # Add functions if they exist
        if self.functions:
            _dict["functions"] = {k: v.to_dict() for k, v in self.functions.items()}
            _dict["tool_call_limit"] = self.tool_call_limit
        return _dict

    def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs) -> Iterator[Any]:
        raise NotImplementedError

    async def ainvoke_stream(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def response(self, messages: List[Message]) -> ModelResponse:
        raise NotImplementedError

    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        raise NotImplementedError

    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        raise NotImplementedError

    async def aresponse_stream(self, messages: List[Message]) -> Any:
        raise NotImplementedError

    def _log_messages(self, messages: List[Message]) -> None:
        """
        Log messages for debugging.
        """
        for m in messages:
            m.log()

    def get_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        if self.tools is None:
            return None

        tools_for_api = []
        for tool in self.tools:
            tools_for_api.append(tool)
        return tools_for_api

    def add_tool(
        self, tool: Union[Toolkit, Callable, Dict, Function], strict: bool = False, agent: Optional[Any] = None
    ) -> None:
        if self.tools is None:
            self.tools = []

        # If the tool is a Dict, add it directly to the Model
        if isinstance(tool, Dict):
            if tool not in self.tools:
                self.tools.append(tool)
                logger.debug(f"Added tool {tool} to model.")

        # If the tool is a Callable or Toolkit, process and add to the Model
        elif callable(tool) or isinstance(tool, Toolkit) or isinstance(tool, Function):
            if self.functions is None:
                self.functions = {}

            if isinstance(tool, Toolkit):
                # For each function in the toolkit, process entrypoint and add to self.tools
                for name, func in tool.functions.items():
                    # If the function does not exist in self.functions, add to self.tools
                    if name not in self.functions:
                        func._agent = agent
                        func.process_entrypoint(strict=strict)
                        if strict and self.supports_structured_outputs:
                            func.strict = True
                        self.functions[name] = func
                        self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {name} from {tool.name} added to model.")

            elif isinstance(tool, Function):
                if tool.name not in self.functions:
                    tool._agent = agent
                    tool.process_entrypoint(strict=strict)
                    if strict and self.supports_structured_outputs:
                        tool.strict = True
                    self.functions[tool.name] = tool
                    self.tools.append({"type": "function", "function": tool.to_dict()})
                    logger.debug(f"Function {tool.name} added to model.")

            elif callable(tool):
                try:
                    function_name = tool.__name__
                    if function_name not in self.functions:
                        func = Function.from_callable(tool, strict=strict)
                        func._agent = agent
                        if strict and self.supports_structured_outputs:
                            func.strict = True
                        self.functions[func.name] = func
                        self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {func.name} added to model.")
                except Exception as e:
                    logger.warning(f"Could not add function {tool}: {e}")

    def deactivate_function_calls(self) -> None:
        # Deactivate tool calls by setting future tool calls to "none"
        # This is triggered when the function call limit is reached.
        self.tool_choice = "none"

    def run_function_calls(
        self, function_calls: List[FunctionCall], function_call_results: List[Message], tool_role: str = "tool"
    ) -> Iterator[ModelResponse]:
        for function_call in function_calls:
            if self.function_call_stack is None:
                self.function_call_stack = []

            # -*- Start function call
            function_call_timer = Timer()
            function_call_timer.start()
            yield ModelResponse(
                content=function_call.get_call_str(),
                tool_call={
                    "role": tool_role,
                    "tool_call_id": function_call.call_id,
                    "tool_name": function_call.function.name,
                    "tool_args": function_call.arguments,
                },
                event=ModelResponseEvent.tool_call_started.value,
            )

            # Track if the function call was successful
            function_call_success = False
            # If True, stop execution after this function call
            stop_execution_after_tool_call = False
            # Additional messages from the function call that will be added to the function call results
            additional_messages_from_function_call = []

            # -*- Run function call
            try:
                function_call_success = function_call.execute()
            except ToolCallException as tce:
                if tce.user_message is not None:
                    if isinstance(tce.user_message, str):
                        additional_messages_from_function_call.append(Message(role="user", content=tce.user_message))
                    else:
                        additional_messages_from_function_call.append(tce.user_message)
                if tce.agent_message is not None:
                    if isinstance(tce.agent_message, str):
                        additional_messages_from_function_call.append(
                            Message(role="assistant", content=tce.agent_message)
                        )
                    else:
                        additional_messages_from_function_call.append(tce.agent_message)
                if tce.messages is not None and len(tce.messages) > 0:
                    for m in tce.messages:
                        if isinstance(m, Message):
                            additional_messages_from_function_call.append(m)
                        elif isinstance(m, dict):
                            try:
                                additional_messages_from_function_call.append(Message(**m))
                            except Exception as e:
                                logger.warning(f"Failed to convert dict to Message: {e}")
                if tce.stop_execution:
                    stop_execution_after_tool_call = True
                    if len(additional_messages_from_function_call) > 0:
                        for m in additional_messages_from_function_call:
                            m.stop_after_tool_call = True

            function_call_output: Optional[Union[List[Any], str]] = ""
            if isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):
                for item in function_call.result:
                    function_call_output += item
                    if function_call.function.show_result:
                        yield ModelResponse(content=item)
            else:
                function_call_output = function_call.result
                if function_call.function.show_result:
                    yield ModelResponse(content=function_call_output)

            # -*- Stop function call timer
            function_call_timer.stop()

            # -*- Create function call result message
            function_call_result = Message(
                role=tool_role,
                content=function_call_output if function_call_success else function_call.error,
                tool_call_id=function_call.call_id,
                tool_name=function_call.function.name,
                tool_args=function_call.arguments,
                tool_call_error=not function_call_success,
                stop_after_tool_call=function_call.function.stop_after_tool_call or stop_execution_after_tool_call,
                metrics={"time": function_call_timer.elapsed},
            )

            # -*- Yield function call result
            yield ModelResponse(
                content=f"{function_call.get_call_str()} completed in {function_call_timer.elapsed:.4f}s.",
                tool_call=function_call_result.model_dump(
                    include={
                        "content",
                        "tool_call_id",
                        "tool_name",
                        "tool_args",
                        "tool_call_error",
                        "metrics",
                        "created_at",
                    }
                ),
                event=ModelResponseEvent.tool_call_completed.value,
            )

            # Add metrics to the model
            if "tool_call_times" not in self.metrics:
                self.metrics["tool_call_times"] = {}
            if function_call.function.name not in self.metrics["tool_call_times"]:
                self.metrics["tool_call_times"][function_call.function.name] = []
            self.metrics["tool_call_times"][function_call.function.name].append(function_call_timer.elapsed)

            # Add the function call result to the function call results
            function_call_results.append(function_call_result)
            if len(additional_messages_from_function_call) > 0:
                function_call_results.extend(additional_messages_from_function_call)
            self.function_call_stack.append(function_call)

            # -*- Check function call limit
            if self.tool_call_limit and len(self.function_call_stack) >= self.tool_call_limit:
                self.deactivate_function_calls()
                break  # Exit early if we reach the function call limit

    def handle_post_tool_call_messages(self, messages: List[Message], model_response: ModelResponse) -> ModelResponse:
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                last_message.role == "assistant"
                and last_message.content is not None
                and isinstance(last_message.content, str)
            ):
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += last_message.content
        else:
            response_after_tool_calls = self.response(messages=messages)
            if response_after_tool_calls.content is not None:
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += response_after_tool_calls.content
            if response_after_tool_calls.parsed is not None:
                # bubble up the parsed object, so that the final response has the parsed object
                # that is visible to the agent
                model_response.parsed = response_after_tool_calls.parsed
            if response_after_tool_calls.audio is not None:
                # bubble up the audio, so that the final response has the audio
                # that is visible to the agent
                model_response.audio = response_after_tool_calls.audio
        return model_response

    async def ahandle_post_tool_call_messages(
        self, messages: List[Message], model_response: ModelResponse
    ) -> ModelResponse:
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                last_message.role == "assistant"
                and last_message.content is not None
                and isinstance(last_message.content, str)
            ):
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += last_message.content
        else:
            response_after_tool_calls = await self.aresponse(messages=messages)
            if response_after_tool_calls.content is not None:
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += response_after_tool_calls.content
            if response_after_tool_calls.parsed is not None:
                # bubble up the parsed object, so that the final response has the parsed object
                # that is visible to the agent
                model_response.parsed = response_after_tool_calls.parsed
            if response_after_tool_calls.audio is not None:
                # bubble up the audio, so that the final response has the audio
                # that is visible to the agent
                model_response.audio = response_after_tool_calls.audio
        return model_response

    def handle_post_tool_call_messages_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                last_message.role == "assistant"
                and last_message.content is not None
                and isinstance(last_message.content, str)
            ):
                yield ModelResponse(content=last_message.content)
        else:
            yield from self.response_stream(messages=messages)

    async def ahandle_post_tool_call_messages_stream(self, messages: List[Message]) -> Any:
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                last_message.role == "assistant"
                and last_message.content is not None
                and isinstance(last_message.content, str)
            ):
                yield ModelResponse(content=last_message.content)
        else:
            async for model_response in self.aresponse_stream(messages=messages):  # type: ignore
                yield model_response

    def _process_image_url(self, image_url: str) -> Dict[str, Any]:
        """Process image (base64 or URL)."""

        if image_url.startswith("data:image") or image_url.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image_url}}
        else:
            raise ValueError("Image URL must start with 'data:image' or 'http(s)://'.")

    def _process_image_path(self, image_path: Union[Path, str]) -> Dict[str, Any]:
        """Process image ( file path)."""
        # Process local file image
        import base64
        import mimetypes

        path = image_path if isinstance(image_path, Path) else Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{base64_image}"
            return {"type": "image_url", "image_url": {"url": image_url}}

    def _process_bytes_image(self, image: bytes) -> Dict[str, Any]:
        """Process bytes image data."""
        import base64

        base64_image = base64.b64encode(image).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    def process_image(self, image: Image) -> Optional[Dict[str, Any]]:
        """Process an image based on the format."""

        if image.url is not None:
            image_payload = self._process_image_url(image.url)

        elif image.filepath is not None:
            image_payload = self._process_image_path(image.filepath)

        elif image.content is not None:
            image_payload = self._process_bytes_image(image.content)

        else:
            logger.warning(f"Unsupported image type: {type(image)}")
            return None

        if image.detail:
            image_payload["image_url"]["detail"] = image.detail

        return image_payload

    def add_images_to_message(self, message: Message, images: Sequence[Image]) -> Message:
        """
        Add images to a message for the model. By default, we use the OpenAI image format but other Models
        can override this method to use a different image format.

        Args:
            message: The message for the Model
            images: Sequence of images in various formats:
                - str: base64 encoded image, URL, or file path
                - Dict: pre-formatted image data
                - bytes: raw image data

        Returns:
            Message content with images added in the format expected by the model
        """
        # If no images are provided, return the message as is
        if len(images) == 0:
            return message

        # Ignore non-string message content
        # because we assume that the images/audio are already added to the message
        if not isinstance(message.content, str):
            return message

        # Create a default message content with text
        message_content_with_image: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]

        # Add images to the message content
        for image in images:
            try:
                image_data = self.process_image(image)
                if image_data:
                    message_content_with_image.append(image_data)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                continue

        # Update the message content with the images
        message.content = message_content_with_image
        return message

    def add_audio_to_message(self, message: Message, audio: Sequence[Audio]) -> Message:
        """
        Add audio to a message for the model. By default, we use the OpenAI audio format but other Models
        can override this method to use a different audio format.
        Args:
            message: The message for the Model
            audio: Pre-formatted audio data like {
                        "data": encoded_string,
                        "format": "wav"
                    }

        Returns:
            Message content with audio added in the format expected by the model
        """
        if len(audio) == 0:
            return message

        # Create a default message content with text
        message_content_with_audio: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]

        for audio_snippet in audio:
            # This means the audio is raw data
            if audio_snippet.content:
                import base64

                encoded_string = base64.b64encode(audio_snippet.content).decode("utf-8")

                # Create a message with audio
                message_content_with_audio.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": audio_snippet.format,
                        },
                    },
                )

        # Update the message content with the audio
        message.content = message_content_with_audio
        message.audio = None  # The message should not have an audio component after this

        return message

    def get_system_message_for_model(self) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.instructions

    def clear(self) -> None:
        """Clears the Model's state."""

        self.metrics = {}
        self.functions = None
        self.function_call_stack = None
        self.session_id = None

    def __deepcopy__(self, memo):
        """Create a deep copy of the Model instance.

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass.

        Returns:
            Model: A new Model instance with deeply copied attributes.
        """
        from copy import deepcopy

        # Create a new instance without calling __init__
        cls = self.__class__
        new_model = cls.__new__(cls)
        memo[id(self)] = new_model

        # Deep copy all attributes
        for k, v in self.__dict__.items():
            if k in {"metrics", "functions", "function_call_stack", "session_id"}:
                continue
            setattr(new_model, k, deepcopy(v, memo))

        # Clear the new model to remove any references to the old model
        new_model.clear()
        return new_model
