import datetime
import hashlib
import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

REASONING_MODELS = [
    "o1",
    "o3-mini",
    "openai/o3",
    "openai/o4-mini-high",
    "anthropic/claude-3.7-sonnet:beta",
    "anthropic/claude-3.7-sonnet:thinking",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-pro-exp-03-25:free",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-flash-preview:thinking",
]

MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "openai/o3",
    "openai/o4-mini-high",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "anthropic/claude-3.7-sonnet:beta",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.7-sonnet",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-pro-exp-03-25:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash-preview:thinking",
    "openrouter/optimus-alpha",
]

# FIXME(Loic): This is a hack to make the images output folder unique for each query
#              (only when program will quit each time)
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # Default values when dimensions aren't available or detail level is unknown
        if detail == "high":
            # Default to a 1024x1024 image calculation for high detail
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # Default to a medium-sized image for medium detail
            return 1024  # This matches the original default
        else:
            # For unknown detail levels, use medium as default
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "clients"):  # Only initialize if not already initialized
            # Initialize clients dictionary
            self.clients = {}

            llm_configs = llm_config or config.llm

            # Ensure default config exists
            if "default" not in llm_configs:
                raise ValueError("Configuration must contain a 'default' entry")

            # Initialize clients_info dictionary to store all configuration related data
            self.clients_info = {}

            # Initialize clients and token counters for all configurations
            for name, cfg in llm_configs.items():
                # Initialize client info dictionary for this config
                self.clients_info[name] = {
                    "total_input_tokens": 0,
                    "total_completion_tokens": 0,
                    "max_input_tokens": (
                        cfg.max_input_tokens
                        if hasattr(cfg, "max_input_tokens")
                        else None
                    ),
                }

                # Initialize tokenizer for this config
                try:
                    tokenizer = tiktoken.encoding_for_model(cfg.model)
                except KeyError:
                    # If the model is not in tiktoken's presets, use cl100k_base as default
                    tokenizer = tiktoken.get_encoding("cl100k_base")

                self.clients_info[name]["tokenizer"] = tokenizer
                self.clients_info[name]["token_counter"] = TokenCounter(tokenizer)

                # Create client for this config
                if self._create_client(name, cfg):
                    logger.debug(f"Initialized LLM client for config: {name}")

            # Set default token counter for backward compatibility
            self.token_counter = self.clients_info["default"]["token_counter"]

    def _create_client(self, name: str, llm_config: LLMSettings) -> bool:
        """Create a client for the given configuration

        Args:
            name: Configuration name
            llm_config: LLM configuration

        Returns:
            bool: True if client was created successfully, False otherwise
        """
        try:
            if llm_config.api_type == "azure":
                self.clients[name] = AsyncAzureOpenAI(
                    base_url=llm_config.base_url,
                    api_key=llm_config.api_key,
                    api_version=llm_config.api_version,
                )
            elif llm_config.api_type == "aws":
                self.clients[name] = BedrockClient()
            else:
                self.clients[name] = AsyncOpenAI(
                    api_key=llm_config.api_key, base_url=llm_config.base_url
                )
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize LLM client for config {name}: {e}")
            return False

    def count_tokens(self, text: str, config_name: str = "default") -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.clients_info[config_name]["tokenizer"].encode(text))

    def count_message_tokens(
        self, messages: List[dict], config_name: str = "default"
    ) -> int:
        return self.clients_info[config_name]["token_counter"].count_message_tokens(
            messages
        )

    def update_token_count(
        self,
        input_tokens: int,
        completion_tokens: int = 0,
        config_name: str = "default",
    ) -> None:
        """Update token counts for the specified configuration"""
        # Only track tokens if max_input_tokens is set for this config
        client_info = self.clients_info[config_name]
        client_info["total_input_tokens"] += input_tokens
        client_info["total_completion_tokens"] += completion_tokens
        logger.info(
            f"Token usage for {config_name}: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={client_info['total_input_tokens']}, "
            f"Cumulative Completion={client_info['total_completion_tokens']}, "
            f"Total={input_tokens + completion_tokens}, "
            f"Cumulative Total={client_info['total_input_tokens'] + client_info['total_completion_tokens']}"
        )

    def check_token_limit(
        self, input_tokens: int, config_name: str = "default"
    ) -> bool:
        """Check if token limits are exceeded for the specified configuration"""
        client_info = self.clients_info[config_name]
        max_tokens = client_info["max_input_tokens"]
        if max_tokens is not None:
            return client_info["total_input_tokens"] + input_tokens <= max_tokens
        # If max_input_tokens is not set for this config, always return True
        return True

    def get_limit_error_message(
        self, input_tokens: int, config_name: str = "default"
    ) -> str:
        """Generate error message for token limit exceeded for the specified configuration"""
        client_info = self.clients_info[config_name]
        max_tokens = client_info["max_input_tokens"]
        if (
            max_tokens is not None
            and (client_info["total_input_tokens"] + input_tokens) > max_tokens
        ):
            return f"Request may exceed input token limit for {config_name} (Current: {client_info['total_input_tokens']}, Needed: {input_tokens}, Max: {max_tokens})"

        return "Token limit exceeded"

    async def _handle_rate_limit_error(self, response):
        """Handle rate limit errors with appropriate backoff strategy

        Args:
            response: The response object from OpenAI API

        This method implements backoff strategies based on the error response:
        1. If retryDelay is present in the response, sleep for that duration
        2. If X-RateLimit-Reset is present in headers, sleep until that timestamp
        3. Otherwise, just let the tenacity retry mechanism handle it

        Returns:
            bool: True if rate limit was detected and handled, False otherwise
        """
        try:
            import asyncio
            import json
            from datetime import datetime

            # Also check if this is a direct ChatCompletion with error
            if hasattr(response, "error"):
                error_obj = getattr(response, "error", {})
                if isinstance(error_obj, dict) and error_obj.get("code", 0) == 429:
                    metadata = error_obj.get("metadata", {})
                    provider_name = metadata.get("provider_name", "")
                    logger.info(f"Rate limit detected from provider: {provider_name}")
                    logger.debug(f"Rate limit error details: {error_obj}")

                    # Case 1: Check for retryDelay in raw response (Google AI Studio format)
                    if "raw" in metadata:
                        try:
                            raw_data = json.loads(metadata["raw"])
                            retry_info = None

                            # Navigate through Google AI Studio error structure
                            if "error" in raw_data and "details" in raw_data["error"]:
                                for detail in raw_data["error"]["details"]:
                                    if (
                                        detail.get("@type", "")
                                        == "type.googleapis.com/google.rpc.RetryInfo"
                                    ):
                                        retry_info = detail
                                        break

                            if retry_info and "retryDelay" in retry_info:
                                delay_str = retry_info["retryDelay"]
                                # Convert "50s" to seconds
                                delay_seconds = int(delay_str.rstrip("s"))
                                logger.info(
                                    f"Rate limit hit. Backing off for {delay_seconds} seconds as specified by provider."
                                )
                                await asyncio.sleep(delay_seconds)
                                return True
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(
                                f"Failed to parse retry delay from response: {e}"
                            )

                    # Case 2: Check for X-RateLimit-Reset in headers
                    if (
                        "headers" in metadata
                        and "X-RateLimit-Reset" in metadata["headers"]
                    ):
                        reset_timestamp = (
                            int(metadata["headers"]["X-RateLimit-Reset"]) / 1000
                        )  # Convert from milliseconds
                        current_time = datetime.now().timestamp()
                        wait_seconds = max(0, reset_timestamp - current_time)

                        if wait_seconds > 0:
                            logger.info(
                                f"Rate limit hit. Waiting until reset time: {datetime.fromtimestamp(reset_timestamp).isoformat()} ({wait_seconds:.2f}s)"
                            )
                            await asyncio.sleep(wait_seconds)
                            return True
                        else:
                            logger.info(
                                f"Rate limit reset time already passed. Proceeding with retry."
                            )
                    return True

            return False
        except Exception as e:
            logger.warning(f"Error in rate limit backoff handling: {e}")
            # Continue with normal retry mechanism
            return False

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        import base64
        import os
        import uuid
        from pathlib import Path

        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Generate a unique ID for the image, use MD5 for dedup
                    image_id = hashlib.md5(
                        str(message["base64_image"]).encode("utf-8")
                    ).hexdigest()
                    logger.debug(f"Processing image with ID: {image_id}")

                    # Save the image to local file for debugging
                    try:
                        # Create images directory if it doesn't exist
                        images_dir = Path(
                            os.path.join(os.getcwd(), "images", CURRENT_TIME)
                        )
                        images_dir.mkdir(exist_ok=True, parents=True)

                        # Decode the image and save it to file
                        image_data = base64.b64decode(message["base64_image"])
                        image_path = images_dir / f"{image_id}.jpg"
                        if not os.path.exists(image_path):
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                            logger.debug(f"Saved image to {image_path}")
                    except Exception as e:
                        logger.exception(f"Failed to save debug image: {e}")

                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]
                    logger.debug("Base64 image ignored as model doesn't support it")

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        name: str = "default",
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Get config for the specified name or fall back to default
            llm_configs = config.llm
            cfg = llm_configs.get(name, llm_configs["default"])
            model = cfg.model
            max_tokens = cfg.max_tokens
            temp = temperature if temperature is not None else cfg.temperature

            # Log request parameters
            logger.debug(
                f"LLM.ask called with: model={model}, stream={stream}, temperature={temp}"
            )

            # Check if the model supports images
            supports_images = model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages, name)
            logger.debug(f"Input token count: {input_tokens}")

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens, name):
                error_message = self.get_limit_error_message(input_tokens, name)
                logger.debug(f"Token limit exceeded: {error_message}")
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": model,
                "messages": messages,
            }

            if model in REASONING_MODELS:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temp

            logger.debug(f"LLM request parameters: {params}")

            # Get the client for the specified configuration
            client = self.clients.get(name)
            if not client:
                logger.warning(f"Client for config '{name}' not found, using default")
                client = self.clients.get("default") or next(
                    iter(self.clients.values())
                )

            if not stream:
                # Non-streaming request
                response = await client.chat.completions.create(**params, stream=False)

                if not response.choices or not response.choices[0].message.content:
                    # Check if response contains rate limit error despite normal format
                    rate_limited = await self._handle_rate_limit_error(response)
                    if rate_limited:
                        raise RateLimitError(
                            "Rate limit hit during non-streaming request"
                        )
                    raise ValueError("Empty or invalid response from LLM")

                # Log response
                logger.debug(f"LLM response: {response}")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens, name
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens, config_name=name)

            # Get the client for the specified configuration
            client = self.clients.get(name)
            if not client:
                logger.warning(f"Client for config '{name}' not found, using default")
                client = self.clients.get("default") or next(
                    iter(self.clients.values())
                )

            response = await client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            try:
                async for chunk in response:
                    # Check if chunk contains rate limit error
                    if not chunk.choices or not chunk.choices[0].delta.content:
                        rate_limited = await self._handle_rate_limit_error(chunk)
                        if rate_limited:
                            raise RateLimitError(
                                "Rate limit hit during streaming request"
                            )

                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)
            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                raise

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # Log full response
            logger.debug(f"Full streaming response: {full_response}")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.exception("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.exception("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.exception(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        name: str = "default",
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Get config for the specified name or fall back to default
            llm_configs = config.llm
            cfg = llm_configs.get(name, llm_configs["default"])
            model = cfg.model
            max_tokens = cfg.max_tokens
            temp = temperature if temperature is not None else cfg.temperature

            # Log request parameters
            logger.debug(
                f"LLM.ask_with_images called with: model={model}, stream={stream}, "
                f"temperature={temp}, "
                f"images_count={len(images)}"
            )

            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content if isinstance(content, list) else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content
            logger.debug(
                f"Updated last message with {len(multimodal_content)} multimodal content items"
            )

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages
            logger.debug(
                f"Final message count for LLM request with images: {len(all_messages)}"
            )

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages, name)
            logger.debug(f"Input token count for request with images: {input_tokens}")
            if not self.check_token_limit(input_tokens, name):
                error_message = self.get_limit_error_message(input_tokens, name)
                logger.debug(f"Token limit exceeded: {error_message}")
                raise TokenLimitExceeded(error_message)

            # Set up API parameters
            params = {
                "model": model,
                "messages": all_messages,
                "stream": stream,
            }
            logger.debug(f"LLM request parameters for images: {params}")

            # Add model-specific parameters
            if model in REASONING_MODELS:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temp

            # Handle non-streaming request
            if not stream:
                # Get the client for the specified configuration
                client = self.clients.get(name)
                if not client:
                    logger.warning(
                        f"Client for config '{name}' not found, using default"
                    )
                    client = self.clients.get("default") or next(
                        iter(self.clients.values())
                    )

                response = await client.chat.completions.create(**params)
                logger.debug(f"LLM response for images (non-streaming): {response}")

                if not response.choices or not response.choices[0].message.content:
                    # Check if response contains rate limit error despite normal format
                    rate_limited = await self._handle_rate_limit_error(response)
                    if rate_limited:
                        raise RateLimitError(
                            "Rate limit hit during non-streaming request"
                        )
                    logger.exception("Empty or invalid response from LLM for image request")
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens, name
                )
                logger.debug(
                    f"Token usage for image request: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} completion"
                )
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens, config_name=name)
            logger.debug(
                f"Starting streaming request with images, estimated input tokens: {input_tokens}"
            )
            # Get the client for the specified configuration
            client = self.clients.get(name)
            if not client:
                logger.warning(f"Client for config '{name}' not found, using default")
                client = self.clients.get("default") or next(
                    iter(self.clients.values())
                )

            response = await client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                if not chunk.choices or not chunk.choices[0].delta.content:
                    # Check if chunk contains rate limit error
                    rate_limited = await self._handle_rate_limit_error(chunk)
                    if rate_limited:
                        raise RateLimitError("Rate limit hit during streaming request")

                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            # Log full response
            logger.debug(f"Full streaming response with images: {full_response}")

            if not full_response:
                logger.error("Empty response from streaming LLM with images")
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(full_response, name)
            logger.info(
                f"Estimated completion tokens for streaming image response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.exception(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.exception("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.exception("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.exception(f"API error: {oe}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        name: str = "default",
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Get config for the specified name or fall back to default
            llm_configs = config.llm
            cfg = llm_configs.get(name, llm_configs["default"])
            model = cfg.model
            max_tokens = cfg.max_tokens
            temp = temperature if temperature is not None else cfg.temperature

            # Log request parameters
            logger.debug(
                f"LLM.ask_tool called with: model={model}, timeout={timeout}, "
                f"tool_choice={tool_choice}, temperature={temp}, "
                f"tools_count={len(tools) if tools else 0}"
            )
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                logger.error(f"Invalid tool_choice: {tool_choice}")
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = model in MULTIMODAL_MODELS
            logger.debug(f"Model {model} supports images: {supports_images}")

            # Format messages
            if system_msgs:
                logger.debug(f"System messages provided: {system_msgs}")
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)
            logger.debug(f"Input token count for messages: {input_tokens}")

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))
                logger.debug(f"Additional tokens for tools: {tools_tokens}")

            input_tokens += tools_tokens
            logger.debug(f"Total input token count for tool request: {input_tokens}")

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                logger.debug(f"Token limit exceeded: {error_message}")
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        logger.error(f"Invalid tool format: {tool}")
                        raise ValueError("Each tool must be a dict with 'type' field")
                logger.debug(f"Validated {len(tools)} tools for request")

            # Set up the completion request
            params = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }
            logger.debug(f"LLM tool request parameters: {params}")

            if model in REASONING_MODELS:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temp

            params["stream"] = False  # Always use non-streaming for tool requests
            logger.debug("Sending tool request to LLM API")
            # Get the client for the specified configuration
            client = self.clients.get(name)
            if not client:
                logger.warning(f"Client for config '{name}' not found, using default")
                client = self.clients.get("default") or next(
                    iter(self.clients.values())
                )

            response: ChatCompletion = await client.chat.completions.create(**params)
            logger.debug(f"LLM tool response received: {response}")

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                # Check if response contains rate limit error despite normal format
                rate_limited = await self._handle_rate_limit_error(response)
                if rate_limited:
                    raise RateLimitError("Rate limit hit during tool request")

                logger.error(f"Invalid or empty response from LLM: {response}")
                # print(response)
                raise ValueError("Invalid or empty response from LLM")
                # return None

            if (
                not response.choices[0].message.content
                and not response.choices[0].message.tool_calls
            ):
                logger.error(f"No content nor tool_calls found in response")
                raise ValueError("No content nor tool_calls found in response")

            # Update token counts
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
            logger.debug(
                f"Token usage for tool request: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} completion"
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.exception(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.exception("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.exception("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.exception(f"API error: {oe}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in ask_tool: {e}")
            raise
