import datetime
from typing import List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.brand_investigator import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message, ToolChoice
from app.tool import (
    Bash,
    BrowserUseTool,
    DeepResearch,
    PythonExecute,
    StrReplaceEditor,
    Terminate,
    ToolCollection,
    UserControlTool,
    WebSearch,
)


class BrandInvestigator(ToolCallAgent):
    """An agent specialized in researching and analyzing brands."""

    name: str = "BrandInvestigator"
    description: str = (
        "An agent that researches and analyzes brands based on user criteria"
    )

    system_prompt: str = SYSTEM_PROMPT.format(
        directory=config.workspace_root,
        current_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A"),
    )
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 50

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            Bash(),
            BrowserUseTool(),
            DeepResearch(),
            PythonExecute(),
            StrReplaceEditor(),
            Terminate(),
            UserControlTool(),
            WebSearch(),
        )
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrandInvestigator":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        # Store original prompt for restoration later
        original_prompt = self.next_step_prompt

        # Check if browser is in use and update prompt with browser context
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use and self.browser_context_helper:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )
            logger.debug(f"Using browser context prompt: {self.next_step_prompt}")

        # Call the parent class's think method
        result = await super().think()

        # Restore original prompt template for next iteration
        self.next_step_prompt = original_prompt

        return result

    async def cleanup(self):
        """Clean up resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
