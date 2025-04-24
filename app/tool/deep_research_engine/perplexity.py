#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PerplexityDeepResearchEngine

The perplexity implementation of deep research engine
"""

import asyncio
import json
import re
import time
from typing import List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.exceptions import ToolError
from app.llm import LLM
from app.logger import logger
from app.schema import Message, ToolChoice
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import SearchResult, WebSearch

from app.tool.deep_research_engine.base import (
    DeepResearchError,
    DeepResearchItem,
    DeepResearchEngine
)


SYSTEM_PROMPT = """
You are a research assistant to help the user to do a deep research on a specific topic.
Please always try all your efforts to be precise and consise,
cross check and verify the result carefully before you show them to the users.
"""

DEEP_RESEARCH_CONFIG_NAME = "deep_research"


class PerplexityDeepResearchEngine(DeepResearchEngine):
    """PerplexityDeepResearchEngine

    Perplexity deep research engine implementation
    """

    llm: LLM = Field(default_factory=LLM)

    def research(self, query: str, *args, **kwargs) -> DeepResearchItem:
        """
        Perform a deep research and return a response of content with citations

        Args:
            query (str): The query to execute deep research
            args: Additional arguments
            kwargs: Additional keyword arguments
        Returns:
            DeepResearchItem: The deep research result item
        Throws:
            DeepResearchError
        """
        # Normalize parameters
        timeout = kwargs.get("timeout", 300)

        try:
            response = asyncio.run(self.llm.ask_tool(
                messages=[Message.user_message(content=query)],
                system_msgs=[Message.system_message(content=SYSTEM_PROMPT)],
                tools=None,
                tool_choice=ToolChoice.NONE,
                stream=False,
                timeout=timeout,
                name=DEEP_RESEARCH_CONFIG_NAME,
            ))

            item = DeepResearchItem(content=response.content, citations=[])
            if getattr(response, "annotations", []):
                for annotation in response.annotations:
                    if getattr(annotation, "type", "url_citation") == "url_citation" and \
                            getattr(annotation, "url_citation", None):
                        item.add_citation(
                            title=annotation.url_citation.title,
                            url=annotation.url_citation.url
                        )
            if getattr(response, "reasoning", ''):
                item.reasoning = response.reasoning
            return item
        except Exception as err:
            logger.error(f"Exception caught {str(err)}")
            raise DeepResearchError(str(err))


if __name__ == "__main__":
    deep_research = PerplexityDeepResearchEngine()
    # result = deep_research.research(
    #     query="What is deep learning", timeout=600
    # )
    result = deep_research.research(
        query="Analyze the Instagram profile @Sarah.FitLife_ (Sarah Dussault). Focus on content themes, style, tone, audience demographics, engagement patterns, and any previous supplement/nutrition brand collaborations."
    )
    print(result)
