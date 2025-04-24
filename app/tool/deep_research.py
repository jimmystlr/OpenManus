import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from app.config import config
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.deep_research_engine import (
    BuiltinDeepResearchEngine,
    DeepResearchEngine,
    DeepResearchError,
    DeepResearchItem,
    PerplexityDeepResearchEngine,
)


class DeepResearchResult(ToolResult):
    """Structured response from the deep research tool, inheriting ToolResult."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="The research query that was executed")
    content: str = Field(default="", description="The research content result")
    reasoning: str = Field(default="", description="The reasoning procedure")
    citations: List[Dict[str, str]] = Field(
        default_factory=list, description="Citations for the research content"
    )

    @model_validator(mode="after")
    def populate_output(self) -> "DeepResearchResult":
        """Populate the output field after validation."""
        if self.error:
            return self

        result_text = [f"DeepResearch results for '{self.query}':"]
        result_text.append(f"[Content]\n{self.content}")
        # result_text.append(f"[Reasoning]\n{self.reasoning}")
        result_text.append(
            f"[Citations]\n"
            + "\n".join(
                f"[{i+1}]Title: {d['title']}, URL: {d['url']}"
                for i, d in enumerate(self.citations)
            )
        )

        # Set the output field to the content for display
        self.output = "\n\n".join(result_text)
        return self


class DeepResearch(BaseTool):
    """Advanced research tool that explores a topic through iterative web searches."""

    name: str = "deep_research"
    description: str = """
    DeepResearch is a specialized tool designed for conducting in-depth research and analytical synthesis on complex topics.
    Unlike general-purpose tools like WebSearch, which are ideal for quick lookups or retrieving real-time information, DeepResearch focuses on generating well-structured, insight-driven reports using curated insights and verified sources provided in advance.
    This tool excels at producing comprehensive, multi-perspective analyses, contextualizing key points, and weaving them into cohesive narratives.
    It should be the preferred choice whenever the task involves critical thinking, synthesis of multiple viewpoints, or writing detailed content based on a set of predefined insights and sources.

    Use DeepResearch when:
    * A research report, strategy memo, or analytical write-up is needed.
    * You have a list of insights and source materials to work from.
    * You need more than surface-level summaries —you're aiming for depth, clarity, and structured reasoning.

    Avoid using WebSearch when the required content involves synthesis, comparison, or thematic organization.
    Instead, delegate such tasks to DeepResearch for higher-quality results.
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research question or topic to investigate.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth of iterative research (1-5). Default is 2.",
                "default": 2,
            },
            "results_per_search": {
                "type": "integer",
                "description": "Number of search results to analyze per search (1-20). Default is 5.",
                "default": 5,
            },
            "max_insights": {
                "type": "integer",
                "description": "Maximum number of insights to return. Default is 20.",
                "default": 20,
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds. Reasonable timeout should take the complexity of the task into account. Default is 600.",
                "default": 600,
            },
        },
        "required": ["query"],
    }

    # Available research engines
    _research_engines: Dict[str, DeepResearchEngine] = PrivateAttr(
        default_factory=lambda: {
            "builtin": BuiltinDeepResearchEngine(),
            "perplexity": PerplexityDeepResearchEngine(),
        }
    )

    async def execute(
        self,
        query: str,
        max_depth: int = 2,
        results_per_search: int = 5,
        max_insights: int = 20,
        timeout: int = 600,
    ) -> DeepResearchResult:
        """Execute deep research on the given query."""

        # Get settings from config
        deep_research_config = config.deep_research_config
        retry_delay = deep_research_config.retry_delay if deep_research_config else 60
        max_retries = deep_research_config.max_retries if deep_research_config else 3

        # Try researching with retries when all engines fail
        for retry_count in range(max_retries + 1):
            try:
                research_item = await self._try_all_engines(
                    query=query,
                    max_depth=max_depth,
                    timeout=timeout,
                )

                if research_item:
                    # Return a successful structured response
                    citations = []
                    if research_item.citations:
                        citations = [
                            {"title": citation.title, "url": citation.url}
                            for citation in research_item.citations
                        ]

                    result = DeepResearchResult(
                        query=query,
                        content=research_item.content,
                        reasoning=research_item.reasoning,
                        citations=citations,
                    )

                    # Save the report to a local file
                    report_path = (
                        config.workspace_root / f"{query[:10]}_deep_research_report.md"
                    )
                    report_path.write_text(result.output)

                    return result

            except Exception as e:
                logger.error(f"Research error: {str(e)}")

            if retry_count < max_retries:
                # All engines failed, wait and retry
                logger.warning(
                    f"All research engines failed. Waiting {retry_delay} seconds before retry {retry_count + 1}/{max_retries}..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"All research engines failed after {max_retries} retries. Giving up."
                )

        # Return an error response
        return DeepResearchResult(
            query=query,
            error="All research engines failed to return results after multiple retries.",
        )

    async def _try_all_engines(
        self, query: str, max_depth: int, timeout: int
    ) -> Optional[DeepResearchItem]:
        """Try all research engines in the configured order."""
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self._research_engines[engine_name]
            logger.info(
                f"🔍 Attempting research with [{engine_name.capitalize()}] with query [{query}]..."
            )

            try:
                # Execute research with the current engine
                research_item = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: engine.research(
                        query=query,
                        max_depth=max_depth,
                        timeout=timeout,
                    ),
                )

                if research_item:
                    if failed_engines:
                        logger.info(
                            f"Research successful with {engine_name.capitalize()} after trying: {', '.join(failed_engines)}"
                        )
                    return research_item

            except DeepResearchError as e:
                logger.warning(f"Engine {engine_name} failed: {str(e)}")
                failed_engines.append(engine_name)
            except Exception as e:
                logger.error(f"Unexpected error with engine {engine_name}: {str(e)}")
                failed_engines.append(engine_name)

        if failed_engines:
            logger.error(f"All research engines failed: {', '.join(failed_engines)}")
        return None

    def _get_engine_order(self) -> List[str]:
        """Determines the order in which to try research engines."""
        deep_research_config = config.deep_research_config

        if deep_research_config:
            preferred = deep_research_config.engine.lower()
            fallbacks = [
                engine.lower() for engine in deep_research_config.fallback_engines
            ]
        else:
            preferred = "builtin"
            fallbacks = []

        # Start with preferred engine, then fallbacks, then remaining engines
        engine_order = [preferred] if preferred in self._research_engines else []
        engine_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self._research_engines and fb not in engine_order
            ]
        )
        engine_order.extend(
            [e for e in self._research_engines if e not in engine_order]
        )

        return engine_order


if __name__ == "__main__":
    deep_research = DeepResearch()
    result = asyncio.run(
        deep_research.execute("What is deep learning", max_depth=1, timeout=600)
    )
    print(result)
