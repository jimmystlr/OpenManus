#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BuiltinDeepResearchEngine

The builtin implementation of deepresearch engine
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


# Prompts for LLM interactions
OPTIMIZE_QUERY_PROMPT = """
You are a research assistant helping to optimize a search query for web research.
Your task is to reformulate the given query to be more effective for web searches.
Make it specific, use relevant keywords, and ensure it's clear and concise.

Original query: {query}

Provide only the optimized query text without any explanation or additional formatting.
"""

EXTRACT_INSIGHTS_PROMPT = """
Analyze the following content and extract key insights related to the research query.
For each insight, assess its relevance to the query on a scale of 0.0 to 1.0.

Research query: {query}
Content to analyze:
{content}

Extract up to 3 most important insights from this content. For each insight:
1. Provide the insight content
2. Provide relevance score (0.0-1.0)
"""

GENERATE_FOLLOW_UPS_PROMPT = """
Based on the insights discovered so far, generate follow-up research queries to explore gaps or related areas.
These should help deepen our understanding of the topic.

Original query: {original_query}
Current query: {current_query}
Key insights so far:
{insights}

Generate up to 3 specific follow-up queries that would help address gaps in our current knowledge.
Each query should be concise and focused on a specific aspect of the research topic.
"""

GENERATE_REPORT_PROMPT = """
I would like you to write a deep research report on the following topic: {query}.

I have already collected several key insights, each supported by specific sources. Please use these insights and their associated sources to create a comprehensive, well-structured, and analytically rich report. The report should include an introduction, main body sections categorized by subtopics or themes, and a conclusion. Ensure that each insight is accurately represented and properly contextualized within the overall topic.

Format requirements:

Use formal academic tone.

Include proper citations using the titles and URLs provided.

Aim for depth, clarity, and critical analysis.

Use headings and subheadings to structure the report logically.

Insights and Sources:
{insights}

And the sources are from citations, please replace the sources attached with insights with the cited number when you mention in the final report.
Citations:
{citations}

Please ensure that the report synthesizes the information rather than simply summarizing each source.
"""


# Constants for insight parsing
DEFAULT_RELEVANCE_SCORE = 1.0
FALLBACK_RELEVANCE_SCORE = 0.7
FALLBACK_CONTENT_LIMIT = 500
# Pattern to detect start of an insight (number., -, *, •) and capture content
INSIGHT_MARKER_PATTERN = re.compile(r"^\s*(?:\d+\.|-|\*|•)\s*(.*)")
# Pattern to detect relevance score, capturing the number (case-insensitive)
RELEVANCE_SCORE_PATTERN = re.compile(r"relevance.*?:.*?(\d\.?\d*)", re.IGNORECASE)


class ResearchInsight(BaseModel):
    """A single insight discovered during research."""

    model_config = ConfigDict(frozen=True)  # Make insights immutable

    content: str = Field(description="The insight content")
    source_url: str = Field(description="URL where this insight was found")
    source_title: Optional[str] = Field(default=None, description="Title of the source")
    relevance_score: float = Field(
        default=1.0, description="Relevance score (0.0-1.0)", ge=0.0, le=1.0
    )

    def __str__(self) -> str:
        """Format insight as string with source attribution."""
        source = self.source_title or self.source_url
        return f"{self.content} [Source: {source}]"


class ResearchContext(BaseModel):
    """Research context for tracking research progress."""

    query: str = Field(description="The original research query")
    insights: List[ResearchInsight] = Field(
        default_factory=list, description="Key insights discovered"
    )
    follow_up_queries: List[str] = Field(
        default_factory=list, description="Generated follow-up queries"
    )
    visited_urls: Set[tuple[str, str]] = Field(
        default_factory=set, description="URLs visited during research and their titles"
    )
    current_depth: int = Field(
        default=0, description="Current depth of research exploration", ge=0
    )
    max_depth: int = Field(
        default=2, description="Maximum depth of research to reach", ge=1
    )


class ResearchSummary(BaseModel):
    """Comprehensive summary of deep research results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="The original research query")
    content: str = Field(default_factory=str, description="The content result")
    insights: List[ResearchInsight] = Field(
        default_factory=list, description="Key insights discovered"
    )
    visited_urls: Set[tuple[str, str]] = Field(
        default_factory=set, description="URLs visited during research and their titles"
    )
    depth_reached: int = Field(
        default=0, description="Maximum depth of research reached", ge=0
    )

    @model_validator(mode="after")
    def populate_output(self) -> "ResearchSummary":
        """Populate the output field after validation."""
        # Group and sort insights by relevance
        grouped_insights = {
            "Key Findings": [i for i in self.insights if i.relevance_score >= 0.8],
            "Additional Information": [
                i for i in self.insights if 0.5 <= i.relevance_score < 0.8
            ],
            "Supplementary Information": [
                i for i in self.insights if i.relevance_score < 0.5
            ],
        }

        sections = [
            f"# Research: {self.query}\n",
            f"**Sources**: {len(self.visited_urls)} | **Depth**: {self.depth_reached + 1}\n",
        ]

        for section_title, insights in grouped_insights.items():
            if insights:
                sections.append(f"## {section_title}")
                for i, insight in enumerate(insights, 1):
                    sections.extend(
                        [
                            insight.content,
                            f"> Source: [{insight.source_title or 'Link'}]({insight.source_url})\n",
                        ]
                    )

        self.content = "\n".join(sections)
        return self

    def to_deep_research_item(self, llm: LLM) -> DeepResearchItem:
        """ Reformulate the result to DeepResearchItem

        Args:
            llm (LLM): LLM used to generate the final report
        Returns:
            DeepResearchItem: The deep research result item
        """
        item = DeepResearchItem(content="", citations=[])

        citations = "\n".join(f"[{i+1}]Title: {d[1]}, URL: {d[0]}"
                              for i, d in enumerate(self.visited_urls))
        prompt = GENERATE_REPORT_PROMPT.format(
                query=self.query,
                insights=self.content,
                citations=citations
        )
        try:
            response = asyncio.run(llm.ask_tool(
                messages=[Message.user_message(content=prompt)],
                tools=None,
                tool_choice=ToolChoice.NONE,
                stream=False,
            ))
            item.content = response.content
        except Exception as err:
            logger.error(f"Exception caught: {str(err)}")
            item.content = self.content

        # Fill out the structure
        for url, title in self.visited_urls:
            item.add_citation(title=title, url=url)

        return item


class BuiltinDeepResearchEngine(DeepResearchEngine):
    """BuiltinDeepResearchEngine

    Builtin deep research engine implementation
    """

    search_tool: WebSearch = Field(default_factory=WebSearch)
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
        max_depth = max(1, min(kwargs.get("max_depth", 2), 5))
        results_per_search = max(1, min(kwargs.get("results_per_search", 5), 20))
        max_insights = kwargs.get("max_insights", 20)
        time_limit_seconds = kwargs.get("timeout", 300)

        # Initialize research context and set deadline
        context = ResearchContext(query=query, max_depth=max_depth)
        deadline = time.time() + time_limit_seconds

        try:
            # Initiate research process with optimized query
            optimized_query = asyncio.run(self._generate_optimized_query(query))
            asyncio.run(self._research_graph(
                context=context,
                query=optimized_query,
                results_count=results_per_search,
                deadline=deadline,
            ))
        except (ToolError, Exception) as e:
            logger.error(f"Research error: {str(e)}")
            raise DeepResearchError(str(e))

        # Prepare final summary
        return ResearchSummary(
            query=query,
            insights=sorted(
                context.insights, key=lambda x: x.relevance_score, reverse=True
            )[:max_insights],
            visited_urls=context.visited_urls,
            depth_reached=context.current_depth,
        ).to_deep_research_item(llm=self.llm)

    async def _generate_optimized_query(self, query: str) -> str:
        """Generate an optimized search query using LLM."""
        try:
            prompt = OPTIMIZE_QUERY_PROMPT.format(query=query)
            response = await self.llm.ask_tool(
                [{"role": "user", "content": prompt}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "optimize_query",
                            "description": "Generate an optimized search query",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The optimized search query",
                                    }
                                },
                                "required": ["query"],
                            },
                        },
                    }
                ],
                tool_choice=ToolChoice.REQUIRED,
                stream=False,
            )

            # Extract the query from the tool_call response
            if response and response.tool_calls and len(response.tool_calls) > 0:
                tool_call = response.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)
                optimized_query = arguments.get("query", "")
            else:
                # Fallback to original query if tool call failed
                logger.warning("Tool call failed to return a valid response")
                return query

            if not optimized_query:
                logger.warning("Generated empty optimized query, using original")
                return query

            logger.info(f"Optimized query: '{optimized_query}'")
            return optimized_query
        except Exception as e:
            logger.warning(f"Failed to optimize query: {str(e)}")
            return query  # Fall back to original query on error

    async def _research_graph(
        self,
        context: ResearchContext,
        query: str,
        results_count: int,
        deadline: float,
    ) -> None:
        """Run a complete research cycle (search, analyze, generate follow-ups)."""
        # Check termination conditions
        if time.time() >= deadline or context.current_depth >= context.max_depth:
            return

        # Log current research step
        logger.info(f"Research cycle at depth {context.current_depth + 1}")

        # 1. Web search
        search_results = await self._search_web(query, results_count)
        if not search_results:
            return

        # 2. Extract insights
        new_insights = await self._extract_insights(
            context, search_results, context.query, deadline
        )
        if not new_insights:
            return

        # 3. Generate follow-up queries
        follow_up_queries = await self._generate_follow_ups(
            new_insights, query, context.query
        )
        context.follow_up_queries.extend(follow_up_queries)

        # Update depth and proceed to next level
        context.current_depth += 1

        # 4. Continue research with follow-up queries
        if follow_up_queries and context.current_depth < context.max_depth:
            tasks = []  # Create a list to hold the tasks
            for follow_up in follow_up_queries[:2]:  # Limit branching factor
                if time.time() >= deadline:
                    break

                # Create a coroutine for the recursive research call
                task = self._research_graph(
                    context=context,
                    query=follow_up,
                    results_count=max(1, results_count - 1),  # Reduce result count
                    deadline=deadline,
                )
                tasks.append(task)  # Add the task to the list

            # Run all the created tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)

    async def _search_web(self, query: str, results_count: int) -> List[SearchResult]:
        """Perform web search for the given query."""
        search_response = await self.search_tool.execute(
            query=query, num_results=results_count, fetch_content=True
        )
        return [] if search_response.error else search_response.results

    async def _extract_insights(
        self,
        context: ResearchContext,
        results: List[SearchResult],
        original_query: str,
        deadline: float,
    ) -> List[ResearchInsight]:
        """Extract insights from search results."""
        all_insights = []

        for rst in results:
            # Skip if URL already visited or time exceeded
            if (rst.url, rst.title) in context.visited_urls or time.time() >= deadline:
                continue

            context.visited_urls.add((rst.url, rst.title))

            # Skip if no content available
            if not rst.raw_content:
                continue

            # Extract insights using LLM
            insights = await self._analyze_content(
                content=rst.raw_content[:10000],  # Limit content size
                url=rst.url,
                title=rst.title,
                query=original_query,
            )

            all_insights.extend(insights)
            context.insights.extend(insights)

            # Log discovered insights
            logger.info(f"Extracted {len(insights)} insights from {rst.url}")

        return all_insights

    async def _generate_follow_ups(
        self, insights: List[ResearchInsight], current_query: str, original_query: str
    ) -> List[str]:
        """Generate follow-up queries based on insights."""
        if not insights:
            return []

        # Format insights for the prompt
        insights_text = "\n".join([f"- {insight.content}" for insight in insights[:5]])

        # Create prompt for generating follow-up queries
        prompt = GENERATE_FOLLOW_UPS_PROMPT.format(
            original_query=original_query,
            current_query=current_query,
            insights=insights_text,
        )

        # Get follow-up queries from LLM using structured output
        response = await self.llm.ask_tool(
            [{"role": "user", "content": prompt}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "generate_follow_ups",
                        "description": "Generate follow-up queries based on research insights",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "follow_up_queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of follow-up queries (max 3) that would help address gaps in current knowledge",
                                    "maxItems": 3,
                                }
                            },
                            "required": ["follow_up_queries"],
                        },
                    },
                }
            ],
            tool_choice=ToolChoice.REQUIRED,
            stream=False,
        )

        # Extract queries from the tool response
        queries = []
        if response and response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            queries = arguments.get("follow_up_queries", [])

        # Ensure we don't return more than 3 queries
        return queries[:3]

    async def _analyze_content(
        self, content: str, url: str, title: str, query: str
    ) -> List[ResearchInsight]:
        """Extract insights from content based on relevance to query."""
        prompt = EXTRACT_INSIGHTS_PROMPT.format(
            query=query, content=content[:5000]  # Limit content size
        )

        response = await self.llm.ask_tool(
            [{"role": "user", "content": prompt}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "extract_insights",
                        "description": "Extract key insights from content with relevance scores",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "insights": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "content": {
                                                "type": "string",
                                                "description": "The insight content",
                                            },
                                            "relevance_score": {
                                                "type": "number",
                                                "description": "Relevance score between 0.0 and 1.0",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                        },
                                        "required": ["content", "relevance_score"],
                                    },
                                    "description": "List of key insights extracted from the content",
                                    "maxItems": 3,
                                }
                            },
                            "required": ["insights"],
                        },
                    },
                }
            ],
            tool_choice=ToolChoice.REQUIRED,
            stream=False,
        )

        insights = []

        # Process structured JSON response
        if response and response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            extracted_insights = arguments.get("insights", [])

            for insight_data in extracted_insights:
                insights.append(
                    ResearchInsight(
                        content=insight_data.get("content", ""),
                        source_url=url,
                        source_title=title,
                        relevance_score=insight_data.get(
                            "relevance_score", FALLBACK_RELEVANCE_SCORE
                        ),
                    )
                )

        # Fallback: if no structured insights found, use fallback approach
        if not insights:
            logger.warning(
                f"Could not parse structured insights from LLM response for {url}. Using fallback."
            )
            insights.append(
                ResearchInsight(
                    content=f"Failed to extract structured insights from content about {title or url}."[
                        :FALLBACK_CONTENT_LIMIT
                    ],
                    source_url=url,
                    source_title=title,
                    relevance_score=FALLBACK_RELEVANCE_SCORE,
                )
            )

        return insights


if __name__ == "__main__":
    deep_research = BuiltinDeepResearchEngine()
    result = deep_research.research(
        query="What is deep learning", max_depth=1, results_per_search=2
    )
    print(result)
