#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Base

The base class of a deepresearch engine implementation
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class DeepResearchError(Exception):
    pass


class DeepResearchItem(BaseModel):
    """DeepResearchItem

    Container for the deep research result
    """

    class Citation(BaseModel):
        """Citation

        Container for the citation result
        """
        title: str = Field(description="The citation title")
        url: str = Field(description="The citation url")

        def __str__(self):
            return f"title: {self.title}; url: {self.url}"

    content: str = Field(description="The content of the deep research result")
    reasoning: Optional[str] = Field(default='', description="The reasoning procedure")
    citations: Optional[List[Citation]] = Field(
            default=None, description="The citations attached with the result")

    def add_citation(self, title: str, url: str) -> None:
        if self.citations is None:
            self.citations = []
        self.citations.append(DeepResearchItem.Citation(title=title, url=url))

    def __str__(self):
        reasoning_str = self.reasoning if self.reasoning else ''
        citations_str = "\n".join(f"[{i+1}]{str(c)}"
                                  for i, c in enumerate(self.citations)) if self.citations else "None"
        return f"Reasoning:\n{reasoning_str}\n\nContent:\n{self.content}\n\nCitations:\n{citations_str}"


class DeepResearchEngine(BaseModel):
    """DeepResearchEngine

    Base class for deep research engines
    """

    model_config = {"arbitrary_types_allowed": True}

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
        raise DeepResearchError("Not implemented!")


if __name__ == "__main__":
    item = DeepResearchItem(content="hello world", citations=[])
    item.add_citation(url="test_url", title="test_title")
    print(item)
