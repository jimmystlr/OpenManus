import asyncio

from prompt_toolkit import PromptSession

from app.agent.brand_investigator import BrandInvestigator
from app.agent.manus import Manus
from app.agent.message_reader import MessageReader
from app.agent.niche_alignment_assistant import NicheAlignmentAssistant
from app.logger import logger


async def main():
    agent = Manus()
    # agent = MessageReader()
    # agent = BrandInvestigator()
    # agent = NicheAlignmentAssistant()
    try:
        session = PromptSession()
        user_input = await session.prompt_async("Enter your prompt: ")
        if not user_input.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request...")
        await agent.run(user_input)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
