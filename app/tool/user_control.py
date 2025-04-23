import asyncio
import sys
from typing import Optional

from pydantic import Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class UserControlTool(BaseTool):
    """
    Tool for transferring control to the user and waiting for user actions.

    This tool is useful in scenarios where the model needs to wait for user input or actions,
    such as asking for more needs or details, web page verification, login, or confirmation before proceeding.
    """

    name: str = "user_control"
    description: str = """
    Transfers control to the user and waits for user input or actions.

    Use this tool when:
    * Ask for more needs or details from the user before taking further actions
    * A web page requires user verification or CAPTCHA solving
    * User login credentials need to be entered manually
    * User confirmation is needed before proceeding with an operation
    * Any scenario where human intervention is required

    The tool will wait for the user to complete their input or action and then return control to the model.
    If the user doesn't respond within the timeout period, the tool will continue execution
    with a timeout message.
    """

    parameters: dict = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to display to the user explaining what action is needed",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum time to wait for user action in seconds (default: 300)",
            },
            "default_action": {
                "type": "string",
                "description": "Action to take if timeout occurs (continue, abort, retry), (default: continue)",
                "enum": ["continue", "abort", "retry"],
            },
        },
        "required": ["message"],
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    # Dictionary to store pending user action futures
    pending_actions: dict = Field(default_factory=dict)

    async def execute(
        self,
        message: str,
        timeout: Optional[int] = 300,
        default_action: Optional[str] = "continue",
    ) -> ToolResult:
        """
        Execute the user control tool.

        Args:
            message: Message to display to the user explaining what action is needed
            timeout: Maximum time to wait for user action in seconds (default: 300)
            default_action: Action to take if timeout occurs (continue, abort, retry)

        Returns:
            ToolResult with the outcome of the user action or timeout
        """
        async with self.lock:
            try:
                logger.info(f"Transferring control to user: {message}")

                # Create a future that will be set when the user completes their action
                user_action_complete = asyncio.Future()

                # Generate a unique ID for this action
                action_id = id(user_action_complete)
                self.pending_actions[action_id] = user_action_complete

                # Log the action ID so it can be used by the UI
                logger.info(f"User control action started with ID: {action_id}")

                # Display message to the user via command line
                print(f"\n[User Action Required] {message}")
                print("Please enter your response (or press Enter to continue):")

                # Function to get user input asynchronously
                async def get_user_input():
                    loop = asyncio.get_event_loop()

                    # Run input in a separate thread to not block the event loop
                    # Using input() instead of sys.stdin.readline() for more reliable input handling
                    def _get_input():
                        try:
                            return input()
                        except EOFError:
                            return ""

                    user_response = await loop.run_in_executor(None, _get_input)
                    if user_response == "":
                        user_response = "Completed"  # Default response for empty input
                    return user_response.strip()

                # Create task for user input
                user_input_task = asyncio.create_task(get_user_input())

                try:
                    # Wait for either user input or timeout
                    done, pending = await asyncio.wait(
                        [user_input_task, user_action_complete],
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if done:
                        if user_input_task in done:
                            # User provided input via command line
                            result = await user_input_task
                            logger.info(
                                f"User provided input via command line: {result}"
                            )
                            # Cancel the other future if it's still pending
                            if not user_action_complete.done():
                                user_action_complete.cancel()
                        else:
                            # Action was completed externally via complete_user_action
                            result = await user_action_complete
                            logger.info(f"User action completed externally: {result}")
                            # Cancel the input task if it's still pending
                            if not user_input_task.done():
                                user_input_task.cancel()

                        # Clean up the reference
                        if action_id in self.pending_actions:
                            del self.pending_actions[action_id]

                        # Return the result - directly use the user input as output
                        return ToolResult(
                            output=result,
                            system=f"action_completed:{action_id}",
                        )
                    else:
                        # If we get here, it means timeout occurred
                        raise asyncio.TimeoutError()

                except asyncio.TimeoutError:
                    logger.warning(f"User action timed out after {timeout} seconds")

                    # Cancel any pending tasks
                    for task in pending:
                        task.cancel()

                    # Clean up the reference
                    if action_id in self.pending_actions:
                        del self.pending_actions[action_id]

                    if default_action == "abort":
                        return ToolResult(
                            error=f"Operation aborted: user did not respond within {timeout} seconds",
                            system=f"action_timeout:{action_id}:abort",
                        )
                    elif default_action == "retry":
                        return ToolResult(
                            output=f"Timeout occurred. Retrying the operation.",
                            system=f"action_timeout:{action_id}:retry",
                        )
                    else:  # default: continue
                        return ToolResult(
                            output=f"Timeout occurred after {timeout} seconds. Continuing with default action.",
                            system=f"action_timeout:{action_id}:continue",
                        )

            except Exception as e:
                logger.error(f"Error in user control tool: {str(e)}")
                return ToolResult(error=f"Error during user control: {str(e)}")

    # Method to be called by the UI when the user completes their action
    async def complete_user_action(
        self, action_id: int, result: str = "Action completed"
    ) -> bool:
        """
        Signal that the user has completed their action.

        This should be called by the UI when the user completes the requested action.

        Args:
            action_id: The ID of the action to complete
            result: Message describing the result of the user action

        Returns:
            bool: True if the action was completed successfully, False otherwise
        """
        async with self.lock:
            if action_id in self.pending_actions:
                future = self.pending_actions[action_id]
                if not future.done():
                    future.set_result(result)
                    logger.info(f"User action completed: {result}")
                # Clean up the reference
                del self.pending_actions[action_id]
                return True
            else:
                logger.warning(f"No pending action found with ID {action_id}")
                return False

    async def cancel_allpending_actions(
        self, reason: str = "Cancelled by system"
    ) -> int:
        """
        Cancel all pending user actions.

        Args:
            reason: The reason for cancellation

        Returns:
            int: Number of actions cancelled
        """
        async with self.lock:
            count = 0
            for action_id, future in list(self.pending_actions.items()):
                if not future.done():
                    future.set_exception(asyncio.CancelledError(reason))
                    count += 1
                del self.pending_actions[action_id]

            logger.info(f"Cancelled {count} pending user actions: {reason}")
            return count
