import asyncio
import os
import re
from typing import List, Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, CLIResult
from app.tool.user_control import UserControlTool

_BASH_DESCRIPTION = """Execute a bash command in the terminal. Use for running code, installing packages, or managing files.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
* Restrictions:
* 1. Only use bash commands that are safe to run in a terminal.
* 2. DO NOT run bash commands that may harm the system or user, for example, running a command that may delete all files on the system.
* 3. DO NOT run bash commands that may expose sensitive information, for example, accessing files out of working directory.
"""


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return CLIResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode()
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """A tool for executing bash commands"""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None

    # 定义危险命令关键字列表
    _DANGEROUS_COMMANDS = [
        r"\brm\b",  # 删除文件
        r"\bsudo\b",  # 超级用户权限
        r"\.\.(?:/|\\\\",  # 访问上级目录
        r"\bmkfs\b",  # 格式化文件系统
        r"\bdd\b",  # 磁盘操作
        r"\bchmod\b",  # 修改权限
        r"\bmv\b",  # 移动文件
        r">/dev",  # 写入设备文件
        r"2>/dev",  # 写入设备文件
        r"\bkill\b",  # 终止进程
        r"\bpkill\b",  # 终止进程
        r"\bwget\b",  # 下载文件
        r"\bcurl\b.*\b-o\b",  # 下载文件并输出
        r"\bchown\b",  # 修改所有者
        r"\bexport\b",  # 设置环境变量
        r"\bsource\b",  # 执行脚本
        r"\beval\b",  # 执行字符串命令
    ]

    def _is_dangerous_command(self, command: str) -> bool:
        """检查命令是否包含危险操作"""
        if not command:
            return False

        for pattern in self._DANGEROUS_COMMANDS:
            if re.search(pattern, command):
                return True

        return False

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return CLIResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            # 检查是否为危险命令
            if self._is_dangerous_command(command):
                # 创建用户控制工具实例
                user_control = UserControlTool()
                # 请求用户确认
                result = await user_control.execute(
                    message=f"检测到潜在危险命令: '{command}'\n是否继续执行? (yes/no, 默认为yes)",
                    timeout=60,
                    default_action="continue",
                )

                # 获取用户输入
                user_input = str(result).strip().lower()

                # 如果用户拒绝执行或明确输入no
                if user_input == "no":
                    return CLIResult(system="命令已被用户取消执行。")

            # 执行命令
            return await self._session.run(command)

        raise ToolError("no command provided.")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)
