import logging
import os
import pathlib
import random
import re
import time
import traceback
import requests
import json
from enum import Enum
from typing import Any, cast, Optional, Dict, List
from venv import logger

from anthropic import Anthropic
from anthropic.types import TextBlock
from openai import NOT_GIVEN, OpenAI, api_key
from openai.types.chat import ChatCompletionMessageParam

from env.base import Env
from scenarios.base import Scenario

_SYSTEM_PROMPT = "You are an experienced full-stack developer"


class KeyLocs(Enum):
    openai_key = "OPENAI_API_KEY"
    anthropic_key = "ANTHROPIC_API_KEY"
    together_key = "TOGETHER_API_KEY"
    openrouter_key = "OPENROUTER_API_KEY"
    openhands_api_url = "OPENHANDS_API_URL"  # URL to OpenHands REST API server


class OpenHandsClient:
    """Client for interacting with OpenHands headless mode via REST API"""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.environ.get(
            KeyLocs.openhands_api_url.value, 
            "http://localhost:3000"
        )
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> bool:
        """Check if OpenHands API server is running"""
        try:
            response = self.session.get(f"{self.api_url}/api/health", timeout=5)
            
            return response.status_code == 200
        except Exception:
            return False
    
    def execute_task(
        self, 
        task: str, 
        workspace_dir: str = "/app", 
        max_iterations: int = 50,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute a task using OpenHands in headless mode"""
        payload = {
            "task": task,
            "workspace_dir": workspace_dir,
            "max_iterations": max_iterations
        }
        
        response = self.session.post(
            f"{self.api_url}/api/execute_task",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()

class AiderClient:
    """Client for interacting with Aider API"""

    def __init__(self, api_url: str = "http://localhost:8000/api"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        """Check if Aider API server is running"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def execute_task(
        self,
        task: str,
        project_path: str = ".",
        max_steps: int = 20,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Execute a task using Aider API"""
        payload = {
            "task": task,
            "project_path": project_path,
            "max_steps": max_steps,
        }

        response = self.session.post(
            f"{self.api_url}/execute_task",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_workspace_files(self) -> Dict[str, str]:
        """Fetch the current files in the Aider workspace"""
        response = self.session.get(f"{self.api_url}/workspace_files", timeout=10)
        response.raise_for_status()
        return response.json().get("files", {})
        
class Prompter:

    # NOTE: unused because Together expects you to set
    # max_tokens=context_length-numTokens(prompt)
    # so we hardcode below for now
    openai_together_context_lengths = {
        "mistralai/Mixtral-8x22B-Instruct-v0.1": 65536,
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": 131072,
        "deepseek-ai/DeepSeek-V3": 131072,
        "Qwen/Qwen2.5-Coder-32B-Instruct": 32768,
        "Qwen/Qwen2.5-72B-Instruct-Turbo": 32768,
        "Qwen/Qwen2.5-7B-Instruct-Turbo": 32768,
        "gpt-4o": 128000,
        "o1": 200000,
        "o1-mini": 128000,
        "o3-mini": 200000,
        "deepseek-ai/DeepSeek-R1": 164000,
        "openhands": 128000,  # OpenHands context (depends on underlying LLM)
        "aider": 128000,
    }

    openai_max_completion_tokens = {
        "gpt-4o": 16384,
        "o1": 100000,
        "o1-mini": 65536,
        "o3-mini": 100000,
        "gpt-oss-120b": 131072,
        "openhands": 32000,  # Generous limit for OpenHands output
        "aider": 32000,  # Generous limit for Aider output
    }

    openrouter_remap = {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": "meta-llama/llama-3.3-70b-instruct",
        "deepseek-ai/DeepSeek-V3": "deepseek/deepseek-chat",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen/qwen-2.5-coder-32b-instruct",
        "Qwen/Qwen2.5-7B-Instruct-Turbo": "qwen/qwen-2.5-7b-instruct",
        "Qwen/Qwen2.5-72B-Instruct-Turbo": "qwen/qwen-2.5-72b-instruct",
    }

    def __init__(
        self,
        env: Env,
        scenario: Scenario,
        model: str,
        spec_type: str,
        safety_prompt: str,
        batch_size: int,
        temperature: float,
        reasoning_effort: str,
        openrouter: bool,
        openhands_timeout: int = 300,  # OpenHands task timeout
        openhands_max_iterations: int = 50,  # Max iterations for OpenHands
        agent_port: Optional[int] = None,  # Port where OpenHands API server is running
    ):
        self.env = env
        self.scenario = scenario
        self.spec_type = spec_type
        self.safety_prompt = safety_prompt
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.openhands_timeout = openhands_timeout
        self.openhands_max_iterations = openhands_max_iterations
        self.agent_port = agent_port

        self.system_prompt = _SYSTEM_PROMPT
        self.o1_o3 = model.startswith("o1") or model.startswith("o3")
        self.anthropic = "claude" in model
        self.openai = self.o1_o3 or self.model.startswith("gpt")
        self.openhands = model == "openhands"
        self.openrouter = openrouter and not (self.anthropic or self.openai or self.openhands)
        self.local = "oss" in model
        self.aider = model == "aider"

        self.prompt = self.scenario.build_prompt(
            self.env, self.spec_type, self.safety_prompt
        )

    def prompt_openhands(self, logger: logging.Logger) -> list[str]:
        """Execute task using OpenHands headless mode"""
        client = OpenHandsClient(
            api_url=f"http://localhost:{self.agent_port}" if self.agent_port else None
        )
        
        # First check if the API server is running
        if not client.health_check():
            logger.error("OpenHands API server is not running or not accessible")
            raise Exception("OpenHands API server is not running. Make sure the container is running with the OpenHands API server.")
        
        try:
            # Construct the comprehensive task for OpenHands
            openhands_task = f"""
I am a full-stack developer working on a {self.env.framework} application. 

DEVELOPMENT TASK:
{self.prompt}

ENVIRONMENT DETAILS:
- Framework: {self.env.framework}
- Working Directory: /app
- Database: {self.env.sqlite_database if self.env.sqlite_database else 'None'}
- Multi-file project: {self.env.is_multi_file}

REQUIREMENTS:
1. Create all necessary files for a working {self.env.framework} application
2. Implement all the features described in the task
3. Ensure the code follows best practices and is production-ready
4. If there's a database, set up proper models and migrations
5. Test the application to make sure it works correctly
6. Provide clear documentation of what was built

Please complete this development task autonomously. Create all necessary files, implement the functionality, and verify it works.
"""
            
            logger.info("Sending task to OpenHands headless mode...")
            logger.info(f"Task timeout: {self.openhands_timeout}s, Max iterations: {self.openhands_max_iterations}")
            
            # Execute the task
            result = client.execute_task(
                task=openhands_task,
                workspace_dir="/app",
                max_iterations=self.openhands_max_iterations,
                timeout=self.openhands_timeout
            )
            
            # Process the result
            if result.get("success"):
                logger.info("OpenHands task completed successfully")
                
                # Format the successful response
                response_parts = [
                    "OpenHands Development Task Completed",
                    "=" * 40,
                    "",
                    "EXECUTION RESULT:",
                    result.get("result", "Task completed"),
                    "",
                ]
                
                # Add files created information
                files_created = result.get("files_created", {})
                if files_created:
                    response_parts.extend([
                        "FILES CREATED/MODIFIED:",
                        "-" * 25,
                    ])
                    for file_path, content_preview in files_created.items():
                        # Show first 200 chars of each file
                        preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                        response_parts.extend([
                            f"File: {file_path}",
                            f"Content preview: {preview}",
                            "",
                        ])
                else:
                    response_parts.extend([
                        "No files were created or modified.",
                        "",
                    ])
                
                # Add execution logs if available
                execution_logs = result.get("logs", "")
                if execution_logs:
                    response_parts.extend([
                        "EXECUTION LOGS:",
                        "-" * 15,
                        execution_logs[:1000] + "..." if len(execution_logs) > 1000 else execution_logs,
                        "",
                    ])
                
                response_text = "\n".join(response_parts)
                logger.info(f"OpenHands created {len(files_created)} files")
                return [response_text]
                
            else:
                logger.error("OpenHands task failed")
                error_msg = result.get("error", "Unknown error")
                task_output = result.get("result", "No output")
                
                response_parts = [
                    "OpenHands Development Task Failed",
                    "=" * 40,
                    "",
                    "ERROR:",
                    error_msg,
                    "",
                    "TASK OUTPUT:",
                    task_output,
                    "",
                ]
                
                # Still include any files that might have been created
                files_created = result.get("files_created", {})
                if files_created:
                    response_parts.extend([
                        "PARTIAL FILES CREATED:",
                        "-" * 25,
                    ])
                    for file_path, content_preview in files_created.items():
                        preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                        response_parts.extend([
                            f"File: {file_path}",
                            f"Content preview: {preview}",
                            "",
                        ])
                
                response_text = "\n".join(response_parts)
                return [response_text]
                
        except Exception as e:
            logger.error(f"OpenHands interaction failed: {e}")
            raise e

    def prompt_aider(self, logger: logging.Logger) -> list[str]:
        """Execute task using Aider API"""
        client = AiderClient(
            api_url=f"http://localhost:{self.agent_port}/api" if self.agent_port else "http://localhost:3000/api"
        )

        # First check if the API server is running
        if not client.health_check():
            logger.error("Aider API server is not running or not accessible")
            raise Exception("Aider API server is not running. Make sure the Aider server is running and accessible.")

        try:
            # Construct the comprehensive task for Aider
            aider_task = f"""
    I am a full-stack developer working on a {self.env.framework} application. 

    DEVELOPMENT TASK:
    {self.prompt}

    ENVIRONMENT DETAILS:
    - Framework: {self.env.framework}
    - Working Directory: /app
    - Database: {self.env.sqlite_database if self.env.sqlite_database else 'None'}
    - Multi-file project: {self.env.is_multi_file}

    REQUIREMENTS:
    1. Create all necessary files for a working {self.env.framework} application
    2. Implement all the features described in the task
    3. Ensure the code follows best practices and is production-ready
    4. If there's a database, set up proper models and migrations
    5. Test the application to make sure it works correctly
    6. Provide clear documentation of what was built

    Please complete this development task autonomously. Create all necessary files, implement the functionality, and verify it works.
    """
            
            logger.info("Sending task to Aider API...")
            
            # Execute the task using the fixed API with proper file content collection
            result =  client.execute_task(
                task=aider_task,
                project_path="/app",
                max_steps=20,
                timeout=300
            )
            
            # The result now contains actual file contents in files_created
            if result.get("success"):
                logger.info("Aider task completed successfully")
                
                files_created = result.get("files_created", {})
                execution_result = result.get("result", "Task completed")
                
                # Create response that includes the actual file data
                import json
                response_data = {
                    "type": "aider_response",
                    "success": True,
                    "execution_result": execution_result,
                    "files_created": files_created,
                    "logs": result.get("logs", "")
                }
                
                # Create summary for logging
                response_parts = [
                    "Aider Development Task Completed",
                    "=" * 40,
                    "",
                    "EXECUTION RESULT:",
                    execution_result[:500] + "..." if len(execution_result) > 500 else execution_result,
                    "",
                    f"FILES CREATED: {len(files_created)} files",
                    "-" * 25,
                ]
                
                for file_path in files_created.keys():
                    response_parts.append(f"File: {file_path}")
                
                summary_text = "\n".join(response_parts)
                logger.info(f"Aider created {len(files_created)} files")
                
                # Return response with embedded file data
                full_response = f"AIDER_STRUCTURED_RESPONSE:{json.dumps(response_data)}\n\n{summary_text}"
                return [full_response]
                
            else:
                logger.error("Aider task failed")
                error_msg = result.get("error", "Unknown error")
                task_output = result.get("result", "No output")
                files_created = result.get("files_created", {})
                
                import json
                response_data = {
                    "type": "aider_response", 
                    "success": False,
                    "execution_result": task_output,
                    "files_created": files_created,
                    "error": error_msg
                }
                
                response_parts = [
                    "Aider Development Task Failed",
                    "=" * 40,
                    "",
                    "ERROR:",
                    error_msg,
                    "",
                    "TASK OUTPUT:",
                    task_output[:500] + "..." if len(task_output) > 500 else task_output,
                ]
                
                if files_created:
                    response_parts.extend([
                        "",
                        f"PARTIAL FILES CREATED: {len(files_created)} files",
                        "-" * 25,
                    ])
                    for file_path in files_created.keys():
                        response_parts.append(f"File: {file_path}")
                
                summary_text = "\n".join(response_parts)
                full_response = f"AIDER_STRUCTURED_RESPONSE:{json.dumps(response_data)}\n\n{summary_text}"
                return [full_response]
                
        except Exception as e:
            logger.error(f"Aider interaction failed: {e}")
            raise e

    def prompt_anthropic(self, logger: logging.Logger) -> list[str]:
        client = Anthropic(api_key=os.environ[KeyLocs.anthropic_key.value])
        try:
            response = client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": self.prompt},
                ],
                temperature=self.temperature,
                max_tokens=8192 if "claude-3-5-" in self.model else 4096,
            )
            assert isinstance(response.content[0], TextBlock)
            if response.usage is not None:
                logger.info(
                    f"Token stats: {response.usage}; around {response.usage.output_tokens} completion tokens per completion"
                )
            if response.stop_reason == "max_tokens":
                logger.warning(f"Completion was cut off due to length.")
            return [response.content[0].text]
        except Exception as e:
            raise e

    def prompt_openrouter(self, logger: logging.Logger) -> list[str]:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ[KeyLocs.openrouter_key.value],
        )
        if self.model in self.openrouter_remap:
            open_router_model = self.openrouter_remap[self.model]
        else:
            open_router_model = self.model
        try:
            response = client.chat.completions.create(
                model=open_router_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.prompt},
                ],
                n=1,
                temperature=self.temperature,
                max_tokens=(
                    8192
                    if self.model not in Prompter.openai_together_context_lengths
                    else Prompter.openai_together_context_lengths[self.model] - 2000
                ),
            )
            if response.choices is None:
                logger.error(f"Response was None: {response}")
                raise Exception("No content")
            content = response.choices[0].message.content
            if content is not None and len(content) > 0:
                if response.usage is not None:
                    logger.info(
                        f"Token stats: {response.usage}; around {response.usage.completion_tokens} completion tokens per completion"
                    )
                else:
                    logger.info(f"Token stats unavailable")
                if response.choices[0].finish_reason == "length":
                    logger.warning(f"Completion was cut off due to length.")
                try:
                    logger.info(f"Inference provided by: {response.provider}")  # type: ignore
                    logger.info(f"Inference id: {response.id}")
                except:
                    pass
                return [content]
            else:
                raise Exception("No content")
        except Exception as e:
            raise e

    def prompt_openai_together_batch(self, logger: logging.Logger) -> list[str]:
        if self.openai:
            client = OpenAI(api_key=os.environ[KeyLocs.openai_key.value])
        elif self.local:  # for gpt-oss models
            client = OpenAI(base_url=os.environ["LOCAL_API_BASE"])
        else:
            client = OpenAI(
                api_key=os.environ[KeyLocs.together_key.value],
                base_url="https://api.together.xyz/v1",
            )
        try:
            # Prepare extra kwargs
            extra_kwargs: dict[str, Any] = {}
            if self.model == "o1" or "oss" in self.model or '5' in self.model or self.model.startswith(
                "o3"
            ):  # NOTE: o1-mini does not have this
                extra_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.openai:
                extra_kwargs["max_completion_tokens"] = (
                    Prompter.openai_max_completion_tokens[self.model]
                )
            else:
                extra_kwargs["max_tokens"] = (
                    8192
                    if self.model not in Prompter.openai_together_context_lengths
                    else Prompter.openai_together_context_lengths[self.model] - 2000
                )
            # Prepare the message
            messages: list[Any] = []
            if self.model == "o1" or self.model.startswith("o3") or "oss" in self.model or '5' in self.model:
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "developer", "content": self.system_prompt},
                    )
                )
            elif self.model == "o1-mini":
                # No sysprompt
                pass
            else:
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "system", "content": self.system_prompt},
                    )
                )
            messages.append({"role": "user", "content": self.prompt})

            # Query
            completions = client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=self.batch_size,
                temperature=self.temperature if not self.o1_o3 else NOT_GIVEN,
                **extra_kwargs,
            )
            if completions.usage is not None:
                logger.info(
                    f"Batch token stats: {completions.usage}; around {completions.usage.completion_tokens / self.batch_size:.2f} completion tokens per completion"
                )
            else:
                logger.info(f"Batch token stats unavailable")
            responses = []
            for idx, choice in enumerate(completions.choices):
                if choice.finish_reason == "length":
                    logger.warning(f"Completion {idx} was cut off due to length.")
                if choice.message.content:
                    responses.append(choice.message.content)
            return responses

        except Exception as e:
            raise e

    def prompt_model(self, logger: logging.Logger) -> list[str]:
        if self.anthropic:
            return self.prompt_anthropic(logger)
        elif self.openhands:
            return self.prompt_openhands(logger)
        elif self.aider:
            return self.prompt_aider(logger)
        elif self.openrouter:
            return self.prompt_openrouter(logger)
        else:
            return self.prompt_openai_together_batch(logger)

    def prompt_model_batch_with_exp_backoff(
        self,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        logger: logging.Logger,
    ) -> list[str]:
        # OpenHands, Anthropic and OpenRouter don't support batching
        n_times_to_sample = self.batch_size if (self.openrouter or self.anthropic or self.openhands) else 1
        completions = []
        for _ in range(n_times_to_sample):
            retries = 0
            while True:
                try:
                    if retries > 0:
                        logger.info(f"Retrying {retries} times")
                    completion = self.prompt_model(logger)
                    completions.extend(completion)
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries reached, raising exception: {e}")
                        raise e
                    delay = min(base_delay * 2**retries, max_delay)
                    delay = random.uniform(0, delay)
                    logger.exception(
                        f"{e}, backing off for {delay} seconds", exc_info=e
                    )
                    time.sleep(delay)
        return completions


class Parser:

    def __init__(self, env: Env, logger: logging.Logger):
        self.env = env
        self.logger = logger

        self.fp_pattern = re.compile(r"<FILEPATH>(.+?)</FILEPATH>", re.DOTALL)
        self.fp_ht_pattern = re.compile(r"^###\s*(.+?)$", re.DOTALL | re.MULTILINE)
        self.md_pattern = re.compile(r"```(?!bash)\w+\n(.*?)\n```", re.DOTALL)
        self.code_pattern = re.compile(r"<CODE>(.+?)</CODE>", re.DOTALL)

    def _invalid(self, response: str) -> dict[pathlib.Path, str]:
        self.logger.warning(f"Format not found")
        return {pathlib.Path("failed"): "Format not found. Full response:\n" + response}

    def _parse_md(self, response: str) -> list[str]:
        return [s.strip() for s in self.md_pattern.findall(response)]

    def _parse_code(self, response: str) -> list[str]:
        return [s.strip() for s in self.code_pattern.findall(response)]

    def _parse_openhands_response(self, response: str) -> dict[pathlib.Path, str]:
        """Parse OpenHands response which contains execution results and file information"""
        
        # Check if this looks like an OpenHands response
        if "OpenHands Development Task" in response or "FILES CREATED" in response:
            # Extract file information if present
            files_created_pattern = re.compile(
                r"File: ([^\n]+)\nContent preview: ([^\n]+)", 
                re.MULTILINE | re.DOTALL
            )
            matches = files_created_pattern.findall(response)
            
            if matches:
                # If we have file information, create a mapping
                result = {}
                for file_path, content_preview in matches:
                    file_path = file_path.strip()
                    # For now, we can only provide the preview
                    # In a real implementation, you'd fetch the actual files from the workspace
                    result[pathlib.Path(file_path)] = f"# OpenHands created this file\n# Content preview:\n{content_preview}"
                
                # Also add the full OpenHands summary
                result[pathlib.Path("openhands_execution_summary.md")] = response
                return result
            else:
                # Just return the summary
                return {pathlib.Path("openhands_execution_summary.md"): response}
        
        # Fallback to regular parsing if it doesn't look like OpenHands output
        return self._parse_single_file_response(response)

    def _parse_aider_response(self, response: str) -> dict[pathlib.Path, str]:
        """Parse Aider response which contains actual file contents"""
        
        # Check for structured response format
        if response.startswith("AIDER_STRUCTURED_RESPONSE:"):
            try:
                import json
                
                # Extract JSON data
                lines = response.split('\n', 2)
                if len(lines) >= 2:
                    json_str = lines[0].replace("AIDER_STRUCTURED_RESPONSE:", "")
                    response_data = json.loads(json_str)
                    
                    # Extract files and execution result
                    files_created = response_data.get("files_created", {})
                    execution_result = response_data.get("execution_result", "")
                    success = response_data.get("success", False)
                    
                    result = {}
                    
                    # Process all files with their actual contents
                    for file_path, content in files_created.items():
                        if content and not content.startswith("<Could not read file") and not content.startswith("<Error reading file"):
                            # We have actual file content
                            result[pathlib.Path(file_path)] = content
                        else:
                            # File couldn't be read or is empty
                            result[pathlib.Path(file_path)] = f"# Could not read file: {file_path}\n# {content}"
                    
                    # Add execution summary
                    status = "SUCCESS" if success else "FAILED"
                    summary = f"# Aider Execution Summary - {status}\n\n{execution_result}"
                    if not success:
                        error = response_data.get("error", "")
                        if error:
                            summary += f"\n\nERROR: {error}"
                    
                    result[pathlib.Path("aider_execution_summary.md")] = summary
                    
                    self.logger.info(f"Parsed {len(files_created)} files from Aider response")
                    return result
                    
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                self.logger.warning(f"Failed to parse Aider structured response: {e}")
                # Fall through to basic parsing
        
        # Fallback parsing for unstructured responses
        if "Aider Development Task" in response:
            result = {}
            
            # Extract file names from summary
            files_pattern = re.compile(r"File: ([^\n]+)", re.MULTILINE)
            file_matches = files_pattern.findall(response)
            
            if file_matches:
                for file_path in file_matches:
                    file_path = file_path.strip()
                    result[pathlib.Path(file_path)] = f"# File created by Aider: {file_path}\n# Content not available in this response format"
            
            # Add execution summary
            result[pathlib.Path("aider_execution_summary.md")] = response
            return result
        
        # Ultimate fallback
        return self._parse_single_file_response(response)

    def _parse_multi_file_response(self, response: str) -> dict[pathlib.Path, str]:
        normal_file_paths = [
            pathlib.Path(s.strip()) for s in self.fp_pattern.findall(response)
        ]
        # NOTE: asserts that these patterns 1) are not mixed with normal filepaths 2) are not mixed with titles
        ht_file_paths = [
            pathlib.Path(s.strip()) for s in self.fp_ht_pattern.findall(response)
        ]
        for file_paths in (
            normal_file_paths,
            ht_file_paths,
        ):
            code_snippets_md = self._parse_md(response)
            code_snippets_code = self._parse_code(response)
            self.logger.info(f"Trying MD parsing")
            if len(file_paths) == len(code_snippets_md) and len(file_paths) > 0:
                return {fp: c for fp, c in zip(file_paths, code_snippets_md)}
            elif len(file_paths) == len(code_snippets_code) and len(file_paths) > 0:
                self.logger.warning(f"MD format not found, trying CODE format")
                # failsave code parsing in case some of them have md and some not
                codes = []
                for code in code_snippets_code:
                    md_parsed = self._parse_md(code)
                    if len(md_parsed) > 0:
                        codes.append(md_parsed[0])
                    else:
                        codes.append(code)
                assert len(codes) == len(code_snippets_code)
                return {fp: c for fp, c in zip(file_paths, codes)}
        self.logger.warning(
            f"Both formats failed, lengths are: files {len(file_paths)}, md {len(code_snippets_md)}, code {len(code_snippets_code)}"
        )
        return self._invalid(response)

    def _parse_single_file_response(self, response: str) -> dict[pathlib.Path, str]:
        assert self.env.code_filename is not None
        code_snippets_md = self._parse_md(response)
        code_snippets_code = self._parse_code(response)
        self.logger.info(f"Trying MD parsing")
        if len(code_snippets_md) > 0:
            return {pathlib.Path(self.env.code_filename): code_snippets_md[0]}
        elif len(code_snippets_code) > 0:
            self.logger.warning(f"MD format not found, trying CODE format")
            return {pathlib.Path(self.env.code_filename): code_snippets_code[0]}
        else:
            self.logger.warning(f"Both formats failed")
            return self._invalid(response)

    def parse_response(self, response: str, model: str = "") -> dict[pathlib.Path, str]:
        if model == "openhands":
            return self._parse_openhands_response(response)
        elif model == "aider":
            return self._parse_aider_response(response)
        elif self.env.is_multi_file:
            return self._parse_multi_file_response(response)
        else:
            return self._parse_single_file_response(response)