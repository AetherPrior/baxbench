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
from docker.models.containers import Container
import docker

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
    openai_org_id = "OPENAI_ORG_ID"


class OpenHandsClient:
    """Client for interacting with OpenHands using local runtime via docker-py"""
    
    def __init__(
        self, 
        container_name: str = None, 
        image_name: str = "python:3.12-slim",
        llm_model: str = "anthropic/claude-3-sonnet-20240229"
    ):
        self.container_name = container_name or f"openhands_container_{int(time.time())}"
        self.image_name = image_name
        self.llm_model = llm_model
        self.docker_client = docker.from_env()
        self.container = None
        self.workspace_volume = None
        self._openhands_installed = False

    def health_check(self) -> bool:
        """Check if Docker daemon is running and we can access it"""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False

    def _create_workspace_volume(self):
        """Create a Docker volume for the workspace"""
        try:
            self.workspace_volume = self.docker_client.volumes.create(
                name=f"openhands_workspace_{int(time.time())}"
            )
            return self.workspace_volume
        except Exception as e:
            raise Exception(f"Failed to create workspace volume: {e}")

    def _get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for OpenHands local runtime"""
        env_vars = {
            "RUNTIME": "local",
            "SANDBOX_VOLUMES": "/workspace:/workspace:rw",
            "LOG_ALL_EVENTS": "true",
            "LLM_MODEL": self.llm_model,
            "PYTHONPATH": "/workspace",
            "PYTHONUNBUFFERED": "1"
        }
        
        # Add API keys based on the model
        if "anthropic" in self.llm_model.lower() or "claude" in self.llm_model.lower():
            api_key = os.environ.get(KeyLocs.anthropic_key.value)
            if api_key:
                env_vars["LLM_API_KEY"] = api_key
                env_vars["ANTHROPIC_API_KEY"] = api_key
        elif "openai" in self.llm_model.lower() or "gpt" in self.llm_model.lower():
            api_key = os.environ.get(KeyLocs.openai_key.value)
            if api_key:
                env_vars["LLM_API_KEY"] = api_key
                env_vars["OPENAI_API_KEY"] = api_key
            org_id = os.environ.get(KeyLocs.openai_org_id.value)
            if org_id:
                env_vars["OPENAI_ORG_ID"] = org_id
        else:
            # Generic fallback
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get(KeyLocs.openai_key.value)
            if api_key:
                env_vars["LLM_API_KEY"] = api_key

        return env_vars

    def _install_openhands(self):
        """Install OpenHands in the container if not already installed"""
        if self._openhands_installed:
            return
            
        try:
            # Update package list and install basic tools
            self.container.exec_run([
                "apt-get", "update"
            ])
            
            self.container.exec_run([
                "apt-get", "install", "-y", "git", "curl", "build-essential"
            ])
            
            # Install OpenHands
            result = self.container.exec_run([
                "pip", "install", "openhands-ai>=0.54.0"
            ])
            
            if result.exit_code != 0:
                raise Exception(f"Failed to install OpenHands: {result.output.decode()}")
            
            # Configure git
            self.container.exec_run([
                "git", "config", "--global", "user.name", "OpenHands User"
            ])
            self.container.exec_run([
                "git", "config", "--global", "user.email", "openhands@localhost"
            ])
            
            self._openhands_installed = True
            
        except Exception as e:
            raise Exception(f"Failed to install OpenHands: {e}")

    def _start_container(self) -> bool:
        """Start the OpenHands container if not already running"""
        try:
            if self.container is None:
                # Check if container already exists
                try:
                    self.container = self.docker_client.containers.get(self.container_name)
                    if self.container.status != 'running':
                        self.container.start()
                except docker.errors.NotFound:
                    # Container doesn't exist, create it
                    if self.workspace_volume is None:
                        self._create_workspace_volume()
                    
                    env_vars = self._get_env_vars()
                    
                    # Create and start container
                    self.container = self.docker_client.containers.run(
                        self.image_name,
                        name=self.container_name,
                        detach=True,
                        tty=True,
                        working_dir="/workspace",
                        volumes={
                            self.workspace_volume.name: {"bind": "/workspace", "mode": "rw"}
                        },
                        environment=env_vars,
                        command="tail -f /dev/null",  # Keep container alive
                        remove=False
                    )
            
            if self.container is None:
                raise Exception("Container is not running")
            
            # Install OpenHands if needed
            self._install_openhands()
            
            return True
        except Exception as e:
            raise Exception(f"Failed to start container: {e}")

    def _initialize_workspace(self):
        """Initialize the workspace with basic setup"""
        try:
            # Create config.toml for OpenHands if it doesn't exist
            config_content = f"""
[core]
max_iterations = 50
max_budget_per_task = 10.0
workspace_base = "/workspace"
runtime = "local"

[llm]
model = "{self.llm_model}"
temperature = 0.0
max_input_tokens = 30000
max_output_tokens = 10000

[security]
confirmation_mode = false
"""
            # Check if config.toml already exists
            result = self.container.exec_run([
                "test", "-f", "/workspace/config.toml"
            ])
            if result.exit_code == 0:
                return  # Already exists
            
            # Write config to container
            self.container.exec_run([
                "sh", "-c", f"echo '{config_content}' > /workspace/config.toml"
            ])
            
            # Initialize git repo if it doesn't exist
            result = self.container.exec_run([
                "git", "status"
            ], workdir="/workspace")
            
            if result.exit_code != 0:
                self.container.exec_run([
                    "git", "init"
                ], workdir="/workspace")
                
        except Exception as e:
            raise Exception(f"Failed to initialize workspace: {e}")

    def _collect_files(self) -> Dict[str, str]:
        """Collect all files from the workspace"""
        files = {}
        
        try:
            # List all files recursively, excluding common patterns
            result = self.container.exec_run([
                "find", "/workspace", "-type", "f", 
                "!", "-path", "*/.*",  # Exclude hidden files and directories
                "!", "-name", "*.log",
                "!", "-name", "*.tmp",
                "!", "-path", "*/__pycache__/*",
                "!", "-path", "*/node_modules/*",
                "!", "-name", "Dockerfile",
                "!", "-name", "config.toml",  # Exclude our config file
            ])
            
            if result.exit_code == 0:
                file_paths = result.output.decode('utf-8').strip().split('\n')
                
                for file_path in file_paths:
                    if file_path and file_path != '/workspace':
                        try:
                            # Read file content
                            cat_result = self.container.exec_run(["cat", file_path])
                            if cat_result.exit_code == 0:
                                # Convert absolute path to relative
                                relative_path = file_path.replace('/workspace/', '', 1) if file_path.startswith('/workspace/') else file_path
                                if relative_path:
                                    files[relative_path] = cat_result.output.decode('utf-8', errors='ignore')
                        except Exception as e:
                            # Log but don't fail for individual file read errors
                            print(f"Warning: Could not read file {file_path}: {e}")
                            
        except Exception as e:
            raise Exception(f"Failed to collect files: {e}")
            
        return files

    def execute_task(
        self,
        task: str,
        workspace_dir: str = "/workspace",
        max_iterations: int = 50,
        timeout: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a task using OpenHands with local runtime"""
        
        try:
            # Start container and initialize
            self._start_container()
            self._initialize_workspace()
            
            # Prepare OpenHands command for headless execution
            openhands_cmd = [
                "python", "-m", "openhands.core.main",
                "-t", task,
                "--max-iterations", str(max_iterations),
                "--config-file", "/workspace/config.toml"
            ]
            
            # Execute OpenHands command
            result = self.container.exec_run(
                openhands_cmd,
                workdir="/workspace",
                environment=self._get_env_vars()
            )
            
            # Collect all files after execution
            files_created = self._collect_files()
            
            # Parse the output to extract useful information
            output = result.output.decode('utf-8', errors='ignore')
            
            # Prepare response
            if result.exit_code == 0:
                return {
                    "success": True,
                    "result": output,
                    "files_created": files_created,
                    "error": None,
                    "exit_code": result.exit_code
                }
            else:
                return {
                    "success": False,
                    "result": output,
                    "files_created": files_created,
                    "error": f"OpenHands command failed with exit code {result.exit_code}",
                    "exit_code": result.exit_code
                }
                
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "files_created": {},
                "error": str(e),
                "exit_code": -1
            }

    def execute_cli_session(
        self,
        workspace_dir: str = "/workspace",
        timeout: int = 60,
        **kwargs
    ) -> Dict[str, Any]:
        """Start an interactive OpenHands CLI session (returns immediately)"""
        
        try:
            # Start container and initialize
            self._start_container()
            self._initialize_workspace()
            
            # Note: CLI mode is interactive, so we can't easily capture its output
            # This method prepares the environment for CLI usage
            
            return {
                "success": True,
                "result": f"OpenHands CLI environment ready. Connect to container '{self.container_name}' and run: python -m openhands.cli.main",
                "container_name": self.container_name,
                "workspace_volume": self.workspace_volume.name if self.workspace_volume else None,
                "error": None
            }
                
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": str(e)
            }

    def get_container_shell(self) -> str:
        """Get command to connect to the container shell"""
        if self.container:
            return f"docker exec -it {self.container_name} /bin/bash"
        return "Container not running"

    def cleanup(self):
        """Clean up container and volume"""
        try:
            if self.container:
                self.container.stop()
                self.container.remove()
            if self.workspace_volume:
                self.workspace_volume.remove()
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.cleanup()
        pass


class AiderClient:
    """Client for interacting with Aider using docker-py directly"""

    def __init__(self, container_name: str = None, image_name: str = "paulgauthier/aider"):
        self.container_name = container_name or f"aider_container_{int(time.time())}"
        self.image_name = image_name
        self.docker_client = docker.from_env()
        self.container = None
        self.workspace_volume = None

    def health_check(self) -> bool:
        """Check if Docker daemon is running and we can access it"""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False

    def _create_workspace_volume(self):
        """Create a Docker volume for the workspace"""
        try:
            self.workspace_volume = self.docker_client.volumes.create(
                name=f"aider_workspace_{int(time.time())}"
            )
            return self.workspace_volume
        except Exception as e:
            raise Exception(f"Failed to create workspace volume: {e}")

    def _start_container(self) -> bool:
        """Start the Aider container if not already running"""

        try:
            if self.container is None:
                # Check if container already exists
                try:
                    self.container = self.docker_client.containers.get(self.container_name)
                    if self.container.status != 'running':
                        self.container.start()
                except docker.errors.NotFound:
                    # Container doesn't exist, create it
                    if self.workspace_volume is None:
                        self._create_workspace_volume()
                    
                    
                    # Create and start container
                    self.container = self.docker_client.containers.run(
                        self.image_name,
                        name=self.container_name,
                        detach=True,
                        tty=True,
                        working_dir="/app",
                        volumes={
                            self.workspace_volume.name: {"bind": "/app", "mode": "rw"}
                        },
                        environment=env_vars,
                        command="tail -f /dev/null",  # Keep container alive
                        remove=False
                    )
            
            # Add environment variable to running container
            if self.container is None:
                raise Exception("Container is not running")

            return True
        except Exception as e:
            raise Exception(f"Failed to start container: {e}")

    def _initialize_git_repo(self):
        """Initialize git repository in the workspace"""
        try:
            # Configure git user
            self.container.exec_run(
                ["git", "config", "--global", "user.name", "Aider User"],
                workdir="/app"
            )
            self.container.exec_run(
                ["git", "config", "--global", "user.email", "aider@localhost"],
                workdir="/app"
            )
            
            # Initialize repo if it doesn't exist
            result = self.container.exec_run(
                ["git", "status"],
                workdir="/app"
            )
            if result.exit_code != 0:
                self.container.exec_run(
                    ["git", "init"],
                    workdir="/app"
                )
        except Exception as e:
            raise Exception(f"Failed to initialize git repo: {e}")

    def _collect_files(self) -> Dict[str, str]:
        """Collect all files from the workspace"""
        files = {}
        
        try:
            # List all files recursively, excluding common patterns
            result = self.container.exec_run([
                "find", "/app", "-type", "f", 
                "!", "-path", "*/.*",  # Exclude hidden files and directories
                "!", "-name", "*.log",
                "!", "-name", "*.tmp",
                "!", "-path", "*/__pycache__/*",
                "!", "-path", "*/node_modules/*",
                "!", "-name", "Dockerfile",
            ])
            
            if result.exit_code == 0:
                file_paths = result.output.decode('utf-8').strip().split('\n')
                
                for file_path in file_paths:
                    if file_path and file_path != '/app':
                        try:
                            # Read file content
                            cat_result = self.container.exec_run(["cat", file_path])
                            if cat_result.exit_code == 0:
                                # Convert absolute path to relative
                                relative_path = file_path.replace('/app/', '', 1) if file_path.startswith('/app/') else file_path
                                if relative_path:
                                    files[relative_path] = cat_result.output.decode('utf-8', errors='ignore')
                        except Exception as e:
                            # Log but don't fail for individual file read errors
                            print(f"Warning: Could not read file {file_path}: {e}")
                            
        except Exception as e:
            raise Exception(f"Failed to collect files: {e}")
            
        return files

    def execute_task(
        self,
        task: str,
        workspace_dir: str = "/app",
        model: str = "gpt-5-mini-2025-08-07",
        timeout: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a task using Aider via docker-py"""
        
        try:
            # Start container
            self._start_container()
            self._initialize_git_repo()
            
            # Prepare aider command
            aider_cmd = [
                "aider",
                "--model", model,
                "--yes",  # Auto-confirm changes
                "--no-show-model-warnings",
                "--cache-prompts",  # Enable prompt caching
                "--no-stream",  # Disable streaming
                "--disable-playwright",  # Disable playwright
                "--no-git",  # Disable git integration to avoid commit issues
                "--reasoning-effort", "low",  # Reasoning effort
                "--message", task,
                "--api-key", f"openai={os.environ.get(KeyLocs.openai_key.value, '')}"
            ]
            # Execute aider command
            result = self.container.exec_run(
                aider_cmd,
                workdir="/app",
                environment={
                    "PYTHONPATH": "/app",
                }
            )
            # Collect all files after execution
            files_created = self._collect_files()
            
            # Prepare response
            if result.exit_code == 0:
                return {
                    "success": True,
                    "result": result.output.decode('utf-8', errors='ignore'),
                    "files_created": files_created,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": result.output.decode('utf-8', errors='ignore'),
                    "files_created": files_created,
                    "error": f"Aider command failed with exit code {result.exit_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "files_created": {},
                "error": str(e)
            }

    def cleanup(self):
        """Clean up container and volume"""
        try:
            if self.container:
                self.container.stop()
                self.container.remove()
            if self.workspace_volume:
                self.workspace_volume.remove()
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # self.cleanup()
        
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
        container: Optional[Container] = None,  # Docker container for Aider
        llm_model: Optional[str] = None,  # LLM model to use for agents that support it
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

        self.llm_model = llm_model
        self.prompt = self.scenario.build_prompt(
            self.env, self.spec_type, self.safety_prompt
        )

        self.container = container  # Docker container for Aider

    def prompt_openhands(self, logger: logging.Logger) -> list[str]:
        """Execute task using OpenHands local runtime via docker-py"""
        
        # Check if Docker is available
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except Exception as e:
            logger.error("Docker is not available or not accessible")
            raise Exception(f"Docker is not available: {e}")

        # Use OpenHandsClient with context manager for automatic cleanup
        with OpenHandsClient(
            llm_model=getattr(self, 'llm_model', 'anthropic/claude-3-sonnet-20240229'),
            container_name=self.container.name
        ) as client:
            if not client.health_check():
                logger.error("Docker daemon is not running or not accessible")
                raise Exception("Docker daemon is not running. Make sure Docker is installed and running.")

            try:
                # Construct the comprehensive task for OpenHands
                openhands_task = f"""
I am a full-stack developer working on a {self.env.framework} application. 

DEVELOPMENT TASK:
{self.prompt}

ENVIRONMENT DETAILS:
- Framework: {self.env.framework}
- Working Directory: /workspace
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
                
                logger.info("Executing task with OpenHands via docker-py...")
                logger.info(f"Model: {client.llm_model}")
                
                # Execute the task using docker-py
                result = client.execute_task(
                    task=openhands_task,
                    workspace_dir="/workspace",
                    max_iterations=getattr(self, 'openhands_max_iterations', 50),
                    timeout=getattr(self, 'openhands_timeout', 300)
                )
                
                # The result now contains actual file contents in files_created
                if result.get("success"):
                    logger.info("OpenHands task completed successfully")
                    
                    files_created = result.get("files_created", {})
                    execution_result = result.get("result", "Task completed")
                    
                    # Create response that includes the actual file data
                    import json
                    response_data = {
                        "type": "openhands_response",
                        "success": True,
                        "execution_result": execution_result,
                        "files_created": files_created,
                        "exit_code": result.get("exit_code", 0),
                        "container_name": client.container_name,
                        "runtime": "local"
                    }
                    
                    # Create summary for logging
                    response_parts = [
                        "OpenHands Development Task Completed (Local Runtime)",
                        "=" * 50,
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
                    logger.info(f"OpenHands created {len(files_created)} files using local runtime")
                    
                    # Return response with embedded file data
                    full_response = f"OPENHANDS_STRUCTURED_RESPONSE:{json.dumps(response_data)}\n\n{summary_text}"
                    return [full_response]
                    
                else:
                    logger.error("OpenHands task failed")
                    error_msg = result.get("error", "Unknown error")
                    task_output = result.get("result", "No output")
                    files_created = result.get("files_created", {})
                    
                    import json
                    response_data = {
                        "type": "openhands_response", 
                        "success": False,
                        "execution_result": task_output,
                        "files_created": files_created,
                        "error": error_msg,
                        "exit_code": result.get("exit_code", -1),
                        "container_name": client.container_name,
                        "runtime": "local"
                    }
                    
                    response_parts = [
                        "OpenHands Development Task Failed (Local Runtime)",
                        "=" * 50,
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
                    full_response = f"OPENHANDS_STRUCTURED_RESPONSE:{json.dumps(response_data)}\n\n{summary_text}"
                    return [full_response]
                    
            except Exception as e:
                logger.error(f"OpenHands interaction failed: {e}")
                raise e

    def prompt_aider(self, logger: logging.Logger) -> list[str]:
        """Execute task using Aider via docker-py"""
        
        # Check if Docker is available
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except Exception as e:
            logger.error("Docker is not available or not accessible")
            raise Exception(f"Docker is not available: {e}")

        # Use AiderClient with context manager for automatic cleanup
        with AiderClient(container_name=self.container.name) as client:
            if not client.health_check():
                logger.error("Docker daemon is not running or not accessible")
                raise Exception("Docker daemon is not running. Make sure Docker is installed and running.")

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
                
                logger.info("Executing task with Aider via docker-py...")
                # Execute the task using docker-py
                result = client.execute_task(
                    task=aider_task,
                    workspace_dir="/app",
                    model=getattr(self, 'llm_model', 'gpt-5-mini-2025-08-07'),
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

    def _parse_aider_response(self, response: str) -> dict[pathlib.Path, str]:
        """Parse Aider response which contains actual file contents from docker-py execution"""
        
        # Check for structured response format from docker-py integration
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
                    
                    # Process all files with their actual contents from docker-py
                    for file_path, content in files_created.items():
                        if content and not content.startswith("<Could not read file") and not content.startswith("<Error reading file"):
                            # We have actual file content from the docker container
                            result[pathlib.Path(file_path)] = content
                        else:
                            # File couldn't be read or is empty
                            result[pathlib.Path(file_path)] = f"# Could not read file: {file_path}\n# {content}"
                    
                    # Add execution summary
                    status = "SUCCESS" if success else "FAILED"
                    execution = f"# Aider Execution - {status}\n\n{execution_result}"
                    if not success:
                        error = response_data.get("error", "")
                        if error:
                            execution += f"\n\nERROR: {error}"

                    result[pathlib.Path("aider_execution_summary.md")] = execution

                    self.logger.info(f"Parsed {len(files_created)} files from docker-py Aider response")
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

    def _parse_openhands_response(self, response: str) -> dict[pathlib.Path, str]:
        """Parse OpenHands response which contains actual file contents from docker-py execution"""
        
        # Check for structured response format from docker-py integration
        if response.startswith("OPENHANDS_STRUCTURED_RESPONSE:"):
            try:
                import json
                
                # Extract JSON data
                lines = response.split('\n', 2)
                if len(lines) >= 2:
                    json_str = lines[0].replace("OPENHANDS_STRUCTURED_RESPONSE:", "")
                    response_data = json.loads(json_str)
                    
                    # Extract files and execution result
                    files_created = response_data.get("files_created", {})
                    execution_result = response_data.get("execution_result", "")
                    success = response_data.get("success", False)
                    exit_code = response_data.get("exit_code", 0)
                    container_name = response_data.get("container_name", "unknown")
                    runtime = response_data.get("runtime", "local")
                    
                    result = {}
                    
                    # Process all files with their actual contents from docker-py
                    for file_path, content in files_created.items():
                        if content and not content.startswith("<Could not read file") and not content.startswith("<Error reading file"):
                            # We have actual file content from the docker container
                            result[pathlib.Path(file_path)] = content
                        else:
                            # File couldn't be read or is empty
                            result[pathlib.Path(file_path)] = f"# Could not read file: {file_path}\n# {content}"
                    
                    # Add execution summary
                    status = "SUCCESS" if success else "FAILED"
                    summary = f"# OpenHands Execution Summary - {status}\n\n"
                    summary += f"Runtime: {runtime}\n"
                    summary += f"Container: {container_name}\n"
                    summary += f"Exit Code: {exit_code}\n\n"
                    summary += f"Execution Result:\n{execution_result}"
                    
                    if not success:
                        error = response_data.get("error", "")
                        if error:
                            summary += f"\n\nERROR: {error}"
                    
                    result[pathlib.Path("openhands_execution_summary.md")] = summary
                    
                    self.logger.info(f"Parsed {len(files_created)} files from docker-py OpenHands response")
                    return result
                    
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                self.logger.warning(f"Failed to parse OpenHands structured response: {e}")
                # Fall through to basic parsing
    
        # Fallback parsing for unstructured responses
        if "OpenHands Development Task" in response:
            result = {}
            
            # Extract file names from summary
            files_pattern = re.compile(r"File: ([^\n]+)", re.MULTILINE)
            file_matches = files_pattern.findall(response)
            
            if file_matches:
                for file_path in file_matches:
                    file_path = file_path.strip()
                    result[pathlib.Path(file_path)] = f"# File created by OpenHands: {file_path}\n# Content not available in this response format"
            
            # Add execution summary
            result[pathlib.Path("openhands_execution_summary.md")] = response
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