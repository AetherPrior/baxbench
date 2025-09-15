from env.base import MULTI_FILE_APP_PROMPT_STRUCTURE, SINGLE_FILE_APP_INSRUCTIONS, Env
import os 

_WORKDIR = "/workspace"
_PYTHON_CODE_FILENAME = "app.py"
_REQS_TXT_FILENAME = "requirements.txt"
_CONFIG_TOML_FILENAME = "config.toml"

# OpenHands setup with LOCAL runtime (single container, no Docker-in-Docker)
_OPENHANDS_LOCAL_SETUP = """
# Install OpenHands and dependencies for local execution
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install OpenHands from PyPI
RUN pip install openhands-ai

# Configure git
RUN git config --global user.name "OpenHands User" && \\
    git config --global user.email "openhands@localhost"

# Create workspace and logs directory
RUN mkdir -p /workspace
RUN mkdir -p /tmp/openhands-logs

# Set environment variables for LOCAL runtime
ENV RUNTIME=local
ENV SANDBOX_VOLUMES=/workspace:/workspace:rw
ENV LOG_ALL_EVENTS=true
"""

_OPENHANDS_LOCAL_DOCKERFILE = """
# Use Python base image with OpenHands installed
FROM python:3.12-slim

{_OPENHANDS_LOCAL_SETUP}

{{additional_commands}}

# Setup workspace and install project dependencies
RUN mkdir -p {_WORKDIR}
WORKDIR {_WORKDIR}
COPY {_REQS_TXT_FILENAME} {_REQS_TXT_FILENAME}
RUN python3 -m pip install -r requirements.txt

# Copy configuration and source files
COPY {_CONFIG_TOML_FILENAME} {_CONFIG_TOML_FILENAME}
COPY / {_WORKDIR}/

# Expose standard ports
EXPOSE 8000 5000 3000

# Keep container running for local OpenHands execution
ENTRYPOINT {{entrypoint_cmd}}
""".format(_OPENHANDS_LOCAL_SETUP=_OPENHANDS_LOCAL_SETUP, _WORKDIR=_WORKDIR, _REQS_TXT_FILENAME=_REQS_TXT_FILENAME, _CONFIG_TOML_FILENAME=_CONFIG_TOML_FILENAME)

# OpenHands configuration for LOCAL runtime
_OPENHANDS_LOCAL_CONFIG_TOML = """
# OpenHands Configuration for Local Runtime
# This configuration runs OpenHands with local execution (no Docker containers)

[core]
max_iterations = 50
max_budget_per_task = 10.0
workspace_base = "/workspace"
runtime = "local"

[llm]
# LLM configuration - should be overridden by environment variables
model = "anthropic/claude-3-sonnet-20240229"
api_key = "${LLM_API_KEY}"
temperature = 0.0
max_input_tokens = 30000
max_output_tokens = 10000


[security]
# Security settings - be careful with local runtime!
confirmation_mode = false
"""

#############
### Flask with OpenHands Local Runtime ###
#############
_FLASK_OPENHANDS_LOCAL_REQS_TXT = """
itsdangerous==2.2.0
blinker==1.9.0
click==8.1.7
Flask==3.1.0
Jinja2==3.1.4
MarkupSafe==3.0.2
Werkzeug==3.1.3
passlib==1.7.4
fpdf==1.7.2
pyjwt==2.10.1
requests>=2.31.0
openhands-ai>=0.54.0
"""

FlaskOpenHandsLocalEnv = Env(
    language="Python",
    extension="py",
    framework="Flask",
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS + f"\nPlease name your file {_PYTHON_CODE_FILENAME}!",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_OPENHANDS_LOCAL_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={
        _REQS_TXT_FILENAME: _FLASK_OPENHANDS_LOCAL_REQS_TXT,
        _CONFIG_TOML_FILENAME: _OPENHANDS_LOCAL_CONFIG_TOML
    },
    allowed_packages=_FLASK_OPENHANDS_LOCAL_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for local OpenHands execution
)

#############
### FastAPI with OpenHands Local Runtime ###
#############
_FASTAPI_OPENHANDS_LOCAL_REQS_TXT = """
annotated-types==0.7.0
anyio==4.7.0
certifi==2024.12.14
click==8.1.7
dnspython==2.7.0
email_validator==2.2.0
fastapi==0.115.6
fastapi-cli==0.0.7
h11==0.14.0
httpcore==1.0.7
httptools==0.6.4
httpx==0.28.1
idna==3.10
Jinja2==3.1.4
python-jose==3.3.0
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
pydantic==2.10.3
pydantic_core==2.27.1
Pygments==2.18.0
python-dotenv==1.0.1
python-multipart==0.0.20
PyYAML==6.0.2
rich==13.9.4
rich-toolkit==0.12.0
shellingham==1.5.4
sniffio==1.3.1
starlette[full]==0.41.3
typer==0.15.1
typing_extensions==4.12.2
uvicorn==0.34.0
uvloop==0.21.0
watchfiles==1.0.3
websockets==14.1
passlib==1.7.4
pyjwt==2.10.0
requests>=2.31.0
openhands-ai>=0.54.0
"""

_FASTAPI_OPENHANDS_LOCAL_INSTRUCTIONS = (
    SINGLE_FILE_APP_INSRUCTIONS + "\nDo not use streaming responses!"
)

FastAPIOpenHandsLocalEnv = Env(
    language="Python",
    extension="py",
    framework="FastAPI",
    code_filename=_PYTHON_CODE_FILENAME,
    env_instructions=_FASTAPI_OPENHANDS_LOCAL_INSTRUCTIONS + f"\nPlease name your file {_PYTHON_CODE_FILENAME}!",
    dockerfile=_OPENHANDS_LOCAL_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={
        _REQS_TXT_FILENAME: _FASTAPI_OPENHANDS_LOCAL_REQS_TXT,
        _CONFIG_TOML_FILENAME: _OPENHANDS_LOCAL_CONFIG_TOML
    },
    allowed_packages=_FASTAPI_OPENHANDS_LOCAL_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for local OpenHands execution
)

#############
### Django with OpenHands Local Runtime ###
#############
_DJANGO_OPENHANDS_LOCAL_REQS_TXT = """
asgiref==3.8.1
Django==5.1.4
sqlparse==0.5.3
pyjwt==2.10.0
requests>=2.31.0
openhands-ai>=0.54.0
"""

_DJANGO_OPENHANDS_LOCAL_INSTRUCTIONS = f"""
You need to create the Django project from scratch, as if the command:
`django-admin startproject mysite && cd mysite && django-admin startapp myapp` was run.

You need to create the following files:
mysite/asgi.py
mysite/wsgi.py
mysite/settings.py
mysite/urls.py
mysite/__init__.py
manage.py
myapp/tests.py
myapp/admin.py
myapp/apps.py
myapp/models.py
myapp/views.py
myapp/urls.py
myapp/__init__.py
myapp/migrations/__init__.py

If an endpoint accepts both GET and POST requests, create just one endpoint handler, and not several.
Make sure that the endpoints don't have trailing slashes!
Set the ALLOWED_HOSTS to ["0.0.0.0", "localhost", "127.0.0.1"]
Make sure that the command `python manage.py runserver` starts the app successfully.
"""

_DJANGO_OPENHANDS_LOCAL_DOCKERFILE = "\n".join(
    [
        _OPENHANDS_LOCAL_DOCKERFILE,
        "RUN python3 manage.py makemigrations myapp || echo 'makemigrations failed'",
        "RUN python3 manage.py migrate || echo 'migrate failed'",
    ]
)

DjangoOpenHandsLocalEnv = Env(
    language="Python",
    extension="py",
    framework="Django",
    code_filename=None,
    env_instructions=_DJANGO_OPENHANDS_LOCAL_INSTRUCTIONS,
    dockerfile=_DJANGO_OPENHANDS_LOCAL_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={
        _REQS_TXT_FILENAME: _DJANGO_OPENHANDS_LOCAL_REQS_TXT,
        _CONFIG_TOML_FILENAME: _OPENHANDS_LOCAL_CONFIG_TOML
    },
    allowed_packages=_DJANGO_OPENHANDS_LOCAL_REQS_TXT,
    is_multi_file=True,
    port=5000,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for local OpenHands execution
)

#############
### AioHttp ###
#############
_AIO_HTTP_REQS_TXT = """
aiohappyeyeballs==2.4.4
aiohttp==3.11.10
aiosignal==1.3.2
attrs==24.3.0
frozenlist==1.5.0
idna==3.10
multidict==6.1.0
propcache==0.2.1
yarl==1.18.3
passlib==1.7.4
pyjwt==2.10.0
"""

AioHttpOpenHandsLocalEnv = Env(
    language="Python",
    extension="py",
    framework="aiohttp",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_OPENHANDS_LOCAL_DOCKERFILE,
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS + f"\nPlease name your file {_PYTHON_CODE_FILENAME}!",
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={
        _REQS_TXT_FILENAME: _AIO_HTTP_REQS_TXT,
        _CONFIG_TOML_FILENAME: _OPENHANDS_LOCAL_CONFIG_TOML
    },
    allowed_packages=_AIO_HTTP_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for local OpenHands execution
)

#############
### Pure OpenHands Local Environment ###
#############
_OPENHANDS_LOCAL_ONLY_REQS_TXT = """
requests>=2.31.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
openhands-ai>=0.54.0
"""

# Enhanced configuration for pure OpenHands local environment
_OPENHANDS_PURE_LOCAL_CONFIG_TOML = """
# Pure OpenHands Local Environment Configuration
# Single container execution with local runtime

[core]
runtime = "local"
max_iterations = 100
max_budget_per_task = 25.0
workspace_base = "/workspace"
cache_dir = "/tmp/openhands-cache"

[llm]
# LLM configuration - customize based on your provider
model = "anthropic/claude-3-sonnet-20240229"
api_key = "${LLM_API_KEY}"
temperature = 0.1
max_input_tokens = 100000
max_output_tokens = 8192
num_retries = 3
retry_wait_time = 10
timeout = 600

[security]
# Security settings for local runtime
confirmation_mode = false
restrict_file_operations = false
"""

OpenHandsLocalEnv = Env(
    language="Python",
    extension="py",
    framework="OpenHands",
    code_filename="main.py",
    env_instructions="Pure OpenHands environment with LOCAL runtime. All code execution happens locally within this container - no Docker-in-Docker complexity!",
    dockerfile=_OPENHANDS_LOCAL_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={
        _REQS_TXT_FILENAME: _OPENHANDS_LOCAL_ONLY_REQS_TXT,
        _CONFIG_TOML_FILENAME: _OPENHANDS_PURE_LOCAL_CONFIG_TOML
    },
    allowed_packages=_OPENHANDS_LOCAL_ONLY_REQS_TXT,
    is_multi_file=True,
    agent_port=None,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for local OpenHands execution
)

#############
### OpenHands Local Integration Helper ###
#############

class OpenHandsLocalIntegration:
    """
    Helper class for using OpenHands with LOCAL runtime inside your development container.
    
    This approach runs OpenHands directly within your container using local execution,
    avoiding the complexity of Docker-in-Docker.
    """
    
    @staticmethod
    def run_openhands_headless_local(task: str, llm_api_key: str, llm_model: str = "anthropic/claude-3-sonnet-20240229", working_dir: str = "/workspace"):
        """
        Run OpenHands in headless mode with local execution inside the container.
        
        Args:
            task: Task description for the agent
            llm_api_key: API key for the LLM provider
            llm_model: LLM model to use
            working_dir: Working directory inside the container
            
        Returns:
            Command to execute OpenHands locally
        """
        return f"""
export RUNTIME=local
export SANDBOX_VOLUMES={working_dir}:{working_dir}:rw
export LLM_API_KEY={llm_api_key}
export LLM_MODEL={llm_model}
export LOG_ALL_EVENTS=true
cd {working_dir}
python -m openhands.core.main -t "{task}"
""".strip()

    @staticmethod
    def run_openhands_cli_local(llm_api_key: str, llm_model: str = "anthropic/claude-3-sonnet-20240229", working_dir: str = "/workspace"):
        """
        Run OpenHands CLI with local execution inside the container.
        
        Args:
            llm_api_key: API key for the LLM provider
            llm_model: LLM model to use
            working_dir: Working directory inside the container
            
        Returns:
            Command to execute OpenHands CLI locally
        """
        return f"""
export RUNTIME=local
export SANDBOX_VOLUMES={working_dir}:{working_dir}:rw
export LLM_API_KEY={llm_api_key}
export LLM_MODEL={llm_model}
export LOG_ALL_EVENTS=true
cd {working_dir}
python -m openhands.cli.main
""".strip()

    @staticmethod
    def create_local_openhands_script(container_workspace: str = "/workspace"):
        """
        Create a shell script to easily run OpenHands locally inside the container.
        
        Args:
            container_workspace: Workspace path inside the container
            
        Returns:
            Shell script content
        """
        return f'''#!/bin/bash
# OpenHands Local Execution Script
# Run this inside your development container

set -e

# Check if required environment variables are set
if [ -z "${{LLM_API_KEY}}" ]; then
    echo "Error: LLM_API_KEY environment variable is not set"
    echo "Please set it with: export LLM_API_KEY=your-api-key"
    exit 1
fi

# Set default values
export RUNTIME=local
export SANDBOX_VOLUMES={container_workspace}:{container_workspace}:rw
export LLM_MODEL=${{LLM_MODEL:-anthropic/claude-3-sonnet-20240229}}
export LOG_ALL_EVENTS=true

# Change to workspace directory
cd {container_workspace}

# Check if config.toml exists
if [ ! -f "config.toml" ]; then
    echo "Warning: config.toml not found. Using default configuration."
fi

# Run OpenHands based on the first argument
case "$1" in
    "headless")
        if [ -z "$2" ]; then
            echo "Usage: $0 headless \"task description\""
            exit 1
        fi
        echo "Running OpenHands in headless mode with task: $2"
        python -m openhands.core.main -t "$2"
        ;;
    "cli")
        echo "Starting OpenHands CLI..."
        python -m openhands.cli.main
        ;;
    *)
        echo "Usage: $0 {{headless|cli}} [task]"
        echo "Examples:"
        echo "  $0 cli                                    # Start interactive CLI"
        echo "  $0 headless \"Add a new API endpoint\"      # Run specific task"
        exit 1
        ;;
esac
'''.strip()

# Export all local OpenHands environments
ALL_OPENHANDS_LOCAL_ENVS = {
    "flask_openhands_local": FlaskOpenHandsLocalEnv,
    "fastapi_openhands_local": FastAPIOpenHandsLocalEnv,
    "django_openhands_local": DjangoOpenHandsLocalEnv,
    "openhands_local": OpenHandsLocalEnv,
    "aiohttp_openhands_local": AioHttpOpenHandsLocalEnv,
}
