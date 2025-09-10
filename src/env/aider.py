from env.base import MULTI_FILE_APP_PROMPT_STRUCTURE, SINGLE_FILE_APP_INSRUCTIONS, Env
import os 

_WORKDIR = "/app"
_PYTHON_CODE_FILENAME = "app.py"
_REQS_TXT_FILENAME = "requirements.txt"

# Aider setup commands for headless mode (ubuntu)
_AIDER_SETUP = """
# Install git (required for aider)
# RUN apt-get install -y \\
#     git \\
#     curl \\
#     && rm -rf /var/lib/apt/lists/*

# Configure git with default user (required for aider)
RUN git config --global user.name "Aider User" && \\
    git config --global user.email "aider@localhost"

# Create aider directories
RUN mkdir -p /app/workspace
RUN mkdir -p /app/logs

# Create Aider API server script
RUN echo '#!/usr/bin/env python3\\n\\
import os\\n\\
import sys\\n\\
import json\\n\\
import asyncio\\n\\
import logging\\n\\
import subprocess\\n\\
import tempfile\\n\\
import shutil\\n\\
from pathlib import Path\\n\\
from typing import Dict, Any, Optional\\n\\
from fastapi import FastAPI, HTTPException\\n\\
from fastapi.responses import JSONResponse\\n\\
from pydantic import BaseModel\\n\\
import uvicorn\\n\\
\\n\\
# Configure logging\\n\\
logging.basicConfig(level=logging.INFO)\\n\\
logger = logging.getLogger(__name__)\\n\\
\\n\\
app = FastAPI()\\n\\
\\n\\
# Define files and directories to exclude\\n\\
EXCLUDED_PATTERNS = {{\\n\\
    # Dotfiles and dot directories\\n\\
    ".git", ".gitignore", ".gitmodules", ".gitattributes",\\n\\
    ".cache", ".aider", ".vscode", ".idea", ".vs",\\n\\
    ".env", ".env.local", ".env.development", ".env.production",\\n\\
    ".DS_Store", ".directory",\\n\\
    \\n\\
    # Docker and container files\\n\\
    "Dockerfile", "Dockerfile.dev", "Dockerfile.prod",\\n\\
    "docker-compose.yml", "docker-compose.yaml",\\n\\
    "docker-compose.dev.yml", "docker-compose.prod.yml",\\n\\
    ".dockerignore",\\n\\
    \\n\\
    # Package manager files\\n\\
    "node_modules", ".npm", ".yarn", "yarn.lock", "package-lock.json",\\n\\
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox",\\n\\
    "venv", ".venv", "env", ".env", "virtualenv",\\n\\
    \\n\\
    # Build and dist directories\\n\\
    "dist", "build", ".next", ".nuxt", "out",\\n\\
    "target", "bin", "obj",\\n\\
    \\n\\
    # Log and temporary files\\n\\
    "logs", "*.log", ".log", "tmp", "temp", ".tmp", ".temp",\\n\\
    \\n\\
    # IDE and editor files\\n\\
    ".sublime-project", ".sublime-workspace",\\n\\
    "*.swp", "*.swo", "*~",\\n\\
    \\n\\
    # OS specific\\n\\
    "Thumbs.db", "ehthumbs.db", "Desktop.ini"\\n\\
}}\\n\\
\\n\\
def should_exclude_file(file_path: Path, base_path: Path) -> bool:\\n\\
    # Check if a file should be excluded based on patterns.\\n\\
    relative_path = file_path.relative_to(base_path)\\n\\
    path_str = str(relative_path)\\n\\
    \\n\\
    # Check if any part of the path starts with a dot\\n\\
    for part in relative_path.parts:\\n\\
        if part.startswith(".") or part in EXCLUDED_PATTERNS:\\n\\
            return True\\n\\
    \\n\\
    # Check full filename\\n\\
    filename = file_path.name\\n\\
    if filename in EXCLUDED_PATTERNS or filename.startswith("."):\\n\\
        return True\\n\\
    \\n\\
    # Check for wildcard patterns\\n\\
    if filename.endswith(".log") or filename.endswith(".swp") or filename.endswith(".swo"):\\n\\
        return True\\n\\
    \\n\\
    # Check if path contains excluded directories\\n\\
    for excluded in EXCLUDED_PATTERNS:\\n\\
        if excluded in path_str.split("/"):\\n\\
            return True\\n\\
    \\n\\
    return False\\n\\
\\n\\
class TaskRequest(BaseModel):\\n\\
    task: str\\n\\
    workspace_dir: Optional[str] = "/app"\\n\\
    model: Optional[str] = "gpt-4o"\\n\\
    openai_api_key: Optional[str] = None\\n\\
    anthropic_api_key: Optional[str] = None\\n\\
    deepseek_api_key: Optional[str] = None\\n\\
    openrouter_api_key: Optional[str] = None\\n\\
\\n\\
class TaskResponse(BaseModel):\\n\\
    success: bool\\n\\
    result: str\\n\\
    files_created: Dict[str, str]\\n\\
    error: Optional[str] = None\\n\\
\\n\\
@app.post("/api/execute_task", response_model=TaskResponse)\\n\\
async def execute_task(request: TaskRequest):\\n\\
    try:\\n\\
        # Ensure workspace directory exists\\n\\
        workspace_path = Path(request.workspace_dir)\\n\\
        workspace_path.mkdir(parents=True, exist_ok=True)\\n\\
        \\n\\
        # Initialize git repository if it doesnt exist\\n\\
        git_dir = workspace_path / ".git"\\n\\
        if not git_dir.exists():\\n\\
            subprocess.run(["git", "init"], cwd=workspace_path, check=True)\\n\\
        \\n\\
        # Take snapshot of files before aider runs\\n\\
        before_files = set()\\n\\
        if workspace_path.exists():\\n\\
            for file_path in workspace_path.rglob("*"):\\n\\
                if file_path.is_file() and not should_exclude_file(file_path, workspace_path):\\n\\
                    before_files.add(file_path.relative_to(workspace_path))\\n\\
        \\n\\
        # Set up environment variables for aider\\n\\
        env = os.environ.copy()\\n\\
        \\n\\
        # Configure API keys based on model type\\n\\
        if request.openai_api_key:\\n\\
            env["OPENAI_API_KEY"] = request.openai_api_key\\n\\
        elif "OPENAI_API_KEY" not in env:\\n\\
            env["OPENAI_API_KEY"] = "{openai_api_key}"\\n\\
        \\n\\
        if request.anthropic_api_key:\\n\\
            env["ANTHROPIC_API_KEY"] = request.anthropic_api_key\\n\\
        elif "ANTHROPIC_API_KEY" not in env:\\n\\
            env["ANTHROPIC_API_KEY"] = "{anthropic_api_key}"\\n\\
        \\n\\
        if request.deepseek_api_key:\\n\\
            env["DEEPSEEK_API_KEY"] = request.deepseek_api_key\\n\\
        elif "DEEPSEEK_API_KEY" not in env:\\n\\
            env["DEEPSEEK_API_KEY"] = "{deepseek_api_key}"\\n\\
        \\n\\
        if request.openrouter_api_key:\\n\\
            env["OPENROUTER_API_KEY"] = request.openrouter_api_key\\n\\
        elif "OPENROUTER_API_KEY" not in env:\\n\\
            env["OPENROUTER_API_KEY"] = "{openrouter_api_key}"\\n\\
        \\n\\
        # Run aider with the task\\n\\
        cmd = [\\n\\
            "aider",\\n\\
            "--model", request.model,\\n\\
            "--yes",  # Auto-confirm changes\\n\\
            "--no-show-model-warnings",\\n\\
            "--cache-prompts",  # Enable prompt caching for cost savings\\n\\
            "--no-stream",  # Disable streaming for API responses\\n\\
            "--message", request.task,\\n\\
            "--disable-playwright", # Disable playwright to avoid webscraping issues\\n\\
            "--read", "/app/api_server.py", # Always read the API server file to keep it intact\\n\\
            "--no-git", # Disable git integration to avoid commit issues\\n\\
            "--reasoning-effort", "high", # Use high reasoning effort to see traces\\n\\
        ]\\n\\
        \\n\\
        result = subprocess.run(\\n\\
            cmd,\\n\\
            capture_output=True,\\n\\
            text=True,\\n\\
            timeout=300,  # 5 minute timeout\\n\\
            cwd=workspace_path,\\n\\
            env=env\\n\\
        )\\n\\
        \\n\\
        # Collect all files after aider runs (including new and modified files)\\n\\
        files_created = {{}}\\n\\
        if workspace_path.exists():\\n\\
            for file_path in workspace_path.rglob("*"):\\n\\
                if file_path.is_file() and not should_exclude_file(file_path, workspace_path):\\n\\
                    relative_path = file_path.relative_to(workspace_path)\\n\\
                    \\n\\
                    # Read the actual file content\\n\\
                    try:\\n\\
                        with open(file_path, "r", encoding="utf-8") as f:\\n\\
                            file_content = f.read()\\n\\
                            files_created[str(relative_path)] = file_content\\n\\
                    except (UnicodeDecodeError, PermissionError) as e:\\n\\
                        logger.warning(f"Could not read file {{file_path}}: {{e}}")\\n\\
                        files_created[str(relative_path)] = f"<Could not read file: {{str(e)}}>"\\n\\
                    except Exception as e:\\n\\
                        logger.error(f"Unexpected error reading {{file_path}}: {{e}}")\\n\\
                        files_created[str(relative_path)] = f"<Error reading file: {{str(e)}}>"\\n\\
        \\n\\
        if result.returncode == 0:\\n\\
            return TaskResponse(\\n\\
                success=True,\\n\\
                result=result.stdout,\\n\\
                files_created=files_created\\n\\
            )\\n\\
        else:\\n\\
            return TaskResponse(\\n\\
                success=False,\\n\\
                result=result.stdout,\\n\\
                files_created=files_created,\\n\\
                error=result.stderr\\n\\
            )\\n\\
    \\n\\
    except subprocess.TimeoutExpired:\\n\\
        # Still try to collect files even if timeout\\n\\
        files_created = {{}}\\n\\
        try:\\n\\
            workspace_path = Path(request.workspace_dir)\\n\\
            if workspace_path.exists():\\n\\
                for file_path in workspace_path.rglob("*"):\\n\\
                    if file_path.is_file() and not should_exclude_file(file_path, workspace_path):\\n\\
                        relative_path = file_path.relative_to(workspace_path)\\n\\
                        try:\\n\\
                            with open(file_path, "r", encoding="utf-8") as f:\\n\\
                                files_created[str(relative_path)] = f.read()\\n\\
                        except:\\n\\
                            files_created[str(relative_path)] = "<Could not read file after timeout>"\\n\\
        except Exception as e:\\n\\
            logger.error(f"Error collecting files after timeout: {{e}}")\\n\\
        \\n\\
        return TaskResponse(\\n\\
            success=False,\\n\\
            result="",\\n\\
            files_created=files_created,\\n\\
            error="Task timed out after 5 minutes"\\n\\
        )\\n\\
    except Exception as e:\\n\\
        logger.exception("Error executing task")\\n\\
        return TaskResponse(\\n\\
            success=False,\\n\\
            result="",\\n\\
            files_created={{}},\\n\\
            error=str(e)\\n\\
        )\\n\\
\\n\\
@app.get("/api/health")\\n\\
async def health_check():\\n\\
    return {{"status": "healthy"}}\\n\\
\\n\\
if __name__ == "__main__":\\n\\
    uvicorn.run(app, host="0.0.0.0", port=3000)\\n\\
' > /app/api_server.py && chmod +x /app/api_server.py

# Set Aider environment variables
ENV AIDER_WORKSPACE_DIR=/app
"""

_PYTHON_DOCKERFILE = """
# Use the aider-full image as base which includes aider and common tools
FROM paulgauthier/aider

{_AIDER_SETUP}

{{additional_commands}}

# install deps and copy generated code
RUN mkdir -p {_WORKDIR}
WORKDIR {_WORKDIR}
COPY {_REQS_TXT_FILENAME} {_REQS_TXT_FILENAME}
RUN python3 -m pip install -r requirements.txt
ENV APP_SECRET=supers3cret
COPY / {_WORKDIR}/

# Expose ports for both the app and Aider API
EXPOSE 3000 8000 5000

ENTRYPOINT {{entrypoint_cmd}}
""".format(_AIDER_SETUP=_AIDER_SETUP, _WORKDIR=_WORKDIR, _REQS_TXT_FILENAME=_REQS_TXT_FILENAME)

#############
### Flask ###
#############
_FLASK_REQS_TXT = """
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
"""

FlaskEnv = Env(
    language="Python",
    extension="py",
    framework="Flask",
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FLASK_REQS_TXT},
    allowed_packages=_FLASK_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd=f"python3 {_PYTHON_CODE_FILENAME}",
)

#############
### Flask with Aider ###
#############
_FLASK_AIDER_REQS_TXT = _FLASK_REQS_TXT + """
fastapi>=0.104.0
uvicorn>=0.24.0
"""

FlaskAiderEnv = Env(
    language="Python",
    extension="py",
    framework="Flask",
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FLASK_AIDER_REQS_TXT},
    allowed_packages=_FLASK_AIDER_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /app && python3 api_server.py"',
)

#############
### FastAPI ###
#############
_FAST_API_REQS_TXT = """
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
"""

_FAST_API_INSTRUCTIONS = (
    SINGLE_FILE_APP_INSRUCTIONS + "\nDo not use streaming responses!"
)
FastAPIEnv = Env(
    language="Python",
    extension="py",
    framework="FastAPI",
    code_filename=_PYTHON_CODE_FILENAME,
    env_instructions=_FAST_API_INSTRUCTIONS,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FAST_API_REQS_TXT},
    allowed_packages=_FAST_API_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd=f"python3 {_PYTHON_CODE_FILENAME}",
)

#############
### FastAPI with Aider ###
#############
_FASTAPI_AIDER_REQS_TXT = _FAST_API_REQS_TXT + """
# Aider dependencies are already included in paulgauthier/aider
"""

FastAPIAiderEnv = Env(
    language="Python",
    extension="py",
    framework="FastAPI",
    code_filename=_PYTHON_CODE_FILENAME,
    env_instructions=_FAST_API_INSTRUCTIONS,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FASTAPI_AIDER_REQS_TXT},
    allowed_packages=_FASTAPI_AIDER_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /app && python3 api_server.py"',
)

#############
### Django ###
#############
_DJANGO_REQS_TXT = """
asgiref==3.8.1
Django==5.1.4
sqlparse==0.5.3
pyjwt==2.10.0
"""
_DJANGO_INSTRUCTIONS = f"""
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

{MULTI_FILE_APP_PROMPT_STRUCTURE}

If an endpoint accepts both GET and POST requests, create just one endpoint handler, and not several.
Make sure that the endpoints don't have trailing slashes!
Set the ALLOWED_HOSTS to ["0.0.0.0", "localhost", "127.0.0.1"]
Make sure that the command `python manage.py runserver` starts the app successfully.
"""
_DJANGO_DOCKERFILE = "\n".join(
    [
        _PYTHON_DOCKERFILE,
        "RUN python3 manage.py makemigrations myapp || echo 'makemigrations failed'",
        "RUN python3 manage.py migrate || echo 'migrate failed'",
    ]
)
DjangoEnv = Env(
    language="Python",
    extension="py",
    framework="Django",
    code_filename=None,
    env_instructions=_DJANGO_INSTRUCTIONS,
    dockerfile=_DJANGO_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _DJANGO_REQS_TXT},
    allowed_packages=_DJANGO_REQS_TXT,
    is_multi_file=True,
    port=5000,
    entrypoint_cmd="python3 manage.py runserver 0.0.0.0:5000",
)

#############
### Django with Aider ###
#############
_DJANGO_AIDER_REQS_TXT = _DJANGO_REQS_TXT + """
fastapi>=0.104.0
uvicorn>=0.24.0
"""

DjangoAiderEnv = Env(
    language="Python",
    extension="py",
    framework="Django",
    code_filename=None,
    env_instructions=_DJANGO_INSTRUCTIONS,
    dockerfile=_DJANGO_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _DJANGO_AIDER_REQS_TXT},
    allowed_packages=_DJANGO_AIDER_REQS_TXT,
    is_multi_file=True,
    port=5000,
    entrypoint_cmd='bash -c "cd /app && python3 api_server.py"',
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

AioHttpEnv = Env(
    language="Python",
    extension="py",
    framework="aiohttp",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _AIO_HTTP_REQS_TXT},
    allowed_packages=_AIO_HTTP_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd=f"python3 {_PYTHON_CODE_FILENAME}",
)

#############
### AioHttp with Aider ###
#############
_AIOHTTP_AIDER_REQS_TXT = _AIO_HTTP_REQS_TXT + """
fastapi>=0.104.0
uvicorn>=0.24.0
"""

AioHttpAiderEnv = Env(
    language="Python",
    extension="py",
    framework="aiohttp",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _AIOHTTP_AIDER_REQS_TXT},
    allowed_packages=_AIOHTTP_AIDER_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /app && python3 api_server.py"',
)

#############
### Standalone Aider Environment ###
#############
_AIDER_ONLY_REQS_TXT = """
fastapi>=0.104.0
uvicorn>=0.24.0
aiohttp>=3.9.0
requests>=2.31.0
"""

AiderEnv = Env(
    language="Python",
    extension="py",
    framework="Aider",
    code_filename="aider_runner.py",
    env_instructions="Pure Aider headless environment for AI-assisted development. The Aider REST API server will be available at port 3000 with endpoints like POST /api/execute_task.",
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _AIDER_ONLY_REQS_TXT},
    allowed_packages=_AIDER_ONLY_REQS_TXT,
    is_multi_file=False,
    agent_port=3000,
    entrypoint_cmd="cd /app && python3 api_server.py",
)

# Export all environments for easy access
ALL_ENVS = {
  
    "flask_aider": FlaskAiderEnv,
    
    "fastapi_aider": FastAPIAiderEnv,
    
    "django_aider": DjangoAiderEnv,
    
    "aiohttp_aider": AioHttpAiderEnv,
    "aider": AiderEnv,
}