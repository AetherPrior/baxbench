from env.base import MULTI_FILE_APP_PROMPT_STRUCTURE, SINGLE_FILE_APP_INSRUCTIONS, Env
import os 

_WORKDIR = "/app"
_PYTHON_CODE_FILENAME = "app.py"
_REQS_TXT_FILENAME = "requirements.txt"

# OpenHands setup commands for headless mode (ubuntu)
_OPENHANDS_SETUP = """
# Install OpenHands and its dependencies
RUN pip install openhands-ai

# Install additional dependencies for OpenHands (ubuntu)
RUN apt-get update && apt-get install -y \\
    curl \\
    sqlite3 \\ 
    && rm -rf /var/lib/apt/lists/*

# Create OpenHands directories
RUN mkdir -p /openhands/workspace
RUN mkdir -p /openhands/logs
RUN mkdir -p /openhands/config

# Create OpenHands config file for headless mode
RUN echo '[core]\\n\\
default_agent = "CodeActAgent"\\n\\
max_iterations = 50\\n\\
max_chars = 10000000\\n\\
\\n\\
[llm]\\n\\
model = "gpt-4o"\\n\\
api_key = "{openai_api_key}"\\n\\
base_url = ""\\n\\
\\n\\
[agent]\\n\\
memory_enabled = false\\n\\
\\n\\
[sandbox]\\n\\
base_container_image = "python:3.12-slim"\\n\\
user_id = 1000\\n\\
timeout = 120\\n\\
' > /openhands/config/config.toml

# Create OpenHands API server script
RUN echo '#!/usr/bin/env python3\\n\\
import os\\n\\
import sys\\n\\
import json\\n\\
import asyncio\\n\\
import logging\\n\\
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
class TaskRequest(BaseModel):\\n\\
    task: str\\n\\
    workspace_dir: Optional[str] = "/app"\\n\\
    max_iterations: Optional[int] = 50\\n\\
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
        # Run OpenHands in headless mode\\n\\
        cmd = [\\n\\
            "python", "-m", "openhands.core.main",\\n\\
            "-d", request.workspace_dir,\\n\\
            "-i", str(request.max_iterations),\\n\\
            "-t", request.task\\n\\
        ]\\n\\
        \\n\\
        import subprocess\\n\\
        result = subprocess.run(\\n\\
            cmd,\\n\\
            capture_output=True,\\n\\
            text=True,\\n\\
            timeout=300,  # 5 minute timeout\\n\\
            cwd="/openhands"\\n\\
        )\\n\\
        \\n\\
        # Collect created files\\n\\
        files_created = {{}}\\n\\
        workspace_path = Path(request.workspace_dir)\\n\\
        if workspace_path.exists():\\n\\
            for file_path in workspace_path.rglob("*"):\\n\\
                if file_path.is_file() and not file_path.name.startswith("."):\\n\\
                    try:\\n\\
                        with open(file_path, "r", encoding="utf-8") as f:\\n\\
                            files_created[str(file_path.relative_to(workspace_path))] = f.read()\\n\\
                    except (UnicodeDecodeError, PermissionError):\\n\\
                        files_created[str(file_path.relative_to(workspace_path))] = "<binary or unreadable file>"\\n\\
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
' > /openhands/api_server.py && chmod +x /openhands/api_server.py

# Set OpenHands environment variables
ENV OPENHANDS_WORKSPACE_DIR=/app
ENV OPENHANDS_CONFIG_DIR=/openhands/config
ENV OPENHANDS_LOG_DIR=/openhands/logs
"""

_PYTHON_DOCKERFILE = """
# setup base - using Alpine for smaller image size
FROM python:3.12-slim

{_OPENHANDS_SETUP}

{{additional_commands}}

# install deps and copy generated code
RUN mkdir -p {_WORKDIR}
WORKDIR {_WORKDIR}
COPY {_REQS_TXT_FILENAME} {_REQS_TXT_FILENAME}
RUN python3 -m pip install -r requirements.txt
ENV APP_SECRET=supers3cret
COPY / {_WORKDIR}/

# Expose ports for both the app and OpenHands API
EXPOSE 3000 8000 5000

ENTRYPOINT {{entrypoint_cmd}}
""".format(_OPENHANDS_SETUP=_OPENHANDS_SETUP, _WORKDIR=_WORKDIR, _REQS_TXT_FILENAME=_REQS_TXT_FILENAME)
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
### Flask with OpenHands ###
#############
_FLASK_OPENHANDS_REQS_TXT = _FLASK_REQS_TXT + """
openhands-ai>=0.55.0
fastapi>=0.104.0
uvicorn>=0.24.0
"""

FlaskOpenHandsEnv = Env(
    language="Python",
    extension="py",
    framework="FlaskOpenHands",
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FLASK_OPENHANDS_REQS_TXT},
    allowed_packages=_FLASK_OPENHANDS_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /openhands && python3 api_server.py"',
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
### FastAPI with OpenHands ###
#############
_FASTAPI_OPENHANDS_REQS_TXT = _FAST_API_REQS_TXT + """
openhands-ai>=0.55.0
"""

FastAPIOpenHandsEnv = Env(
    language="Python",
    extension="py",
    framework="FastAPIOpenHands",
    code_filename=_PYTHON_CODE_FILENAME,
    env_instructions=_FAST_API_INSTRUCTIONS,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FASTAPI_OPENHANDS_REQS_TXT},
    allowed_packages=_FASTAPI_OPENHANDS_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /openhands && python3 api_server.py"',
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
### Django with OpenHands ###
#############
_DJANGO_OPENHANDS_REQS_TXT = _DJANGO_REQS_TXT + """
openhands-ai>=0.55.0
fastapi>=0.104.0
uvicorn>=0.24.0
"""

DjangoOpenHandsEnv = Env(
    language="Python",
    extension="py",
    framework="DjangoOpenHands",
    code_filename=None,
    env_instructions=_DJANGO_INSTRUCTIONS,
    dockerfile=_DJANGO_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _DJANGO_OPENHANDS_REQS_TXT},
    allowed_packages=_DJANGO_OPENHANDS_REQS_TXT,
    is_multi_file=True,
    port=5000,
    entrypoint_cmd='bash -c "cd /app && python3 manage.py runserver 0.0.0.0:5000 & cd /openhands && python3 api_server.py"',
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
### AioHttp with OpenHands ###
#############
_AIOHTTP_OPENHANDS_REQS_TXT = _AIO_HTTP_REQS_TXT + """
openhands-ai>=0.55.0
fastapi>=0.104.0
uvicorn>=0.24.0
"""

AioHttpOpenHandsEnv = Env(
    language="Python",
    extension="py",
    framework="aiohttpOpenHands",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _AIOHTTP_OPENHANDS_REQS_TXT},
    allowed_packages=_AIOHTTP_OPENHANDS_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='bash -c "cd /openhands && python3 api_server.py"',
)

#############
### Standalone OpenHands Environment ###
#############
_OPENHANDS_ONLY_REQS_TXT = """
openhands-ai>=0.55.0
fastapi>=0.104.0
uvicorn>=0.24.0
aiohttp>=3.9.0
requests>=2.31.0
"""

OpenHandsEnv = Env(
    language="Python",
    extension="py",
    framework="OpenHands",
    code_filename="openhands_runner.py",
    env_instructions="Pure OpenHands headless environment for AI-assisted development. The OpenHands REST API server will be available at port 3000 with endpoints like POST /api/execute_task.",
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _OPENHANDS_ONLY_REQS_TXT},
    allowed_packages=_OPENHANDS_ONLY_REQS_TXT,
    is_multi_file=False,
    agent_port=3000,
    entrypoint_cmd="cd /openhands && python3 api_server.py",
)

# Export all environments for easy access
ALL_ENVS = {
    "flask": FlaskEnv,
    "flask_openhands": FlaskOpenHandsEnv,
    "fastapi": FastAPIEnv,
    "fastapi_openhands": FastAPIOpenHandsEnv,
    "django": DjangoEnv,
    "django_openhands": DjangoOpenHandsEnv,
    "aiohttp": AioHttpEnv,
    "aiohttp_openhands": AioHttpOpenHandsEnv,
    "openhands": OpenHandsEnv,
}