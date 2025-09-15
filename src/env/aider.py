from env.base import MULTI_FILE_APP_PROMPT_STRUCTURE, SINGLE_FILE_APP_INSRUCTIONS, Env
import os 

_WORKDIR = "/app"
_PYTHON_CODE_FILENAME = "app.py"
_REQS_TXT_FILENAME = "requirements.txt"

# Simplified Aider setup without REST API server
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

# Expose ports for the app
EXPOSE 8000 5000

# Keep container running for docker-py execution
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
_FLASK_AIDER_REQS_TXT = _FLASK_REQS_TXT

FlaskAiderEnv = Env(
    language="Python",
    extension="py",
    framework="Flask",
    env_instructions=SINGLE_FILE_APP_INSRUCTIONS + f"\nPlease name your file {_PYTHON_CODE_FILENAME}!",
    code_filename=_PYTHON_CODE_FILENAME,
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FLASK_AIDER_REQS_TXT},
    allowed_packages=_FLASK_AIDER_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for docker-py
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
_FASTAPI_AIDER_REQS_TXT = _FAST_API_REQS_TXT

FastAPIAiderEnv = Env(
    language="Python",
    extension="py",
    framework="FastAPI",
    code_filename=_PYTHON_CODE_FILENAME,
    env_instructions=_FAST_API_INSTRUCTIONS + f"\nPlease name your file {_PYTHON_CODE_FILENAME}!",
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _FASTAPI_AIDER_REQS_TXT},
    allowed_packages=_FASTAPI_AIDER_REQS_TXT,
    is_multi_file=False,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for docker-py
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
_DJANGO_AIDER_REQS_TXT = _DJANGO_REQS_TXT

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
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for docker-py
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
_AIOHTTP_AIDER_REQS_TXT = _AIO_HTTP_REQS_TXT

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
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for docker-py
)

#############
### Standalone Aider Environment ###
#############
_AIDER_ONLY_REQS_TXT = """
requests>=2.31.0
"""

AiderEnv = Env(
    language="Python",
    extension="py",
    framework="Aider",
    code_filename="placeholder.py",
    env_instructions="Pure Aider headless environment for AI-assisted development. Uses docker-py to execute aider commands directly.",
    dockerfile=_PYTHON_DOCKERFILE,
    workdir=_WORKDIR,
    sqlite_database="db.sqlite3",
    manifest_files={_REQS_TXT_FILENAME: _AIDER_ONLY_REQS_TXT},
    allowed_packages=_AIDER_ONLY_REQS_TXT,
    is_multi_file=False,
    agent_port=None,
    entrypoint_cmd='tail -f /dev/null',  # Keep container alive for docker-py
)

# Export all environments for easy access
ALL_ENVS = {
    "flask_aider": FlaskAiderEnv,
    "fastapi_aider": FastAPIAiderEnv,
    "django_aider": DjangoAiderEnv,
    "aiohttp_aider": AioHttpAiderEnv,
    "aider": AiderEnv,
}