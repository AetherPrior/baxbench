from env.base import Env
from env.aider import (
    FlaskAiderEnv,
    AioHttpAiderEnv,
    FastAPIAiderEnv,
    DjangoAiderEnv,
)
all_aider_envs: list[Env] = [
    FlaskAiderEnv,
    AioHttpAiderEnv,
    FastAPIAiderEnv,
    DjangoAiderEnv,
]
