from env.base import Env
from env.openhands import (
    FlaskOpenHandsLocalEnv,
    FastAPIOpenHandsLocalEnv,
    DjangoOpenHandsLocalEnv,
    AioHttpOpenHandsLocalEnv,
)
all_openhands_envs: list[Env] = [
    FlaskOpenHandsLocalEnv,
    FastAPIOpenHandsLocalEnv,
    DjangoOpenHandsLocalEnv,
    AioHttpOpenHandsLocalEnv,
]
