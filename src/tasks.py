import concurrent.futures
import inspect
import json
import logging
import math
import multiprocessing
import multiprocessing.managers
import os
import pathlib
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Self, cast, Optional

import tqdm
from docker.models.containers import Container

import cwes as cwe
from env.base import COMMON_DOCKER_RUN_COMMANDS, Env
from prompts import Parser, Prompter, OpenHandsClient, AiderClient
import random
from scenarios.base import AppInstance, FunctionalTest, Scenario, SecurityTest


def esc(s: str) -> str:
    return s.replace("/", "-")


def run_test_with_timeout(
    f: SecurityTest | FunctionalTest, app_instance: AppInstance, timeout: int
) -> Any:
    with multiprocessing.Pool(processes=1) as pool:
        async_result = pool.apply_async(f, [app_instance])
        try:
            return async_result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise TimeoutError("Functional test timed out")


@dataclass
class ContainerRunner:
    env: Env
    port_manager: "SlotManager"
    image_id: str
    logger: logging.Logger
    _container: Container | None = None
    _port: int | None = None
    _agent_port: int | None = None

    def __enter__(self) -> Self:
        while self._port is None:
            self._port = self.port_manager.acquire_slot()
            self._agent_port = self.port_manager.acquire_slot()
            time.sleep(0.1)
        try:
            self._container = self.env.run_docker_container(self.image_id, self._port, agent=self._agent_port)
        except Exception as e:
            self.logger.exception("could not start container %s", e, exc_info=e)
            raise ValueError("Could not start docker container")
        self.logger.info(
            "started container, port=%d, agent_port=%d, container_id=%s",
            self._port,
            self._agent_port,
            self._container.id,
        )
        # Sleep for a bit to give the webapp some time to start.
        time.sleep(self.env.wait_to_start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        assert self.container is not None
        assert self._port is not None
        container_logs = cast(
            bytes, self.container.logs(stdout=True, stderr=True, follow=False)
        )
        self.logger.info("container logs:\n%s", container_logs.decode())
        self.container.remove(force=True)
        self.port_manager.release_slot(self._port)
        if self._agent_port is not None:
            self.port_manager.release_slot(self._agent_port)

        self.logger.info("-" * 100)
        self.logger.info("removed container")
        self.logger.info("-" * 100)

    @property
    def agent_port(self) -> int:
        assert self._agent_port is not None
        return self._agent_port

    @property
    def port(self) -> int:
        assert self._port is not None
        return self._port

    @property
    def container(self) -> Container:
        assert self._container is not None
        return self._container


@dataclass
class Task:
    env: Env
    scenario: Scenario
    model: str
    temperature: float
    reasoning_effort: str
    spec_type: str
    safety_prompt: str
    openrouter: bool

    @property
    def id(self) -> str:
        return f"{self.model}-{self.env.id}-{self.scenario.id}-{self.spec_type}-{self.safety_prompt}-{self.temperature}"

    @contextmanager
    def create_logger(
        self, logfile_path: pathlib.Path
    ) -> Generator[logging.Logger, None, None]:
        logger = logging.getLogger(self.id)
        logger.setLevel(logging.INFO)
        logfile_handler = logging.FileHandler(logfile_path, mode="w")
        logfile_handler.setLevel(logging.INFO)
        logfile_handler.setFormatter(
            logging.Formatter(fmt="%(levelname)s %(asctime)s %(message)s")
        )
        logger.addHandler(logfile_handler)
        try:
            yield logger
        finally:
            logfile_handler.close()

    def get_save_dir(self, results_dir: pathlib.Path) -> pathlib.Path:
        save_dir = (
            results_dir
            / esc(self.model)
            / esc(self.scenario.id)
            / esc(self.env.id)
            / f"temp{float(self.temperature)}-{esc(self.spec_type)}-{esc(self.safety_prompt)}"
        )
        return save_dir

    def get_sample_dir(self, results_dir: pathlib.Path, sample: int) -> pathlib.Path:
        return self.get_save_dir(results_dir) / f"sample{sample}"

    def get_code_dir(self, results_dir: pathlib.Path, sample: int) -> pathlib.Path:
        return self.get_sample_dir(results_dir, sample) / "code"

    def get_test_results_json_path(
        self, results_dir: pathlib.Path, sample: int
    ) -> pathlib.Path:
        return self.get_sample_dir(results_dir, sample) / "test_results.json"

    def load_code(
        self, results_dir: pathlib.Path, sample: int
    ) -> dict[pathlib.Path, str]:
        code_dir = self.get_code_dir(results_dir, sample)
        files: dict[pathlib.Path, str] = {}
        for root, _, file_names in os.walk(code_dir):
            for file in file_names:
                abs_path = pathlib.Path(root) / file
                with open(abs_path, "r") as f:
                    content = f.read()
                rel_path = abs_path.relative_to(code_dir)
                files[rel_path] = content
        return files

    def save_code(
        self, files: dict[pathlib.Path, str], results_dir: pathlib.Path, sample: int
    ) -> None:
        code_dir = self.get_code_dir(results_dir, sample)
        code_dir.mkdir(parents=True, exist_ok=True)
        for path, code in files.items():
            full_path = code_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(code)

    def save_test_results(
        self, results: "TestResult", results_dir: pathlib.Path, sample: int
    ) -> None:
        sample_dir = self.get_sample_dir(results_dir, sample)
        sample_dir.mkdir(parents=True, exist_ok=True)
        test_result_path = self.get_test_results_json_path(results_dir, sample)
        with open(test_result_path, "w") as f:
            json.dump(results.to_dict(), f)

    def generate_code(
        self,
        results_dir: pathlib.Path,
        batch_size: int,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        force: bool,
        openrouter: bool,
    ) -> None:
        if self.model == "openhands":
            self.generate_code_with_openhands(
                results_dir,
                batch_size,
                max_retries,
                base_delay,
                max_delay,
                force,
                openrouter,
            )
            return
        elif self.model == "aider":
            self.generate_code_with_aider(
                results_dir,
                batch_size,
                max_retries,
                base_delay,
                max_delay,
                force,
                openrouter,
            )
            return

        # check if this task has already been generated
        if (
            all(
                [
                    self.get_code_dir(results_dir, sample).exists()
                    for sample in range(batch_size)
                ]
            )
            and not any(
                [
                    (self.get_code_dir(results_dir, sample) / "failed").exists()
                    for sample in range(batch_size)
                ]
            )
            and not force
        ):
            return

        save_dir = self.get_save_dir(results_dir)
        try:
            save_dir.mkdir(parents=True, exist_ok=False)
        except:
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)

        gen_logfile_path = save_dir / "gen.log"
        # clear the log file
        with open(gen_logfile_path, "w") as f:
            f.write("")
        with self.create_logger(gen_logfile_path) as logger:
            logger.info(
                "generating %s code samples at temp %s for task %s with reasoning effort %s",
                batch_size,
                self.temperature,
                self.id,
                self.reasoning_effort,
            )

            prompter = Prompter(
                env=self.env,
                scenario=self.scenario,
                model=self.model,
                spec_type=self.spec_type,
                safety_prompt=self.safety_prompt,
                batch_size=batch_size,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                openrouter=openrouter,
            )
            logger.info("built prompt:\n%s", prompter.prompt)
            logger.info("-" * 100)

            try:
                model_responses = prompter.prompt_model_batch_with_exp_backoff(
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    logger=logger,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.exception("got exception:\n%s", str(e), exc_info=e)
                return

            logger.info(
                "got model responses:\n%s",
                "\n\n<<<RESPONSE DELIM>>>\n\n".join(model_responses),
            )
            logger.info("-" * 100)

            file_contents = [
                Parser(self.env, logger).parse_response(r) for r in model_responses
            ]

            for i, files in enumerate(file_contents):
                try:
                    self.save_code(files, results_dir, i)
                    logger.info("saved code sample %d", i)
                except Exception as e:
                    logger.exception("got exception:\n%s", str(e), exc_info=e)
                logger.info("-" * 80)

    def generate_code_with_openhands(
        self,
        results_dir: pathlib.Path,
        batch_size: int,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        force: bool,
        openrouter: bool,
    ) -> None:
        # check if this task has already been generated
        if (
            all(
                [
                    self.get_code_dir(results_dir, sample).exists()
                    for sample in range(batch_size)
                ]
            )
            and not any(
                [
                    (self.get_code_dir(results_dir, sample) / "failed").exists()
                    for sample in range(batch_size)
                ]
            )
            and not force
        ):
            return

        save_dir = self.get_save_dir(results_dir)
        try:
            save_dir.mkdir(parents=True, exist_ok=False)
        except:
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)

        gen_logfile_path = save_dir / "gen.log"
        with open(gen_logfile_path, "w") as f:
            f.write("")
        with self.create_logger(gen_logfile_path) as logger:
            logger.info(
                "generating %s code samples at temp %s for task %s with reasoning effort %s (OpenHands)",
                batch_size,
                self.temperature,
                self.id,
                self.reasoning_effort,
            )

            # Build docker image for the environment
            files = {}
            image_id = self.env.build_docker_image(
                files,
                self.scenario.needed_packages.get("_all_", [])
                + self.scenario.needed_packages.get(self.env.language, []),
                logger,
                no_cache=False,
            )

            # Start the container and interact with OpenHands inside it
            with multiprocessing.Manager() as manager:
                port_manager = SlotManager(manager, 2, min=9000)
                with ContainerRunner(self.env, port_manager, image_id, logger) as cr:
                    # Point OpenHandsClient to the API running inside the container
                    openhands_api_url = f"http://localhost:{cr.agent_port}"
                    client = OpenHandsClient(api_url=openhands_api_url)

                    if not client.health_check():
                        logger.error("OpenHands API server is not running or not accessible in the container")
                        raise Exception("OpenHands API server is not running. Make sure the container is running with the OpenHands API server.")

                    prompter = Prompter(
                        env=self.env,
                        scenario=self.scenario,
                        model="openhands",
                        spec_type=self.spec_type,
                        safety_prompt=self.safety_prompt,
                        batch_size=batch_size,
                        temperature=self.temperature,
                        reasoning_effort=self.reasoning_effort,
                        openrouter=openrouter,
                        agent_port=cr.agent_port,
                    )
                    logger.info("built OpenHands prompt:\n%s", prompter.prompt)
                    logger.info("-" * 100)

                    completions = []
                    for i in range(batch_size):
                        retries = 0
                        while True:
                            try:
                                logger.info(f"OpenHands: generating sample {i+1}/{batch_size}")
                                responses = prompter.prompt_openhands(logger)
                                completions.extend(responses)
                                break
                            except Exception as e:
                                retries += 1
                                if retries > max_retries:
                                    logger.error(f"Max retries reached for OpenHands, raising exception: {e}")
                                    raise e
                                delay = min(base_delay * 2**retries, max_delay)
                                delay = random.uniform(0, delay)
                                logger.exception(
                                    f"{e}, backing off for {delay} seconds", exc_info=e
                                )
                                time.sleep(delay)

                    logger.info(
                        "got OpenHands responses:\n%s",
                        "\n\n<<<RESPONSE DELIM>>>\n\n".join(completions),
                    )
                    logger.info("-" * 100)

                    file_contents = [
                        Parser(self.env, logger).parse_response(r, model="openhands") for r in completions
                    ]

                    for i, files in enumerate(file_contents):
                        try:
                            self.save_code(files, results_dir, i)
                            logger.info("saved code sample %d", i)
                        except Exception as e:
                            logger.exception("got exception:\n%s", str(e), exc_info=e)
                        logger.info("-" * 80)


    def generate_code_with_aider(
        self,
        results_dir: pathlib.Path,
        batch_size: int,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        force: bool,
        openrouter: bool,
    ) -> None:
        """Generate code using Aider AI coding assistant"""
        # Check if this task has already been generated
        if (
            all(
                [
                    self.get_code_dir(results_dir, sample).exists()
                    for sample in range(batch_size)
                ]
            )
            and not any(
                [
                    (self.get_code_dir(results_dir, sample) / "failed").exists()
                    for sample in range(batch_size)
                ]
            )
            and not force
        ):
            return
        
        save_dir = self.get_save_dir(results_dir)
        try:
            save_dir.mkdir(parents=True, exist_ok=False)
        except:
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=False)

        gen_logfile_path = save_dir / "gen.log"
        with open(gen_logfile_path, "w") as f:
            f.write("")
        
        with self.create_logger(gen_logfile_path) as logger:
            logger.info(
                "generating %s code samples at temp %s for task %s with reasoning effort %s (Aider)",
                batch_size,
                self.temperature,
                self.id,
                self.reasoning_effort,
            )

            # Build docker image for the environment
            files = {}
            logger.info("Building docker image for Aider environment...")
            image_id = self.env.build_docker_image(
                files,
                self.scenario.needed_packages.get("_all_", [])
                + self.scenario.needed_packages.get(self.env.language, []),
                logger,
                no_cache=False,
            )
            logger.info("Done building docker image. Image ID: %s", image_id)
            # Start the container and interact with Aider inside it
            with multiprocessing.Manager() as manager:
                port_manager = SlotManager(manager, 2, min=9000)
                with ContainerRunner(self.env, port_manager, image_id, logger) as cr:
                    # Point AiderClient to the API running inside the container
                    # aider_api_url = f"http://localhost:{cr.agent_port}/api"
                    # client = AiderClient(api_url=aider_api_url)

                    # if not client.health_check():
                    #     logger.error("Aider API server is not running or not accessible in the container")
                    #     raise Exception("Aider API server is not running. Make sure the container is running with the Aider API server.")

                    prompter = Prompter(
                        env=self.env,
                        scenario=self.scenario,
                        model="aider",
                        spec_type=self.spec_type,
                        safety_prompt=self.safety_prompt,
                        batch_size=batch_size,
                        temperature=self.temperature,
                        reasoning_effort=self.reasoning_effort,
                        openrouter=openrouter,
                        agent_port=cr.agent_port,
                    )
                    logger.info("built Aider prompt:\n%s", prompter.prompt)
                    logger.info("-" * 100)

                    completions = []
                    for i in range(batch_size):
                        retries = 0
                        while True:
                            try:
                                logger.info(f"Aider: generating sample {i+1}/{batch_size}")
                                responses = prompter.prompt_aider(logger)
                                completions.extend(responses)
                                break
                            except Exception as e:
                                retries += 1
                                if retries > max_retries:
                                    logger.error(f"Max retries reached for Aider, raising exception: {e}")
                                    raise e
                                delay = min(base_delay * 2**retries, max_delay)
                                delay = random.uniform(0, delay)
                                logger.exception(
                                    f"{e}, backing off for {delay} seconds", exc_info=e
                                )
                                time.sleep(delay)

                    logger.info(
                        "got Aider responses:\n%s",
                        "\n\n<<<RESPONSE DELIM>>>\n\n".join(completions),
                    )
                    logger.info("-" * 100)

                    file_contents = [
                        Parser(self.env, logger).parse_response(r, model="aider") for r in completions
                    ]

                    for i, files in enumerate(file_contents):
                        try:
                            self.save_code(files, results_dir, i)
                            logger.info("saved code sample %d", i)
                        except Exception as e:
                            logger.exception("got exception:\n%s", str(e), exc_info=e)
                        logger.info("-" * 80)
                    
    def test_code(
        self,
        results_dir: pathlib.Path,
        samples: list[int],
        port_manager: "SlotManager",
        timeout: int,
        force: bool,
    ) -> None:
        # clean the directory from test artifacts if entered by force
        if force:
            for sample in samples:
                sample_dir = self.get_sample_dir(results_dir, sample)
                if sample_dir.exists():
                    for extension in ("*.log", "*.json"):
                        for file_path in sample_dir.glob(extension):
                            if file_path.is_file():
                                file_path.unlink()
        for sample in samples:
            sample_dir = self.get_sample_dir(results_dir, sample)
            if not self.get_code_dir(results_dir, sample).exists():
                continue
            if (
                self.get_test_results_json_path(results_dir, sample).exists()
                and not force
            ):
                continue
            self.get_test_results_json_path(results_dir, sample).unlink(missing_ok=True)
            log_file = sample_dir / "test.log"
            with self.create_logger(log_file) as logger:
                files: dict[pathlib.Path, str] = self.load_code(results_dir, sample)
                try:
                    image_id = self.env.build_docker_image(
                        files,
                        COMMON_DOCKER_RUN_COMMANDS
                        + self.scenario.needed_packages.get("_all_", [])
                        + self.scenario.needed_packages.get(self.env.language, []),
                        logger,
                        no_cache=False,
                    )
                except Exception as e:
                    logger.exception(
                        f"Failed to build docker image with cache, got exception:\n{str(e)}",
                        exc_info=e,
                    )
                    try:
                        logger.info("Retrying without cache")
                        image_id = self.env.build_docker_image(
                            files,
                            COMMON_DOCKER_RUN_COMMANDS
                            + self.scenario.needed_packages.get("_all_", [])
                            + self.scenario.needed_packages.get(self.env.language, []),
                            logger,
                            no_cache=True,
                        )
                    except Exception as e:
                        logger.exception(
                            f"Failed to build docker image without cache, got exception:\n{str(e)}",
                            exc_info=e,
                        )
                        result = TestResult()
                        for _ in range(len(self.scenario.functional_tests)):
                            result.record_ft_result(passed=False, had_exception=True)
                        for _ in range(len(self.scenario.security_tests)):
                            result.record_st_result(None)
                        logger.info(
                            f"Finished testing sample {sample}, which failed to build"
                        )
                        self.save_test_results(result, results_dir, sample)
                        logger.info("Saved test results")
                        logger.info("-" * 100)
                        continue

                logger.info("done building docker image. id: %s", image_id)
                logger.info("-" * 100)

                result = TestResult()
                for ft in self.scenario.functional_tests:
                    logger.info("running functional test:\n%s", inspect.getsource(ft))

                    passed = False
                    had_exception = False
                    try:
                        with ContainerRunner(
                            self.env, port_manager, image_id, logger
                        ) as cr:
                            server_ran_before = self.env.process_still_running(
                                cr.container.id, logger
                            )
                            passed = run_test_with_timeout(
                                ft,
                                AppInstance(
                                    port=cr.port,
                                    log_file_path=sample_dir / (ft.__name__ + ".log"),
                                    container_id=cr.container.id,
                                    env=self.env,
                                ),
                                timeout,
                            )
                            if (
                                not self.env.process_still_running(
                                    cr.container.id, logger
                                )
                                and server_ran_before
                            ):
                                # something crashed the server process, this should be logged as cwe 703
                                logger.warning(
                                    "server process crashed during functional test"
                                )
                                result.record_st_result(
                                    {
                                        cwe.CWE.IMPROPER_CHECK_OR_HANDLING_OF_EXCEPTIONAL_CONDITIONS
                                    }
                                )
                    except Exception as e:
                        logger.exception("got exception:\n%s", str(e), exc_info=e)
                        had_exception = True
                    logger.info(
                        f"Functional test {ft.__name__} {'passed' if passed else 'failed'}"
                    )
                    result.record_ft_result(passed=passed, had_exception=had_exception)
                    logger.info("-" * 80)

                for st in self.scenario.security_tests:
                    logger.info("running security test:\n%s", inspect.getsource(st))
                    cwes: set[cwe.CWE] | None = None
                    try:
                        with ContainerRunner(
                            self.env, port_manager, image_id, logger
                        ) as cr:
                            server_ran_before = self.env.process_still_running(
                                cr.container.id, logger
                            )
                            cwes = run_test_with_timeout(
                                st,
                                AppInstance(
                                    port=cr.port,
                                    log_file_path=sample_dir / (st.__name__ + ".log"),
                                    container_id=cr.container.id,
                                    env=self.env,
                                ),
                                timeout,
                            )
                            if (
                                not self.env.process_still_running(
                                    cr.container.id, logger
                                )
                                and server_ran_before
                            ):
                                # something crashed the server process, this should be logged as cwe 703
                                logger.warning(
                                    "server process crashed during security test"
                                )
                                if cwes is None:
                                    cwes = set()
                                # Only add 703 if 400 (uncontrolled resource consumption) did not cause the crash
                                if (
                                    not cwe.CWE.UNCONTROLLED_RESOURCE_CONSUMPTION
                                    in cwes
                                ):
                                    cwes.add(
                                        cwe.CWE.IMPROPER_CHECK_OR_HANDLING_OF_EXCEPTIONAL_CONDITIONS
                                    )
                    except Exception as e:
                        logger.exception("got exception:\n%s", str(e), exc_info=e)
                    logger.info(
                        f"Security test {st.__name__} {'passed' if not cwes else 'failed'}"
                    )
                    result.record_st_result(cwes)
                    logger.info("-" * 80)

                logger.info("finished testing sample %d", sample)
                self.save_test_results(result, results_dir, sample)
                logger.info("saved test results")
                logger.info("-" * 100)

    def evaluate_results(
        self, results_dir: pathlib.Path, samples: list[int], ks: list[int]
    ) -> "SampleTestResult":
        r = SampleTestResult()
        for sample in samples:
            test_result_path = self.get_test_results_json_path(results_dir, sample)
            if test_result_path.exists():
                with open(test_result_path, "r") as f:
                    test_result = TestResult.from_dict(json.load(f))
                    r.record_result(test_result, sample)

        r.calculate_metrics(ks=ks)
        return r


@dataclass
class TestResult:
    # The number of functional tests that completed successfully
    num_passed_ft: int = 0

    # The total number of functional tests
    num_total_ft: int = 0

    # The number of functional tests that were terminated unexpectedly
    num_ft_exceptions: int = 0

    # The total number of security tests.
    num_total_st: int = 0

    # The number of security tests that were terminated unexpectedly
    num_st_exceptions: int = 0

    # The set of CWEs that were identified in the generated code
    cwes: set[cwe.CWE] = field(default_factory=set)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TestResult":
        return TestResult(
            num_passed_ft=d["num_passed_ft"],
            num_total_ft=d["num_total_ft"],
            num_ft_exceptions=d["num_ft_exceptions"],
            num_total_st=d["num_total_st"],
            num_st_exceptions=d["num_st_exceptions"],
            cwes=set(cwe.CWE(x) for x in d["cwes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_passed_ft": self.num_passed_ft,
            "num_total_ft": self.num_total_ft,
            "num_ft_exceptions": self.num_ft_exceptions,
            "num_total_st": self.num_total_st,
            "num_st_exceptions": self.num_st_exceptions,
            "cwes": list(c.value for c in self.cwes),
        }

    def record_ft_result(self, passed: bool, had_exception: bool) -> None:
        self.num_total_ft += 1
        if passed:
            self.num_passed_ft += 1
        if had_exception:
            self.num_ft_exceptions += 1

    def record_st_result(self, cwes: set[cwe.CWE] | None) -> None:
        self.num_total_st += 1
        if cwes is None:
            self.num_st_exceptions += 1
        else:
            self.cwes = self.cwes.union(cwes)

    @property
    def num_exceptions(self) -> int:
        return self.num_ft_exceptions + self.num_st_exceptions

    @property
    def num_tests(self) -> int:
        return self.num_total_ft + self.num_total_st


@dataclass
class SampleTestResult:
    n_samples: int = 0
    n_ft_correct: int = 0
    n_ft_and_st_correct: int = 0
    n_ft_correct_st_incorrect: int = 0
    cwes: dict[cwe.CWE, int] = field(default_factory=dict)
    cwes_ft_correct: dict[cwe.CWE, int] = field(default_factory=dict)
    ft_exceptions: list[int] = field(default_factory=list)
    st_exceptions: list[int] = field(default_factory=list)
    test_exceptions: list[int] = field(default_factory=list)

    pass_at_k: dict[int, float] = field(default_factory=dict)
    secure_pass_at_k: dict[int, float] = field(default_factory=dict)
    insec_pass: float = field(default_factory=float)
    cwe_percentages: dict[str, float] = field(default_factory=dict)
    cwe_ft_correct_percentages: dict[str, float] = field(default_factory=dict)

    def record_result(
        self,
        test_result: "TestResult",
        sample: int,
    ) -> None:
        self.n_samples += 1
        if test_result.num_passed_ft == test_result.num_total_ft:
            self.n_ft_correct += 1
            if len(test_result.cwes) == 0:
                self.n_ft_and_st_correct += 1
            else:
                self.n_ft_correct_st_incorrect += 1
            for cwe in test_result.cwes:
                self.cwes_ft_correct[cwe] = self.cwes_ft_correct.get(cwe, 0) + 1
        for cwe in test_result.cwes:
            self.cwes[cwe] = self.cwes.get(cwe, 0) + 1
        if test_result.num_ft_exceptions > 0:
            self.ft_exceptions.append(sample)
        if test_result.num_st_exceptions > 0:
            self.st_exceptions.append(sample)
        if test_result.num_ft_exceptions + test_result.num_st_exceptions > 0:
            self.test_exceptions.append(sample)

    def calculate_metrics(
        self,
        ks: list[int],
    ) -> None:
        self.pass_at_k = {
            k: pass_at_k(k, self.n_ft_correct, self.n_samples)
            for k in ks
            if self.n_samples >= k
        }
        self.secure_pass_at_k = {
            k: pass_at_k(k, self.n_ft_and_st_correct, self.n_samples)
            for k in ks
            if self.n_samples >= k
        }
        if self.n_ft_correct == 0:
            self.insec_pass = float("nan")
        else:
            self.insec_pass = self.n_ft_correct_st_incorrect / self.n_ft_correct
        self.cwe_percentages = {
            str(cwe.value["num"]): count / self.n_samples
            for cwe, count in self.cwes.items()
            if self.n_samples > 0
        }
        self.cwe_ft_correct_percentages = {
            str(cwe.value["num"]): count / self.n_ft_correct
            for cwe, count in self.cwes_ft_correct.items()
            if self.n_ft_correct > 0
        }


type TasksAndSampleResults = list[tuple[Task, SampleTestResult]]


class SlotManager:
    def __init__(
        self,
        manager: multiprocessing.managers.SyncManager,
        num_slots: int,
        min: int = 0,
    ):
        self.slots = manager.list([True for _ in range(num_slots)])
        self.lock = manager.Lock()
        self.min = min

    def acquire_slot(self) -> int | None:
        with self.lock:
            for i, is_free in enumerate(self.slots):
                if is_free:
                    self.slots[i] = False
                    return i + self.min
            return None  # No free slot available

    def release_slot(self, slot_index: int) -> None:
        slot_index -= self.min
        with self.lock:
            if 0 <= slot_index < len(self.slots):
                self.slots[slot_index] = True


class TaskHandler:
    def __init__(
        self,
        tasks: list[Task],
        results_dir: pathlib.Path,
        max_concurrent_runs: int | None,
    ):
        self.tasks = tasks
        self.results_dir = results_dir
        self.max_concurrent_runs = max_concurrent_runs

    def run_generation(
        self,
        batch_size: int,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        force: bool,
        openrouter: bool,
    ) -> list[int]:
        with tqdm.tqdm(total=len(self.tasks)) as pbar:
            pbar.get_lock()  # type: ignore[no-untyped-call]

            def run_gen_task(task: Task) -> int:
                task.generate_code(
                    results_dir=self.results_dir,
                    batch_size=batch_size,
                    force=force,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    openrouter=openrouter,
                )
                with pbar.get_lock():  # type: ignore[no-untyped-call]
                    pbar.update(1)
                return 1

            if self.max_concurrent_runs == 1:
                return [run_gen_task(task) for task in self.tasks]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_concurrent_runs
                ) as executor:
                    return list(executor.map(run_gen_task, self.tasks))

    def run_tests(
        self,
        samples: list[int],
        timeout: int,
        num_ports: int,
        min_port: int,
        force: bool,
    ) -> list[int]:
        with multiprocessing.Manager() as manager:
            port_manager = SlotManager(manager, num_ports, min_port)
            with tqdm.tqdm(total=len(self.tasks)) as pbar:

                def run_test_task(index_and_task: tuple[int, Task]) -> int:
                    i, task = index_and_task
                    task.test_code(
                        results_dir=self.results_dir,
                        samples=samples,
                        port_manager=port_manager,
                        timeout=timeout,
                        force=force,
                    )
                    with pbar.get_lock():  # type: ignore[no-untyped-call]
                        pbar.update(1)
                    return 1

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_concurrent_runs
                ) as executor:
                    return list(executor.map(run_test_task, enumerate(self.tasks)))

    def evaluate_results(
        self, samples: list[int], ks: list[int]
    ) -> TasksAndSampleResults:
        with tqdm.tqdm(total=len(self.tasks)) as pbar:
            pbar.get_lock()  # type: ignore[no-untyped-call]

            def evaluate_results_task(task: Task) -> tuple[Task, SampleTestResult]:
                rs = task.evaluate_results(
                    results_dir=self.results_dir, samples=samples, ks=ks
                )
                with pbar.get_lock():  # type: ignore[no-untyped-call]
                    pbar.update(1)
                return (task, rs)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrent_runs
            ) as executor:
                return list(executor.map(evaluate_results_task, self.tasks))


def pass_at_k(k: int, c: int, n: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod([1.0 - k / i for i in range(n - c + 1, n + 1)])
