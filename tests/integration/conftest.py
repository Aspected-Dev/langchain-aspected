"""Shared fixtures for integration tests that run against a real Aspected server.

The Aspected server is started in Docker via `testcontainers`.  The image
(and tag) used can be configured through the ``ASPECTED_IMAGE`` environment
variable, defaulting to ``xillio/aspected:latest``. This makes it easy to
pin/override the version from CI (e.g. GitHub Actions repository/organization
variables) without touching test code.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator

import pytest
from aspected_client import AspectedClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

# Image (and tag) of the Aspected server to test against. Configurable via
# environment variable so CI can pin a specific version.
ASPECTED_IMAGE = os.environ.get("ASPECTED_IMAGE", "xillio/aspected:latest")

# Port the Aspected server listens on inside the container.
ASPECTED_CONTAINER_PORT = 8080

# How long to wait for the server to become ready before giving up.
ASPECTED_STARTUP_TIMEOUT = int(os.environ.get("ASPECTED_STARTUP_TIMEOUT", "120"))

# Substring logged by the server once it is ready to accept connections.
_READY_LOG_PATTERN = "listening at"


@pytest.fixture(scope="session")
def aspected_container() -> Iterator[DockerContainer]:
    """Start a single Aspected server container for the whole test session."""
    container = (
        DockerContainer(ASPECTED_IMAGE)
        .with_env("ASPECTED_ACCEPT_EULA", "true")
        .with_exposed_ports(ASPECTED_CONTAINER_PORT)
        .waiting_for(
            LogMessageWaitStrategy(_READY_LOG_PATTERN).with_startup_timeout(
                ASPECTED_STARTUP_TIMEOUT
            )
        )
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def aspected_url(aspected_container: DockerContainer) -> str:
    """Base URL of the running Aspected server."""
    host = aspected_container.get_container_host_ip()
    port = aspected_container.get_exposed_port(ASPECTED_CONTAINER_PORT)
    return f"http://{host}:{port}"


@pytest.fixture
def aspected_client(aspected_url: str) -> Iterator[AspectedClient]:
    """A fresh :class:`~aspected_client.AspectedClient` pointed at the test server."""
    client = AspectedClient(url=aspected_url)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def index_name(aspected_client: AspectedClient) -> Iterator[str]:
    """A unique index name, automatically deleted after the test.

    The community edition of the Aspected server enforces a small maximum
    number of indexes, so every test must clean up after itself instead of
    leaving indexes lying around.
    """
    name = f"test-index-{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        # Index may not exist (never created, or already deleted by the test
        # itself) -- that's fine, nothing to clean up.
        with contextlib.suppress(Exception):
            aspected_client.delete_index(name)
