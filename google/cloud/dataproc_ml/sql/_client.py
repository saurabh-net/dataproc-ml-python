# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process-level Vertex AI clients and event loop used by the AI functions.

Spark imports a module only once per Python worker process, so the state held
here is shared by every task the worker runs. That is what lets one HTTPS
connection pool serve all the partitions a worker processes, instead of being
rebuilt for each one.
"""

import asyncio
import logging
import os
import threading
from typing import Awaitable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The location used when neither the caller nor the environment supplies one.
# The global endpoint is used because the current Gemini models are served
# there and it spreads load across regions; a single region can be requested
# with the location argument when data residency requires it.
DEFAULT_LOCATION = "global"

_LOOP_LOCK = threading.Lock()
_LOOP: Optional[asyncio.AbstractEventLoop] = None

_CLIENT_LOCK = threading.Lock()
_CLIENTS: Dict[Tuple[Optional[str], Optional[str]], object] = {}

# Counts how often a cached client was reused. Only read by tests, to confirm
# that clients really are shared across batches and tasks.
_client_cache_hits = 0


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Returns the event loop shared by every task in this worker process.

    The asynchronous Vertex AI client owns an ``httpx.AsyncClient`` bound to
    the loop that created it. Spark calls a pandas UDF once per Arrow batch, so
    running :func:`asyncio.run` per batch would build and tear down a loop for
    every batch and leave any cached client bound to a closed one. Instead a
    single loop runs on a daemon thread for the life of the process.

    Returns:
        The running event loop.
    """
    global _LOOP  # pylint: disable=global-statement
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_LOOP.run_forever,
                name="dataproc-ml-ai-functions",
                daemon=True,
            ).start()
        return _LOOP


def run_coroutine(coro: Awaitable[_T]) -> _T:
    """Runs a coroutine on the shared loop and waits for its result.

    Args:
        coro: The coroutine to execute.

    Returns:
        Whatever the coroutine returns.
    """
    return asyncio.run_coroutine_threadsafe(coro, get_event_loop()).result()


def resolve_project_and_location(
    project: Optional[str], location: Optional[str]
) -> Tuple[Optional[str], str]:
    """Fills in the project from the environment and the location from the
    default when they are unset.

    Workers on Dataproc, Dataproc Serverless and GKE resolve their identity
    from Application Default Credentials, so the project normally does not
    have to be passed explicitly.

    The location deliberately ignores ``GOOGLE_CLOUD_LOCATION`` and
    ``GOOGLE_CLOUD_REGION``. A Dataproc cluster sets those to the region it
    runs in, which is rarely a statement about where a model should be
    called, and the current Gemini models are served only from the global
    endpoint. Honoring them would route the default model to a region that
    does not serve it, and because a failed row reports its error in
    ``status`` rather than raising, that would surface as a column of NULLs.
    A region can still be requested explicitly with the ``location``
    argument.

    Args:
        project: An explicitly configured project, or None.
        location: An explicitly configured location, or None.

    Returns:
        A (project, location) tuple. The project may still be None when it
        cannot be discovered, in which case the client reports the problem.
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not project:
        try:
            # Imported lazily: only needed when the project is not configured.
            import google.auth  # pylint: disable=import-outside-toplevel

            _, project = google.auth.default()
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Could not infer the project from ADC: %s", e)

    return project, location or DEFAULT_LOCATION


def get_client(project: Optional[str], location: Optional[str]):
    """Returns a client for the given project and location, creating it once.

    The first call in a worker process builds the client; later calls in that
    process, including calls from other Spark tasks, reuse it.

    Args:
        project: The Google Cloud project, or None to infer it.
        location: The Vertex AI location, or None to infer it.

    Returns:
        A ``google.genai.Client`` configured for Vertex AI.
    """
    global _client_cache_hits  # pylint: disable=global-statement

    key = (project, location)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is not None:
            _client_cache_hits += 1
            return client

        resolved_project, resolved_location = resolve_project_and_location(
            project, location
        )
        logger.info(
            "Creating a Vertex AI client for project=%s location=%s",
            resolved_project,
            resolved_location,
        )

        # Imported lazily so that importing this package does not pull in the
        # Vertex AI dependency until an AI function is evaluated.
        from google import genai  # pylint: disable=import-outside-toplevel

        client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=resolved_location,
        )
        _CLIENTS[key] = client
        return client


def get_generate_content(project: Optional[str], location: Optional[str]):
    """Returns the asynchronous ``generate_content`` of the cached client.

    Args:
        project: The Google Cloud project, or None to infer it.
        location: The Vertex AI location, or None to infer it.

    Returns:
        An awaitable callable accepting ``model``, ``contents`` and ``config``.
    """
    return get_client(project, location).aio.models.generate_content


def clear_cache() -> None:
    """Drops all cached clients. Intended for tests."""
    global _client_cache_hits  # pylint: disable=global-statement
    with _CLIENT_LOCK:
        _CLIENTS.clear()
        _client_cache_hits = 0
