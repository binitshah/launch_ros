# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Launching Nodes on other machines.

A machine running `ros2 launch --serve` exposes two services on a node named after the machine:

- `/<machine>/launch_node` takes a JSON node spec and launches it on that machine.
- `/<machine>/stop_nodes` stops every node launched on behalf of a given requester.

For this prototype both use `diagnostic_msgs/srv/AddDiagnostics`, which is just
`string --- bool success, string message`: the JSON goes in `load_namespace`.
"""

import asyncio
import json
import os
import pathlib
import re
import tempfile
import threading
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

from diagnostic_msgs.srv import AddDiagnostics
import launch
from launch.event_handlers import OnShutdown
import launch.logging
import yaml

from .ros_adapters import get_ros_node

LAUNCH_NODE_SERVICE = 'launch_node'
STOP_NODES_SERVICE = 'stop_nodes'

_LOCAL_MACHINE_NAME_GLOBAL = 'launch_ros_local_machine_name'
_REMOTE_NODE_CLIENT_GLOBAL = 'launch_ros_remote_node_client'

# How long launch shutdown waits for each machine to acknowledge stopping its nodes.
_STOP_TIMEOUT_SEC = 2.0


def machine_name_to_node_name(machine_name: Text) -> Text:
    """Turn an arbitrary machine name (e.g. a hostname) into a valid ROS node name."""
    node_name = re.sub(r'[^A-Za-z0-9_]', '_', machine_name)
    if not node_name or node_name[0].isdigit():
        node_name = 'machine_' + node_name
    return node_name


def machine_service_name(machine_name: Text, service: Text) -> Text:
    return '/{}/{}'.format(machine_name_to_node_name(machine_name), service)


def set_local_machine_name(context: launch.LaunchContext, machine_name: Text) -> None:
    """Record which machine this launch is running on; Nodes for it are run locally."""
    context.extend_globals({_LOCAL_MACHINE_NAME_GLOBAL: machine_name})


def get_local_machine_name(context: launch.LaunchContext) -> Optional[Text]:
    return getattr(context.locals, _LOCAL_MACHINE_NAME_GLOBAL, None)


def launch_node_remotely(
    context: launch.LaunchContext,
    machine_name: Text,
    spec: Dict[Text, Any]
) -> None:
    """Ask the given machine to launch a node from `spec`, without blocking the launch loop."""
    if not hasattr(context.locals, _REMOTE_NODE_CLIENT_GLOBAL):
        context.extend_globals({_REMOTE_NODE_CLIENT_GLOBAL: _RemoteNodeClient(context)})
    getattr(context.locals, _REMOTE_NODE_CLIENT_GLOBAL).launch_node(context, machine_name, spec)


def node_from_spec(spec: Dict[Text, Any]):
    """Build a local launch_ros Node from a spec made by Node on the requesting machine."""
    # Here to avoid cyclic import
    from .actions import Node

    parameters = []
    for param in spec['parameters']:
        if 'file' in param:
            # Parameter files only exist on the requesting machine, so their contents are sent.
            fd, path = tempfile.mkstemp(prefix='launch_params_', suffix='.yaml')
            with os.fdopen(fd, 'w') as f:
                f.write(param['file'])
            parameters.append(pathlib.Path(path))
        else:
            name, _, value = param['rule'].partition(':=')
            parameters.append({name: yaml.safe_load(value)})
    return Node(
        package=spec['package'],
        executable=spec['executable'],
        name=spec['name'],
        namespace=spec['namespace'],
        parameters=parameters or None,
        remappings=[tuple(rule) for rule in spec['remappings']],
        arguments=spec['arguments'] or None,
        ros_arguments=spec['ros_arguments'] or None,
        output='screen',
    )


class _RemoteNodeClient:
    """Sends launch requests to other machines and stops their nodes on launch shutdown."""

    def __init__(self, context: launch.LaunchContext):
        self.__logger = launch.logging.get_logger('launch_ros.distributed')
        # Getting the node first registers the ROS adapter's shutdown handler before ours, and
        # since handlers run newest first, ours runs while the node is still usable.
        self.__node = get_ros_node(context)
        self.__requester = self.__node.get_fully_qualified_name()
        self.__launch_clients = {}
        self.__stop_clients = {}
        # Remote nodes don't keep the launch service busy the way local processes do, so once
        # one is launched this future stands in for them until shutdown.
        self.__keep_alive = None
        context.register_event_handler(OnShutdown(on_shutdown=self._on_shutdown))

    def launch_node(self, context, machine_name, spec):
        if machine_name not in self.__launch_clients:
            self.__launch_clients[machine_name] = self.__node.create_client(
                AddDiagnostics, machine_service_name(machine_name, LAUNCH_NODE_SERVICE))
            self.__stop_clients[machine_name] = self.__node.create_client(
                AddDiagnostics, machine_service_name(machine_name, STOP_NODES_SERVICE))
        request = AddDiagnostics.Request()
        request.load_namespace = json.dumps({'requester': self.__requester, 'node': spec})
        context.add_completion_future(context.asyncio_loop.run_in_executor(
            None, self._launch_node, context, machine_name, spec, request))

    def _launch_node(self, context, machine_name, spec, request):
        client = self.__launch_clients[machine_name]
        description = "'{}' on machine '{}'".format(spec['executable'], machine_name)
        if not client.service_is_ready():
            self.__logger.info("Waiting for machine '{}' (is `ros2 launch --serve` running "
                               'there?)'.format(machine_name))
        while not client.wait_for_service(timeout_sec=1.0):
            if context.is_shutdown:
                self.__logger.warning('Abandoning launch of {}, due to shutdown.'
                                      .format(description))
                return

        event = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: event.set())
        while not event.wait(1.0):
            if context.is_shutdown:
                self.__logger.warning('Abandoning launch of {}, due to shutdown.'
                                      .format(description))
                future.cancel()
                return

        if future.exception() is not None:
            raise future.exception()
        response = future.result()
        if response.success:
            # Wait for the keep alive to exist before this future completes, so launch never
            # looks idle in between.
            asyncio.run_coroutine_threadsafe(
                self._keep_alive_until_shutdown(context), context.asyncio_loop).result()
            self.__logger.info('Launched {}'.format(description))
        else:
            self.__logger.error('Failed to launch {}: {}'.format(description, response.message))

    async def _keep_alive_until_shutdown(self, context):
        if self.__keep_alive is None and not context.is_shutdown:
            self.__keep_alive = context.asyncio_loop.create_future()
            context.add_completion_future(self.__keep_alive)

    def _on_shutdown(self, event, context):
        request = AddDiagnostics.Request()
        request.load_namespace = json.dumps({'requester': self.__requester})
        pending = {}
        for machine_name, client in self.__stop_clients.items():
            if client.service_is_ready():
                pending[machine_name] = client.call_async(request)
        for machine_name, future in pending.items():
            done = threading.Event()
            future.add_done_callback(lambda _, done=done: done.set())
            if not done.wait(_STOP_TIMEOUT_SEC):
                # The machine also stops our nodes once it sees this launch leave the graph.
                self.__logger.warning(
                    "Machine '{}' did not confirm stopping its nodes".format(machine_name))
        if self.__keep_alive is not None and not self.__keep_alive.done():
            self.__keep_alive.set_result(None)
