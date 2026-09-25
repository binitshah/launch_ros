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

"""Per-machine rclpy node that launches Nodes on behalf of other machines' launch files."""

import json
import os
import threading
from typing import Optional
from typing import Text

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import PackageNotFoundError
from diagnostic_msgs.srv import AddDiagnostics
import launch
from launch.events import IncludeLaunchDescription
from launch.events.process import ShutdownProcess
import launch.logging
from launch_ros.distributed import LAUNCH_NODE_SERVICE
from launch_ros.distributed import machine_name_to_node_name
from launch_ros.distributed import node_from_spec
from launch_ros.distributed import set_local_machine_name
from launch_ros.distributed import STOP_NODES_SERVICE
import rclpy
from rclpy.executors import SingleThreadedExecutor

MACHINE_ENV_VAR = 'MACHINE'


def get_machine_name(override: Optional[Text] = None) -> Optional[Text]:
    """Return the machine name from `override`, falling back to the $MACHINE env var."""
    if override:
        return override
    return os.environ.get(MACHINE_ENV_VAR) or None


def check_executable(package: Optional[Text], executable: Text) -> Optional[Text]:
    """Return why `executable` in `package` can't be run on this machine, or None if it can."""
    if package is None:
        return None
    try:
        prefix = get_package_prefix(package)
    except PackageNotFoundError:
        return "package '{}' not found".format(package)
    if not os.path.isfile(os.path.join(prefix, 'lib', package, executable)):
        return "executable '{}' not found in package '{}'".format(executable, package)
    return None


class MachineNode:
    """
    An rclpy node named after the machine, spun in its own context and thread.

    It serves `launch_ros.distributed`'s launch_node and stop_nodes services, launching Nodes
    into the given launch context. Nodes launched for a requester are stopped when it asks, or
    when it leaves the ROS graph (e.g. its launch was killed).
    """

    def __init__(self, machine_name: Text):
        self.__machine_name = machine_name
        self.__logger = launch.logging.get_logger('machine.' + machine_name)
        self.__launch_context = None
        self.__context = None
        self.__node = None
        self.__executor = None
        self.__thread = None
        self.__is_running = False
        # Only touched from the executor thread.
        self.__nodes_by_requester = {}
        self.__requesters_seen_in_graph = set()

    def start(self, launch_context: launch.LaunchContext):
        if self.__is_running:
            raise RuntimeError('Cannot start a MachineNode that is already running')
        self.__launch_context = launch_context
        # A non-default context keeps rclpy from installing signal handlers that
        # would fight with the ones installed by the LaunchService.
        self.__context = rclpy.Context()
        rclpy.init(args=[], context=self.__context)
        self.__node = rclpy.create_node(
            machine_name_to_node_name(self.__machine_name), context=self.__context)
        self.__node.create_service(
            AddDiagnostics, '~/' + LAUNCH_NODE_SERVICE, self._on_launch_node)
        self.__node.create_service(
            AddDiagnostics, '~/' + STOP_NODES_SERVICE, self._on_stop_nodes)
        self.__node.create_timer(1.0, self._stop_nodes_of_departed_requesters)
        self.__executor = SingleThreadedExecutor(context=self.__context)
        self.__executor.add_node(self.__node)
        self.__is_running = True
        self.__thread = threading.Thread(target=self._spin, daemon=True)
        self.__thread.start()
        self.__logger.info("machine node '{}' is serving '{}'".format(
            self.__node.get_fully_qualified_name(),
            self.__node.resolve_service_name('~/' + LAUNCH_NODE_SERVICE)))

    def _spin(self):
        while self.__is_running:
            self.__executor.spin_once(timeout_sec=0.1)

    def _emit_event(self, event):
        # Queued from the launch loop's own thread, without waiting on it: the loop may be
        # waiting on this thread in shutdown().
        context = self.__launch_context
        context.asyncio_loop.call_soon_threadsafe(context.emit_event_sync, event)

    def _on_launch_node(self, request, response):
        try:
            data = json.loads(request.load_namespace)
            requester = data['requester']
            spec = data['node']
            error = check_executable(spec['package'], spec['executable'])
            node = None if error else node_from_spec(spec)
        except Exception as e:
            error = 'invalid request: {!r}'.format(e)
        if error:
            self.__logger.error('Refusing to launch a node: {}'.format(error))
            response.success = False
            response.message = error
            return response
        self.__logger.info("launching '{}' for '{}'".format(spec['executable'], requester))
        self.__nodes_by_requester.setdefault(requester, []).append(node)
        self._emit_event(IncludeLaunchDescription(launch.LaunchDescription([node])))
        response.success = True
        return response

    def _on_stop_nodes(self, request, response):
        try:
            requester = json.loads(request.load_namespace)['requester']
        except Exception as e:
            response.success = False
            response.message = 'invalid request: {!r}'.format(e)
            return response
        self._stop_nodes(requester, 'on request')
        response.success = True
        return response

    def _stop_nodes_of_departed_requesters(self):
        in_graph = {
            (ns.rstrip('/') + '/' + name)
            for name, ns in self.__node.get_node_names_and_namespaces()
        }
        for requester in list(self.__nodes_by_requester):
            if requester in in_graph:
                self.__requesters_seen_in_graph.add(requester)
            elif requester in self.__requesters_seen_in_graph:
                self._stop_nodes(requester, 'as it left the ROS graph')

    def _stop_nodes(self, requester, reason):
        self.__requesters_seen_in_graph.discard(requester)
        nodes = self.__nodes_by_requester.pop(requester, [])
        if nodes:
            self.__logger.info("stopping {} node(s) of '{}' {}".format(
                len(nodes), requester, reason))
        for node in nodes:
            self._emit_event(ShutdownProcess(
                process_matcher=launch.events.matches_action(node)))

    def shutdown(self):
        if not self.__is_running:
            return
        self.__is_running = False
        self.__thread.join()
        self.__executor.shutdown()
        self.__node.destroy_node()
        rclpy.shutdown(context=self.__context)
        self.__thread = None
        self.__executor = None
        self.__node = None
        self.__context = None
        self.__launch_context = None


def machine_node_actions(machine_name: Text):
    """
    Return launch actions that run a MachineNode for the lifetime of the LaunchService.

    The actions also mark the launch as running on `machine_name`, so Nodes with that machine
    run locally; include them before any launch file.
    The node is started when the actions are executed and shut down on the
    launch system's Shutdown event.
    """
    machine_node = MachineNode(machine_name)

    def start(context):
        set_local_machine_name(context, machine_name)
        machine_node.start(context)
        return None

    return [
        launch.actions.RegisterEventHandler(launch.event_handlers.OnShutdown(
            on_shutdown=lambda *args, **kwargs: machine_node.shutdown())),
        launch.actions.OpaqueFunction(function=start),
    ]
