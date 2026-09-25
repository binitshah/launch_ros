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

"""Per-machine rclpy node that lives for as long as a LaunchService is running."""

import os
import re
import threading
from typing import Optional
from typing import Text

import launch
import launch.logging
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

MACHINE_ENV_VAR = 'MACHINE'
TEST_TOPIC = '~/test_string'


def get_machine_name(override: Optional[Text] = None) -> Optional[Text]:
    """Return the machine name from `override`, falling back to the $MACHINE env var."""
    if override:
        return override
    return os.environ.get(MACHINE_ENV_VAR) or None


def machine_name_to_node_name(machine_name: Text) -> Text:
    """Turn an arbitrary machine name (e.g. a hostname) into a valid ROS node name."""
    node_name = re.sub(r'[^A-Za-z0-9_]', '_', machine_name)
    if not node_name or node_name[0].isdigit():
        node_name = 'machine_' + node_name
    return node_name


class MachineNode:
    """An rclpy node named after the machine, spun in its own context and thread."""

    def __init__(self, machine_name: Text):
        self.__machine_name = machine_name
        self.__logger = launch.logging.get_logger('machine.' + machine_name)
        self.__context = None
        self.__node = None
        self.__executor = None
        self.__thread = None
        self.__is_running = False

    def start(self):
        if self.__is_running:
            raise RuntimeError('Cannot start a MachineNode that is already running')
        # A non-default context keeps rclpy from installing signal handlers that
        # would fight with the ones installed by the LaunchService.
        self.__context = rclpy.Context()
        rclpy.init(args=[], context=self.__context)
        self.__node = rclpy.create_node(
            machine_name_to_node_name(self.__machine_name), context=self.__context)
        # TODO: replace this test subscription with one that receives the nodes to launch.
        self.__node.create_subscription(String, TEST_TOPIC, self._on_test_string, 10)
        self.__executor = SingleThreadedExecutor(context=self.__context)
        self.__executor.add_node(self.__node)
        self.__is_running = True
        self.__thread = threading.Thread(target=self._spin, daemon=True)
        self.__thread.start()
        self.__logger.info("machine node '{}' subscribed to '{}'".format(
            self.__node.get_fully_qualified_name(),
            self.__node.resolve_topic_name(TEST_TOPIC)))

    def _spin(self):
        while self.__is_running:
            self.__executor.spin_once(timeout_sec=0.1)

    def _on_test_string(self, msg: String):
        self.__logger.info('received test string: {}'.format(msg.data))

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


def machine_node_actions(machine_name: Text):
    """
    Return launch actions that run a MachineNode for the lifetime of the LaunchService.

    The node is started when the actions are executed and shut down on the
    launch system's Shutdown event.
    """
    machine_node = MachineNode(machine_name)

    def start(context):
        machine_node.start()
        return None

    return [
        launch.actions.RegisterEventHandler(launch.event_handlers.OnShutdown(
            on_shutdown=lambda *args, **kwargs: machine_node.shutdown())),
        launch.actions.OpaqueFunction(function=start),
    ]
