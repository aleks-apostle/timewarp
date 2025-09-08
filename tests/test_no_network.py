from __future__ import annotations

import socket

import pytest

from timewarp.replay.no_network import NetworkBlocked, no_network


def test_no_network_blocks_socket_connect() -> None:
    with no_network():
        s = socket.socket()
        with pytest.raises(NetworkBlocked):
            s.connect(("127.0.0.1", 9))  # discard port, should be blocked before syscall
