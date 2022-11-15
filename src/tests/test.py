# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""
import numpy as np
import time
from time import perf_counter
import unittest
import logging
from src.threaded_videocapture import ThreadedVideoCapture

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TestThreadedVideoCapture(unittest.TestCase):
    def assert_normal_behavior(self, tvc: ThreadedVideoCapture):
        assert tvc.isOpened()
        timeout = tvc.timeout
        get_frame_within = perf_counter() + timeout / 2 if timeout > 0 else perf_counter() + 1
        while perf_counter() < get_frame_within:
            ret, frame = tvc.read()
            if ret:
                break
        assert ret
        assert isinstance(frame, np.ndarray)
        ret, frame = tvc.read()
        assert not ret
        assert frame is None
        time.sleep(tvc.timeout + 0.1)
        ret, frame = tvc.read()
        assert ret is None
        assert frame is None

    def test_basic(self):
        timeout = 0.2
        with ThreadedVideoCapture("testimage_1.jpg", timeout=timeout) as tvc:
            self.assert_normal_behavior(tvc)

    def test_timeout(self):
        with ThreadedVideoCapture("testimage_1.jpg", timeout=1) as tvc:
            assert tvc.timeout == 1
            time.sleep(0.8)
            assert tvc.is_alive
            tvc.timeout = 2
            assert tvc.timeout == 2
            time.sleep(0.5)
            assert tvc.is_alive
            tvc.timeout = 0
            assert tvc.timeout == 0
            time.sleep(0.5)
            assert tvc.is_alive
            tvc.timeout = -1
            assert tvc.timeout == -1
            time.sleep(0.5)
            assert tvc.is_alive
            tvc.timeout = 0.1
            time.sleep(0.2)
            assert not tvc.is_alive

    def test_pollrate(self):
        with ThreadedVideoCapture("testimage_1.jpg", timeout=10, poll_rate=10) as tvc:
            assert tvc.poll_rate == 10
            time.sleep(1.1)
            assert 9 < tvc.actual_poll_rate < 11
            tvc.poll_rate = 100
            assert tvc.poll_rate == 100
            time.sleep(2.1)
            assert 95 < tvc.actual_poll_rate < 105



if __name__ == '__main__':
    unittest.main()
