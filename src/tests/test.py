# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""
from src.threaded_videocapture import ThreadedVideoCapture
import io
import logging
import time
import unittest
from time import perf_counter

import numpy as np

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("tests")


class TestThreadedVideoCapture(unittest.TestCase):
    def assert_normal_behavior(self, tvc: ThreadedVideoCapture):
        assert tvc.isOpened()
        assert tvc.is_alive
        timeout = tvc.timeout
        get_frame_within = perf_counter() + timeout / 2 if timeout > 0 else perf_counter() + 1
        assert get_frame_within > 0, f"get_frame_within={get_frame_within}"
        while perf_counter() < get_frame_within:
            ret, frame = tvc.read()
            if ret:
                break
        assert ret, f"ret={ret}"
        assert isinstance(frame, np.ndarray), f"type of frame={type(frame)}"
        ret, frame = tvc.read()
        assert not ret, f"ret={ret}"
        assert frame is None, f"frame={frame}"
        time.sleep(tvc.timeout + 1)
        assert not tvc.is_alive
        ret, frame = tvc.read()
        assert ret is None, f"ret={ret}"
        assert frame is None, f"frame={frame}"

    def test_basic(self):
        timeout = 0.2
        with ThreadedVideoCapture("testimage_1.jpg", timeout=timeout) as tvc:
            self.assert_normal_behavior(tvc)

    def test_timeout(self):
        """ Test that ThreadedVideoCapture will use the timeout values we set it to. """
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
        """ Test that ThreadedVideoCapture uses the poll rates we set it to. """
        with ThreadedVideoCapture("testimage_1.jpg", timeout=10, poll_rate=10) as tvc:
            assert tvc.poll_rate == 10
            time.sleep(1.1)
            assert 9 < tvc.actual_poll_rate < 11
            tvc.poll_rate = 100
            assert tvc.poll_rate == 100
            time.sleep(2.1)
            assert 95 < tvc.actual_poll_rate < 105

    def test_logging(self):
        """ Test that logging works correctly with or without providing a Logger object when instantiating
        ThreadedVideoCapture."""
        log_capture_string = io.StringIO()
        handler = logging.StreamHandler(log_capture_string)
        handler.setLevel(logging.DEBUG)
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            tvc.logger.setLevel(logging.DEBUG)
            tvc.logger.addHandler(handler)
        assert log_capture_string.getvalue() == "Received QUIT signal.\n", f"Contents is " \
                                                                           f"\"{repr(log_capture_string.getvalue())}\"."
        tvc.logger.removeHandler(handler)
        log_capture_string.truncate(0)
        log_capture_string.seek(0)
        testlogger = logging.getLogger("testlogger")
        with ThreadedVideoCapture("testimage_1.jpg", logger=testlogger) as tvc:
            tvc.logger.setLevel(logging.DEBUG)
            tvc.logger.addHandler(handler)
        assert log_capture_string.getvalue() == "Received QUIT signal.\n", f"Contents is " \
                                                                           f"\"{repr(log_capture_string.getvalue())}\"."
        log_capture_string.close()

    def test_isalive(self):
        """ Test that the isalive() method returns expected results. """
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            assert tvc.is_alive
        assert not tvc.is_alive

    def test_release(self):
        """ Test that the release method can close the ThreadedVideoCapture"""
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            assert tvc.is_alive
            tvc.release()
            assert not tvc.is_alive

    def test_open(self):
        """ Test that we can succesfully re-open a ThreadedVideoCapture that has been released. """
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            tvc.logger.setLevel(logging.INFO)
            self.assert_normal_behavior(tvc)
            assert tvc.open("testimage_1.jpg")
            self.assert_normal_behavior(tvc)


if __name__ == '__main__':
    unittest.main()
