# -*- coding: utf-8 -*-
"""
@author: magne.lauritzen
"""
from src.threaded_videocapture import ThreadedVideoCapture
import logging
import time
import unittest
from time import perf_counter
from math import inf

import numpy as np

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("tests")


class TestThreadedVideoCapture(unittest.TestCase):
    def assert_normal_behavior(self, tvc: ThreadedVideoCapture):
        self.assertTrue(tvc.isOpened())
        self.assertTrue(tvc.is_alive)
        timeout = tvc.timeout
        get_frame_within = perf_counter() + timeout / 2 if timeout > 0 else perf_counter() + 1
        self.assertGreater(get_frame_within, 0, f"get_frame_within={get_frame_within}")
        while perf_counter() < get_frame_within:
            ret, frame = tvc.read()
            if ret:
                break
        self.assertTrue(ret)
        self.assertIsInstance(frame, np.ndarray)
        ret, frame = tvc.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        time.sleep(tvc.timeout + 1)
        self.assertFalse(tvc.is_alive)
        ret, frame = tvc.read()
        self.assertIsNone(ret)
        self.assertIsNone(frame)

    def test_basic(self):
        timeout = 0.2
        with ThreadedVideoCapture("testimage_1.jpg", timeout=timeout) as tvc:
            self.assert_normal_behavior(tvc)

    def test_timeout(self):
        """ Test that ThreadedVideoCapture will use the timeout values we set it to. """
        with ThreadedVideoCapture("testimage_1.jpg", timeout=1) as tvc:
            time.sleep(0.1)  # Permit time for VideoCaptureThread to initialize
            tvc_timeout = tvc.timeout
            self.assertEqual(tvc_timeout, 1)
            time.sleep(0.8)
            self.assertTrue(tvc.is_alive)
            tvc.timeout = 2
            time.sleep(0.1)  # Permit time for VideoCaptureThread to receive the new setting
            tvc_timeout = tvc.timeout
            self.assertEqual(tvc_timeout, 2)
            time.sleep(0.5)
            self.assertTrue(tvc.is_alive)
            tvc.timeout = 0
            time.sleep(0.1)  # Permit time for VideoCaptureThread to receive the new setting
            tvc_timeout = tvc.timeout
            self.assertEqual(tvc_timeout, inf)
            time.sleep(0.5)
            self.assertTrue(tvc.is_alive)
            tvc.timeout = -1
            time.sleep(0.1)  # Permit time for VideoCaptureThread to receive the new setting
            tvc_timeout = tvc.timeout
            self.assertEqual(tvc_timeout, inf)
            time.sleep(0.5)
            self.assertTrue(tvc.is_alive)
            tvc.timeout = 0.1
            time.sleep(0.1)  # Permit time for VideoCaptureThread to receive the new setting
            assert not tvc.is_alive

    def test_pollrate(self):
        """ Test that ThreadedVideoCapture uses the poll rates we set it to. """
        with ThreadedVideoCapture("testimage_1.jpg", timeout=10, poll_rate=10) as tvc:
            self.assertEqual(tvc.poll_rate, 10)
            time.sleep(1.1)
            self.assertTrue(9 < tvc.actual_poll_rate < 11, f"actual_poll_rate = {tvc.actual_poll_rate}")
            tvc.poll_rate = 100
            time.sleep(0.1)  # Permit time for VideoCaptureThread to receive the new setting
            self.assertEqual(tvc.poll_rate, 100)
            time.sleep(2.1)
            self.assertTrue(95 < tvc.actual_poll_rate < 105, f"actual_poll_rate = {tvc.actual_poll_rate}")

    def test_logging(self):
        """ Test that logging works correctly with or without providing a Logger object when instantiating
        ThreadedVideoCapture."""
        with self.assertLogs('TVC', level='DEBUG') as lc:
            with ThreadedVideoCapture("testimage_1.jpg"):
                time.sleep(0.1)  # Permit time for logger to do its thing
                self.assertEqual(lc.output[-1], 'INFO:TVC:Timeout set to 1 seconds.')

        testlogger = logging.getLogger("testlogger")
        with self.assertLogs('testlogger', level='DEBUG') as lc:
            with ThreadedVideoCapture("testimage_1.jpg", logger=testlogger):
                time.sleep(0.1)  # Permit time for logger to do its thing
                self.assertEqual(lc.output[-1], 'INFO:testlogger:Timeout set to 1 seconds.')

    def test_isalive(self):
        """ Test that the isalive() method returns expected results. """
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            self.assertTrue(tvc.is_alive)
        self.assertFalse(tvc.is_alive)

    def test_release(self):
        """ Test that the release method can close the ThreadedVideoCapture"""
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            self.assertTrue(tvc.is_alive)
            tvc.release()
            self.assertFalse(tvc.is_alive)

    def test_open(self):
        """ Test that we can succesfully re-open a ThreadedVideoCapture that has been released. """
        with ThreadedVideoCapture("testimage_1.jpg") as tvc:
            tvc.logger.setLevel(logging.INFO)
            self.assert_normal_behavior(tvc)
            self.assertTrue(tvc.open("testimage_1.jpg"))
            self.assert_normal_behavior(tvc)


if __name__ == '__main__':
    unittest.main()
