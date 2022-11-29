# -*- coding: utf-8 -*-
"""
Author: Magne Lauritzen
"""

import logging
import queue
import threading
import time
from collections import deque
from enum import Enum, auto
from math import inf
from statistics import mean
from typing import Any, Optional, Tuple, Union, Deque, Dict

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]

module_logger = logging.getLogger("TVC")


# noinspection PyArgumentList
class InputVariables(Enum):
    TIMEOUT = auto()
    POLLRATE = auto()
    QUIT = auto()


# noinspection PyArgumentList
class OutputVariables(Enum):
    POLLRATE = auto()
    FPS = auto()


def queue_put(q: Union[queue.Queue, queue.LifoQueue], data: Any) -> None:
    """ Places an item in a Queue. If the queue is full, it removes one element to make room. """
    if q.full():
        q.get(block=False)
    q.put(data, block=False)


class VideoCaptureThread(threading.Thread):
    def __init__(self, name: str = None, args: Tuple = (), kwargs: Dict[str, Any] = None,
                 *, daemon: bool = None):

        super().__init__(target=self.reader, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.poll_period = 0.
        self.timeout = 0.
        self.prev_frame_timestamp = 0.
        self.prev_poll_timestamp = 0.
        self.prev_emit_timestamp = 0.
        self.prev_settings_check_timestamp = 0.
        self.poll_period_deque: Deque[float] = deque()
        self.frame_period_deque: Deque[float] = deque()
        self.quitflag = False
        self.logger: logging.Logger
        self.capture: cv2.VideoCapture
        self.frame_queue: queue.Queue
        self.out_queue: queue.Queue
        self.in_queue: queue.Queue

    @property
    def time_since_prev_frame(self) -> float:
        return time.perf_counter() - self.prev_frame_timestamp

    def _set_timeout(self, new_timeout: float):
        self.timeout = new_timeout
        infostr = f"Timeout set to {self.timeout} seconds."
        if self.timeout <= 0:
            self.timeout = inf
            infostr += " VideoCaptureThread will never timeout and must be stopped with " \
                       "a QUIT signal."
        self.logger.info(infostr)

    def _set_pollrate(self, poll_rate: float):
        poll_rate = poll_rate
        if poll_rate is None:
            self.poll_period = -inf
            self.logger.info("Poll rate set to unlimited.")
        elif poll_rate <= 0:
            self.logger.warning(
                f"Attempted to set poll rate less or equal to 0: {poll_rate}. Poll "
                f"rate remains unchanged at {1 / self.poll_period} Hz")
        else:
            self.poll_period = 1 / poll_rate
            self.logger.info(f"Poll rate set to {poll_rate} Hz")

    def _read_frame(self):
        this_poll_timestamp = time.perf_counter()
        if self.prev_poll_timestamp != 0:
            self.poll_period_deque.append(this_poll_timestamp - self.prev_poll_timestamp)
        self.prev_poll_timestamp = this_poll_timestamp
        frame_available = self.capture.grab()
        if frame_available:
            this_frame_timestamp = time.perf_counter()
            if self.prev_frame_timestamp != 0:
                self.frame_period_deque.append(this_frame_timestamp - self.prev_frame_timestamp)
            ret, frame = self.capture.retrieve()
            queue_put(self.frame_queue, (ret, frame))
            self.prev_frame_timestamp = this_frame_timestamp

    def _emit_statistics(self):
        if self.prev_poll_timestamp - self.prev_emit_timestamp > 1:
            if len(self.poll_period_deque) > 0:
                mean_poll_rate = 1 / mean(self.poll_period_deque)
            else:
                mean_poll_rate = None
            self.poll_period_deque.clear()
            if len(self.frame_period_deque) > 0:
                mean_fps = 1 / mean(self.frame_period_deque)
            else:
                mean_fps = None
            self.frame_period_deque.clear()
            queue_put(self.out_queue, {OutputVariables.POLLRATE: mean_poll_rate, OutputVariables.FPS: mean_fps})
            self.prev_emit_timestamp = time.perf_counter()

    def _poll_for_settings(self):
        # Get time since last poll and sleep if needed to reach poll_rate
        time_since_prev_poll = time.perf_counter() - self.prev_poll_timestamp
        sleep_until = time.perf_counter() + max(0., self.poll_period - time_since_prev_poll)
        # Check for new settings at up to 100hz while sleeping
        while True:
            now = time.perf_counter()
            if now - self.prev_settings_check_timestamp >= 0.01:
                try:
                    in_data: Tuple[InputVariables, Any] = self.in_queue.get(block=False)
                    if in_data[0] == InputVariables.TIMEOUT:
                        self._set_timeout(in_data[1])
                    elif in_data[0] == InputVariables.POLLRATE:
                        self._set_pollrate(in_data[1])
                    if in_data[0] == InputVariables.QUIT:
                        self.logger.info("Received QUIT signal.")
                        self.quitflag = True
                        break
                except queue.Empty:
                    pass
                self.prev_settings_check_timestamp = now

            if time.perf_counter() > sleep_until:
                break

    def reader(self, capture: cv2.VideoCapture, frame_queue: queue.Queue, out_queue: queue.Queue,
               in_queue: queue.Queue, timeout: float, poll_rate: Optional[float], logger: logging.Logger) -> None:
        self.logger = logger
        self.capture = capture
        self.frame_queue = frame_queue
        self.out_queue = out_queue
        self.in_queue = in_queue
        try:
            if self.capture.isOpened():
                self._set_pollrate(poll_rate)
                self._set_timeout(timeout)
                self.prev_frame_timestamp = time.perf_counter()
                while self.time_since_prev_frame < self.timeout and not self.quitflag:
                    # Poll for new settings or a QUIT flag
                    self._poll_for_settings()

                    # Read a frame from the VideoCapture if one is available
                    self._read_frame()

                    # Emit statistics: poll rate and fps
                    self._emit_statistics()

                if self.time_since_prev_frame >= self.timeout:
                    logger.info("VideoCaptureThread timed out. Seconds since last frame: "
                                f"{self.time_since_prev_frame:.1f} Timeout setting: {self.timeout:.1f}s.")
        finally:
            queue_put(frame_queue, (None, None))

    def run(self) -> None:
        self.exc = None
        try:
            super().run()
        except BaseException as e:
            self.exc = e

    def join(self, timeout: float = None) -> None:
        super().join(timeout)
        if self.exc:
            raise self.exc


class ThreadedVideoCapture:
    """
    This is a class that can be used in place of cv2.VideoCapture. It performs frame reads in a separate thread, which
    frees up the main thread to do other tasks.

    ThreadedVideoCapture can be used in the same way as VideoCapture. It takes a few extra keyword arguments. See
    the __init__ method for details.

    ThreadedVideoCapture can be used as a context manager. If you do not use it as a context manager, you must
    ensure to call release() when you are done with it.
    """

    def __init__(self, *args: Any, timeout: float = 1, poll_rate: Optional[float] = None, frame_queue_size: int = 1,
                 logger: logging.Logger = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        args             : Positional arguments to cv2.VideoCapture
        timeout          : Threaded reader timeout. If no new frames are received within 'timeout' seconds, the thread
            quits. Default is 1 second. Set it to 0 or a negative number to never time out.
        poll_rate        : Threaded reader polling rate. The fastest rate (in calls per second) to capture.grab(). Set
            this at least as high as your expected frame rate, preferably twice as large for headroom. Default None,
            which means unlimited poll rate.
        frame_queue_size : The length of the queue that holds frames fetched by the threaded reader. If the threaded
            reader has grabbed a new frame and the queue is full, the oldest element is removed to make room. Default 1.
        logger           : Custom logger. If not provided, ThreadedVideoCapture will create its own logger which can be
            accessed with ThreadedVideoCapture.logger.
        kwargs           : Keyword arguments to cv2.VideoCapture
        """
        self.logger = logger if logger is not None else module_logger
        self.frame_queue: queue.Queue[Tuple[Optional[bool], Optional[np.ndarray]]] = queue.Queue(
            maxsize=frame_queue_size)
        self._capture = cv2.VideoCapture()
        self._output_queue: queue.Queue[Dict[OutputVariables, Any]] = queue.Queue(maxsize=1)
        self._input_queue: queue.Queue[Tuple[InputVariables, Any]] = queue.Queue()
        self._timeout = timeout
        self._poll_rate = poll_rate
        self._return_data: Dict[OutputVariables, Any] = {}
        self._threaded_reader: Optional[VideoCaptureThread] = None
        self.open(*args, **kwargs)

    @property
    def is_alive(self) -> bool:
        if self._threaded_reader is not None:
            return self._threaded_reader.is_alive()
        else:
            return False

    def read(self) -> Tuple[Optional[bool], Optional[np.ndarray]]:
        """
        Returns one frame from the frame queue. If the queue is empty, returns (False, None). If the threaded reader
        has quit, returns (None, None).

        Returns
        -------
        ret     : Optional bool. True if a frame is available. If the threaded reader has quit, it is None.
        frame   : np.ndarray or None.
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return False, None

    def open(self, *args: Any, **kwargs: Any) -> bool:
        """
        Opens a video stream. Wraps VideoCapture.open and takes identical arguments.

        Parameters
        ----------
        args    : Arguments for VideoCapture.open
        kwargs  : Keyword arguments for VideoCapture.open

        Returns
        -------
        success : bool. Whether the source was successfully opened.
        """
        self.release()
        success: bool = self._capture.open(*args, **kwargs)
        if success:
            self.logger.info(f"Sucessfully opened VideoCapture with args={args}, kwargs={kwargs}")
        else:
            self.logger.info(f"Failed at opening VideoCapture with args={args}, kwargs={kwargs}")
        self._create_thread()
        return success

    def set(self, propId: int, value: int) -> bool:
        """ Wrapper for VideoCapture.set. """
        retval: bool = self._capture.set(propId, value)
        return retval

    def get(self, propId: int) -> float:
        """ Wrapper for VideoCapture.get. """
        retval: float = self._capture.get(propId)
        return retval

    def getBackendName(self) -> str:
        """ Wrapper for VideoCapture.getBackendName. """
        retval: str = self._capture.getBackendName()
        return retval

    def isOpened(self) -> bool:
        """ Wrapper for VideoCapture.isOpened. """
        retval: bool = self._capture.isOpened()
        return retval

    def release(self) -> None:
        """
        Stops the reader thread and releases the VideoCapture. Clears all queues.
        """
        if self._threaded_reader is not None and self._threaded_reader.is_alive():
            self._input_queue.put((InputVariables.QUIT, None))
            self.logger.info("Waiting for VideoCaptureThread to quit.")
            self._threaded_reader.join()
        self._capture.release()
        # Clear the queues
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        with self._output_queue.mutex:
            self._output_queue.queue.clear()
        with self._output_queue.mutex:
            self._output_queue.queue.clear()

    @property
    def actual_poll_rate(self) -> Optional[float]:
        """ Returns the most recent actual poll rate reported by the reader thread. The poll rate value is updated
        once per second."""
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.POLLRATE, None)

    @property
    def fps(self) -> Optional[float]:
        """ Returns the most recent frames per second reported by the reader thread. The FPS value is updated once per
        second."""
        try:
            self._return_data = self._output_queue.get(block=False)
        except queue.Empty:
            pass
        finally:
            return self._return_data.get(OutputVariables.FPS, None)

    @property
    def timeout(self) -> float:
        """ Return the reader thread timeout value. If no reader thread exists, returns the timeout value that will be
         applied to the next reader thread that is created. """
        if self._threaded_reader is not None:
            return self._threaded_reader.timeout
        else:
            return self._timeout

    @timeout.setter
    def timeout(self, timeout: Optional[float]) -> None:
        """
        Set the reader thread timeout in seconds. If no new frames have been received from the source (camera, file,
        stream, etc), within this time, the reader thread will quit. Set timeout to None to never time out.

        Parameters
        ----------
        timeout : Timeout value in seconds. Optional.
        """
        self._input_queue.put((InputVariables.TIMEOUT, timeout))
        self._timeout = timeout

    @property
    def poll_rate(self) -> float:
        """ Returns the reader thread poll rate. If no reader thread exists, returns the poll rate value that will be
         applied to the next reader thread that is created. """
        if self._threaded_reader is not None:
            return 1/self._threaded_reader.poll_period
        else:
            return self._poll_rate

    @poll_rate.setter
    def poll_rate(self, poll_rate: Optional[float]) -> None:
        """
        Set the reader thread poll rate. This value is the highest rate (in calls per second) to capture.grab() that
        the reader thread is allowed to make. For unlimited poll rate, set it to None.

        Parameters
        ----------
        poll_rate   : Poll rate in calls per second
        """
        if poll_rate is not None:
            assert poll_rate > 0, "Poll rate may not be 0 or negative."
        self._input_queue.put((InputVariables.POLLRATE, poll_rate))
        self._poll_rate = poll_rate

    def _create_thread(self):
        self._threaded_reader = VideoCaptureThread(args=(self._capture,
                                                         self.frame_queue,
                                                         self._output_queue,
                                                         self._input_queue,
                                                         self._timeout,
                                                         self._poll_rate,
                                                         self.logger), daemon=True)
        self._threaded_reader.start()

    def __enter__(self) -> 'ThreadedVideoCapture':
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()

if __name__ == "__main__":
    """ Example code that opens up a webcam stream. """
    with ThreadedVideoCapture(0) as tvc:  # Open webcam stream
        while True:
            ret, frame = tvc.read()
            if ret:  # ret is True if a frame was obtained with tvc.read()
                cv2.imshow('frame', frame)
            if ret is None:  # ret is None if tvc has stopped.
                print("End of stream.")
                break
            if cv2.waitKey(1) == ord('q'):
                break
