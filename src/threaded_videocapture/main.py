# -*- coding: utf-8 -*-
"""
Author: Magne Lauritzen
"""

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]
import threading
import queue
import time
from typing import Any, Optional, Tuple, Union, Deque, Dict
import logging
from enum import Enum, auto
from collections import deque
from statistics import mean
from math import inf


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

    def reader(self, capture: cv2.VideoCapture, frame_queue: queue.Queue, out_queue: queue.Queue,
               in_queue: queue.Queue, timeout: float, poll_rate: Optional[float]) -> None:
        try:
            poll_period_deque: Deque[float] = deque()
            frame_period_deque: Deque[float] = deque()
            if capture.isOpened():
                quitflag = False
                poll_period: Optional[float] = 1 / poll_rate if poll_rate else None
                time_since_frame = 0.
                last_emit_timestamp = 0.
                last_settings_poll_timestamp = 0.
                prev_frame_timestamp = time.time()
                prev_read_timestamp = time.perf_counter()
                while time_since_frame < timeout and not quitflag:
                    # Get time since last call and sleep if needed to reach poll_rate
                    time_since_last_read = time.perf_counter() - prev_read_timestamp
                    sleep_until = time.perf_counter() + max(0., poll_period - time_since_last_read)
                    # Poll new settings at 10hz while sleeping
                    while True:
                        now = time.perf_counter()
                        if now - last_settings_poll_timestamp > 0.1:
                            try:
                                in_data: Tuple[InputVariables, Any] = in_queue.get(block=False)
                                if in_data[0] == InputVariables.TIMEOUT:
                                    timeout = in_data[1]
                                    infostr = f"Timeout set to {timeout} seconds."
                                    if timeout <= 0:
                                        timeout = inf
                                        infostr += " VideoCaptureThread will never timeout and must be stopped with " \
                                                   "a QUIT signal."
                                    logging.info(infostr)
                                elif in_data[0] == InputVariables.POLLRATE:
                                    poll_rate = in_data[1]
                                    if poll_rate is None:
                                        poll_period = -inf
                                        logging.info(f"Poll rate set to unlimited.")
                                    elif in_data[1] <= 0:
                                        logging.warning(
                                            f"Attempted to set poll rate less or equal to 0: {in_data[1]}. Poll "
                                            f"rate remains unchanged at {1 / poll_period}")
                                    else:
                                        poll_period = 1 / in_data[1]
                                        logging.info(f"Poll rate set to {in_data[1]} Hz")
                                if in_data[0] == InputVariables.QUIT:
                                    logging.info(f"Received QUIT signal.")
                                    quitflag = True
                                    break
                            except queue.Empty:
                                pass
                            last_settings_poll_timestamp = now

                        if time.perf_counter() > sleep_until:
                            break

                    poll_period_deque.append(time.perf_counter() - prev_read_timestamp)
                    # Read frame
                    prev_read_timestamp = time.perf_counter()
                    frame_available = capture.grab()
                    if frame_available:
                        this_frame_timestamp = time.perf_counter()
                        frame_period_deque.append(this_frame_timestamp - prev_frame_timestamp)
                        ret, frame = capture.retrieve()
                        queue_put(frame_queue, (ret, frame))
                        prev_frame_timestamp = this_frame_timestamp
                    time_since_frame = prev_read_timestamp - prev_frame_timestamp

                    # Emit statistics every 1 second
                    if prev_read_timestamp - last_emit_timestamp > 1:
                        if len(poll_period_deque) > 0:
                            mean_poll_rate = 1 / mean(poll_period_deque)
                        else:
                            mean_poll_rate = None
                        poll_period_deque.clear()
                        if len(frame_period_deque) > 0:
                            mean_fps = 1 / mean(frame_period_deque)
                        else:
                            mean_fps = None
                        frame_period_deque.clear()
                        queue_put(out_queue, {OutputVariables.POLLRATE: mean_poll_rate,
                                              OutputVariables.FPS: mean_fps})
                        last_emit_timestamp = prev_read_timestamp
                if time_since_frame >= timeout:
                    logging.info(f"VideoCaptureThread timed out after {timeout} seconds of no frames.")
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

    def __init__(self, *args: Any, timeout: float = 1, poll_rate: Optional[float] = 100, frame_queue_size: int = 1,
                 **kwargs: Any) -> None:
        """

        Parameters
        ----------
        args             : Positional arguments to cv2.VideoCapture
        timeout          : Threaded reader timeout. If no new frames are received within 'timeout' seconds, the thread
            quits. Default is 1 second.
        poll_rate        : Threaded reader polling rate. The fastest rate (in calls per second) to capture.grab(). Set
            this at least as high as your expected frame rate, preferably twice as large for headroom. Default 100.
            Set it to None for unlimited poll rate.
        frame_queue_size : The length of the queue that holds frames fetched by the threaded reader. If the threaded
            reader has grabbed a new frame and the queue is full, the oldest element is removed to make room. Default 1.
        kwargs
        """
        self._capture = cv2.VideoCapture()
        self._frame_queue: queue.Queue[Tuple[Optional[bool], Optional[np.ndarray]]] = queue.Queue(maxsize=frame_queue_size)
        self._output_queue: queue.Queue[Dict[OutputVariables, Any]] = queue.Queue(maxsize=1)
        self._input_queue: queue.Queue[Tuple[InputVariables, Any]] = queue.Queue()
        self._timeout = timeout
        self._poll_rate = poll_rate
        self._return_data: Dict[OutputVariables, Any] = {}
        self.threaded_reader: Optional[threading.Thread] = None
        self.open(*args, **kwargs)

    @property
    def is_alive(self):
        return self.threaded_reader.is_alive()

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
            return self._frame_queue.get(block=False)
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
            self.threaded_reader = VideoCaptureThread(args=(self._capture,
                                                            self._frame_queue,
                                                            self._output_queue,
                                                            self._input_queue,
                                                            self._timeout,
                                                            self._poll_rate), daemon=True)
            self.threaded_reader.start()
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
        Stops the reader thread and releases the VideoCapture.
        """
        if self.threaded_reader:
            self._input_queue.put((InputVariables.QUIT, None))
            logging.info("Waiting for threaded reader to quit.")
            self.threaded_reader.join()
        self._capture.release()

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
        """ Return the reader thread timeout value. """
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: float) -> None:
        """
        Set the reader thread timeout. If no new frames have been received from the source (camera, file, stream,
        etc), the reader thread will quit.

        Parameters
        ----------
        timeout : Timeout value in seconds.
        """
        self._input_queue.put((InputVariables.TIMEOUT, timeout))
        self._timeout = timeout

    @property
    def poll_rate(self) -> float:
        """ Returns the reader thread poll rate. """
        return self._poll_rate

    @poll_rate.setter
    def poll_rate(self, poll_rate: Optional[float]) -> None:
        """
        Set the reader thread poll rate. This value is the highest rate (in calls per second) to capture.grab() that
        the reader thread is allowed to make.

        Parameters
        ----------
        poll_rate   : Poll rate in calls per second
        """
        if poll_rate is not None:
            assert poll_rate > 0, "Poll rate may not be 0 or negative."
        self._input_queue.put((InputVariables.POLLRATE, poll_rate))
        self._poll_rate = poll_rate

    def __enter__(self) -> 'ThreadedVideoCapture':
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()

def main():
    import os
    phone_ips = {'work': "10.0.49.50",
                 'home': "192.168.1.21"}
    phone_ip = phone_ips['work']
    capture_port = 8080

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    #with ThreadedVideoCapture(rf"http://{phone_ip}:{capture_port}/video", cv2.CAP_FFMPEG) as tvc:
    with ThreadedVideoCapture("../tests/testimage_1.jpg") as tvc:
        if not tvc.isOpened():
            print(f'Cannot open RTSP stream.')
            exit(-1)

        counter = 0
        while True:
            ret, frame = tvc.read()
            if ret:
                cv2.imshow("Frame", frame)
            if ret is None:
                break
            if cv2.waitKey(1) == ord("q"):
                break
            if cv2.waitKey(1) == ord("f"):
                tvc.poll_rate = (tvc.poll_rate + 10) % 50 + 5
            counter += 1


if __name__ == "__main__":
    main()
