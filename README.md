![LICENCE](https://img.shields.io/github/license/nup002/Threaded-VideoCapture)
[![Flake8](https://github.com/nup002/Threaded-VideoCapture/actions/workflows/flake8.yml/badge.svg)](https://github.com/nup002/Threaded-VideoCapture/actions/workflows/flake8.yml)
[![PyTest](https://github.com/nup002/Threaded-VideoCapture/actions/workflows/PyTest.yml/badge.svg)](https://github.com/nup002/Threaded-VideoCapture/actions/workflows/PyTest.yml)
![Version](https://img.shields.io/pypi/v/Threaded-VideoCapture)
![Python](https://img.shields.io/pypi/pyversions/Threaded-VideoCapture)

# Threaded-VideoCapture
A direct drop-in replacement for OpenCV's `VideoCapture` that runs in a background thread, allowing the main thread to 
do useful work instead of waiting on frames. 

This library is useful if your code spend a lot of time waiting for new frames, or if you are processing a stream in 
realtime and cannot process frames fast enough to keep up with the stream.

`Threaded-VideoCapture` requires `opencv-python` 4.0.0.21 or greater. It has been tested on Python 3.6, 3.7, 3.8. 3.9, 
3.10, and 3.11. 

It is a young library. Therefore bugs may exist, and useful features may be missing. Bug reports, 
feature requests, and pull requests are therefore highly appreciated!


## Installation
`Threaded-VideoCapture` is available from PyPi. Run the following in a command line terminal:

`pip install threaded-videocapture`

## Simple example
`Threaded-VideoCapture` can be used exactly like the normal `VideoCapture`:
```
import cv2
from threaded_videocapture import ThreadedVideoCapture

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
```

# Documentation
`ThreadedVideoCapture` creates a background thread with a `VideoCapture` instance in it. This instance will 
continuously read frames and place them on a FIFO queue. When you call `ThreadedVideoCapture.read()`, the oldest
frame is returned from the queue. If there are no frames in the queue, a `(False, None)` tuple is returned.

An instantiated `ThreadedVideoCapture` will eventually stop yielding frames. This is normal, and occurs for example 
when:
  * There are no more frames in the video file
  * A stream times out
  * An exception occurs

When `ThreadedVideoCapture` stops, it will place a `(None, None)` tuple on the queue. This signifies that
`ThreadedVideoCapture.read()` will never yield new frames until a new source has been opened with
`ThreadedVideoCapture.open()`. This can also be checked with `ThreadedVideoCapture.is_alive()`.

## Instantiation parameters
`ThreadedVideoCapture` takes additional parameters compared to `VideoCapture` during instantiation: `frame_queue_size`, 
`timeout`, `poll_rate`, and `logger`. 
They are explained here.

### Frame queue size
Frames read by the `VideoCapture` instance in the background thread will be placed on a queue, as explained above. When 
the queue becomes full, the oldest item is deleted to make room for the new frame. The length of the queue can be 
specified when instantiating `ThreadedVideoCapture` to suit your needs. Be default, the queue length is 1, meaning that 
only the most recent frame is available.

### Timeout
If `ThreadedVideoCapture` does not receive a new frame within a specified time, it will time out and quit. This is 
useful for 
example when you are capturing a stream and you do not know when it will end. You can set the timeout value both when
instantiating, and at any other time. By default, the timeout is set to 1 second. The following example shows how to 
start a `ThreadedVideoCapture` that will wait indefinitely for a single frame, then change its timeout and quit if 
no frames are received within the timeout value.
```
with ThreadedVideoCapture(0, timeout=0) as tvc:  # Open webcam stream with timeout disabled.
    # Poll for a single frame for eternity due to no timeout
    while True:
        ret, frame = cap.read() 
        if ret:
            break
    
    tvc.timeout = 2.5 # Set timeout to 2.5 seconds
    # Poll for frames for 2.5 seconds before ThreadedVideoCapture times out
    while True:
        ret, frame = tvc.read() 
        if ret is None:  # ret is only None if tvc has stopped.
            print("ThreadedVideoCapture has timed out.")
            break
    
```

### Polling rate
You can limit how often the `VideoCapture` instance calls `grab()` by specifying the polling rate at instantiation, or 
at any other time. By default, it is not limited. 

### Logger
The `Threaded-VideoCapture` library uses Pythons excellent `logging` library to log events. By default 
`ThreadedVideoCapture` uses its own logger named 'TVC', but you can provide it with a custom logger object at 
instantiation. The logger is found at `ThreadedVideoCapture.logger`.

## Reusing a ThreadedVideoCapture instance
You can open a new video source without having to close your original `ThreadedVideoCapture` instance and creating a 
new one. Simply call `ThreadedVideoCapture.open()` with your new source parameters. This will release the `VideoCapture`
 instance for your old source, `join` the background thread, and create a new `VideoCapture` in its own thread for the 
new video source. Example:

```
# Example showing how to switch to a different webcam after 1 second with the same ThreadedVideoCapture instance.

from time import time
with ThreadedVideoCapture(0) as tvc:  # Open webcam 0 stream
    tick = time()
    while True:
        ret, frame = tvc.read()
        if ret:  # ret is True if a frame was obtained with tvc.read()
            cv2.imshow('frame', frame) 
        if cv2.waitKey(1) == ord('q'):
            break
        # After one second of opening the stream from webcam 0, we switch seamlessly to webcam 1.
        if time() - tick > 1:
            tvc.open(1)
```

## Statistics
The current frames per second (FPS) and actual polling rate can be obtained with `ThreadedVideoCapture.fps` and 
`ThreadedVideoCapture.actual_poll_rate`. These values are updated once per second.
