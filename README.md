# ThreadedVideoCapture
A direct drop-in replacement for OpenCV's `VideoCapture` that runs in a background thread, allowing the main thread to 
do useful work instead of waiting on frames. 

`ThreadedVideoCapture` is a new library. Bugs may exist, and useful features may be missing. Bug reports, 
feature requests, and pull requests are therefore highly appreciated!


## How to install
PyPi package is currently being worked on and should be ready before December 2022. For now, you can
download this repository directly.

## Simple example
`ThreadedVideoCapture` can be used exactly like the normal `VideoCapture`:
```
from threadedvideocapture import ThreadedVideoCapture

with ThreadedVideoCapture(0) as tvc:  # Open webcam stream
    while True:
        ret, frame = tvc.read()
        if ret is None:  # ret is only None if tvc has stopped.
            print("End of stream.")
            break
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
```

# Documentation
`ThreadedVideoCapture` creates a background thread with a `VideoCapture` instance in it. This instance will 
continuously read frames and place them on a FIFO queue. When you call `ThreadedVideoCapture.read()`, the oldest
frame is returned from the queue. 

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
start a `ThreadedVideoCapture` that will wait until infinity for a single frame, then change its timeout and quit if 
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
        ret, frame = cap.read() 
        if ret is None:  # ret is only None if tvc has stopped.
            print("The ThreadedVideoCapture has timed out.")
            break
    
```

### Polling rate
You can limit how often the `VideoCapture` instance calls `grab()` by specifying the polling rate at instantiation, or 
at any other time. By default, it is not limited. 

### Logger
`ThreadedVideoCapture` uses Pythons excellent `logging` library to log events. By default `ThreadedVideoCapture` 
creates its own logger named 'ThreadedVideoCapture', but you can provide it with a custom logger object if you wish at 
instantiation. The logger is directly exposed as `ThreadedVideoCapture.logger`.

