"""Bounded, lossless handoff to the existing encoder; close includes queue drain."""
import queue
import threading
import time


class AsyncVideoWriter:
    def __init__(self, writer, capacity=16):
        self.writer = writer
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.submitted = 0
        self.written = 0
        self.worker_wall_s = 0.
        self.close_wall_s = None
        self.closed = False
        self.worker = threading.Thread(target=self._run, name='replay-video-encoder', daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    start = time.perf_counter()
                    try:
                        self.writer.append_data(item)
                        self.written += 1
                    except BaseException as error:
                        self.error = error
                    finally:
                        self.worker_wall_s += time.perf_counter()-start
            finally:
                self.queue.task_done()

    def append_data(self, frame):
        if self.closed:
            raise RuntimeError('Cannot append after video close')
        if self.error is not None:
            raise RuntimeError('Video encoder failed') from self.error
        # Caller gives a distinct RGB copy whose ownership is retained by queue.
        self.queue.put(frame)
        self.submitted += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        start = time.perf_counter()
        self.queue.put(None)
        self.queue.join()
        self.worker.join()
        self.writer.close()
        self.close_wall_s = time.perf_counter()-start
        if self.error is not None:
            raise RuntimeError('Video encoder failed') from self.error
        assert self.written == self.submitted

    def report(self):
        return dict(frames_submitted=self.submitted, frames_encoded=self.written,
                    capacity_frames=self.queue.maxsize, worker_wall_s=self.worker_wall_s,
                    close_and_drain_wall_s=self.close_wall_s, dropped_frames=0,
                    note='Worker time overlaps rollout; do not add it to elapsed wall time')
