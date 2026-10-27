"""Optional ABC IPC integration, imported only for ABC teleoperation."""

import threading
import time
import numpy as np
from edith.eval_utils.sync import Sample


class ABCSubscriber:
    def __init__(self, names, sink):
        self.names, self.sink = names, sink
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="abc-local-state"
        )
        self.thread.start()

    def _run(self):
        import zmq
        from deploy.robot import communication as comms

        context = zmq.Context()
        sockets = {}
        try:
            poller = zmq.Poller()
            for name in self.names:
                sock = comms.create_subscriber(context, name)
                sock.setsockopt(zmq.RCVHWM, 32)
                sockets[sock] = name
                poller.register(sock, zmq.POLLIN)
            sequence = dict.fromkeys(self.names, 0)
            while not self.stop_event.is_set():
                for sock in dict(poller.poll(100)):
                    data, metadata = comms.subscribe(sock)
                    timestamp = round(float(metadata["timestamp"]) * 1e9)
                    name = sockets[sock]
                    if data.shape != (7,) or not np.all(np.isfinite(data)):
                        raise ValueError(
                            f"Invalid YAM joint vector on {name}: {data.shape}"
                        )
                    sequence[name] += 1
                    self.sink.add(
                        name,
                        Sample(
                            timestamp,
                            time.perf_counter_ns(),
                            timestamp,
                            sequence[name],
                            data,
                            "abc_host_publish",
                        ),
                    )
        except Exception as error:
            self.sink.fail(error)
        finally:
            for sock in sockets:
                sock.close(linger=0)
            context.term()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
