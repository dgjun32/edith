import time
class Timer:
    def __init__(
        self,
        hz: float,
        warn_on_miss: bool = True,
        miss_warn_threshold: float = 0.05,
        miss_warn_interval: float = 1.0,
    ):
        self.hz = hz
        self.dt = 1 / hz
        self.last_time = time.perf_counter()
        self.warn_on_miss = warn_on_miss
        self.miss_warn_threshold = miss_warn_threshold
        self.miss_warn_interval = miss_warn_interval
        self.last_miss_warn_time = 0.0

    def sleep(self):
        sleep_time = self.dt - (time.perf_counter() - self.last_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            now = time.perf_counter()
            miss = -sleep_time
            if (
                self.warn_on_miss
                and miss >= self.miss_warn_threshold
                and now - self.last_miss_warn_time >= self.miss_warn_interval
            ):
                print(f"Missed target control frequency by {-miss:.6f} seconds")
                self.last_miss_warn_time = now
        self.last_time = time.perf_counter()

class TimeCounter:
    def __init__(self):
        self.last_time = time.perf_counter()

    def count_time(self):
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time
        return dt
