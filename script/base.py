import asyncio
import random
import time
from abc import ABC, abstractmethod


class RateLimiter:
    def __init__(self, calls_per_minute: int = 10, retry_after: int = 60):
        self.calls_per_minute = calls_per_minute
        self.retry_after = retry_after
        self.call_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < 60]

            if len(self.call_times) >= self.calls_per_minute:
                wait_time = 60 - (now - self.call_times[0]) + 1
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                await self.acquire()  # Recursively call acquire after waiting

            self.call_times.append(time.time())

    async def execute_with_retry(self, coroutine, max_retries=2):
        retries = 0
        while retries < max_retries:
            try:
                await self.acquire()
                return await coroutine
            except Exception as e:
                retries += 1
                wait_time = self.retry_after * (1 + random.random())
                print(
                    f"Error: {e}. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

        raise Exception(f"Failed after {max_retries} retries")


class Query(ABC):
    def __init__(self):
        self.client = None
        self.rate_limiter = None  # API didn't support async calls

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
