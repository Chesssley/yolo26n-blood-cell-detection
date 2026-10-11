"""Exercise upload cancellation through Gradio's real routes and inference queue."""
import asyncio
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from app.app import READY, build_demo
from app.inference import DetectionResult

import httpx
from fastapi import FastAPI
from gradio import mount_gradio_app
from PIL import Image


class UploadCancellationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.started, self.release, self.finished = Event(), Event(), Event()

        def predict(image, confidence):
            if Path(image).name == "a.png":
                self.started.set()
                try:
                    if not self.release.wait(10):
                        raise TimeoutError("Test did not release inference")
                finally:
                    self.finished.set()
            return DetectionResult(Image.new("RGB", (4, 3)),
                                   {"RBC": 1, "WBC": 0, "Platelets": 0}, 1, 1.0, "CPU")

        self.detector = Mock(predict=Mock(side_effect=predict))
        self.demo = build_demo(self.detector).queue()
        self.completed_outputs = {}
        self.observed = []
        server = mount_gradio_app(FastAPI(), self.demo, path="/")
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server),
                                        base_url="http://test")
        self.dependencies = self.demo.config["dependencies"]
        self.detect = next(d for d in self.dependencies if d["api_name"] == "detect")
        self.change = next(d for d in self.dependencies
                           if d["backend_fn"] and any(t[1] == "change" for t in d["targets"]))
        response = await self.client.get("/gradio_api/startup-events")
        response.raise_for_status()
        self.images = {}
        for name in ("a.png", "b.png"):
            image = BytesIO()
            Image.new("RGB", (4, 3)).save(image, format="PNG")
            response = await self.client.post("/gradio_api/upload",
                                              files={"files": (name, image.getvalue(), "image/png")})
            response.raise_for_status()
            self.images[name] = {"path": response.json()[0], "orig_name": name,
                                 "meta": {"_type": "gradio.FileData"}}

    async def asyncTearDown(self):
        self.release.set()
        if self.started.is_set():
            await asyncio.to_thread(self.finished.wait, 5)
        await self.client.aclose()
        self.demo._queue.close()
        self.demo._queue._cancel_asyncio_tasks()
        self.demo.is_running = False

    async def submit(self, dependency, data, session):
        response = await self.client.post("/gradio_api/queue/join", json={
            "fn_index": dependency["id"], "data": data, "session_hash": session})
        response.raise_for_status()
        return response.json()["event_id"]

    async def completed(self, session, event_id):
        if event_id in self.completed_outputs:
            return self.completed_outputs.pop(event_id)
        messages = self.demo._queue.pending_messages_per_session[session]
        async with asyncio.timeout(10):
            while True:
                message = (await messages.get()).model_dump()
                self.observed.append(message)
                if message["msg"] == "process_completed":
                    self.assertTrue(message["success"], message)
                    if message["event_id"] == event_id:
                        return message["output"]
                    self.completed_outputs[message["event_id"]] = message["output"]

    async def change_upload(self, session, old_event):
        # Emulate the frontend: dispatch cancellation listeners and the preview
        # registered on the same upload.change target, using their actual config.
        for dependency in self.dependencies:
            if dependency["targets"] == self.change["targets"]:
                for fn_index in dependency["cancels"]:
                    response = await self.client.post("/gradio_api/cancel", json={
                        "fn_index": fn_index, "session_hash": session, "event_id": old_event})
                    response.raise_for_status()
        preview = await self.submit(self.change, [self.images["b.png"]], session)
        output = await self.completed(session, preview)
        self.assertEqual(output["data"][1:7], [None] * 6)
        self.assertEqual(output["data"][-1], READY)

    async def test_changed_upload_discards_running_result_and_allows_new_detection(self):
        old = await self.submit(self.detect, [self.images["a.png"], 0.25], "changing")
        self.assertTrue(await asyncio.to_thread(self.started.wait, 5))
        await self.change_upload("changing", old)
        self.release.set()
        self.assertTrue(await asyncio.to_thread(self.finished.wait, 5))
        self.assertEqual(await self.completed("changing", old), {})
        new = await self.submit(self.detect, [self.images["b.png"], 0.25], "changing")
        self.assertEqual((await self.completed("changing", new))["data"][-1], "Detection complete.")
        self.assertFalse(any(m["event_id"] == old and m.get("output", {}).get("data")
                             for m in self.observed))
        self.assertEqual(self.detector.predict.call_count, 2)

    async def test_changed_upload_cancels_queued_job_without_affecting_other_session(self):
        other = await self.submit(self.detect, [self.images["a.png"], 0.25], "other")
        self.assertTrue(await asyncio.to_thread(self.started.wait, 5))
        old = await self.submit(self.detect, [self.images["a.png"], 0.25], "changing")
        await self.change_upload("changing", old)
        self.release.set()
        self.assertEqual(await self.completed("changing", old), {})
        self.assertEqual((await self.completed("other", other))["data"][-1], "Detection complete.")
        self.detector.predict.assert_called_once()
