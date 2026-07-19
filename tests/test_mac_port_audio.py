"""Mac (Apple Silicon) port: the sounddevice recording backend used on
non-Windows platforms.

The dev machine is Windows, so IS_WINDOWS is patched to False to force
AudioRecorder down the sounddevice path; sounddevice itself is a real
dependency already used unconditionally for playback (AudioPlayer), so it
is mocked here only where real hardware would otherwise be touched.
"""
from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from otoweave_app import audio as audio_module
from otoweave_app.audio import (
    AudioRecorder,
    AudioSource,
    SAMPLE_RATE,
    _NullAudioBackend,
    _SoundDeviceInputStream,
    _classify_input_level,
)


def _fake_device(name: str, max_input_channels: int, default_samplerate: int = SAMPLE_RATE) -> dict:
    return {
        "name": name,
        "max_input_channels": max_input_channels,
        "default_samplerate": default_samplerate,
    }


class ClassifyInputLevelTests(unittest.TestCase):
    """Shared by the pyaudio and sounddevice measurement paths."""

    def test_silence_is_poor(self) -> None:
        result = _classify_input_level(np.zeros(1600, dtype=np.float32))
        self.assertEqual(result["state"], "Poor")

    def test_clipping_peak_is_poor(self) -> None:
        samples = np.full(1600, 0.999, dtype=np.float32)
        result = _classify_input_level(samples)
        self.assertEqual(result["state"], "Poor")

    def test_moderate_signal_is_good(self) -> None:
        t = np.arange(1600) / SAMPLE_RATE
        samples = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = _classify_input_level(samples)
        self.assertEqual(result["state"], "Good")


class SoundDeviceInputStreamAdapterTests(unittest.TestCase):
    """_SoundDeviceInputStream must expose the same three methods
    AudioRecorder calls on the pyaudio stream object."""

    def test_start_stop_close_delegate_to_the_wrapped_stream(self) -> None:
        wrapped = MagicMock()
        adapter = _SoundDeviceInputStream(wrapped)

        adapter.start_stream()
        wrapped.start.assert_called_once()

        adapter.stop_stream()
        wrapped.stop.assert_called_once()

        adapter.close()
        wrapped.close.assert_called_once()


class NullAudioBackendTests(unittest.TestCase):
    def test_terminate_is_a_harmless_noop(self) -> None:
        _NullAudioBackend().terminate()


class AvailableAudioSourcesSdTests(unittest.TestCase):
    def test_blackhole_device_surfaces_as_pc_audio_loopback(self) -> None:
        devices = [
            _fake_device("MacBook Pro Microphone", 1),
            _fake_device("BlackHole 2ch", 2),
        ]
        with patch("sounddevice.query_devices", return_value=devices), patch(
            "sounddevice.default"
        ) as default:
            default.device = 0
            sources = audio_module._available_audio_sources_sd()
        self.assertEqual(len(sources), 2)
        by_kind = {s.kind for s in sources}
        self.assertEqual(by_kind, {"microphone", "loopback"})
        loopback = next(s for s in sources if s.kind == "loopback")
        self.assertTrue(loopback.label.startswith("PC音声"))
        microphone = next(s for s in sources if s.kind == "microphone")
        self.assertTrue(microphone.label.startswith("マイク"))

    def test_soundflower_and_loopback_named_devices_are_also_detected(self) -> None:
        devices = [
            _fake_device("Loopback Audio", 2),
            _fake_device("Soundflower (2ch)", 2),
        ]
        with patch("sounddevice.query_devices", return_value=devices), patch(
            "sounddevice.default"
        ) as default:
            default.device = -1
            sources = audio_module._available_audio_sources_sd()
        self.assertTrue(all(s.kind == "loopback" for s in sources))

    def test_output_only_devices_are_excluded(self) -> None:
        devices = [
            _fake_device("Built-in Output", 0),
            _fake_device("Built-in Microphone", 1),
        ]
        with patch("sounddevice.query_devices", return_value=devices), patch(
            "sounddevice.default"
        ) as default:
            default.device = 1
            sources = audio_module._available_audio_sources_sd()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].device_index, 1)

    def test_default_device_is_marked_and_sorted_first(self) -> None:
        devices = [
            _fake_device("Second Mic", 1),
            _fake_device("Default Mic", 1),
        ]
        with patch("sounddevice.query_devices", return_value=devices), patch(
            "sounddevice.default"
        ) as default:
            default.device = 1
            sources = audio_module._available_audio_sources_sd()
        self.assertEqual(sources[0].device_index, 1)
        self.assertIn("（既定）", sources[0].label)

    def test_missing_sounddevice_module_returns_empty_list(self) -> None:
        with patch("sounddevice.query_devices", side_effect=RuntimeError("no portaudio")):
            self.assertEqual(audio_module._available_audio_sources_sd(), [])

    def test_import_error_returns_empty_list(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(audio_module._available_audio_sources_sd(), [])


class MeasureAudioInputSdTests(unittest.TestCase):
    def test_classifies_a_recorded_block(self) -> None:
        source = AudioSource(
            id="microphone:0",
            label="テストマイク",
            device_index=0,
            sample_rate=SAMPLE_RATE,
            channels=1,
            kind="microphone",
        )
        t = np.arange(int(SAMPLE_RATE * 0.5)) / SAMPLE_RATE
        fake_recording = (0.2 * np.sin(2 * np.pi * 440 * t) * 32768).astype(np.int16).reshape(-1, 1)
        with patch("sounddevice.rec", return_value=fake_recording) as rec, patch(
            "sounddevice.wait"
        ):
            result = audio_module._measure_audio_input_sd(source, duration_seconds=0.5)
        rec.assert_called_once()
        self.assertEqual(result["state"], "Good")

    def test_measure_audio_input_dispatches_to_sd_when_not_windows(self) -> None:
        source = AudioSource(
            id="microphone:0",
            label="テストマイク",
            device_index=0,
            sample_rate=SAMPLE_RATE,
            channels=1,
            kind="microphone",
        )
        with patch.object(audio_module, "IS_WINDOWS", False), patch.object(
            audio_module, "_measure_audio_input_sd", return_value={"state": "Good"}
        ) as measure:
            result = audio_module.measure_audio_input(source)
        measure.assert_called_once()
        self.assertEqual(result["state"], "Good")


class DefaultInputIndexSdTests(unittest.TestCase):
    def test_scalar_default_device(self) -> None:
        with patch("sounddevice.default") as default:
            default.device = 3
            self.assertEqual(audio_module._default_input_index_sd(), 3)

    def test_tuple_default_device_uses_input_half(self) -> None:
        with patch("sounddevice.default") as default:
            default.device = (2, 5)
            self.assertEqual(audio_module._default_input_index_sd(), 2)

    def test_unresolvable_device_returns_negative_one(self) -> None:
        with patch("sounddevice.default") as default:
            default.device = "not-a-number"
            self.assertEqual(audio_module._default_input_index_sd(), -1)


class AudioRecorderSoundDeviceBackendTests(unittest.TestCase):
    """AudioRecorder.start() must pick the sounddevice backend on
    non-Windows without touching pyaudiowpatch."""

    def _source(self) -> AudioSource:
        return AudioSource(
            id="microphone:0",
            label="テストマイク",
            device_index=0,
            sample_rate=SAMPLE_RATE,
            channels=1,
            kind="microphone",
        )

    def test_start_uses_raw_input_stream_and_enqueues_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = AudioRecorder(
                source=self._source(),
                output_pcm=Path(temporary) / "out.pcm",
                on_speech_chunk=lambda chunk: None,
                on_error=lambda message: None,
            )
            fake_stream = MagicMock()
            with patch.object(audio_module, "IS_WINDOWS", False), patch(
                "sounddevice.RawInputStream", return_value=fake_stream
            ) as raw_stream:
                recorder.start()
                # start() calls stream.start_stream() -> adapter -> fake_stream.start()
                fake_stream.start.assert_called_once()
                self.assertIsInstance(recorder._pa, _NullAudioBackend)
                self.assertIsInstance(recorder._stream, _SoundDeviceInputStream)

                callback = raw_stream.call_args.kwargs["callback"]
                block = (np.ones(160, dtype=np.int16) * 1000).tobytes()
                callback(block, 160, None, None)
                queued = recorder._queue.get(timeout=2)
                self.assertEqual(queued, block)
            recorder._signal_worker_stop()
            if recorder._worker is not None:
                recorder._worker.join(timeout=5)
            recorder._file.close()

    def test_start_failure_on_sounddevice_releases_pcm_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = AudioRecorder(
                source=self._source(),
                output_pcm=Path(temporary) / "out.pcm",
                on_speech_chunk=lambda chunk: None,
                on_error=lambda message: None,
            )
            with patch.object(audio_module, "IS_WINDOWS", False), patch(
                "sounddevice.RawInputStream", side_effect=OSError("device busy")
            ):
                with self.assertRaises(OSError):
                    recorder.start()
            self.assertIsNone(recorder._file)
            (Path(temporary) / "out.pcm").unlink()


if __name__ == "__main__":
    unittest.main()
