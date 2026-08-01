import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from factory.config import Settings
from factory.llm_runtime import LLMRuntimeError, _command, managed_llama_server


class LLMRuntimeTests(unittest.TestCase):
    def settings(self, extra=None):
        env = extra or {}
        with patch.dict("os.environ", env, clear=True):
            return Settings.from_env()

    def test_command_rejects_non_local_managed_endpoint(self):
        settings = self.settings({"LLAMA_CPP_BASE_URL": "https://remote.example/v1"})
        with self.assertRaisesRegex(LLMRuntimeError, "localhost"):
            _command(settings)

    def test_existing_server_is_reused_and_not_terminated(self):
        settings = self.settings()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "factory.llm_runtime._is_healthy", return_value=True
        ), patch("factory.llm_runtime.subprocess.Popen") as popen:
            with managed_llama_server(settings, Path(temporary)):
                pass
        popen.assert_not_called()

    def test_managed_process_is_stopped_after_generation_window(self):
        settings = self.settings({"LLAMA_CPP_STARTUP_TIMEOUT_SECONDS": "30"})
        process = Mock()
        process.poll.side_effect = [None, None, None]
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temporary, patch(
            "factory.llm_runtime._is_healthy", side_effect=[False, True]
        ), patch("factory.llm_runtime._command", return_value=["llama-server"]), patch(
            "factory.llm_runtime.subprocess.Popen", return_value=process
        ):
            with managed_llama_server(settings, Path(temporary)):
                self.assertFalse(process.terminate.called)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)


if __name__ == "__main__":
    unittest.main()
