# Reviewer runtime probe

This change adds a real Modal T4 import probe before the expensive production canary. It verifies the production import order for Qwen3-TTS and Qwen2.5-Omni, records package and CUDA versions, and fails with the exact exception and traceback when the runtime is inconsistent.

The workflow continues to use persistent `modal-production` GitHub environment credentials. It does not use browser token authorization.
