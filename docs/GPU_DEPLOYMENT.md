# GPU deployment contract

The production proof worker runs on Modal with one T4 GPU.

## Required owner actions

1. Create a Modal account and run `modal setup`.
2. Set a hard workspace budget before deployment.
3. Create `ai-growth-factory-secrets` with `YOUTUBE_OAUTH_JSON`, `PUBLISH_ENABLED=true`, and `YOUTUBE_PRIVACY_STATUS=private`.
4. Create the protected GitHub environment `modal-production`.
5. Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to that environment.
6. Run the manual deploy workflow.
7. Inspect three private canaries before changing privacy status.

The pipeline does not auto-promote private videos to public.
