# Pattern Model Deployment

The Research Trend Analysis and Pattern Intelligence sections require two runtime
artifacts:

- `model_xgb.joblib`
- `model_meta.json`

`models.artifact_store.load_model_artifacts()` loads them from
`PATTERN_ARTIFACT_DIR` when set, otherwise from `artifacts/`.

Production mounts `/home/ubuntu/sas-pattern-artifacts` into the container at
`/app/artifacts` and sets `PATTERN_ARTIFACT_DIR=/app/artifacts`. The artifacts
are intentionally git-ignored and are not copied into the Docker image.

## Build And Deploy

Run the GitHub workflow `Train Pattern Model`. It trains via
`scripts/train_tradeable_model.py`, uploads the artifacts, copies them to
`/home/ubuntu/sas-pattern-artifacts`, and restarts `sas-server`.

The main deploy workflow now fails before starting `sas-server` if the VM does
not already have both required artifacts.

## Verify In Container

```bash
docker exec sas-server python scripts/smoke_pattern_runtime.py --symbol NVDA --direct-only
```

With a paid-user JWT, verify the HTTP endpoints too:

```bash
docker exec sas-server python scripts/smoke_pattern_runtime.py \
  --symbol NVDA \
  --base-url http://localhost:8000/api/v1 \
  --access-token "$PAID_USER_JWT"
```

`/health` returns HTTP 503 when paid users are configured but the pattern model
cannot be loaded.
