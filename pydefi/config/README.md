# pydefi chain configuration

Snapshots of chain metadata consumed at import time by bridge modules.

## `ccip-chains.json`

**Used by:** `pydefi.bridge.ccip`.

### Update

```bash
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' | jq '.' > pydefi/config/ccip-chains.json
```

Automated weekly via `.github/workflows/sync-ccip-config.yml`, which opens a
PR only when `.data` drifts — `metadata.timestamp` / `requestId` change on
every fetch and are ignored.
