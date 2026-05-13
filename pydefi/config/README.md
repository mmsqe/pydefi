# pydefi chain configuration

Snapshots of chain metadata consumed at import time by bridge modules.

## `ccip-chains.json`

**Used by:** `pydefi.bridge.ccip`.

### Update

```bash
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' | jq '.' > pydefi/config/ccip-chains.json
```
