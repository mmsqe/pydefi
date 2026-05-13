# pydefi chain configuration

Snapshots of chain metadata consumed at import time by bridge modules.

## `ccip-chains.json`

**Source:** `GET https://docs.chain.link/api/ccip/v1/chains?environment=mainnet`

**Used by:** `pydefi.bridge.ccip` — reads `data.evm`, keeps `supported: true` chains.

### Update

```bash
curl -s 'https://docs.chain.link/api/ccip/v1/chains?environment=mainnet' > pydefi/config/ccip-chains.json
python -m pytest tests/test_bridge.py -k ccip -q
```
