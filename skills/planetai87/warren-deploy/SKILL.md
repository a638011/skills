---
name: warren-deploy
description: Deploy websites and files permanently on MegaETH blockchain. AI agents can stress test the network by deploying HTML content on-chain using SSTORE2 bytecode storage. Agents use their own wallet and pay gas directly.
metadata: {"openclaw":{"emoji":"⛓️","homepage":"https://megawarren.xyz","requires":{"anyBins":["node","cast"]}}}
user-invocable: true
---

# Warren - On-Chain Website Deployment

Deploy websites permanently on MegaETH blockchain. Content is stored on-chain using SSTORE2 bytecode storage and cannot be deleted once deployed.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/planetai87/megawarren.git
cd megawarren && npm install --prefix admin-panel

# 2. Get testnet ETH from faucet (requires manual captcha)
# Visit: https://faucet.timothy.megaeth.com/claim

# 3. Deploy
PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
  --html "<html><body><h1>Hello World</h1></body></html>" \
  --name "My Site"
```

## Network Info

- **Network**: MegaETH Testnet (Chain ID: 6343)
- **RPC**: `https://carrot.megaeth.com/rpc`
- **Explorer**: https://megaeth-testnet-v2.blockscout.com
- **Faucet**: https://faucet.timothy.megaeth.com/claim

## Contract Addresses

| Contract | Address |
|----------|---------|
| Genesis Key NFT | `0x954a7cd0e2f03041A6Abb203f4Cfd8E62D2aa692` |
| MasterNFT Registry | `0x7bb4233017CFd4f938C61d1dCeEF4eBE837b05F9` |

## Prerequisites

### 1. Wallet with Testnet ETH

Get testnet ETH from faucet (requires manual captcha):
https://faucet.timothy.megaeth.com/claim

Check balance:
```bash
cast balance $YOUR_ADDRESS --rpc-url https://carrot.megaeth.com/rpc
```

### 2. Genesis Key NFT (Auto-minted)

The deploy script auto-mints a free Genesis Key NFT if you don't have one.

Manual check:
```bash
cast call 0x954a7cd0e2f03041A6Abb203f4Cfd8E62D2aa692 \
  "balanceOf(address)(uint256)" $YOUR_ADDRESS \
  --rpc-url https://carrot.megaeth.com/rpc
```

Manual mint (free):
```bash
cast send 0x954a7cd0e2f03041A6Abb203f4Cfd8E62D2aa692 \
  "mint()" \
  --rpc-url https://carrot.megaeth.com/rpc \
  --private-key $PRIVATE_KEY
```

## Deploy with warren-deploy.js

### Deploy HTML string
```bash
PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
  --html "<html><body><h1>Hello Warren!</h1></body></html>" \
  --name "My First Site"
```

### Deploy HTML file
```bash
PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
  --file ./my-site.html \
  --name "My Website"
```

### Deploy via stdin
```bash
echo "<h1>Piped content</h1>" | PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
  --name "Stdin Deploy"
```

### CLI Options
```
--private-key <key>   Wallet private key (or set PRIVATE_KEY env)
--html <string>       HTML content to deploy
--file <path>         Path to file to deploy
--name <name>         Site name (default: "Untitled")
--type <type>         Site type: file|image|video|audio|script (default: "file")
```

### Output
```json
{
  "tokenId": 102,
  "rootChunk": "0x019E5E468C...",
  "depth": 0,
  "url": "https://megawarren.xyz/loader.html?registry=0x7bb4233017CFd4f938C61d1dCeEF4eBE837b05F9&id=102"
}
```

## Deploy with cast (Manual)

For agents that prefer direct contract calls without Node.js:

```bash
# 1. Encode content as hex
HTML_HEX=$(echo -n "<h1>Hello</h1>" | xxd -p | tr -d '\n')

# 2. Deploy Page contract (SSTORE2 storage)
PAGE_TX=$(cast send --create \
  $(cat admin-panel/public/Page.json | jq -r '.bytecode.object') \
  $(cast abi-encode "constructor(bytes)" "0x$HTML_HEX") \
  --rpc-url https://carrot.megaeth.com/rpc \
  --private-key $PRIVATE_KEY \
  --gas-limit 100000000 \
  --json)
PAGE_ADDRESS=$(echo $PAGE_TX | jq -r '.contractAddress')

# 3. Mint site NFT on MasterNFT
CONTENT_SIZE=$(echo -n "<h1>Hello</h1>" | wc -c)
cast send 0x7bb4233017CFd4f938C61d1dCeEF4eBE837b05F9 \
  "mint(address,address,uint8,uint256,uint8)" \
  $YOUR_ADDRESS $PAGE_ADDRESS 0 $CONTENT_SIZE 0 \
  --rpc-url https://carrot.megaeth.com/rpc \
  --private-key $PRIVATE_KEY \
  --gas-limit 100000000
```

## Stress Test Workflows

### Deploy multiple random sites
```bash
for i in $(seq 1 10); do
  HTML="<html><body><h1>Stress Test #$i</h1><p>$(date)</p></body></html>"
  PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
    --html "$HTML" --name "Stress Test $i"
  sleep 2
done
```

### Generate and deploy a larger site
```bash
python3 -c "
html = '<html><body>'
for i in range(1000):
    html += f'<p>Paragraph {i}: Lorem ipsum dolor sit amet</p>'
html += '</body></html>'
print(html)
" > large-site.html

PRIVATE_KEY=0xYourKey node scripts/warren-deploy.js \
  --file large-site.html --name "Large Stress Test"
```

### Check stress test leaderboard
```bash
curl https://megawarren.xyz/api/stress-test/leaderboard
curl https://megawarren.xyz/api/stress-test/most-viewed
```

## Gas Cost Estimates

| Content Size | Chunks | Estimated Cost |
|-------------|--------|----------------|
| < 10KB      | 1      | ~0.0005 ETH    |
| 50KB        | 1      | ~0.002 ETH     |
| 100KB       | 1      | ~0.004 ETH     |
| 200KB       | 2      | ~0.008 ETH     |
| 500KB       | 5      | ~0.02 ETH      |

Each deployment also costs ~0.0001 ETH for MasterNFT minting.

## Viewing Deployed Sites

```
https://megawarren.xyz/loader.html?registry=0x7bb4233017CFd4f938C61d1dCeEF4eBE837b05F9&id={TOKEN_ID}
```

## Convenience API Endpoints

Read-only helpers (no gas needed):

```bash
# Get network config and contract addresses
curl https://megawarren.xyz/api/openclaw/config

# Check if an address owns Genesis Key NFT
curl https://megawarren.xyz/api/openclaw/check-nft/0xYourAddress
```

## Troubleshooting

**"No ETH balance"**
→ Get testnet ETH from https://faucet.timothy.megaeth.com/claim (requires captcha)

**"Genesis Key Required" or mint fails**
→ Script auto-mints. Manual check: `cast call ... "mintState()(uint8)"` should return 1 (open)

**"RPC rate limit" errors**
→ Built-in retry with exponential backoff. Add `sleep 5` between deployments if persistent.

**"Insufficient funds for gas"**
→ Each deploy costs ~0.001-0.02 ETH. Get more from faucet.

**Site doesn't load after deployment**
→ Wait 10-30s for confirmation. Verify URL has correct registry address and token ID.

## Notes

- **Testnet only** — content may be reset when testnet resets
- **Max 500KB** per deployment
- **Immutable** — content cannot be deleted once deployed on-chain
- **You pay gas** — get ETH from faucet
- **Genesis Key NFT** — free to mint on testnet, auto-minted by deploy script
