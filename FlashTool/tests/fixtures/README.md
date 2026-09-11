# FlashTool Test Fixtures

Fixtures must be synthetic, minimal and non-sensitive.

Do not commit proprietary firmware, device dumps, private keys, signed vendor images containing confidential material, or user data.

Current tests construct minimal Android boot, sparse-image and CrAU headers in memory and create a temporary OTA ZIP. Future parser fixtures should follow the same principle unless an upstream public test vector has a compatible license.
