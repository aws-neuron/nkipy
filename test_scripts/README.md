# Test Scripts

These scripts were used during the optimization work to test various hypotheses and verify fixes.

## Organization

- `test_*.sh` - Test scripts for specific scenarios
- `run_*.sh` - Scripts to run engines in various configurations
- `test_*.py` - Python test scripts

Most of these are historical and kept for reference. The main test cases that should be preserved:

1. Basic sleep/wake cycle test
2. P2P transfer followed by sleep test
3. Wake from checkpoint test

The scripts reference specific instance IPs and configurations from the development environment and will need adjustment for other environments.
