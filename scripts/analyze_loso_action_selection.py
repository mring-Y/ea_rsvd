#!/usr/bin/env python3
"""Compatibility entry point for common-start candidate analysis.

This script now uses one baseline for both candidate gains. Outputs report
descriptive matrix/seed grouping; no leave-one-out classifier is fitted.
"""
from analyze_common_start_actions import main

if __name__ == "__main__":
    raise SystemExit(main())
