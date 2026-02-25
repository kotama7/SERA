#!/usr/bin/env python
"""Run SERA research pipeline directly."""
import sys
sys.path.insert(0, "/home/t-kotama/workplace/SERA/src")
from sera.commands.research_cmd import run_research
run_research("/home/t-kotama/workplace/SERA/sera_workspace", resume=False, skip_phase0=True, skip_paper=True)
