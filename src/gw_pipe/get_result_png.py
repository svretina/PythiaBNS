#!/usr/bin/env python3

import os
import numpy as np
import post_analysis

directories, _ = post_analysis.json_checker()

os.system("mkdir send_pdf")

for item in directories:
    abspath = item.split('/results')[0]
    subdir = item.split('results/')[-1]
    
    os.system(f"mkdir -p send_pdf/{subdir}")
    os.system(f"cp {item}/*.pdf {abspath}/send_pdf/{subdir}/")
    os.system(f"cp {item}/*corner* {abspath}/send_pdf/{subdir}/")
    os.system(f"cp {item}/*trace* {abspath}/send_pdf/{subdir}/")

comps = os.popen("find results -type f -name '*comparison*'").read().split('\n')[:-1]
for comp in comps:
    os.system(f"cp {comp} {abspath}/send_pdf/{'/'.join(comp.split('/')[1:-1])}")
