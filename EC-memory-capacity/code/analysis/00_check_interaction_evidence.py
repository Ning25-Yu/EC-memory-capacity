"""Check that the evidence table covers exactly the 168 signed interactions of the logical rules."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import pandas as pd

import netdraw
from paths import DATA

E = pd.read_csv(DATA + "/network_evidence/interaction_evidence.csv", dtype=str)
R = pd.read_csv(DATA + "/network_evidence/references.csv", dtype=str)
rules = {(s, t): "inhibition" if neg else "activation" for s, t, neg in netdraw.signed_edges()}
table = {(r.regulator, r.target): r.sign for r in E.itertuples()}
assert len(rules) == 168, len(rules)
assert table == rules, set(table.items()) ^ set(rules.items())
assert set(E.reference_id) <= set(R.reference_id)
print(f"{len(rules)} interactions, {len(E)} evidence rows, {E.reference_id.nunique()} studies: consistent with the rules")
