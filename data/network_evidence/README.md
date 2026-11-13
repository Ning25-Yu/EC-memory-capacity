# Literature evidence for the 168 interactions

| File | Content |
|---|---|
| `interaction_evidence.csv` | one row per interaction and supporting study (170 rows; 168 interactions, two of them with two studies) |
| `references.csv` | the 134 studies: full citation, DOI and PMID |
| `references.bib` | the same studies in BibTeX format (keys = `reference_id`) |

Columns of `interaction_evidence.csv`:

- `regulator`, `target`: node names as in `data/EC_network.bnet` (CDKN1A = p21, CDKN1B = p27, CDKN2A = p16)
- `sign`: `activation` or `inhibition`, as the regulator appears in the logical rule of the target
- `reference_id`: key into `references.csv` and `references.bib`
- `study`: first author and year
- `interaction_as_reported`: the interaction as curated from the study
- `experimental_system`: cells or tissue in which it was observed
- `doi`, `pmid`: identifiers of the study

Evidence from endometrial cancer cells or tissue was preferred. Where none was available, the interaction is supported
by another system, which is stated in `experimental_system`. The signs and the regulator sets are checked against
`data/EC_network.bnet` by `python code/analysis/00_check_interaction_evidence.py`.
