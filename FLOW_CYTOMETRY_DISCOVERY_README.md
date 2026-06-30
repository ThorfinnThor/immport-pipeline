# Add-only flow cytometry dataset discovery overlay

This overlay adds a new GitHub Actions workflow and new Python scripts. It does **not** replace or modify your existing pipeline files.

## Files added

```text
.github/workflows/flow_cytometry_discovery.yml
requirements_flow_discovery.txt
scripts/run_flow_cytometry_discovery.py
scripts/send_flow_discovery_email.py
scripts/flow_discovery/__init__.py
scripts/flow_discovery/config.py
scripts/flow_discovery/models.py
scripts/flow_discovery/http_utils.py
scripts/flow_discovery/text_utils.py
scripts/flow_discovery/file_evidence.py
scripts/flow_discovery/classify.py
scripts/flow_discovery/scoring.py
scripts/flow_discovery/pipeline.py
scripts/flow_discovery/sources/__init__.py
scripts/flow_discovery/sources/zenodo.py
scripts/flow_discovery/sources/figshare.py
scripts/flow_discovery/sources/europepmc.py
scripts/flow_discovery/sources/immport.py
scripts/flow_discovery/sources/clinicaltrials.py
```

## What it does

The new workflow searches for newly published or recently released candidate datasets from:

- Zenodo
- Figshare
- Europe PMC literature/accession clues
- ImmPort clinical-trial cytometry studies
- ClinicalTrials.gov enrichment for extracted NCT identifiers

It ranks candidates using explicit evidence:

- raw `.fcs` files
- archive files with `.fcs`/cytometry evidence
- FlowJo/LMD/workspace-like files
- FlowRepository `FR-FCM-*` accessions
- spectral-flow terms such as Cytek Aurora, Sony ID7000, spectral cytometry, unmixing
- multicolor/high-parameter terms such as 18-color, 30-parameter, high-dimensional flow
- clinical-trial evidence from NCT identifiers, ClinicalTrials.gov, and ImmPort `clinicalTrial=Y`

## Output files

The workflow writes:

```text
output/flow_cytometry_discovery_all_candidates.csv
output/flow_cytometry_datasets_ranked.csv
output/flow_cytometry_high_confidence_raw_or_trial.csv
```

`flow_cytometry_datasets_ranked.csv` is the main file to inspect first.

## How to install in GitHub

1. Unzip this overlay.
2. Copy the folders/files into the root of your existing repository.
3. Do not delete your existing files.
4. Commit and push.
5. Go to **Actions** in GitHub.
6. Run **Flow Cytometry Dataset Discovery** manually with **Run workflow**.

## Optional secrets

The workflow works without repository tokens for public data. Optional:

```text
ZENODO_TOKEN
FIGSHARE_TOKEN
```

Email uses the same SMTP-style secrets as your existing workflow:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
MAIL_TO
MAIL_FROM
```

If all email secrets are present, the workflow sends the three CSV files as email attachments. If not, it still uploads the CSV files as a GitHub Actions artifact.

## Tuning search breadth

Edit these values in `.github/workflows/flow_cytometry_discovery.yml` if needed:

```yaml
FLOW_DISCOVERY_MAX_RESULTS: "250"
ZENODO_MAX_PAGES: "4"
FIGSHARE_MAX_PAGES: "4"
FIGSHARE_MAX_DETAILS: "120"
EUROPEPMC_MAX_PAGES: "2"
IMMPORT_MAX_STUDIES: "250"
```

To add custom search queries without editing code, set this workflow environment variable as a `||` separated list:

```yaml
FLOW_DISCOVERY_QUERIES: '"spectral flow cytometry" fcs||"Cytek Aurora" NCT||"30-color flow cytometry" clinical trial'
```

## Scoring logic

The main score is:

```text
final_score =
  0.30 * raw_data_score
+ 0.25 * clinical_trial_score
+ 0.20 * spectral_or_multicolor_score
+ 0.15 * newness_score
+ 0.10 * downloadability_score
```

The design intentionally favors verified raw-data evidence over vague text mentions.
