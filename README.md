# car-listing-visual-verification

Application-side repository for car listing visual verification.

## Repository split

Scraping, filtering, and dataset upload pipeline were moved to:

- `/Users/ruslan/Projects/car-listing-data-pipeline`

This repository no longer contains Drom pipeline code.

## Data location

Local dataset artifacts remain here under:

- `data/`

`data/` is ignored by git, so local files stay available for the app without polluting repository history.

## Quick start

Install dependencies:

```bash
make requirements
```

Run quality checks:

```bash
make lint
```

Example module command:

```bash
python3 -m car_listing_visual_verification.plots --help
```

## Pipeline commands

If you need scraping/filtering/Hugging Face upload, run them in:

- `/Users/ruslan/Projects/car-listing-data-pipeline`

You can also run:

```bash
make pipeline-location
```
