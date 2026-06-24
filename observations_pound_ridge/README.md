# Pound Ridge Observations — iNaturalist (via GBIF)

This is the **field-observation** counterpart to the vocabulary sources. The
sibling folders (`manual_of_woody_plants/`, `newcomb_wildflower_guide/`,
`gobotany/`) define *vocabulary* — what features and values exist, and which
species have which features. This folder contains the **actual observations of
plants at a specific place**: Ward Pound Ridge Reservation in Westchester
County, NY.

Pound Ridge is Arborphy's primary test site for prototyping. Pairing real
geo-located observations with the descriptive vocabulary is what turns the
graph from a static dictionary into a queryable model of "what plants are
where, when, and what do they look like."

## Files

| File                                         | Source                                                  | Rows  | Notes |
| -------------------------------------------- | ------------------------------------------------------- | ----- | ----- |
| `ward_pound_ridge_species.csv`               | Direct iNaturalist export                               | 11,881 | Rich schema: photo URLs, common names, observer info, observation timestamps, license. Best for visual / image-based work. |
| `gbif_inaturalist_observations.csv`          | iNaturalist mirrored via GBIF (TSV-formatted, `.csv` extension is historical) | 4,225  | Canonical GBIF taxon keys, full taxonomy hierarchy. Best for joining to the GBIF backbone taxonomy already loaded in `arq-research`. |

Both files cover the same place (Ward Pound Ridge Reservation) but the iNat
direct export goes back to 2016 and is more complete; the GBIF mirror is a
filtered subset with the canonical taxonomy IDs attached.

## How this connects to the vocabulary sources

Each observation has a `scientific_name` (iNat) or `species` (GBIF). Joining
that name to any source's `species_tags` matrix (e.g.
`manual_of_woody_plants/species_tags/`) gives you the descriptive feature
profile for the observed plant.

Example query pattern:

```
observation (lat, lon, date, species_name, photo_url)
    join species_tags  on species_name
    join vocabulary    on value_label    (after unpivot)
```

→ produces rows of the form
`(date, lat, lon, photo_url, feature_name, value_label, definition)`.

## Out of scope here

- **GIS layers** (parcel boundaries, trails, ecosites) — large (~235 MB),
  lives in `arq-refdata/pound_ridge/gis/`. Not needed for vocabulary work.
- **Demo HTML maps** — visualization scratch from arq-refdata.
- **Other test sites** — Pound Ridge is the only site staged here. Future
  sites will be siblings (e.g. `observations_<site_name>/`).

## Source of truth

These files are snapshots from `arq-refdata/pound_ridge/`. 
