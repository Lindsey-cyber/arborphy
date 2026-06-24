# GoBotany Illustrations — Placeholder

GoBotany is a live web service operated by the New England Wild Flower Society.
Each character value in the GoBotany identification key is accompanied by an
illustration served from the gobotany.nativeplanttrust.org image CDN.

These illustrations have **not been mirrored locally** in the arq-refdata
extraction. The `character_value_labels.json` API response in
`../raw_pages/api_raw/` contains `image_url` fields pointing to the originals.

## Target output

Following the Dirr pattern (`../../manual_of_woody_plants/illustrations/`):

- One PNG per character value, fetched from the GoBotany CDN
- `label_map.csv` mapping each PNG to the GoBotany `(character, value)` pair
- A licensing note — GoBotany content is CC-licensed; verify terms before mirroring

This is a candidate task for the summer extraction work.
