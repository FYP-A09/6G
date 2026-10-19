# Slice Twin

A monitor for the unified environment: watch how three controllers split each cell's
bandwidth between eMBB, URLLC and mMTC through one held-out day of real Milan traffic
(Wed 6 Nov 2013), with scripted incidents. It borrows from Cisco Packet Tracer:
Realtime and Simulation modes, an event list, and step-to-next-event playback.

- **Block view:** the 100 central cells. Each ring is one cell's slice split, moving
  packets show served traffic, and the inspector shows a cell's demand, allocation,
  latency against target and SLA.
- **City view:** all 10,000 grid cells, coloured by real traffic, load or SLA misses, with
  landmarks and city-wide incidents. Click a cell to compare the controllers on it.

Controllers: Static (equal thirds), Reactive (expert rules on current demand), and
Predictive (the same rules fed the TGNN forecast of the next 10 minutes).

## Run locally

```bash
python -m http.server 8765 --directory app/slice_twin
```

Then open http://localhost:8765.

## Regenerate the data

You need the Milan files on the project drive.

```bash
python src/unified/showcase.py app/slice_twin/data.json   # block view: data.json
python src/unified/forecast_city.py                        # city forecaster (once)
python src/unified/showcase_city.py app/slice_twin         # city view: city.json, city.bin.gz
```
