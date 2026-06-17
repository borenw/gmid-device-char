# gmid-device-char

**PDK-agnostic** MOSFET **gm/I_D device characterization** driven from Python + Cadence
**Spectre**. It **auto-discovers** the NMOS/PMOS models, their type, minimum L, supply VDD, and
PDK include files **from any prior Spectre run** — nothing process-specific is hardcoded — then
builds a standalone single-device sweep bench, extracts the small-signal operating point
(`gm, gds, gmbs, cgs, cgd, cgg, vth, vdsat, ft, …`) over a **geometry × bias** matrix, and produces
gm/I_D design plots and a reusable lookup table.

Verified across multiple bulk and FinFET process nodes (180 nm down to 15 nm) from the same
notebook with no code changes — only `RUN_LOG` differs. It handles spectre-syntax cards and **HSPICE
`.model … nmos` cards pulled in via `simulator lang=spice` / `.include` / `.lib`** (re-wrapped
automatically, with `.param` corner knobs carried along), **binned models** (instances use the base
name, cards are `<base>.1`/`.2`… — both registered), **FinFETs sized by `nfin`** (planar by `w`),
normalizes op-point name variants across BSIM4 / BSIM-CMG, and **capitalizes on the run's own
analyses** — taking the swept DC bias range as VDD and matching the run temperature.

## Notebooks

| file | what it does |
|---|---|
| `device_char.ipynb` | The main deliverable. Auto-discovers the devices from your run, validates them with a quick Spectre op, then runs gm/I_D characterization with per-step PASS/FAIL gates, a `grep`-the-raw-output worked example, and design plots. |
| `rerun_spectre_sim.ipynb` | Re-runs a previous Spectre transient sim from its run directory, non-destructively, and summarizes the result. |
| `make_nb.py` | Generator script that builds `device_char.ipynb` (edit this, then re-run to regenerate the notebook). |

## Auto-discovery

From `RUN_LOG` (a `spectre.out` log or an `input.scs`), the notebook:
- locates the netlist (via the log's working dir / `Reading file:`),
- rewrites all `include`/`.include` lines to absolute paths — preserving `simulator lang=spice`
  context so **HSPICE model cards re-resolve correctly** — so models load from any working dir,
- scans the netlist + included cards for MOSFET models — spectre `model … bsim4 type=n|p`,
  `(inline) subckt …` wrappers, **and HSPICE `.model … nmos|pmos level=54`** — tags each n/p, and
  picks the **most-instantiated** NMOS and PMOS,
- takes **Lmin** = the smallest instantiated L of each, and **VDD** = the larger of the supply
  source and the run's swept DC bias range (`idvds`/`idvgs` `stop=`/`values=`), and **temperature**
  from the run's `options temp=`,
- **validates** each chosen device by running a one-point Spectre op (the "command-line query"
  fallback) before the full sweep.

If auto-discovery can't pick an NMOS/PMOS pair, Step 1 **doesn't just fail** — it prints the
candidate MOSFET model names found in the netlist (with inferred type and use-count) plus a
ready-to-edit `DEVICES_OVERRIDE = [(model, type, VDD, [L…]), …]` template, and halts so you can
paste the right device names in and re-run. (On success it still lists the candidates so you can
override a wrong pick the same way.)

Sweep: nested **L (Lmin→2·Lmin, 7 steps) × VDS × VGS**, `W = 2 µm`, `nf = 1`, `VSB = 0`, 27 °C.
PMOS biases flipped automatically.

## How it works

1. **Standalone input** — set `RUN_LOG` at the top to any prior Spectre run's `spectre.out`
   (or `input.scs`). Everything else (devices, includes, supply, tool path) is auto-discovered.
2. Generates one Spectre netlist per device, runs it (`-format psfascii` so the PSF is
   readable), caches results, and skips re-simulation when nothing changed.
3. Parses the ASCII PSF with [`psf_utils`](https://pypi.org/project/psf-utils/) into a tidy
   `pandas` DataFrame, adds figures of merit (`gm/I_D`, intrinsic gain `gm/gds`, `Vov`, `fT`).
4. Plots: **gm/I_D vs Vov** (fixed scales), **fT vs gm/I_D** (with the L sweep and a gm/I_D = 15
   design line), intrinsic gain, and output characteristics.
5. Exports `<process>_devchar.parquet` (full dataset) and `<process>_gmid_lut.npz` (N-D lookup
   grids per device). The `<process>` prefix is auto-derived from the run's sim directory name so
   characterizing several processes into the same `devchar_runs/` folder doesn't overwrite —
   override with `PROCESS = "…"` in step 1.

## Requirements

- Cadence **Spectre** (developed against SPECTRE 20.1) and a BSIM / BSIM-CMG MOSFET PDK.
- Run on any host where the tools/PDK are installed. `spectre` is often **not** on `$PATH`, so
  the notebook resolves it in this order: **`SPECTRE_BIN` env var → the exact binary recorded in
  `RUN_LOG`'s command line (matches the original run) → `$PATH` → common Cadence install dirs**
  (`/usr/local/packages/*/SPECTRE*/…`, `/opt/cadence*/…`). Licensing is taken from your
  environment (`CDS_LIC_FILE` / `LM_LICENSE_FILE`); if unset it's auto-matched to the chosen
  install's `license.dat`, or set `CDS_LIC_OVERRIDE` to force a specific file.
- Python: `pip install --user psf_utils pandas numpy matplotlib pyarrow` + Jupyter.

## Configuration (the only required edit)

Open `device_char.ipynb`, set `RUN_LOG` (top cell) to any prior Spectre run from **your**
environment — its `psf/spectre.out` log or its `input.scs` netlist — then **Run All**. Everything
else is auto-discovered; the sweep ranges (`W`, `DVGS`, `DVDS`, …) and `DEVICES_OVERRIDE` are plain
variables in step 1 if you want to retarget geometry or models.

> Note: `device_char.ipynb` is committed as a **clean template** (no executed outputs) so it stays
> PDK-neutral. Running it regenerates everything. The large simulation data (`devchar_runs/`,
> `*.parquet`, `*.npz`) is regenerable and is **git-ignored**.
