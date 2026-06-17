import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):  cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Device characterization — gm/ID matrix (PDK-agnostic)  ·  standalone

Self-contained Spectre **gm/ID sweep bench**. It **auto-discovers** the NMOS + PMOS device models,
their type, minimum L, supply VDD, and PDK include files **from your run** — nothing PDK-specific
is hardcoded — spectre-syntax model cards, HSPICE `.model … nmos` cards (via `simulator lang=spice`
/ `.include` / `.lib` + `.param` corner knobs), **binned models** (instance uses the base name,
cards are `<base>.1`/`.2`…), and **FinFETs sized by `nfin`** are all handled — planar BSIM and
FinFET BSIM-CMG alike, across a range of bulk and FinFET process nodes. It also **capitalizes on
the run's own analyses** — taking the swept bias range as VDD and matching the run temperature.
Extracts `gm, gds, cgs, cgd, cgg, vth, vdsat, ft …` over a **geometry × bias** matrix.

**No Claude needed to re-run.** Point `RUN_LOG` below at *any* previous Spectre run (its
`spectre.out` log **or** its `input.scs`) and **Cell → Run All**. The notebook resolves the netlist,
absolutizes the PDK includes, picks the most-instantiated N/P devices, and validates each by running
a quick Spectre op. Swap the log + Run All to re-characterize a different process.

Each step shows a yellow **RUNNING** banner that clears to a green **PASS** (red **FAIL** halts).
**Run wherever Cadence Spectre and your PDK are installed** — Spectre is auto-detected
(`SPECTRE_BIN` → the binary named in `RUN_LOG` → `$PATH` → common install dirs) and licensing is
taken from your environment.""")

md("""## ⓘ USER INPUT — provide an existing Spectre run
Give the path to a previous run's **log** (`…/psf/spectre.out`) or its **netlist** (`…/input.scs`).
That's the only edit needed to retarget the notebook.""")
code('''# REQUIRED: path to ANY existing Spectre run from YOUR environment -- either its log
# ("<run>/psf/spectre.out") or its netlist ("<run>/input.scs"). The notebook lifts that
# run's PDK `include` lines so the SAME device models resolve here. This is the only edit
# most users need.
#   example:  RUN_LOG = "/path/to/your/run/psf/spectre.out"
RUN_LOG = ""   # <-- SET ME''')

md("## 1. Setup · model resolution · PASS/FAIL + RUNNING helpers")
code('''%matplotlib inline
import os, glob, re, shutil, subprocess, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from psf_utils import PSF
from IPython.display import display, HTML, clear_output

# ---------- live status: yellow RUNNING -> cleared -> green PASS ----------
RUNNING_MIN_S = 0.2                   # keep the yellow banner visible at least this long
_RUN_T0 = [0.0]
def running(step):
    _RUN_T0[0] = time.time()
    display(HTML(f'<div style="background:#ffd400;color:#111;font-weight:700;'
                 f'padding:4px 12px;border-radius:4px;display:inline-block">'
                 f'\\u25b6 RUNNING \\u2014 {step} \\u2026</div>'))
def finish():
    dt = time.time() - _RUN_T0[0]     # guarantee a visible dwell even for cached/fast steps
    if dt < RUNNING_MIN_S:
        time.sleep(RUNNING_MIN_S - dt)
    clear_output(wait=True)           # wipes the yellow banner when the PASS lines print

def check(label, ok, detail=""):
    tag = "\\033[1;42;30m PASS \\033[0m" if ok else "\\033[1;41;97m FAIL \\033[0m"
    print(f"{tag}  {label}" + (f"  \\u2014 {detail}" if detail else ""))
    return bool(ok)
def gate(step, oks):
    allok = all(oks)
    bar = "\\033[1;42;30m" if allok else "\\033[1;41;97m"
    print(bar + f"  {step}: {'PASS' if allok else 'FAIL'}  ({sum(oks)}/{len(oks)} checks)  " + "\\033[0m")
    if not allok:
        raise AssertionError(f"{step} FAILED \\u2014 fix above before continuing.")

def show_latest(globpat, label="latest output"):
    """ls -lt: newest file matching an absolute glob, with full path + timestamp + size."""
    cmd = f"ls -lt --time-style=long-iso {globpat} 2>/dev/null | head -1"
    print(f"$ {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.rstrip()
    print(out if out else "(no files yet)")

# ---------- AUTO-DISCOVER the run: netlist, absolutized includes, and devices ----------
# Generalizes across PDKs (no hardcoded model names/supply). Reads the log, then the run
# directory's netlist + included model cards. Picks the most-instantiated NMOS and PMOS,
# their Lmin (min instantiated L), and the supply VDD.
def _si(s):
    suf = {'T':1e12,'G':1e9,'M':1e6,'K':1e3,'k':1e3,'m':1e-3,
           'u':1e-6,'n':1e-9,'p':1e-12,'f':1e-15,'a':1e-18}
    s = (s or "").strip()
    try:
        return float(s[:-1])*suf[s[-1]] if s and s[-1] in suf else float(s)
    except Exception:
        return None

def _kv(text):
    d = {}
    for tok in text.replace(" =", "=").replace("= ", "=").split():
        if "=" in tok:
            k, v = tok.split("=", 1); d[k.strip().lower()] = v.strip()
    return d

def discover_run(run_ref):
    txt = open(run_ref, errors="ignore").read() if run_ref and os.path.exists(run_ref) else ""
    netlist = None
    if "Reading file:" in txt:                                   # run_ref is a log
        cwd = rd = None
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith("Current working directory:"): cwd = s.split(":", 1)[1].strip()
            elif s.startswith("Reading file:") and s.endswith(".scs") and rd is None:
                rd = s.split(":", 1)[1].strip()
        for c in ([os.path.join(cwd, "input.scs")] if cwd else []) + ([rd] if rd else []):
            if c and os.path.exists(c): netlist = c; break
    elif run_ref and os.path.exists(run_ref):
        netlist = run_ref
    if not netlist or not os.path.exists(netlist):
        return dict(netlist=None, include_block="", devices=[], info="no netlist found")
    ndir = os.path.dirname(os.path.abspath(netlist)); ntxt = open(netlist, errors="ignore").read()

    # includes -> (lang, kind, abspath, tail); track `simulator lang=` and handle spectre
    # `include`, spice `.include`, and `.lib "file" SECTION` — re-wrapped so
    # HSPICE/spice model files resolve in the right language.
    inc = []
    lang = "spectre"
    for ln in ntxt.splitlines():
        s = ln.strip(); low = s.lower().replace(" ", "")
        if low.startswith("simulatorlang=spice"): lang = "spice"; continue
        if low.startswith("simulatorlang=spectre"): lang = "spectre"; continue
        kind = kw = None
        if s.startswith("include "):    kind, kw = "include", "include "
        elif s.startswith(".include "): kind, kw = "include", ".include "
        elif s.startswith(".lib ") and '"' in s: kind, kw = "lib", ".lib "
        if kw:
            rest = s[len(kw):].strip()
            if rest and rest[0] in ('"', "'"):
                q = rest[0]; path = rest[1:rest.index(q, 1)]; tail = rest[rest.index(q, 1)+1:]
            else:
                path = rest.split()[0]; tail = rest[len(path):]
            if not os.path.isabs(path): path = os.path.normpath(os.path.join(ndir, path))
            inc.append((lang, kind, path, tail))
        elif s.startswith(".param ") or s.startswith("parameters "):
            inc.append((lang, "param", s, ""))   # model-corner params (corner knobs)
    parts = []
    for lg, kind, path, tail in inc:
        if kind == "param":
            parts.append('simulator lang=spice\\n%s\\nsimulator lang=spectre' % path if lg == "spice" else path)
        elif kind == "lib":
            parts.append('simulator lang=spice\\n.lib "%s"%s\\nsimulator lang=spectre' % (path, tail))
        elif lg == "spice":
            parts.append('simulator lang=spice\\n.include "%s"%s\\nsimulator lang=spectre' % (path, tail))
        else:
            parts.append('include "%s"%s' % (path, tail))
    include_block = "\\n".join(parts) + ("\\n" if parts else "")

    blob = ntxt                                                  # model text = netlist + includes
    for lg, kind, path, tail in inc:
        if kind in ("include", "lib") and os.path.exists(path):
            try: blob += "\\n" + open(path, errors="ignore").read()
            except Exception: pass
    lines = blob.splitlines()

    def _new_stmt(line):
        h = line.split()[:1]
        return bool(h) and h[0].lstrip(".").lower() in ("model", "subckt", "inline", "ends")
    models = {}                                                  # name -> 'n'/'p'
    def _register(name, t):
        models.setdefault(name, t)
        if "." in name and name.rsplit(".", 1)[1].isdigit():    # binned model: <base>.<n> -> base
            models.setdefault(name.rsplit(".", 1)[0], t)
    for i, ln in enumerate(lines):
        toks = ln.strip().split()
        tok0 = toks[0].lstrip(".").lower() if toks else ""
        if tok0 == "model" and len(toks) >= 3:                   # spectre bsim + HSPICE .model
            name = toks[1]; kw = toks[2].lower()
            if kw.startswith("nmos") or kw == "n":   t = "n"
            elif kw.startswith("pmos") or kw == "p": t = "p"
            elif kw.startswith("bsim") or "mos" in kw or kw.startswith("psp"):
                body = [ln]; j = i + 1                            # gather '+' and '{ }' card styles
                while j < len(lines) and j < i + 60 and not _new_stmt(lines[j]):
                    body.append(lines[j]); j += 1
                b = " ".join(body).replace(" ", "").lower()
                t = "n" if "type=n" in b else ("p" if "type=p" in b else None)
            else: t = None
            if t: _register(name, t)
        if toks and (toks[0] == "subckt" or (len(toks) >= 2 and toks[0] == "inline" and toks[1] == "subckt")):
            name = toks[2] if toks[0] == "inline" else toks[1]
            body = []; j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("ends"):
                body.append(lines[j].strip()); j += 1
            bt = " ".join(body).lower()
            if "bsim" not in bt: continue
            btn = bt.replace(" ", "")
            t = "n" if "type=n" in btn else ("p" if "type=p" in btn else None)
            if t is None:
                nm = name.lower()
                t = "n" if (nm[:1] == "n" or "nmos" in nm or "nch" in nm) else \\
                    ("p" if (nm[:1] == "p" or "pmos" in nm or "pch" in nm) else None)
            if t: _register(name, t)

    inst = {}; sizing = {}                                       # instantiated MOS -> [L]; size param
    for ln in ntxt.splitlines():
        s = ln.strip()
        if not s or s.startswith("//") or s.startswith("*") or "(" not in s or ")" not in s: continue
        rest = s.split(")", 1)[1].split()
        if rest and rest[0] in models:
            kv = _kv(" ".join(rest[1:]))
            inst.setdefault(rest[0], []).append(_si(kv.get("l")) if kv.get("l") else None)
            if "nfin" in kv: sizing[rest[0]] = "nfin"            # FinFET (BSIM-CMG)
            elif "w" in kv:  sizing.setdefault(rest[0], "w")

    # candidate device names — every model token used by an M*/m* instance, with inferred type.
    # Shown to the user when auto-pick fails (or to override a wrong pick) via DEVICES_OVERRIDE.
    mos_inst = {}
    for ln in ntxt.splitlines():
        s = ln.strip()
        if not s or s[0] not in "mM" or "(" not in s or ")" not in s: continue
        rest = s.split(")", 1)[1].split()
        if rest: mos_inst[rest[0]] = mos_inst.get(rest[0], 0) + 1
    def _gtype(nm):
        if nm in models: return models[nm]
        low = nm.lower()
        if low[:1] == "n" or "nmos" in low or "nch" in low: return "n"
        if low[:1] == "p" or "pmos" in low or "pch" in low: return "p"
        return "?"
    candidates = sorted(({"name": k, "type": _gtype(k), "uses": v} for k, v in mos_inst.items()),
                        key=lambda d: -d["uses"])

    def pick(t):
        used = [k for k in inst if models[k] == t]
        if used: return max(used, key=lambda k: len(inst[k]))
        cand = [k for k in models if models[k] == t]
        return cand[0] if cand else None
    nmos, pmos = pick("n"), pick("p")

    def lmin_for(m):
        Ls = [x for x in inst.get(m, []) if x]
        return min(Ls) if Ls else 0.18e-6

    vdd = None                                                   # supply-named net, else max dc
    for ln in ntxt.splitlines():
        s = ln.strip()
        if "vsource" not in s or "(" not in s or ")" not in s: continue
        nodes = s[s.find("(")+1:s.find(")")].split()
        v = _si(_kv(s.split(")", 1)[1]).get("dc"))
        if v is not None and nodes and nodes[0].rstrip("!").lower() in \\
                ("vdd","vcc","vdda","vddio","vpwr","vcca","vddd","avdd"):
            vdd = v; break
    if vdd is None:
        allv = [_si(_kv(s.split(")", 1)[1]).get("dc")) for s in
                (x.strip() for x in ntxt.splitlines()) if "vsource" in s and ")" in s]
        allv = [v for v in allv if v is not None]
        vdd = max(allv) if allv else 1.2
    # CAPITALIZE on the run's own analyses: the bias actually swept (DC sweep `stop=` and
    # idvds `values=[...]`) is the characterization rail -> take its max as VDD.
    vbias = []
    for ln in ntxt.splitlines():
        s = ln.strip().lower()
        if "param=dc" in s.replace(" ", "") and "stop=" in s.replace(" ", ""):
            v = _si(_kv(ln).get("stop"))
            if v is not None: vbias.append(v)
        if "values=[" in s:
            seg = s[s.find("values=[")+8:]; seg = seg[:seg.find("]")]
            vbias += [v for v in (_si(t) for t in seg.split()) if v is not None]
    if vbias: vdd = max(vdd, max(vbias))
    vdd = min(max(round(vdd, 3), 0.6), 5.0)

    temp = 27.0                                                  # CAPITALIZE: match run temperature
    for ln in ntxt.splitlines():
        s = ln.strip().lower()
        if "options" in s and "temp=" in s.replace(" ", ""):
            v = _si(_kv(ln).get("temp"))
            if v is not None: temp = v; break

    devices = []
    for model, t in [(nmos, "n"), (pmos, "p")]:
        if model:
            lm = lmin_for(model)
            Ls = [round(x*1e6, 4) for x in np.linspace(lm, 2*lm, 7)]
            devices.append((model, t, vdd, Ls))
    return dict(netlist=netlist, include_block=include_block, devices=devices,
                nmos=nmos, pmos=pmos, vdd=vdd, temp=temp, sizing=sizing,
                candidates=candidates, n_models=len(models), info="ok")

def process_tag(netlist):
    # short process label from the run's sim directory (strip a leading "spectre_"),
    # used to prefix output files so different processes don't overwrite each other.
    if not netlist: return "run"
    d = os.path.dirname(os.path.abspath(netlist)); name = os.path.basename(d)
    if name in ("netlist", "", "."): name = os.path.basename(os.path.dirname(d))
    name = re.sub("^[Ss]pectre[_-]?", "", name)
    return re.sub("[^A-Za-z0-9._-]+", "_", name).strip("_") or "run"

# ---------- environment (portable auto-detect; override if needed) ----------
# NOTE: at many sites `spectre` is NOT on $PATH, so we also (a) reuse the exact spectre
# binary recorded in RUN_LOG's command line (best: matches the original run), and
# (b) scan common Cadence install dirs. Override anytime with the SPECTRE_BIN env var.
def find_spectre(run_ref=""):
    if os.environ.get("SPECTRE_BIN"):
        return os.environ["SPECTRE_BIN"]
    if run_ref and os.path.exists(run_ref):                # the spectre that made this run
        try:
            m = re.search(r"(/[^\\s'\\\"]+/bin/spectre)\\b", open(run_ref).read())
            if m and os.path.exists(m.group(1)):
                return m.group(1)
        except Exception:
            pass
    if shutil.which("spectre"):
        return shutil.which("spectre")
    pats = ["/usr/local/packages/*/SPECTRE*/tools.lnx86/bin/spectre",
            "/usr/local/cadence*/*/SPECTRE*/tools*/bin/spectre",
            "/opt/cadence*/*/SPECTRE*/tools*/bin/spectre",
            os.path.expanduser("~/*/SPECTRE*/tools*/bin/spectre")]
    hits = sorted({h for p in pats for h in glob.glob(p)})
    return hits[-1] if hits else "spectre"                 # newest install; or set SPECTRE_BIN

def find_license(spectre_path):
    # already provided by the environment? leave it.
    if os.environ.get("CDS_LIC_FILE") or os.environ.get("LM_LICENSE_FILE"):
        return os.environ.get("CDS_LIC_FILE") or os.environ.get("LM_LICENSE_FILE")
    if os.environ.get("CDS_LIC_OVERRIDE"):
        os.environ["CDS_LIC_FILE"] = os.environ["CDS_LIC_OVERRIDE"]; return os.environ["CDS_LIC_FILE"]
    d = os.path.dirname(os.path.abspath(spectre_path))     # walk up to a sibling license.dat
    for _ in range(6):
        cand = os.path.join(d, "license.dat")
        if os.path.exists(cand):
            os.environ["CDS_LIC_FILE"] = cand; return cand
        d = os.path.dirname(d)
    return None

SPECTRE = find_spectre(RUN_LOG)
LICENSE = find_license(SPECTRE)

# ---------- sweep settings (edit to taste) ----------
# To bypass auto-discovery, set DEVICES_OVERRIDE = [(model, "n"/"p", VDD, [L_um,...]), ...]
DEVICES_OVERRIDE = None
PROCESS = ""                                               # output-file prefix; blank = auto from sim dir
W_TOTAL, NF, VSB, DVGS, DVDS = 2e-6, 1, 0.0, 0.025, 0.05   # planar: W/finger = W_TOTAL/NF
NFIN = 10                                                  # FinFET fin count (BSIM-CMG, nfin-sized)
OUTDIR = os.path.expanduser("~/devchar_runs"); os.makedirs(OUTDIR, exist_ok=True)

running("STEP 1 \\u2014 Setup, tool + device auto-discovery")
disc = discover_run(RUN_LOG) if (RUN_LOG and os.path.exists(RUN_LOG)) else \\
       dict(netlist=None, include_block="", devices=[])
INCLUDE_BLOCK = disc.get("include_block", "")
DEVICES = DEVICES_OVERRIDE if DEVICES_OVERRIDE else disc.get("devices", [])
TEMP = disc.get("temp", 27.0)          # characterize at the run's own temperature
SIZING = disc.get("sizing", {})        # {model: "nfin"|"w"} — FinFET vs planar
PROC = PROCESS or process_tag(disc.get("netlist"))   # output-file prefix
def device_line(model):                # size FinFETs by nfin, planar devices by w/nf
    if SIZING.get(model) == "nfin":
        return f"MD ( d g 0 b ) {model} l=pL nfin={NFIN}"
    return f"MD ( d g 0 b ) {model} l=pL w=pW nf=pNF"
finish()
SPECTRE_OK = bool(shutil.which(SPECTRE) or os.path.exists(SPECTRE))
ntypes = {d[1] for d in DEVICES}
oks = []
oks.append(check("RUN_LOG set & exists", bool(RUN_LOG) and os.path.exists(RUN_LOG),
                 RUN_LOG or "set RUN_LOG above"))
oks.append(check("spectre found", SPECTRE_OK,
                 SPECTRE if SPECTRE_OK else "not on PATH/install dirs; set SPECTRE_BIN"))
oks.append(check("license (FYI)", True, LICENSE or "ambient env"))
oks.append(check("netlist located", bool(disc.get("netlist")), disc.get("netlist") or "not found"))
oks.append(check("include block resolved (FYI)", True,
                 f"{INCLUDE_BLOCK.count(chr(10))} absolute include line(s)"))
cand = disc.get("candidates", [])
oks.append(check("devices selected (auto or DEVICES_OVERRIDE)", len(DEVICES) >= 1,
                 ", ".join(f"{d[0]}({d[1]})" for d in DEVICES)
                 or "none - see candidate list below / set DEVICES_OVERRIDE"))
oks.append(check("NMOS + PMOS coverage (FYI)", True,
                 "n+p" if {"n", "p"} <= ntypes else f"only {sorted(ntypes) or 'none'}"))
oks.append(check("supply VDD / temp (FYI)", True,
                 f"{DEVICES[0][2]} V @ {TEMP} C" if DEVICES else "?"))
oks.append(check("process tag / output prefix (FYI)", True, PROC))

# Fallback for the auto-discovery error: list candidate device names + a fill-in template
if not DEVICES:
    print("\\n\\033[1;41;97m  AUTO-DISCOVERY could not select an NMOS/PMOS pair.  \\033[0m")
    if cand:
        print("MOSFET models used by the netlist (pick one NMOS and one PMOS):")
        for c in cand:
            print(f"    {c['name']:24s} type={c['type']}   (used {c['uses']}x)")
    else:
        print("  (no MOSFET instances found - is RUN_LOG pointing at the right run?)")
    _vg = disc.get("vdd", 1.0)
    _ls = ", ".join(str(round(x, 4)) for x in np.linspace(0.05, 0.10, 7))
    print("\\nThen set DEVICES_OVERRIDE at the TOP of this cell and re-run, e.g.:")
    print("    DEVICES_OVERRIDE = [")
    print(f'        ("<nmos_model>", "n", {_vg}, [{_ls}]),   # L values in microns')
    print(f'        ("<pmos_model>", "p", {_vg}, [{_ls}]),')
    print("    ]")

gate("STEP 1 \\u2014 Setup & discovery", oks)
for d in DEVICES:
    print(f"   * {d[1].upper()} {d[0]:14s} VDD={d[2]} V  L={d[3][0]}..{d[3][-1]} um (7 steps)")
if DEVICES and cand:
    print("   candidates seen (set DEVICES_OVERRIDE to change): "
          + ", ".join(f"{c['name']}[{c['type']}]" for c in cand))''')

md("""## 1b. Validate each discovered device with a quick Spectre op (command-line query)
If file-based discovery is wrong, this catches it: we actually instantiate the model and run a
one-point Spectre operating point. A clean run confirms the model name + params are valid.""")
code('''def probe_device(model, mtype, vdd, Ls):
    sgn = 1 if mtype == "n" else -1
    d = os.path.join(OUTDIR, "_probe"); os.makedirs(d, exist_ok=True)
    nl = f"""simulator lang=spectre
global 0
{INCLUDE_BLOCK}
parameters pL={Ls[0]}u pW={W_TOTAL} pNF={NF}
{device_line(model)}
Vg ( g 0 ) vsource dc={round(sgn*vdd*0.7, 3)}
Vd ( d 0 ) vsource dc={round(sgn*vdd*0.5, 3)}
probeOpts options temp={TEMP} tnom={TEMP}
opp dc
"""
    open(os.path.join(d, "probe.scs"), "w").write(nl)
    subprocess.run([SPECTRE, "-64", "probe.scs", "-format", "psfascii", "-raw", "./praw",
                    "+log", "probe.out"], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(d, "probe.out")).read() if os.path.exists(os.path.join(d, "probe.out")) else ""
    ok = "0 errors" in log
    return ok, ("0 errors - model valid" if ok else "spectre rejected it (see _probe/probe.out)")

running("STEP 1b \\u2014 device validation via Spectre op")
probe = [(d[0],) + probe_device(*d) for d in DEVICES]
finish()
oks = ([check(f"{m}: simulates", ok, msg) for (m, ok, msg) in probe]
       or [check("at least one device discovered", False, "none")])
gate("STEP 1b \\u2014 device validation", oks)''')

md("## 2. Netlist generator")
code('''def gen_netlist(model, mtype, vdd, Ls):
    sgn  = 1 if mtype == "n" else -1
    Lstr = " ".join(f"{l*1e-6:.6g}" for l in Ls)
    return f"""// auto-generated gm/ID characterization bench: {model}
simulator lang=spectre
global 0
{INCLUDE_BLOCK}
parameters pL=0.35u pW={W_TOTAL} pNF={NF} VGS=0 VDS=0 VSB={VSB}
{device_line(model)}
Vd ( d 0 ) vsource dc={sgn}*VDS
Vg ( g 0 ) vsource dc={sgn}*VGS
Vb ( b 0 ) vsource dc={-sgn}*VSB
charOpts options temp={TEMP} tnom={TEMP}
save MD:oppoint
swpL sweep param=pL values=[{Lstr}] {{
  swpVDS sweep param=VDS start=0 stop={vdd} step={DVDS} {{
    dcVGS dc param=VGS start=0 stop={vdd} step={DVGS}
  }}
}}
"""

running("STEP 2 \\u2014 Netlist generator")
_nl = gen_netlist(*DEVICES[0]); finish()
oks = []
oks.append(check("netlist non-empty", len(_nl) > 500, f"{len(_nl)} chars"))
oks.append(check("model instantiated", DEVICES[0][0] in _nl))
oks.append(check("op-point saved", "save MD:oppoint" in _nl))
oks.append(check("nested L/VDS/VGS sweep", all(k in _nl for k in ("swpL", "swpVDS", "dcVGS"))))
gate("STEP 2 \\u2014 Netlist generator", oks)''')

md("## 3. Run Spectre for each device  (cached unless geometry/grid changes)")
code('''FORCE_RESIM = False
def expected_files(vdd, Ls):
    return len(Ls) * len(np.arange(0, vdd + 1e-9, DVDS))
def run_device(model, mtype, vdd, Ls):
    d = os.path.join(OUTDIR, model); raw = os.path.join(d, "raw"); os.makedirs(d, exist_ok=True)
    have = len(glob.glob(os.path.join(raw, "*_dcVGS.dc")))
    if not FORCE_RESIM and have == expected_files(vdd, Ls):
        return 0, have, "cached (skipped sim)"
    if os.path.isdir(raw): shutil.rmtree(raw)
    open(os.path.join(d, "char.scs"), "w").write(gen_netlist(model, mtype, vdd, Ls))
    r = subprocess.run([SPECTRE, "-64", "char.scs", "-format", "psfascii",
                        "-raw", "./raw", "+escchars", "+log", "char.out"],
                       cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summ = [l for l in open(os.path.join(d, "char.out")) if "spectre completes" in l]
    return r.returncode, len(glob.glob(os.path.join(raw, "*_dcVGS.dc"))), \\
           (summ[-1].strip() if summ else "no summary")

running("STEP 3 \\u2014 Spectre runs (may take minutes if not cached)")
res = [(d[0],) + run_device(*d) + (expected_files(d[2], d[3]),) for d in DEVICES]
finish()
oks = [check(f"{name:9s} rc={rc} files={n}/{exp}",
            rc == 0 and n == exp and ("cached" in s or "0 errors" in s), s)
       for (name, rc, n, s, exp) in res]
gate("STEP 3 \\u2014 Spectre runs", oks)
show_latest(f"{OUTDIR}/*/raw/*_dcVGS.dc", "newest sim output")''')

md("## 4. Load every operating point into a tidy DataFrame")
code('''def load_device(model, mtype, vdd, Ls):
    VDS_axis = np.round(np.arange(0, vdd + 1e-9, DVDS), 6); frames = []
    for f in sorted(glob.glob(os.path.join(OUTDIR, model, "raw", "swpL-*_swpVDS-*_dcVGS.dc"))):
        m = re.search(r"swpL-(\\d+)_swpVDS-(\\d+)", os.path.basename(f))
        iL, iV = int(m.group(1)), int(m.group(2)); psf = PSF(f)
        vgs = np.real(np.asarray(psf.sweeps[0].abscissa))
        sig = {s.name.split(":", 1)[1]: np.real(np.asarray(s.ordinate))
               for s in psf.all_signals() if ":" in s.name}
        rec = {"device": model, "type": mtype, "vdd": vdd, "L_um": Ls[iL],
               "VDS": VDS_axis[iV], "VGS": vgs}
        for k, v in sig.items():
            if np.ndim(v) == 1 and len(v) == len(vgs): rec[k] = v
        frames.append(pd.DataFrame(rec))
    return pd.concat(frames, ignore_index=True)

running("STEP 4 \\u2014 Load & compute figures of merit")
df = pd.concat([load_device(*d) for d in DEVICES], ignore_index=True)
# normalize op-point name variants across model types (BSIM4 vs BSIM-CMG FinFET, etc.)
for prim, alts in {"vdsat": ["vdssat"], "ids": ["id"], "ft": ["fug"]}.items():
    if prim not in df.columns:
        for a in alts:
            if a in df.columns: df[prim] = df[a]; break
df["Id"]     = df["ids"].abs()
df["gm_id"]  = df["gm"] / df["Id"].replace(0, np.nan)     # gm/Id [S/A] (== 1/V)
df["gain"]   = df["gm"] / df["gds"].replace(0, np.nan)    # intrinsic gain gm/gds
df["Id_W"]   = df["Id"] / W_TOTAL
df["Vov"]    = df["VGS"] - df["vth"].abs()                # overdrive = |Vgs| - |Vth|
df["ft_GHz"] = (df["ft"] if "ft" in df else df["gm"]/(2*np.pi*df["cgg"].abs())) / 1e9
try:    df.to_parquet(os.path.join(OUTDIR, f"{PROC}_devchar.parquet")); saved = f"{PROC}_devchar.parquet"
except Exception: df.to_csv(os.path.join(OUTDIR, f"{PROC}_devchar.csv"), index=False); saved = f"{PROC}_devchar.csv"
finish()
essential = ["ids","gm","gds","gm_id","vth","cgg","ft_GHz","Vov"]   # needed for all FOMs/plots
optional  = ["cgs","cgd","vdsat","gmbs"]                            # nice-to-have, model-dependent
oks = []
oks.append(check("dataframe non-empty", len(df) > 0, f"{len(df):,} rows"))
oks.append(check("all devices present", set(df.device) == {d[0] for d in DEVICES}))
oks.append(check("essential params present", all(c in df for c in essential),
                 ",".join(c for c in essential if c not in df) or "all"))
oks.append(check("optional params (FYI)", True,
                 "present: " + (",".join(c for c in optional if c in df) or "none")))
oks.append(check("gm finite & non-zero", np.isfinite(df.gm).any() and (df.gm.abs() > 0).any()))
oks.append(check("results saved", os.path.exists(os.path.join(OUTDIR, saved)), saved))
gate("STEP 4 \\u2014 Load & FOM", oks)
show_latest(f"{OUTDIR}/{PROC}_devchar.*", "newest dataset")
df.head()''')

md("""## 4b. Worked example — `grep` the **gm** parameter straight from the raw output

Proves the data is really in the Spectre ASCII PSF, and ties one grep'd number to a point on the
design plot (6a). The op-point dump stores one `"MD:gm" <value>` line per VGS step.""")
code('''running("4b \\u2014 grep gm from raw simulation output")
EX_DEV = DEVICES[0][0]; EX_VDD = DEVICES[0][2]      # first discovered device, VDD-scaled bias
EX_VDS = round((EX_VDD / 2) / DVDS) * DVDS
EX_VGS = round((EX_VDD * 0.6) / DVGS) * DVGS
g  = df[df.device == EX_DEV]; Ls = sorted(g.L_um.unique()); EX_L = Ls[0]
iL, iV = Ls.index(EX_L), int(round(EX_VDS / DVDS))
EX_FILE = glob.glob(os.path.join(OUTDIR, EX_DEV, "raw",
                    f"swpL-{iL:03d}_swpVDS-{iV:03d}_dcVGS.dc"))[0]
finish()

def grep_nums(pat, f):
    out = subprocess.run(f"grep {pat!r} {f}", shell=True, capture_output=True, text=True).stdout
    vals = [float(t) for l in out.splitlines() for t in [l.split()[-1]]
            if t.lstrip("-")[:1].isdigit()]
    return out, np.array(vals)

print(f"# file under test:\\n{EX_FILE}\\n")
gm_out, gm_v   = grep_nums('"MD:gm"', EX_FILE)
_,      vgs_v  = grep_nums('"VGS"',   EX_FILE)
print(f"$ grep '\\"MD:gm\\"' <file>      # 1st line = unit, then one gm per VGS step")
print("\\n".join(gm_out.splitlines()[:5]) + "\\n   ... (" + str(len(gm_v)) + " values)\\n")

j = int(round(EX_VGS / DVGS))
EX_GM, EX_VGSv = gm_v[j], vgs_v[j]
print(f">>> example point:  VGS = {EX_VGSv:.3f} V   ->   grep'd gm = {EX_GM:.4e} S")

row = df[(df.device == EX_DEV) & (df.L_um == EX_L) &
         np.isclose(df.VDS, EX_VDS, atol=DVDS/2) &
         np.isclose(df.VGS, EX_VGS, atol=DVGS/2)].iloc[0]
EX_VOV, EX_GMID = float(row.Vov), float(row.gm_id)
oks = [check("grep'd gm matches psf_utils-loaded gm", np.isclose(EX_GM, row.gm, rtol=1e-9),
             f"{EX_GM:.4e} S vs {row.gm:.4e} S"),
       check("example maps to a plotted point", np.isfinite(EX_VOV) and np.isfinite(EX_GMID),
             f"Vov={EX_VOV:.3f} V, gm/Id={EX_GMID:.2f} S/A")]
gate("4b \\u2014 grep gm", oks)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(vgs_v, gm_v, "-o", ms=3, label=f"grep'd gm  ({EX_DEV}, L={EX_L}µm, VDS={EX_VDS}V)")
ax.semilogy([EX_VGSv], [EX_GM], "r*", ms=18, zorder=5)
ax.annotate(f"grep example: VGS={EX_VGSv:.2f}V, gm={EX_GM:.3e}S",
            (EX_VGSv, EX_GM), textcoords="offset points", xytext=(8, -22), color="r", fontsize=8)
ax.set(xlabel="VGS [V]", ylabel="gm [S]", title="gm vs VGS  =  the data returned by the grep command")
ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8); fig.tight_layout()''')

md("## 5. Device matrix — gm, gds, caps at a design bias (gm/Id, VDS=VDD/2)")
code('''def device_matrix(df, gm_id_target=10.0):
    recs = []
    for (dev, L), g in df.groupby(["device", "L_um"]):
        vdd = g["vdd"].iloc[0]; vds = round((vdd/2)/DVDS)*DVDS
        sub = g[np.isclose(g["VDS"], vds, atol=DVDS/2)]; sub = sub[sub["Id"] > 0].sort_values("gm_id")
        if len(sub) < 3: continue
        x = sub["gm_id"].values; rec = {"device": dev, "L_um": L, "VDS": vds}
        for p in ["Vov","Id","gm","gds","gain","cgs","cgd","cgg","vth","vdsat","ft_GHz"]:
            if p in sub: rec[p] = float(np.interp(gm_id_target, x, sub[p].values))
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["device", "L_um"]).reset_index(drop=True)

running("STEP 5 \\u2014 Device matrix")
GM_ID = 10.0; matrix = device_matrix(df, GM_ID); finish()
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
oks = []
oks.append(check("matrix has rows", len(matrix) > 0, f"{len(matrix)} rows"))
oks.append(check("one row per device×L", len(matrix) == sum(len(d[3]) for d in DEVICES)))
oks.append(check("gm > 0 all rows", bool((matrix.gm > 0).all())))
oks.append(check("intrinsic gain finite", bool(np.isfinite(matrix.gain).all())))
gate("STEP 5 \\u2014 Device matrix", oks)
_size = f"nfin={NFIN}" if "nfin" in SIZING.values() else f"W={W_TOTAL*1e6:g}um"
print(f"Device matrix @ gm/Id = {GM_ID} S/A, VDS = VDD/2, {_size}:")
matrix''')

md("## 6. gm/ID design plots")
code('''def at_vds_half(g):
    vdd = g["vdd"].iloc[0]; vds = round((vdd/2)/DVDS)*DVDS
    return g[np.isclose(g["VDS"], vds, atol=DVDS/2)]''')

md("### 6a. gm/Id [S/A] vs overdrive Vov  — y 0–35, x −0.4…1.0 (0.2 grid); red ★ = the grep example")
code('''running("STEP 6a \\u2014 gm/Id vs Vov")
ncol = min(len(DEVICES), 2); nrow = int(np.ceil(len(DEVICES) / ncol))
fig, axarr = plt.subplots(nrow, ncol, figsize=(6.2*ncol, 4.6*nrow), squeeze=False)
axes = axarr.flat
for ax, (dev, *_ ) in zip(axes, DEVICES):
    gv = at_vds_half(df[df.device == dev])
    for L, gl in gv.groupby("L_um"):
        gl = gl.sort_values("Vov")
        ax.plot(gl["Vov"].values, gl["gm_id"].values, label=f"L={L:g} µm")
    if dev == EX_DEV:                                  # pinpoint the grep worked-example datum
        ax.plot([EX_VOV], [EX_GMID], "r*", ms=16, zorder=6)
        ax.annotate(f"grep ex: gm={EX_GM:.2e}S", (EX_VOV, EX_GMID),
                    textcoords="offset points", xytext=(6, 6), color="r", fontsize=7)
    ax.set(title=dev, xlabel="Vov [V]", ylabel="gm/Id [S/A]")
    ax.set_xlim(-0.4, 1.0); ax.set_ylim(0, 35)
    ax.set_xticks(np.arange(-0.4, 1.0001, 0.2)); ax.set_yticks(np.arange(0, 35.001, 5))
    ax.grid(True, which="both", alpha=.35); ax.legend(fontsize=7)
for k in range(len(DEVICES), nrow*ncol): axes[k].axis("off")
fig.suptitle(f"gm/Id vs overdrive Vov  @ VDS = VDD/2   (red star on {EX_DEV} = grep example point)")
fig.tight_layout()
finish()
gate("STEP 6a \\u2014 gm/Id vs Vov",
     [check("data within axis ranges", bool(df.gm_id.between(0,35).any() and df.Vov.between(-0.4,1.0).any())),
      check("grep example star placed", np.isfinite(EX_VOV) and np.isfinite(EX_GMID),
            f"({EX_VOV:.3f} V, {EX_GMID:.2f} S/A)")])''')

md("### 6b. Ft [GHz] vs gm/Id [S/A]  — x 4–20 (grid 2), y 0–70 (grid 10), L sweep, dashed @ gm/Id=15")
code('''running("STEP 6b \\u2014 Ft vs gm/Id")
ncol = min(len(DEVICES), 2); nrow = int(np.ceil(len(DEVICES) / ncol))
fig, axarr = plt.subplots(nrow, ncol, figsize=(6.2*ncol, 4.6*nrow), squeeze=False)
axes = axarr.flat
for ax, (dev, *_ ) in zip(axes, DEVICES):
    gv = at_vds_half(df[df.device == dev])
    for L, gl in gv.groupby("L_um"):
        gl = gl[gl.Id > 0].sort_values("gm_id")
        ax.plot(gl["gm_id"].values, gl["ft_GHz"].values, label=f"L={L:g} µm")
    ax.axvline(15, color="k", ls="--", lw=1.2)
    ax.set(title=dev, xlabel="gm/Id [S/A]", ylabel="Ft [GHz]")
    ax.set_xlim(4, 20); ax.set_ylim(0, 70)
    ax.set_xticks(np.arange(4, 20.001, 2)); ax.set_yticks(np.arange(0, 70.001, 10))
    ax.grid(True, which="both", alpha=.35); ax.legend(fontsize=6, title="L (Lmin→2·Lmin)")
for k in range(len(DEVICES), nrow*ncol): axes[k].axis("off")
fig.suptitle("Ft vs gm/Id  (L swept Lmin→2·Lmin, 7 steps)  @ VDS=VDD/2  —  dashed: gm/Id = 15")
fig.tight_layout()
finish()
nL = {d[0]: df[df.device == d[0]].L_um.nunique() for d in DEVICES}
gate("STEP 6b \\u2014 Ft vs gm/Id",
     [check("7 L curves per device", all(v == 7 for v in nL.values()), str(nL)),
      check("Ft data > 0", bool((df.ft_GHz > 0).any())),
      check("gm/Id range covers the 15 line", bool(df.gm_id.between(4, 20).any()))])''')

md("### 6c. (bonus) intrinsic gain & output characteristics")
code('''running("STEP 6c \\u2014 bonus plots")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for dev, *_ in DEVICES:
    g0 = at_vds_half(df[df.device == dev]); g0 = g0[g0.L_um == g0.L_um.min()]
    g0 = g0[g0.Id > 0].sort_values("gm_id")
    axes[0].plot(g0["gm_id"].values, g0["gain"].values, label=f"{dev} L={g0.L_um.iloc[0]:g}µm")
axes[0].set(xlabel="gm/Id [S/A]", ylabel="intrinsic gain gm/gds", title="Intrinsic gain vs gm/Id (Lmin)")
axes[0].set_xlim(4, 20); axes[0].grid(True, which="both", alpha=.3); axes[0].legend(fontsize=8)
dev0 = DEVICES[0][0]
g = df[(df.device == dev0) & (df.L_um == df[df.device == dev0].L_um.min())]
for vg in np.round(np.arange(0.4, g.vdd.iloc[0] + 1e-9, 0.2), 3):
    gg = g[np.isclose(g["VGS"], vg, atol=DVGS/2)].sort_values("VDS")
    if not gg.empty: axes[1].plot(gg["VDS"].values, gg["Id"].values*1e3, label=f"VGS={vg:.1f}")
axes[1].set(xlabel="VDS [V]", ylabel="Id [mA]", title=f"Output char  {dev0}  L={g.L_um.iloc[0]:g}µm")
axes[1].grid(True, alpha=.3); axes[1].legend(fontsize=8, title="VGS")
fig.tight_layout(); finish()
gate("STEP 6c \\u2014 bonus plots", [check("rendered", True)])''')

md("## 7. Lookup-table export + deliverables listing")
code('''def export_lut(df, path):
    out = {}; params = ["Id","gm","gds","gmbs","cgs","cgd","cgg","cdd",
                         "vth","vdsat","Vov","gm_id","gain","ft_GHz"]
    for dev, g in df.groupby("device"):
        Ls = np.sort(g.L_um.unique()); VDSs = np.sort(g.VDS.unique()); VGSs = np.sort(g.VGS.unique())
        iL = {v:i for i,v in enumerate(Ls)}; iD = {v:i for i,v in enumerate(VDSs)}
        iG = {round(v,6):i for i,v in enumerate(VGSs)}
        cube = {p: np.full((len(Ls), len(VDSs), len(VGSs)), np.nan) for p in params}
        for _, r in g.iterrows():
            a, b, c = iL[r.L_um], iD[r.VDS], iG[round(r.VGS,6)]
            for p in params:
                if p in r and pd.notna(r[p]): cube[p][a,b,c] = r[p]
        out[f"{dev}__L"] = Ls; out[f"{dev}__VDS"] = VDSs; out[f"{dev}__VGS"] = VGSs
        for p in params: out[f"{dev}__{p}"] = cube[p]
    np.savez_compressed(path, **out); return path

running("STEP 7 \\u2014 LUT export")
LUT = export_lut(df, os.path.join(OUTDIR, f"{PROC}_gmid_lut.npz")); _z = np.load(LUT); finish()
oks = []
oks.append(check("npz written", os.path.exists(LUT), f"{os.path.getsize(LUT)/1e6:.2f} MB"))
oks.append(check("LUT reloads", len(_z.files) > 0, f"{len(_z.files)} arrays"))
oks.append(check("gm grid present for every device", all(f"{d[0]}__gm" in _z.files for d in DEVICES)))
gate("STEP 7 \\u2014 LUT export", oks)

# ---- final deliverables: full path + size, standard `ls -l` long format ----
csvparq = f"{PROC}_devchar.parquet" if os.path.exists(os.path.join(OUTDIR, f"{PROC}_devchar.parquet")) else f"{PROC}_devchar.csv"
cmd = f"ls -l --time-style=long-iso {OUTDIR}/{csvparq} {OUTDIR}/{PROC}_gmid_lut.npz"
print(f"\\nOutput data files (process prefix: {PROC}):")
print(f"$ {cmd}")
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.rstrip())''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_char.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
