"""Build docs/report/index.html: copy media/plots from outputs/, inline the data JSON.

Run `make report` (renders the policy videos, exports the data, then runs this).
"""

import datetime
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
O = REPO / "outputs"

media = {
    "m1_offset_0mm.mp4": "m1/offset_0mm.mp4",
    "m1_offset_4mm.mp4": "m1/offset_4mm.mp4",
    "m4_press_slide.mp4": "m4/press_slide_comp.mp4",
    "m6_nut.mp4": "m6/nut_seed0.mp4",
    "m6_pih_kp150_4mm.mp4": "m6_pih/kp150_offset_4mm.mp4",
}
img = {
    "m2_sine_2_comp.png": "m2/sine_2_comp.png",
    "m2_tilt_raw.png": "m2/tilt_raw.png",
    "m3_timestep_x_solref.png": "m3/heat_peak_F_hf_N__timestep_x_solref_tc.png",
    "m3_kp_x_speed.png": "m3/heat_peak_F_hf_N__kp_x_speed_mmps.png",
    "m5_variable_kp.png": "m5/variable_kp.png",
    "m5_stiffness_tradeoff.png": "m5/stiffness_tradeoff.png",
}
data = json.loads((O / "report/report_data.json").read_text())
seeds = sorted({r["seed"] for r in data["videos"]["runs"]})
for s in seeds:
    media[f"seed{s}_compare.mp4"] = f"s1/videos/seed{s}_compare.mp4"
data["traces"] = {
    str(s): json.loads((O / f"s1/videos/seed{s}_traces.json").read_text()) for s in seeds
}

files = {}
for sub, table in (("media", media), ("img", img)):
    (HERE / sub).mkdir(exist_ok=True)
    for dst, src in table.items():
        shutil.copyfile(O / src, HERE / sub / dst)
        files[f"{sub}/{dst}"] = f"{sub}/{dst}"

sha = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()
html = (HERE / "template.html").read_text()
html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
html = html.replace("__DATE__", datetime.date.today().isoformat()).replace("__SHA__", sha)
(HERE / "index.html").write_text(html)
size = sum((HERE / p).stat().st_size for p in files.values()) / 1e6
print(f"index.html {len(html) / 1e3:.0f} kB, {len(files)} media files, {size:.1f} MB")
