"""Independent SLE cohort: Nehar-Belaid et al. 2020 (GEO GSE135779, childhood and adult SLE and healthy
donors, 10x PBMC). Downloads the GEO supplementary RAW tar and the series matrix into data/raw/nb/."""
import os, re, subprocess, urllib.request
B = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135779/"
os.makedirs("data/raw/nb", exist_ok=True)
def get(url, dst):
    if not os.path.exists(dst):
        print("get", url, flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "8", "-o", dst, url], check=True)
idx = urllib.request.urlopen(B + "suppl/").read().decode()
files = sorted(set(re.findall(r'href="(GSE135779[^"/]+)"', idx)))
print("suppl files", files, flush=True)
for f in files:
    get(B + "suppl/" + f, "data/raw/nb/" + f)
midx = urllib.request.urlopen(B + "matrix/").read().decode()
for f in sorted(set(re.findall(r'href="(GSE135779[^"/]+)"', midx))):
    get(B + "matrix/" + f, "data/raw/nb/" + f)
for f in os.listdir("data/raw/nb"):
    if f.endswith(".tar"):
        os.makedirs("data/raw/nb/raw", exist_ok=True)
        subprocess.run(["tar", "-xf", "data/raw/nb/" + f, "-C", "data/raw/nb/raw"], check=True)
print("NB_DOWNLOAD_DONE", len(os.listdir("data/raw/nb/raw")) if os.path.isdir("data/raw/nb/raw") else 0, flush=True)
