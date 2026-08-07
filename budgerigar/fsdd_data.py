from __future__ import annotations
import json,re,subprocess,wave
from pathlib import Path

FSDD_REPOSITORY="https://github.com/Jakobovski/free-spoken-digit-dataset.git"
_NAME=re.compile(r"^(?P<label>[0-9])_(?P<speaker>[^_]+)_(?P<index>[0-9]+)\.wav$")

def prepare_fsdd(root,revision="master"):
    """Clone/version FSDD in Colab and return the resolved repository commit."""
    root=Path(root)
    if not (root/".git").is_dir():
        root.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["git","clone","--depth=1",FSDD_REPOSITORY,str(root)],check=True)
    subprocess.run(["git","-C",str(root),"fetch","--tags","--depth=1","origin",revision],check=True)
    subprocess.run(["git","-C",str(root),"checkout","--detach","FETCH_HEAD"],check=True)
    return subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()

def build_fsdd_manifest(root):
    root=Path(root);rows=[]
    for path in sorted((root/"recordings").glob("*.wav")):
        match=_NAME.match(path.name)
        if not match:continue
        label=int(match.group("label"));speaker=match.group("speaker");index=int(match.group("index"))
        split="test" if index<=4 else "validation" if index<=9 else "train"
        with wave.open(str(path),"rb") as handle:
            rate=handle.getframerate();duration=handle.getnframes()/rate;channels=handle.getnchannels();width=handle.getsampwidth()
        rows.append({"id":path.stem,"audio":str(path.resolve()),"label":label,"text":str(label),"speaker":speaker,"recording_index":index,"sample_rate":rate,"duration":duration,"channels":channels,"sample_width":width,"split":split,"dataset":"fsdd"})
    if not rows:raise ValueError(f"no FSDD recordings found below {root}")
    return rows

def write_fsdd_manifest(rows,destination,revision):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in rows)+"\n",encoding="utf-8")
    counts={name:sum(row["split"]==name for row in rows) for name in ("train","validation","test")}
    return {"manifest":str(destination),"records":len(rows),"splits":counts,"revision":revision,"speakers":sorted({row["speaker"] for row in rows})}
