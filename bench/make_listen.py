"""Generate a local blind A/B listening test page.

Usage: make_listen.py CAND_DIR OUT_DIR [--n 20] [--ref baseline/ref_fp32]

Copies N randomized pairs (teacher vs candidate, sides shuffled per pair) into
OUT_DIR and writes index.html: keyboard-driven blind AB with a preference tally.
Open OUT_DIR/index.html in a browser. Results shown on screen + downloadable CSV.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ref", default=str(ROOT / "baseline/ref_fp32"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ref_dir, cand_dir, out = Path(args.ref), Path(args.cand_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)

    items = sorted(f.stem for f in ref_dir.glob("*.wav") if (cand_dir / f.name).exists())
    # skip very long items for listenability
    import soundfile as sf

    items = [i for i in items if sf.info(ref_dir / f"{i}.wav").duration < 25]
    rng.shuffle(items)
    items = items[: args.n]

    manifest = []
    for k, stem in enumerate(items):
        flip = rng.random() < 0.5
        a_src = (cand_dir if flip else ref_dir) / f"{stem}.wav"
        b_src = (ref_dir if flip else cand_dir) / f"{stem}.wav"
        shutil.copy(a_src, out / "audio" / f"p{k:02d}_A.wav")
        shutil.copy(b_src, out / "audio" / f"p{k:02d}_B.wav")
        manifest.append({"pair": k, "item": stem, "A": "cand" if flip else "ref", "B": "ref" if flip else "cand"})
    (out / "key.json").write_text(json.dumps(manifest, indent=1))

    html = """<!doctype html><meta charset="utf-8"><title>Blind AB — Kokoro</title>
<style>body{font-family:system-ui;margin:40px auto;max-width:640px}button{font-size:18px;padding:10px 18px;margin:6px}
#tally{margin-top:24px;white-space:pre;font-family:monospace}</style>
<h2>Blind A/B — which sounds better?</h2>
<p>Keys: <b>1</b> play A · <b>2</b> play B · <b>A</b>/<b>B</b> prefer · <b>T</b> tie. One of each pair is the fp32 reference, sides randomized.</p>
<div id="prog"></div>
<audio id="pa"></audio><audio id="pb"></audio>
<div>
<button onclick="play('a')">▶ A (1)</button><button onclick="play('b')">▶ B (2)</button>
<button onclick="vote('A')">Prefer A</button><button onclick="vote('B')">Prefer B</button><button onclick="vote('T')">Tie (T)</button>
</div>
<div id="tally"></div>
<script>
let key=null, i=0, votes=[];
fetch('key.json').then(r=>r.json()).then(k=>{key=k;load();});
function load(){
  if(i>=key.length){finish();return;}
  document.getElementById('prog').textContent=`Pair ${i+1} / ${key.length}  (item ${key[i].item})`;
  document.getElementById('pa').src=`audio/p${String(i).padStart(2,'0')}_A.wav`;
  document.getElementById('pb').src=`audio/p${String(i).padStart(2,'0')}_B.wav`;
}
function play(w){const a=document.getElementById('pa'),b=document.getElementById('pb');a.pause();b.pause();(w=='a'?a:b).currentTime=0;(w=='a'?a:b).play();}
function vote(v){votes.push({pair:i,vote:v,winner:v=='T'?'tie':key[i][v]});i++;load();}
function finish(){
  const n=votes.length, cand=votes.filter(v=>v.winner=='cand').length, ref=votes.filter(v=>v.winner=='ref').length, tie=n-cand-ref;
  let t=`DONE  n=${n}\\nprefer candidate: ${cand}\\nprefer reference: ${ref}\\nties: ${tie}\\n\\n`;
  t+='pair,item,vote,winner\\n'+votes.map(v=>`${v.pair},${key[v.pair].item},${v.vote},${v.winner}`).join('\\n');
  document.getElementById('tally').textContent=t;
  const blob=new Blob([t],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ab_results.csv';a.textContent='download csv';document.body.appendChild(a);
}
window.addEventListener('keydown',e=>{
  if(e.key=='1')play('a'); else if(e.key=='2')play('b');
  else if(e.key=='a'||e.key=='A')vote('A'); else if(e.key=='b'||e.key=='B')vote('B');
  else if(e.key=='t'||e.key=='T')vote('T');
});
</script>"""
    (out / "index.html").write_text(html)
    print(f"wrote {out}/index.html with {len(manifest)} pairs")


if __name__ == "__main__":
    main()
