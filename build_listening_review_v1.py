"""Build a local, non-technical listening review for learned neighbours."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from representation_learning_v1 import TARGET_SR, load_sound_objects, read_fragment


def safe_clip_name(stable_id: str) -> str:
    return "".join(char for char in stable_id if char.isalnum() or char in "_-") + ".wav"


def build_listening_review(
    index_path: str | Path,
    memory_dir: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    *,
    trials: int = 12,
    title: str = "Hyponoia Listening Review v1",
    review_id: str = "hyponoia-listening-review-v1",
) -> dict:
    output_dir = Path(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    objects = load_sound_objects(index_path, memory_dir)
    object_by_id = {}
    for item in objects:
        object_by_id.setdefault(item.stable_id, item)

    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    examples = evaluation["example_neighbors"][:trials]
    review_trials = []
    exported = set()

    def export(stable_id: str) -> str:
        if stable_id not in object_by_id:
            raise ValueError(f"Missing audio object for {stable_id}")
        filename = safe_clip_name(stable_id)
        if stable_id not in exported:
            audio = read_fragment(object_by_id[stable_id])
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 0.99:
                audio = audio * (0.99 / peak)
            sf.write(clips_dir / filename, audio, TARGET_SR, subtype="PCM_16")
            exported.add(stable_id)
        return f"clips/{filename}"

    for number, example in enumerate(examples, start=1):
        anchor = example["anchor"]
        review_trials.append({
            "number": number,
            "anchor": {
                "id": anchor["stable_id"],
                "audio": export(anchor["stable_id"]),
                "recording": anchor.get("recording", ""),
                "category": anchor.get("review_category", ""),
            },
            "neighbors": [
                {
                    "id": neighbor["stable_id"],
                    "audio": export(neighbor["stable_id"]),
                    "recording": neighbor.get("recording", ""),
                    "similarity": round(float(neighbor["cosine_similarity"]), 4),
                }
                for neighbor in example["neighbors"]
            ],
        })

    data = json.dumps(review_trials, ensure_ascii=False).replace("</", "<\\/")
    escaped_title = html_lib.escape(title)
    storage_key = json.dumps(review_id)
    html = f"""<!doctype html>
<html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped_title}</title>
<style>
:root{{--bg:#f5f4ef;--panel:#fff;--text:#24231f;--muted:#69675f;--line:#d9d6cc;--accent:#4d5d53;--soft:#e8ece9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 system-ui,sans-serif}}
main{{max-width:960px;margin:auto;padding:28px 18px 60px}} h1{{font-size:1.65rem;margin:0 0 8px}} h2{{font-size:1.15rem;margin:0}}
.intro,.trial{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:0 0 16px}}
.muted{{color:var(--muted)}} .anchor{{background:var(--soft);border-radius:9px;padding:13px;margin:12px 0}}
.neighbor{{border-top:1px solid var(--line);padding:13px 0}} .row{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
audio{{width:min(390px,100%);height:36px}} .choices{{display:flex;gap:7px;flex-wrap:wrap;margin-left:auto}}
button{{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:7px;padding:8px 11px;cursor:pointer}}
button.selected{{background:var(--accent);color:#fff;border-color:var(--accent)}} .toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.progress{{font-weight:600}} @media(max-width:640px){{.choices{{margin-left:0;width:100%}} audio{{width:100%}}}}
</style></head><body><main>
<section class="intro"><h1>{escaped_title}</h1>
<p>Άκου πρώτα τον <strong>αρχικό ήχο</strong> και μετά τους πέντε ήχους που το μοντέλο θεωρεί κοντινούς. Για κάθε ζεύγος διάλεξε μία απάντηση.</p>
<div class="toolbar"><span class="progress" id="progress"></span><button id="download">Κατέβασε τις απαντήσεις</button><button id="reset">Καθαρισμός</button></div>
</section><div id="trials"></div>
<script>const trials={data}; const key={storage_key}; let answers=JSON.parse(localStorage.getItem(key)||'{{}}');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function save(){{localStorage.setItem(key,JSON.stringify(answers));renderProgress()}}
function renderProgress(){{const total=trials.reduce((n,t)=>n+t.neighbors.length,0);document.getElementById('progress').textContent=`Απαντήσεις: ${{Object.keys(answers).length}} / ${{total}}`}}
function choose(pair,value){{answers[pair]=value;save();document.querySelectorAll(`[data-pair="${{pair}}"]`).forEach(b=>b.classList.toggle('selected',b.dataset.value===value))}}
const root=document.getElementById('trials');
trials.forEach(t=>{{const section=document.createElement('section');section.className='trial';
section.innerHTML=`<h2>Ομάδα ${{t.number}}</h2><div class="anchor"><strong>Αρχικός ήχος</strong>${{t.anchor.category?`<div class="muted">Κατηγορία: ${{esc(t.anchor.category)}}</div>`:''}}<div class="row"><audio controls preload="none" src="${{esc(t.anchor.audio)}}"></audio><span class="muted">${{esc(t.anchor.recording)}}</span></div></div>`;
t.neighbors.forEach((n,i)=>{{const pair=t.anchor.id+'__'+n.id;const div=document.createElement('div');div.className='neighbor';div.innerHTML=`<div class="row"><audio controls preload="none" src="${{esc(n.audio)}}"></audio><span class="muted">Γείτονας ${{i+1}}</span><div class="choices"><button data-pair="${{esc(pair)}}" data-value="related">Συγγενικός</button><button data-pair="${{esc(pair)}}" data-value="unsure">Αβέβαιο</button><button data-pair="${{esc(pair)}}" data-value="different">Διαφορετικός</button></div></div>`;section.appendChild(div)}});root.appendChild(section)}});
document.querySelectorAll('[data-pair]').forEach(b=>{{b.addEventListener('click',()=>choose(b.dataset.pair,b.dataset.value));if(answers[b.dataset.pair]===b.dataset.value)b.classList.add('selected')}});
document.getElementById('download').addEventListener('click',()=>{{const blob=new Blob([JSON.stringify({{version:1,created_at:new Date().toISOString(),answers}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='hyponoia_listening_feedback.json';a.click();URL.revokeObjectURL(a.href)}});
document.getElementById('reset').addEventListener('click',()=>{{if(confirm('Να καθαριστούν όλες οι απαντήσεις;')){{answers={{}};save();document.querySelectorAll('.selected').forEach(b=>b.classList.remove('selected'))}}}});renderProgress();</script>
</main></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    manifest = {
        "title": title,
        "review_id": review_id,
        "trials": len(review_trials),
        "comparisons": sum(len(item["neighbors"]) for item in review_trials),
        "unique_clips": len(exported),
        "sample_rate": TARGET_SR,
        "entrypoint": "index.html",
    }
    (output_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--title", default="Hyponoia Listening Review v1")
    parser.add_argument("--review-id", default="hyponoia-listening-review-v1")
    args = parser.parse_args()
    print(json.dumps(build_listening_review(
        args.index, args.memory_dir, args.evaluation, args.output_dir,
        trials=args.trials, title=args.title, review_id=args.review_id,
    ), indent=2))


if __name__ == "__main__":
    main()
