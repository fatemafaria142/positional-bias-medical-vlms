import pandas as pd, numpy as np, nibabel as nib, json, sys
POS={'left','right'}
def roles(o):
    p=o.split(); return (p[0].lower(),' '.join(p[1:])) if p[0].lower() in POS else (None,o)
def build_prompt(organ,present,landmark=None):
    pos,core=roles(organ); w=['Segment','the']
    if present.get('position',True): w.append(pos)
    if present.get('organ',True): w.append(core)
    elif landmark: w+=['structure','near','the',roles(landmark)[1]]
    else: w.append('structure')
    w+=['in','this','image']; s=' '.join(w); return s[0].upper()+s[1:]+'.'
def landmarks_for(person,sid):
    m=pd.read_csv(f'{person}/samples/neurite_oasis/manifest_neurite_oasis.csv'); m=m[(m.subject_id==sid)&(m.rotation_angle_deg==0)]
    lab=np.asarray(nib.load(f'{person}/samples/neurite_oasis/'+m.label_path.iloc[0]).dataobj)
    cents={}
    for r in m.itertuples():
        idx=np.nonzero(lab==int(r.label_id))
        if len(idx[0]): cents[r.organ]=np.array([c.mean() for c in idx])
    return m,cents
def nearest(organ,cents):
    tc=roles(organ)[1]; cand={n:c for n,c in cents.items() if roles(n)[1]!=tc}
    d={n:float(np.linalg.norm(cents[organ]-c)) for n,c in cand.items()}; return min(d,key=d.get)
def prompts(person,sid,cores):
    m,cents=landmarks_for(person,sid); out=[]
    for r in m.itertuples():
        pos,core=roles(r.organ)
        if pos is None or core not in cores: continue
        lm=nearest(r.organ,cents)
        for kind,pres in [('both_present',{}),('position_removed',{'position':False}),('organ_removed',{'organ':False}),('both_removed',{'position':False,'organ':False})]:
            out.append(dict(dataset='neurite_oasis',person=person,subject_id=sid,organ=r.organ,label_id=int(r.label_id),
              condition=kind,landmark_organ=lm if not pres.get('organ',True) else '',prompt_text=build_prompt(r.organ,{'position':pres.get('position',True),'organ':pres.get('organ',True)},lm)))
    return out
SUBJECTS_BATCH3 = [("hridoy", "OASIS_OAS1_0123_MR1"), ("faria", "OASIS_OAS1_0057_MR1"), ("nipu", "OASIS_OAS1_0151_MR1")]
CORES_BATCH3 = {"cerebellum white matter", "thalamus", "putamen", "caudate",
                "hippocampus", "ventral DC", "amygdala", "pallidum"}


def main():
    """Writes voxtell-oasis-batch3-prompts.csv: the 4 distinct prompts (both present / position removed /
    organ removed / both removed) for every subject x organ in OASIS batch 3. The landmark in the
    organ-removed prompts is picked exactly like the notebook does (nearest other organ by GT centroid,
    excluding the target's own left/right pair). The same logic reproduced the prompts in the existing
    batch-1 results CSV 80/80 times when checked."""
    rows = []
    for person, sid in SUBJECTS_BATCH3:
        rows += prompts(person, sid, CORES_BATCH3)
    d = pd.DataFrame(rows)
    d.insert(0, "prompt_id", range(1, len(d) + 1))
    d.to_csv("voxtell-oasis-batch3-prompts.csv", index=False)
    print(len(d), "rows written")


if __name__ == "__main__":
    main()
