"""Label-based validation sample selection and visual review only, never training."""
import numpy as np
from PIL import Image,ImageDraw
from topk_diagnostics import NAMES


def select_review_groups(probabilities,truth,count=20,min_confidence=.8):
    p=np.asarray(probabilities);truth=np.asarray(truth);pred=p.argmax(1)
    failed=np.flatnonzero((truth==1)&np.isin(pred,[4,6])&(p.max(1)>=min_confidence))
    correct=np.flatnonzero((truth==1)&(pred==1))
    failed=failed[np.argsort(-p[failed].max(1),kind='stable')][:count]
    correct=correct[np.argsort(-p[correct,1],kind='stable')][:count]
    return dict(failed_fear=failed,correct_fear=correct)


def nearest_source(features,queries,k=3):
    features=np.asarray(features,dtype=np.float32);queries=np.asarray(queries,dtype=np.float32)
    if not np.isfinite(features).all() or not np.isfinite(queries).all():raise ValueError('nonfinite features')
    if (np.linalg.norm(features,axis=1)<=1e-8).any() or (np.linalg.norm(queries,axis=1)<=1e-8).any():raise ValueError('zero features')
    x=features/np.linalg.norm(features,axis=1,keepdims=True)
    z=queries/np.linalg.norm(queries,axis=1,keepdims=True)
    similarity=z@x.T
    idx=np.argsort(-similarity,axis=1,kind='stable')[:,:min(k,len(x))]
    return idx,np.take_along_axis(similarity,idx,axis=1)


def model_crop_image(tensor_array):
    # Invert only normalization to display the ACTUAL model input crop.
    x=np.asarray(tensor_array).transpose(1,2,0)
    x=x*np.array([.229,.224,.225])+np.array([.485,.456,.406])
    return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8'))


def render_pages(rows,outdir,prefix,per_page=5):
    from pathlib import Path
    outdir=Path(outdir);outputs=[]
    # Each row has raw query + exact crop + up to three nearest source crops.
    for start in range(0,len(rows),per_page):
        page=rows[start:start+per_page]
        canvas=Image.new('RGB',(1120,len(page)*225+35),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),prefix+' | Query raw / actual model crop / nearest SOURCE references',fill='black')
        for j,row in enumerate(page):
            y=35+j*225
            draw.text((8,y),row['caption'],fill='black')
            with Image.open(row['path']) as im:
                raw=im.convert('RGB');raw.thumbnail((165,165))
            canvas.paste(raw,(8,y+45))
            canvas.paste(row['crop'].resize((165,165)),(185,y+45))
            for n,neighbor in enumerate(row.get('neighbors',[])[:3]):
                x=370+n*245
                draw.text((x,y+25),neighbor['caption'],fill='black')
                canvas.paste(neighbor['crop'].resize((165,165)),(x,y+45))
        filename=outdir/('%s_%02d.png'%(prefix,start//per_page+1))
        canvas.save(filename);outputs.append(str(filename))
    return outputs
