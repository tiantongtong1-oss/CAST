"""Small deterministic image sample audit, without using labels."""
from pathlib import Path
import numpy as np


def audit_images(paths,limit=32):
    from PIL import Image
    records=[]
    for path in list(paths)[:limit]:
        with Image.open(path) as image:
            rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
            spread=rgb.max(axis=2)-rgb.min(axis=2)
            records.append(dict(path=str(path),width=image.width,height=image.height,
                stored_mode=image.mode,mean_channel_spread=float(spread.mean()),
                fraction_pixels_channel_spread_gt2=float((spread>2).mean())))
    return dict(sample_count=len(records),selection='first N loader paths; not a random representative sample',
                records=records)


def save_preview(reference_paths,query_paths,destination,count=6):
    from PIL import Image,ImageDraw
    rows=[]
    for domain,paths in [('RAF reference',reference_paths),('query',query_paths)]:
        for i,path in enumerate(list(paths)[:count]):rows.append((domain,i,path))
    canvas=Image.new('RGB',(640, max(len(rows),1)*180),'white');draw=ImageDraw.Draw(canvas)
    for row,(domain,i,path) in enumerate(rows):
        with Image.open(path) as im:
            rgb=im.convert('RGB');size=rgb.size
            raw=rgb.copy();raw.thumbnail((150,150))
            crop=rgb.resize((256,256)).crop((16,16,240,240)).resize((150,150))
            gray=crop.convert('L').convert('RGB')
        y=row*180
        canvas.paste(raw,(0,y+25));canvas.paste(crop,(160,y+25));canvas.paste(gray,(320,y+25))
        draw.text((0,y+5),f'{domain} #{i} {size[0]}x{size[1]} | raw / center-crop / gray',fill='black')
    canvas.save(destination)
