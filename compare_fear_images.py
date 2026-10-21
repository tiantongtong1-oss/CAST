"""Create reproducible RAF/FER Fear contact sheets without model inference."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def test_view(image):
    # Matches PIL Resize((256,256)) + CenterCrop(224) in train.build_transforms.
    return image.resize((256, 256), Image.Resampling.BILINEAR).crop((16,16,240,240))


def image_stats(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    gray = .299*rgb[:,:,0] + .587*rgb[:,:,1] + .114*rgb[:,:,2]
    channel_gap = rgb.max(2)-rgb.min(2)
    return dict(width=image.width, height=image.height, original_mode=image.mode,
        brightness=float(gray.mean()), contrast_std=float(gray.std()),
        mean_channel_gap=float(channel_gap.mean()),
        near_gray_pixel_fraction=float((channel_gap <= 2).mean()))


def read_groups(source_path, prediction_csv):
    source = Path(source_path)
    raf = []
    with (source/'EmoLabel/list_patition_label.txt').open() as f:
        for line in f:
            parts = line.split()
            if len(parts)==2 and parts[0].startswith('train') and int(parts[1])==2:
                path = source/'Image/aligned'/(Path(parts[0]).stem+'_aligned.jpg')
                raf.append(dict(path=str(path), true_class='Fear', predicted_class='',
                                fear_probability='', fear_rank=''))
    with Path(prediction_csv).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'path','true_class','predicted_class','fear_probability','fear_rank'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('CSV must be from diagnose_fear.py and contain '+str(sorted(required)))
        fer = [r for r in reader if r['true_class']=='Fear']
    if not raf or not fer:
        raise ValueError('No RAF training Fear or FER Fear rows found')
    return dict(raf_train_fear=raf,
        fer_correct_fear=[r for r in fer if r['predicted_class']=='Fear'],
        fer_fear_as_sad_neutral=[r for r in fer if r['predicted_class'] in ('Sad','Neutral')],
        fer_fear_as_surprise=[r for r in fer if r['predicted_class']=='Surprise'])


def load_image(path):
    # Same decoding as dataset.py; original mode is inspected separately.
    import cv2
    with Image.open(path) as original:
        mode = original.mode
    array = cv2.imread(str(path))
    if array is None:
        raise ValueError('Cannot decode '+str(path))
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)), mode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_path', required=True)
    p.add_argument('--prediction_csv', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--samples_per_group', type=int, default=20)
    p.add_argument('--seed', type=int, default=1314)
    a = p.parse_args()
    if a.samples_per_group < 1:
        p.error('samples_per_group must be positive')
    groups = read_groups(a.source_path, a.prediction_csv)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    stats_rows, manifests, errors, summaries = [], [], [], {}
    for gi, (name, rows) in enumerate(groups.items()):
        rows = sorted(rows, key=lambda r:r['path'])
        picks = random.Random(a.seed+gi).sample(rows, min(len(rows), a.samples_per_group))
        # Statistics cover the FULL group, not only displayed examples.
        valid_images = {}
        for row in rows:
            try:
                image, mode = load_image(row['path'])
                for stage, view in [('original',image), ('test_input',test_view(image))]:
                    stats = image_stats(view)
                    if stage=='original':
                        stats['original_mode'] = mode
                    stats_rows.append(dict(group=name, path=row['path'], stage=stage, **stats))
                if row in picks:
                    valid_images[row['path']] = image
            except Exception as e:
                errors.append(dict(group=name,path=row['path'],error=str(e)))
        # Five examples per page, two columns each: original aspect ratio / actual test crop.
        for page_start in range(0, len(picks), 5):
            page = picks[page_start:page_start+5]
            canvas = Image.new('RGB',(720,90+len(page)*290),'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((12,10),name,fill='black')
            draw.text((12,30),'LEFT: original (aspect preserved); RIGHT: 256 resize + 224 center crop',fill='black')
            draw.text((12,48),'Display enlargement only. Normalization omitted for RGB visualization.',fill='black')
            for j,row in enumerate(page):
                y = 80+j*290
                manifests.append(dict(group=name, page=page_start//5+1, **{k:row.get(k,'') for k in ['path','true_class','predicted_class','fear_probability','fear_rank']}))
                if row['path'] not in valid_images:
                    draw.text((10,y),'IMAGE READ FAILED: '+Path(row['path']).name, fill='red'); continue
                image = valid_images[row['path']]
                left = ImageOps.contain(image,(224,224),Image.Resampling.BILINEAR)
                canvas.paste(left,(12+(224-left.width)//2,y+(224-left.height)//2))
                canvas.paste(test_view(image),(372,y))
                draw.text((12,y+228),Path(row['path']).name[:52],fill='black')
                draw.text((12,y+244),'Original: %dx%d'%(image.width,image.height),fill='black')
                detail = 'Source training Fear' if not row['predicted_class'] else 'Pred=%s  P(Fear)=%.4f  rank=%s'%(row['predicted_class'],float(row['fear_probability']),row['fear_rank'])
                draw.text((12,y+260),detail,fill='black')
            canvas.save(out/('%s_page%02d.png'%(name,page_start//5+1)))
        summaries[name] = dict(total=len(rows), selected=len(picks), stages={})
        for stage in ['original','test_input']:
            subset=[r for r in stats_rows if r['group']==name and r['stage']==stage]
            summaries[name]['stages'][stage] = dict(readable=len(subset), metrics={
                key:dict(zip(['p05','p50','p95'],np.quantile([r[key] for r in subset],[.05,.5,.95]).tolist()))
                for key in ['width','height','brightness','contrast_std','mean_channel_gap','near_gray_pixel_fraction']}) if subset else dict(readable=0)
    for filename,rows in [('image_statistics.csv',stats_rows),('selected_images.csv',manifests)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            if rows:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result=dict(seed=a.seed,prediction_csv=str(Path(a.prediction_csv).resolve()),groups=summaries,errors=errors,
        note='Descriptive comparison only; no training, prediction or automatic label correction. Image statistics do not establish causality.')
    (out/'comparison_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    for name,summary in summaries.items():
        print(name, 'total=',summary['total'],'selected=',summary['selected'])
    print('Unreadable images:',len(errors))
    print('Output:',out.resolve())


if __name__=='__main__':
    main()
