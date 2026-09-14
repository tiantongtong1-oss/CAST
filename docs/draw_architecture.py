"""Render the manually audited CAST graph tied to architecture_manifest.json.

Run from the repository root: python docs/draw_architecture.py
Requires reportlab and PyMuPDF. No generated photographs or model results are used.
"""
from pathlib import Path
import hashlib
import json
import math

import fitz
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
manifest = json.loads((OUT / 'architecture_manifest.json').read_text())
for filename, expected in manifest['source_sha256'].items():
    actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('Re-audit diagram after source changes: ' + filename)

W, H = 2400, 1420
PDF = OUT / 'cast_mobilenetv2_framework.pdf'
font_dir = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('DiagramSans', str(font_dir / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont('DiagramBold', str(font_dir / 'VeraBd.ttf')))
c = canvas.Canvas(str(PDF), pagesize=(W, H))
c.setTitle('MobileNetV2 Dual-View EMA CAST - code-audited training architecture')
c.setAuthor('CAST')
P = '#411C70'
PURPLE = '#7745B1'
LAV = '#AA85D5'
LINE = '#C7B1E4'
BG = '#FBF9FE'
INK = '#29203A'
MUTED = '#746686'
LIGHT = '#F0E9FA'
PEACH = '#FFE5DD'
REGIONS = []
COLORS = ['#7F9AE8', '#9D87D4', '#F6AE83', '#FFD775', '#A4ACBA', '#EFC1B3', '#B5A5E4']


def box(x, y, w, h, fill='#FFFFFF', stroke=LINE, radius=18, dash=None, width=1.6):
    REGIONS.append((x, y, w, h))
    c.saveState()
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(width)
    if dash:
        c.setDash(dash)
    c.roundRect(x, H-y-h, w, h, radius, stroke=1, fill=1)
    c.restoreState()


def text(x, y, s, size=23, color=INK, bold=False, align='left'):
    font = 'DiagramBold' if bold else 'DiagramSans'
    containing = [r for r in REGIONS if r[0] <= x <= r[0]+r[2] and r[1] <= y <= r[1]+r[3]]
    region = min(containing, key=lambda r:r[2]*r[3]) if containing else (0, 0, W, H)
    left, right = region[0]+12, region[0]+region[2]-12
    available = {'left':right-x, 'center':2*min(x-left,right-x), 'right':x-left}[align]
    if available > 0:
        size = min(size, size * available / max(pdfmetrics.stringWidth(s, font, size), 1))
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    getattr(c, {'left':'drawString', 'center':'drawCentredString', 'right':'drawRightString'}[align])(x, H-y, s)


def lines(x, y, strings, size=23, gap=33, color=INK, bold=False, align='left'):
    for i, s in enumerate(strings):
        text(x, y+i*gap, s, size, color, bold, align)


def arrow(points, color=P, width=3, dash=None, head=12):
    c.saveState()
    c.setStrokeColor(HexColor(color))
    c.setFillColor(HexColor(color))
    c.setLineWidth(width)
    if dash:
        c.setDash(dash)
    p=c.beginPath()
    p.moveTo(points[0][0], H-points[0][1])
    for x,y in points[1:]:
        p.lineTo(x,H-y)
    c.drawPath(p)
    if head:
        x0,y0=points[-2]; x,y=points[-1]
        angle=math.atan2(y-y0,x-x0)
        p=c.beginPath(); p.moveTo(x,H-y)
        for delta in (.43,-.43):
            px=x-head*math.cos(angle+delta)
            py=y-head*math.sin(angle+delta)
            p.lineTo(px,H-py)
        p.close()
        c.setDash([])
        c.drawPath(p,stroke=0,fill=1)
    c.restoreState()


def panel(x,y,w,h,number,title,subtitle):
    box(x,y,w,h,BG,LINE,20,[7,5])
    box(x+3,y+3,w-6,76,P,P,17)
    text(x+24,y+36,number+'  '+title,30,'#FFFFFF',True)
    text(x+24,y+65,subtitle,20,'#E8DAFA')


def chip(x,y,w,label,fill=LIGHT):
    box(x,y,w,43,fill,LINE,10)
    text(x+w/2,y+29,label,21,P,True,'center')


def stack(x,y):
    # Stylized feature tensors; widths encode no measured quantity.
    for i,(hh,ww) in enumerate([(78,20),(69,27),(58,34),(47,41)]):
        xx=x+i*58
        yy=y+(78-hh)/2
        c.setFillColor(HexColor(['#B9B0EC','#A49ADF','#9287D3','#8073C6'][i]))
        c.setStrokeColor(HexColor('#E0D9F5'))
        c.rect(xx,H-yy-hh,ww,hh,stroke=1,fill=1)
        path=c.beginPath();path.moveTo(xx,H-yy);path.lineTo(xx+17,H-yy+17)
        path.lineTo(xx+ww+17,H-yy+17);path.lineTo(xx+ww,H-yy);path.close()
        c.setFillColor(HexColor('#D4CDF2'));c.drawPath(path,stroke=1,fill=1)
        path=c.beginPath();path.moveTo(xx+ww,H-yy);path.lineTo(xx+ww+17,H-yy+17)
        path.lineTo(xx+ww+17,H-yy-hh+17);path.lineTo(xx+ww,H-yy-hh);path.close()
        c.setFillColor(HexColor('#8B7DC2'));c.drawPath(path,stroke=1,fill=1)


c.setFillColor(white);c.rect(0,0,W,H,stroke=0,fill=1)
text(40,65,'MobileNetV2 + Dual-View EMA CAST',45,P,True)
text(40,106,'RAF-DB to FER2013  |  Target adaptation after source pre-training  |  Re-read from GitHub commit '+manifest['code_commit'][:7],23,MUTED)
text(2360,67,'CODE-AUDITED FRAMEWORK',20,PURPLE,True,'right')

# Teacher is an independent branch, never a sequential layer after the student.
panel(40,145,2320,405,'2.','Dual-View EMA Teacher','Independent copy of the entire student  |  eval mode  |  no gradient')
box(65,248,325,247)
text(227,281,'Input views',24,P,True,'center')
chip(80,300,140,'Weak 1')
chip(235,300,140,'Weak 2')
text(227,375,'FER2013 target (unlabeled)',20,MUTED,align='center')
chip(80,408,295,'RAF-DB source weak')
text(227,478,'Source labels anchor memory',19,MUTED,align='center')

box(435,248,390,247,LIGHT)
text(630,285,'EMA MobileNetV2 + head',26,P,True,'center')
stack(505,323)
text(630,449,'512-D features  +  7 logits',23,P,True,'center')
text(630,478,'Dropout / BN in eval mode',20,MUTED,align='center')
arrow([(390,327),(435,327)])
arrow([(390,430),(435,430)])

box(870,248,575,247)
text(894,284,'Consistency & confidence',27,P,True)
lines(896,324,[
    'Temperature softmax on each weak view',
    'Same per-view prior correction in both paths',
    'Global class thresholds on full target train',
    'Agreement + mean / per-view confidence',
    'Fallback: BOTH views must be confident',
],22,33)
arrow([(825,360),(870,360)])

box(1490,248,350,247,LIGHT)
text(1665,284,'Reliable pseudo labels',26,P,True,'center')
for i,color in enumerate(COLORS):
    box(1519+i*43,320,31,59,color,color,4)
    text(1534+i*43,403,str(i),17,MUTED,align='center')
text(1665,442,'labels + mask + bounded weights',20,P,align='center')
text(1665,476,'Empty selection stays empty',20,MUTED,align='center')
arrow([(1445,360),(1490,360)])

box(1895,248,435,247)
text(2112,285,'Detached prototype inputs',26,P,True,'center')
lines(1920,332,[
    'Source teacher features + true labels',
    'Mean normalized target weak features',
    'Target updates use reliable samples',
    'EMA prototype memory: 7 x 512',
],22,37)
arrow([(1840,380),(1895,380)],color=PURPLE,dash=[4,5])
# Feature route from teacher to memory inputs; no student gradient on this route.
arrow([(630,495),(630,525),(2110,525),(2110,495)],color=LAV,width=2,dash=[3,5])
text(1050,518,'detached teacher features',18,MUTED)

# Supervision and memory routes above the three trainable/constraint panels.
arrow([(1665,495),(1665,580),(1080,580),(1080,665)],color=PURPLE,dash=[4,5])
arrow([(1665,580),(1665,665)],color=PURPLE,dash=[4,5])
text(1130,572,'Reliable pseudo-label supervision',21,PURPLE,True)
arrow([(2300,495),(2300,665)],color=LAV,dash=[3,5])
text(2280,582,'memory',19,MUTED,align='right')

panel(40,675,705,455,'1.','Shared Student Backbone','Source weak view + target strong view share parameters')
panel(815,675,610,455,'3.','DDRL Module','Loss constraints on the same 512-D student features')
panel(1495,675,865,455,'4.','CCDR Module','Classifier regularization + stable target affinity')

# Student, including its source and target minibatch inputs.
chip(65,786,235,'RAF-DB: source')
text(182,861,'weak view + true label',20,MUTED,align='center')
chip(65,900,235,'FER2013: target',PEACH)
text(182,975,'strong view + pseudo label',19,MUTED,align='center')
box(375,778,335,198,LIGHT)
text(542,815,'MobileNetV2.features',27,P,True,'center')
stack(421,858)
arrow([(300,808),(335,808),(335,875),(375,875)])
arrow([(300,922),(335,922),(335,900),(375,900)])
box(375,1000,335,99)
text(542,1036,'GAP -> 1280 -> Linear -> 512',23,P,True,'center')
text(542,1070,'Dropout before / after projection',18,MUTED,align='center')
arrow([(542,976),(542,1000)])
text(65,1049,'Source replay is cycled',20,MUTED)
text(65,1080,'throughout adaptation',20,MUTED)

# Shared feature bus branches into BOTH DDRL and CCDR, not a sequential layer.
arrow([(710,1049),(780,1049),(780,625),(1990,625)],head=0)
arrow([(1325,625),(1325,665)])
arrow([(1990,625),(1990,665)])
text(810,615,'Student features: f_s, f_t  (512-D)',23,P,True)
# Hollow crossing makes the label-supervision and feature wires distinct.
c.setFillColor(white);c.circle(1080,H-625,6,stroke=0,fill=1)
arrow([(1068,625),(1092,625)],head=0)
c.setFillColor(white);c.circle(1665,H-625,6,stroke=0,fill=1)
arrow([(1653,625),(1677,625)],head=0)

# DDRL only emits loss constraints; it does not modify the forward tensor.
box(840,780,560,137)
text(865,819,'Domain alignment',28,P,True)
text(865,858,'Class-conditional MK-MMD',25,INK)
text(865,893,'L_align: source vs reliable target, by class',20,MUTED)
box(840,943,560,137,LIGHT)
text(865,982,'Dual class enhancement',28,P,True)
text(865,1021,'Class vs other classes: negative MK-MMD',22,INK)
text(865,1056,'L_sep: strengthen class structure',21,MUTED)
text(1120,1110,'Bounded density weights  |  Valid-class normalization',20,MUTED,align='center')

# CCDR implements a classifier and memory-based affinity on features.
box(1520,780,340,132)
text(1690,819,'Classifier',28,P,True,'center')
text(1690,856,'Linear(512, 7) + BN(7)',23,INK,align='center')
text(1690,889,'bias=False; shared across domains',18,MUTED,align='center')
box(1900,780,435,132,LIGHT)
text(2117,819,'EMA class prototypes',27,P,True,'center')
for i,color in enumerate(COLORS):
    box(1967+i*44,837,32,32,color,color,4)
text(2117,896,'Source anchors + reliable target updates',20,MUTED,align='center')

box(1520,947,340,137,LIGHT)
text(1690,987,'Classification loss',25,P,True,'center')
text(1690,1021,'L_cls = L_s + lambda_t * L_t',22,INK,align='center')
text(1690,1061,'Weighted target CE; separate means',19,MUTED,align='center')
arrow([(1690,912),(1690,947)])
box(1900,947,435,137)
text(2117,987,'Differentiable target affinity',25,P,True,'center')
text(2117,1021,'Pull to own class + negative-class margin',21,INK,align='center')
text(2117,1061,'Reliable only; warm-up + capped ramp',21,MUTED,align='center')
arrow([(2117,912),(2117,947)])
text(1690,1111,'L_mod: classifier weight modulation',19,MUTED,align='center')
text(2117,1111,'Missing prototypes skipped; student receives gradient',18,MUTED,align='center')

# Parameter EMA route; update is not part of the gradient graph.
arrow([(375,675),(375,593),(450,593),(450,505)],color=LAV,width=2.5,dash=[3,5])
text(60,588,'Student -> teacher EMA',23,PURPLE,True)
text(60,620,'after each optimizer step',20,MUTED)

# True total objective includes the original classifier modulation term.
box(375,1220,1985,106,LIGHT,LINE,18)
text(1367,1254,'TOTAL TARGET-STAGE TRAINING OBJECTIVE',23,P,True,'center')
text(1367,1301,'L = w1 (L_s + lambda_t L_t) + w2 (L_align + L_sep) + lambda_aff(epoch) L_aff + w3 L_mod',30,P,True,'center')
arrow([(1120,1130),(1120,1220)])
arrow([(1925,1130),(1925,1220)])
text(1140,1187,'DDRL loss',19,MUTED)
text(1945,1187,'Classification + affinity + modulation',19,MUTED)
arrow([(375,1272),(18,1272),(18,1030),(40,1030)],color=PURPLE,width=3,dash=[9,5])
text(47,1206,'Backpropagate to student',23,PURPLE,True)

arrow([(45,1370),(115,1370)])
text(130,1378,'Forward / loss input',20,INK)
arrow([(405,1370),(475,1370)],color=PURPLE,dash=[4,5])
text(490,1378,'Pseudo-label supervision',20,INK)
arrow([(805,1370),(875,1370)],color=LAV,dash=[3,5])
text(890,1378,'EMA / memory (no gradient)',20,INK)
arrow([(1275,1370),(1345,1370)],color=PURPLE,dash=[9,5])
text(1360,1378,'Gradient update',20,INK)
text(2360,1378,'Inference: student backbone + classifier only',21,P,True,'right')
c.showPage();c.save()

with fitz.open(PDF) as document:
    page = document[0]
    (OUT / 'cast_mobilenetv2_framework.svg').write_text(page.get_svg_image(text_as_path=True))
    (OUT / 'cast_mobilenetv2_framework.png').write_bytes(
        page.get_pixmap(matrix=fitz.Matrix(1.25,1.25), alpha=False).tobytes('png'))
print(json.dumps({'pdf':str(PDF), 'png_size':[3000,1775], 'source_commit':manifest['code_commit']}))
