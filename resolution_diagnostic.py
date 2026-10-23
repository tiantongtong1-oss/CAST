"""Deterministic PIL low-resolution bottleneck before normal preprocessing."""
class LowResolutionBottleneck:
    def __init__(self,size=0):
        if isinstance(size,bool) or int(size)!=size or size<0:
            raise ValueError('low-resolution size must be a nonnegative integer')
        self.size=int(size)

    def __call__(self,image):
        if not self.size:return image
        from PIL import Image
        resampling=getattr(Image,'Resampling',Image)
        return image.resize((self.size,self.size),resample=resampling.BILINEAR)


def save_resolution_preview(paths,destination,size,count=6):
    from PIL import Image,ImageDraw
    rows=list(paths)[:count]
    canvas=Image.new('RGB',(640,max(len(rows),1)*190),'white');draw=ImageDraw.Draw(canvas)
    bilinear=getattr(Image,'Resampling',Image).BILINEAR
    for row,path in enumerate(rows):
        with Image.open(path) as image:
            rgb=image.convert('RGB');low=LowResolutionBottleneck(size)(rgb)
            original=rgb.resize((256,256),bilinear).crop((16,16,240,240))
            reduced=low.resize((256,256),bilinear).crop((16,16,240,240))
            y=row*190
            draw.text((0,y+5),f'query #{row} native={rgb.size}, bottleneck={low.size}; original / reduced crop',fill='black')
            canvas.paste(original.resize((160,160),bilinear),(0,y+25))
            canvas.paste(reduced.resize((160,160),bilinear),(180,y+25))
    canvas.save(destination)
