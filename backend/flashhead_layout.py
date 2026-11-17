"""Place generated head pixels into a real photo at the requested export framing."""
import cv2
import numpy as np
from PIL import Image
from head_composite import head_mask,crop_window,composite_head


class Layout:
    def __init__(self,photo,size,bounds,box):
        width,height=size;sw,sh=photo.size
        if any(type(v) is not int or v<128 or v>3840 or v%2 for v in size):
            raise ValueError('Use even export dimensions between 128 and 3840 pixels')
        x,y,w,h=bounds
        # Validate the source bounds and ensure no generated body pixels enter.
        alpha=head_mask(photo,photo.size,bounds)*crop_window(photo.size,box)
        scale=max(width/sw,height/sh);cw,ch=width/scale,height/scale
        cx=(x+w/2)*sw
        left=max(0,min(sw-cw,cx-cw/2));top=(sh-ch)/2
        face=(x*sw,(1-y-h)*sh,(x+w)*sw,(1-y)*sh)
        self.mode='crop'
        if not(left<=face[0] and top<=face[1] and face[2]<=left+cw and face[3]<=top+ch):
            # A landscape crop of a close-up can cut the face. Preserve it with
            # ordinary padding instead of reconstructing absent body or hair.
            self.mode='fit';scale=min(width/sw,height/sh)
            fw,fh=round(sw*scale),round(sh*scale)
            ox,oy=(width-fw)//2,(height-fh)//2
            self.source=np.zeros((height,width,3),dtype=np.uint8)
            self.source[oy:oy+fh,ox:ox+fw]=np.asarray(photo.resize((fw,fh),Image.Resampling.LANCZOS))
            self.mask=np.zeros((height,width),dtype=np.float32)
            self.mask[oy:oy+fh,ox:ox+fw]=np.asarray(Image.fromarray(alpha).resize((fw,fh),Image.Resampling.BILINEAR))
            sx,sy=fw/sw,fh/sh
            self.source_transform=np.array([[sx,0,ox+(sx-1)/2],[0,sy,oy+(sy-1)/2],[0,0,1.]])
        else:
            crop=(left,top,min(sw,left+cw),min(sh,top+ch))
            self.source=np.asarray(photo.resize(size,Image.Resampling.LANCZOS,box=crop))
            self.mask=np.asarray(Image.fromarray(alpha).resize(size,Image.Resampling.BILINEAR,box=crop))
            self.source_transform=np.array([[scale,0,-left*scale+(scale-1)/2],
                                            [0,scale,-top*scale+(scale-1)/2],[0,0,1.]])
        self.size=size;self.box=box
        yy,xx=np.nonzero(self.mask>0)
        if not len(xx):raise ValueError('The selected framing hides the animated face')
        self.region=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)

    def frame(self,generated,correction=None):
        mh,mw=generated.shape[:2];left,top,right,bottom=self.box
        sx,sy=(right-left)/mw,(bottom-top)/mh
        projection=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1.]])
        rigid=np.eye(3)
        if correction is not None:rigid[:2]=correction
        # One resampling from model pixels to final output, including optional
        # rigid stabilization. The static source body is prepared only once.
        matrix=self.source_transform@projection@rigid
        x0,y0,x1,y1=self.region
        matrix[:2,2]-=[x0,y0]
        projected=cv2.warpAffine(generated,matrix[:2],(x1-x0,y1-y0),
                                 flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT101)
        result=self.source.copy()
        result[y0:y1,x0:x1]=composite_head(self.source[y0:y1,x0:x1],projected,self.mask[y0:y1,x0:x1])
        if not np.array_equal(result[self.mask==0],self.source[self.mask==0]):
            raise RuntimeError('Appearance changed outside the animated head region')
        return result
