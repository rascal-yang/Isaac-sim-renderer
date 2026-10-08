#!/usr/bin/env python3
"""Same-state image comparison; raw RGB files avoid MP4 compression differences."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2

def gray(x):return cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32)
def ssim(x,y):
    a,b=gray(x),gray(y);ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb;co=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    return float((((2*ma*mb+6.5025)*(2*co+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225)))[5:-5,5:-5].mean())
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=np.load(a.reference/'raw_rgb.npy',mmap_mode='r');y=np.load(a.candidate/'raw_rgb.npy',mmap_mode='r');assert x.shape==y.shape
    assert json.loads((a.reference/'renderer_settings.json').read_text())==json.loads((a.candidate/'renderer_settings.json').read_text()),'Renderer settings differ'
    n=len(x);mae=[];mse=[]
    for i in range(n):
        d=x[i].astype(np.float32)-y[i].astype(np.float32);mae.append(float(abs(d).mean()));mse.append(float((d*d).mean()))
    sample=list(range(0,n,10));scores=[ssim(x[i],y[i]) for i in sample]
    moving=list(range(2,max(3,n-2),max(1,n//100)))
    lag={str(k):float(np.mean([abs(gray(y[i])-gray(x[i+k])).mean() for i in moving if 0<=i+k<n])) for k in [-2,-1,0,1,2]}
    report=dict(frames=n,mean_absolute_channel_error=float(np.mean(mae)),psnr_db=float(10*np.log10(255**2/max(float(np.mean(mse)),1e-12))),grayscale_ssim_mean=float(np.mean(scores)),grayscale_ssim_min=min(scores),gray_error_by_reference_frame_offset=lag,best_relative_frame_offset=int(min(lag,key=lag.get)),notes='Every tenth frame SSIM; +/-2 relative offsets only, not absolute sensor latency certification; stochastic RT2 pixels need not be byte-identical.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'image_comparison.json').write_text(json.dumps(report,indent=2))
    for i in sorted({0,n//4,n//2,min(n-1,699),min(n-1,700),min(n-1,701),3*n//4,n-1}):
        cv2.imwrite(str(a.output/f'frame_{i:04d}.png'),cv2.cvtColor(np.concatenate([x[i],y[i]],axis=1),cv2.COLOR_RGB2BGR))
    assert report['best_relative_frame_offset']==0,report
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
