# IFTA
Interstitial fibrosis and tubular atrophy measurement via hierarchical extractions of kidney and atrophy regions with deep learning method

GCPANet进行肺实质分割：首先使用GCPANet分割肺部的实质区域。
Mask-RCNN用于病灶检测和分割：将GCPANet生成的肺实质掩码和原始图像一起传递给Mask-RCNN，结合病灶标签进行目标检测与实例分割。
