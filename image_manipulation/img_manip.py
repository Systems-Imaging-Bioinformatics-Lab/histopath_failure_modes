import numpy as np
from scipy import interpolate
from scipy.ndimage import morphology
from PIL import Image, ImageDraw
import cv2

def rand_spline(dim,inPts= None, nPts = 5,random_seed =None,startEdge = True,endEdge = True):
    np.random.seed(seed=random_seed)

    # print(dim)
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    if inPts is None:
        inPts = np.concatenate((np.random.randint(invDim[0],size=(nPts,1)),
                                 np.random.randint(invDim[1],size=(nPts,1))),
                                axis=1)
        if startEdge == True:
            edgeNum = np.random.randint(4)
            LR_v_TB = edgeNum % 2 # left/right vs top/bottom
            LT_V_RB = edgeNum // 2 # left/top vs right/bottom
            inPts[0,LR_v_TB] = LT_V_RB * invDim[LR_v_TB] # one edge or the other
        if endEdge == True:
            edgeNum = np.random.randint(4)
            LR_v_TB = edgeNum % 2 # left/right vs top/bottom
            LT_V_RB = edgeNum // 2 # left/top vs right/bottom    
            inPts[nPts-1,LR_v_TB] = LT_V_RB * invDim[LR_v_TB] # one edge or the other
        # print(inPts)
    else:
        if isinstance(inPts,list):
            inPts = np.array(inPts)
        nPts = inPts.shape[0]

    distXY = np.sqrt(np.sum(np.diff(inPts,axis=0)**2,axis=1))
    cdXY = np.concatenate((np.zeros((1)),np.cumsum(distXY)),axis=0)
    iDist = np.arange(np.floor(cdXY[-1])+1)
    print(max(iDist),cdXY)
    splXY = interpolate.pchip_interpolate(cdXY,inPts,iDist)
    return splXY


def add_fold(input_im,samp_arr =None, sampSpl=None, inPts = None,random_seed =None,scaleXY =[1,1],fold_width = 100,
             samp_shiftXY = None,randEdge=False, nLayers = 1):
    np.random.seed(seed=random_seed)

    im_arr = np.array(input_im)
    dim = input_im.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy

    if sampSpl is None:
        if inPts is None:
            sampSpl = rand_spline(dim, inPts = inPts,random_seed = random_seed)
        else:
            sampSpl = rand_spline(dim, nPts = 3,random_seed = random_seed)
    
    if samp_arr is None:
        samp_arr = np.copy(im_arr)
        
    if samp_shiftXY is None: # randomly initialized if empty
        shiftXY = np.random.randint(-int(dim[0]/2),int(dim[0]/2),size=(2,1))
    else:
        shiftXY = samp_shiftXY
            

    pad_szXY = (512,512,0) # pad x, pad y, no pad z (have to reshape for np.pad, which takes y,x,z)
    sampBlur = (((fold_width//20)*2)+1,((fold_width//20)*2)+1) # has to be odd kernel
#     scaleXY = [2/3, 1]
    
    pad_amt = np.transpose(np.tile(np.array(pad_szXY)[[1,0,2]],(2,1)))
    samp_pad_arr = np.pad(samp_arr,pad_amt,mode='symmetric')

    sampSplBBox = np.vstack((np.amin(sampSpl,axis=0),np.amax(sampSpl,axis=0)))
    sampSplBBSz = np.diff(sampSplBBox,axis=0)
    rsSplBBox = np.zeros((2,2))

    signTup = (-1,1)
    for di in range(2):
        rsSplBBox[di,:] = np.mean(sampSplBBox,axis=0) + (((sampSplBBSz/2) / scaleXY) * signTup[di])

    sampSplBBPts = np.zeros((4,2),dtype=np.float32)
    outBBPts = np.zeros((4,2),dtype=np.float32)

    for di in range(sampSplBBPts.shape[0]):    
        LR_v_TB = di % 2 # left/right vs top/bottom
        LT_V_RB = di // 2 # left/top vs right/bottom
        sampSplBBPts[di,0] = sampSplBBox[LR_v_TB,0]
        sampSplBBPts[di,1] = sampSplBBox[LT_V_RB,1]
        outBBPts[di,0] = rsSplBBox[LR_v_TB,0] + pad_szXY[0] + shiftXY[0]
        outBBPts[di,1] = rsSplBBox[LT_V_RB,1] + pad_szXY[1] + shiftXY[1]

    M = cv2.getPerspectiveTransform(outBBPts,sampSplBBPts)

    warp_im = cv2.warpPerspective(samp_pad_arr,M,dim)

    #
    mask = np.ones(dim)
    mask[(sampSpl[:,1].astype(int)),sampSpl[:,0].astype(int)] = 0
    bw_dist = morphology.distance_transform_edt(mask)
    if randEdge == True:
        distRand = np.random.randint(-int(fold_width/2),int(fold_width/2),size=invDim)
        bw_dist = cv2.blur(bw_dist+distRand,(5,5))
    
    im_L =  input_im.convert("L")
    im_L_arr = np.array(im_L)
    bw_reg = bw_dist <= fold_width
    
#     alpha_arr = np.minimum(255,(((255-im_L_arr)*.25) * (bw_reg)))
#     alpha_mask = Image.fromarray(alpha_arr.astype(np.uint8),'L')

    # multiplicative combination.  Makes things darker
    unit_dst_arr = np.ones(warp_im.shape)
    for i in range(warp_im.shape[2]):
        unit_dst_arr[:,:,i] = np.where(bw_reg,warp_im[:,:,i]/255,1)
    unit_dst_arr = cv2.GaussianBlur(unit_dst_arr,sampBlur,0)
    comb_arr = unit_dst_arr * im_arr
    comb_img = Image.fromarray(comb_arr.astype(np.uint8),'RGB')
    
    if nLayers > 1: # recursive addition
        comb_img = add_fold(comb_img,samp_arr=samp_arr, sampSpl=sampSpl,inPts=inPts,random_seed = random_seed+1,
                 scaleXY=scaleXY,fold_width=fold_width,samp_shiftXY=samp_shiftXY,randEdge=randEdge,
                 nLayers=nLayers-1)
    return comb_img
    
#     fig,ax = plt.subplots(figsize=(9, 6))
#     plt.imshow(shifted_layer)
#     plt.plot(sampSpl[:,0],sampSpl[:,1])
#     plt.show()