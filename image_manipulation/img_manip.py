import sys
import os
import numpy as np
from scipy import interpolate
from scipy.ndimage import morphology
from scipy.stats import multivariate_normal
from PIL import Image, ImageDraw
from cv2 import GaussianBlur, blur, getPerspectiveTransform, warpPerspective

def rand_spline(dim,inPts= None, nPts = 5,random_seed =None,startEdge = True,endEdge = True):
    np.random.seed(seed=random_seed)

#     print(nPts)
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    if inPts is None:
        inPts = np.concatenate((np.random.randint(invDim[0],size=(nPts,1)),
                                 np.random.randint(invDim[1],size=(nPts,1))),
                                axis=1)
        startEdgeFlag = (startEdge == True) or (startEdge in range(0,4))
        if startEdgeFlag == True:
            if (startEdge in range(0,4)): # allow for manual specification of edge
                edgeNum = startEdge
            else:
                edgeNum = np.random.randint(4)
            LR_v_TB = edgeNum % 2 # left/right vs top/bottom
            LT_V_RB = edgeNum // 2 # left/top vs right/bottom
            inPts[0,LR_v_TB] = LT_V_RB * dim[LR_v_TB] # one edge or the other
        if endEdge == True or (endEdge in range(0,4)) or (endEdge in range(-4,0) and startEdgeFlag):
            if (endEdge in range(0,4)):  # allow for manual specification of edge
                edgeNum = endEdge
            elif(endEdge in range(-4,0) and startEdgeFlag): 
                # allow for relative specification of end edge compared to the start edge
                # -2 is opposite side, -4 is the same side
                edgeNum = ((endEdge + edgeNum + 4) % 4)
            else:
                edgeNum = np.random.randint(4)
            LR_v_TB = edgeNum % 2 # left/right vs top/bottom
            LT_V_RB = edgeNum // 2 # left/top vs right/bottom    
            inPts[nPts-1,LR_v_TB] = LT_V_RB * dim[LR_v_TB] # one edge or the other
        # print(inPts)
    else:
        if isinstance(inPts,list):
            inPts = np.array(inPts)
        nPts = inPts.shape[0]

    distXY = np.sqrt(np.sum(np.diff(inPts,axis=0)**2,axis=1))
    cdXY = np.concatenate((np.zeros((1)),np.cumsum(distXY)),axis=0)
    iDist = np.arange(np.floor(cdXY[-1])+1)
#     print(max(iDist),cdXY)
    splXY = interpolate.pchip_interpolate(cdXY,inPts,iDist)
    return splXY

def rand_gauss(dim,zeroToOne = False, maxCov = 50, nNorms = 25, random_seed = None,centXY = None,
              minMaxX = None, minMaxY = None, minCovScale = .1,minDiagCovScale = .25, maxCrCovScale = .7):
    np.random.seed(seed=random_seed)
    invDim = (dim[1],dim[0])
     
    if zeroToOne == True:
        xV = np.linspace(0,1,num= dim[0])
        yV = np.linspace(0,1,num= dim[1])
        if minMaxX is None:
            minMaxX = [0,1]
        if minMaxY is None:
            minMaxY = [0,1]
    else:
        xV = np.arange(dim[0])
        yV = np.arange(dim[1])
        
        if minMaxX is None:
            minMaxX = [0,dim[0]]
        if minMaxY is None:
            minMaxY = [0,dim[1]]

    xM, yM = np.meshgrid(xV,yV)
    pos = np.dstack((xM, yM))
    
    if centXY is None:
        centXY = np.concatenate((np.random.uniform(minMaxX[0],minMaxX[1],size=(nNorms,1)),
                        np.random.uniform(minMaxY[0],minMaxY[1],size=(nNorms,1))), axis=1)
    
    sumMap = np.zeros(invDim)
    for i in range(nNorms):
        cent = centXY[i,:]
        cov = np.zeros((2,2))
        # need to make a symmetric positive semidefinite covariance matrix
        cMaxCov = np.random.uniform(maxCov* minCovScale,maxCov,size=(1,1))
        cov = np.diag(np.random.uniform(cMaxCov * minDiagCovScale,cMaxCov,size=(1,2)).flatten())
        maxCrCov = np.sqrt(np.product(np.diag(cov)))
        cov[[0,1],[1,0]] = np.random.uniform(-maxCrCov* maxCrCovScale,maxCrCov*maxCrCovScale) 
        rv = multivariate_normal(cent.flatten(), cov)
        
        sumMap += (rv.pdf(pos) * (cMaxCov * 2 * np.pi))

    sumMap = sumMap * maxCov
    return sumMap

def add_marker(inputIm,random_seed = None,nPts = 3, sampSpl = None, inPts = None, 
              width = 50, alpha = .75, rgbVal= None,
              rgbRange = np.array([[0,50],[0,50],[0,100]])):
    np.random.seed(seed=random_seed)
    if rgbVal is None:
        rgbVal = np.zeros((3,1))
        for i in range(3):
            rgbVal[i] = np.random.randint(rgbRange[i,0],rgbRange[i,1])
        rgbVal = rgbVal.flatten()

    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
#     print(sampSpl is None, inPts is None)
    if sampSpl is None:
        if inPts is None:
            sampSpl = rand_spline(dim, nPts = nPts,random_seed = random_seed)
        else:
            sampSpl = rand_spline(dim, inPts = inPts,random_seed = random_seed)
    
#     print(nPts)
    
    mask = np.ones(invDim)
    mask[(sampSpl[:,1].astype(int)),sampSpl[:,0].astype(int)] = 0
    bw_dist = morphology.distance_transform_edt(mask)

    bw_reg = bw_dist <= width
    im_rgba = inputIm.convert("RGBA")
    alpha_mask = Image.fromarray((bw_reg*alpha*255).astype(np.uint8),'L')
    color_arr = np.zeros((invDim[0],invDim[1],3),dtype=np.uint8)
    for i in range(len(rgbVal)):
        color_arr[:,:,i] = rgbVal[i]
    color_layer = Image.fromarray(color_arr,'RGB')

    comp_im = Image.composite(color_layer, im_rgba, alpha_mask)
    comp_im = comp_im.convert("RGB")
    return comp_im


def add_fold(inputIm,samp_arr =None, sampSpl=None, inPts = None,random_seed =None,scaleXY =[1,1],fold_width = 100,
             samp_shiftXY = None,randEdge=False, nLayers = 2, nPts = 3,endEdge = -2):
    np.random.seed(seed=random_seed)

    im_arr = np.array(inputIm)
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy

    if sampSpl is None:
        if inPts is None:
            sampSpl = rand_spline(dim, nPts = nPts,random_seed = random_seed, endEdge = endEdge)
        else:
            sampSpl = rand_spline(dim, inPts = inPts,random_seed = random_seed)
    
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
        
    rsSplBBSz = np.diff(rsSplBBox,axis=0)

    sampSplBBPts = np.zeros((4,2),dtype=np.float32)
    outBBPts = np.zeros((4,2),dtype=np.float32)
    randShiftX = np.random.randint(-int(rsSplBBSz[0,0]/4),int(rsSplBBSz[0,0]/4),size=(4,1))
    randShiftY = np.random.randint(-int(rsSplBBSz[0,1]/4),int(rsSplBBSz[0,1]/4),size=(4,1))
    
    for di in range(sampSplBBPts.shape[0]):    
        LR_v_TB = di % 2 # left/right vs top/bottom
        LT_V_RB = di // 2 # left/top vs right/bottom
        sampSplBBPts[di,0] = sampSplBBox[LR_v_TB,0]
        sampSplBBPts[di,1] = sampSplBBox[LT_V_RB,1]
        outBBPts[di,0] = rsSplBBox[LR_v_TB,0] + pad_szXY[0] + shiftXY[0] + randShiftX[di]
        outBBPts[di,1] = rsSplBBox[LT_V_RB,1] + pad_szXY[1] + shiftXY[1] + randShiftY[di]

    M = getPerspectiveTransform(outBBPts,sampSplBBPts)

    warp_im = warpPerspective(samp_pad_arr,M,dim)

    #
    mask = np.ones(invDim)
    mask[(sampSpl[:,1].astype(int)),sampSpl[:,0].astype(int)] = 0
    bw_dist = morphology.distance_transform_edt(mask)
    if randEdge == True:
        distRand = np.random.randint(-int(fold_width/2),int(fold_width/2),size=invDim)
        bw_dist = blur(bw_dist+distRand,(5,5))
    
    im_L =  inputIm.convert("L")
    im_L_arr = np.array(im_L)
    bw_reg = bw_dist <= fold_width
    

    # multiplicative combination.  Makes things darker
    unit_dst_arr = np.ones(warp_im.shape)
    for i in range(warp_im.shape[2]):
        unit_dst_arr[:,:,i] = np.where(bw_reg,warp_im[:,:,i]/255,1)
    unit_dst_arr = GaussianBlur(unit_dst_arr,sampBlur,0)
    comb_arr = unit_dst_arr * im_arr
    comb_img = Image.fromarray(comb_arr.astype(np.uint8),'RGB')
    
    if nLayers > 1: # recursive addition
        comb_img = add_fold(comb_img,samp_arr=samp_arr, sampSpl=sampSpl,inPts=inPts,random_seed = random_seed+1,
                 scaleXY=scaleXY,fold_width=fold_width,samp_shiftXY=samp_shiftXY,randEdge=randEdge,
                 nLayers=nLayers-1)
    return comb_img

def add_sectioning(inputIm, sliceWidth = 120, random_seed = None, scaleMin = .5, scaleMax = .8, randEdge = True,
                  sampSpl = None, inPts = None):
    # Uneven sectioning due to different thicknesses of slide
    np.random.seed(seed=random_seed)
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    
    if sampSpl is None:
        if inPts is None:
            sampSpl = rand_spline(dim, inPts = inPts,random_seed = random_seed)
        else:
            sampSpl = rand_spline(dim, nPts = 2, endEdge = -2,random_seed = random_seed)

    mask = np.ones(invDim)
    mask[(sampSpl[:,1].astype(int)),sampSpl[:,0].astype(int)] = 0
    bw_dist = morphology.distance_transform_edt(mask)
    if randEdge == True:
        distRand = np.random.randint(-int(sliceWidth),int(sliceWidth),size=invDim)
        bw_dist = blur(bw_dist+distRand,(5,5))

    bw_reg = bw_dist <= sliceWidth
    nDistRng = bw_dist / sliceWidth
    halfScale = (scaleMin + scaleMax)/2
    scaleRandMin = np.random.uniform(scaleMin,(halfScale+scaleMin)/2,size=(1,1))
    scaleRandMax = np.random.uniform((halfScale+scaleMax)/2,scaleMax,size=(1,1))
    scaleRMinMax = np.concatenate((scaleRandMin,scaleRandMax),axis = 1).flatten()

    nDistRng = np.interp(nDistRng,np.array([0,1],dtype=np.float64),scaleRMinMax)

    nDistRng[np.logical_not(bw_reg)] = 1

    imHSV = inputIm.convert("HSV")
    imHSV_arr = np.array(imHSV)
    imHSV_arr[:,:,1] = np.minimum(255,np.multiply(imHSV_arr[:,:,1],nDistRng))
    imHSV_arr[:,:,2] = np.minimum(255,np.divide(imHSV_arr[:,:,2],(nDistRng+1)/2))
    # increase the lightness by half the factor of decreased saturation
    imSatHSV = Image.fromarray(imHSV_arr,"HSV")
    imSat = imSatHSV.convert("RGB")
    return imSat

def add_bubbles(inputIm,random_seed = None,nBubbles = 25, maxWidth = 50,alpha = .75, edgeWidth = 2,
               edgeColorMult = .75, rgbVal = (225,225,225)):
    np.random.seed(seed=random_seed)
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
#     maxCov = maxWidth
    
    sumMap = rand_gauss(dim,random_seed = random_seed, nNorms=nBubbles, maxCov = maxWidth, zeroToOne = False,
                       minCovScale = .1,minDiagCovScale = .25, maxCrCovScale = .7)
    bw_reg = sumMap >= 1
    # mask = 1-bw_reg # invert the mask
    bw_dist = morphology.distance_transform_edt(bw_reg)
    edge_area = np.logical_and(bw_dist <= edgeWidth,bw_reg)

    alpha_mask = Image.fromarray((bw_reg*alpha*255).astype(np.uint8),'L')
    color_arr = np.zeros((invDim[0],invDim[1],3),dtype=np.uint8)

    meanColor = np.mean(np.array(inputIm),axis=(0,1))
    for i in range(len(rgbVal)):
        color_arr[:,:,i] = rgbVal[i]
        color_arr[edge_area,i] = np.uint8(meanColor[i] * edgeColorMult)

    color_layer = Image.fromarray(color_arr,'RGB')
    comp_im = Image.composite(color_layer, inputIm, alpha_mask)
    return comp_im

def add_illumination(inputIm,random_seed = None, maxCov = 15, nNorms = 3,scaleMin = .8,scaleMax = 1.1,
                    minCovScale = .5,minDiagCovScale = .1, maxCrCovScale = .2):
    np.random.seed(seed=random_seed)
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0])

    xV = np.linspace(0,1,num= dim[0])
    yV = np.linspace(0,1,num= dim[1])
    xM, yM = np.meshgrid(xV,yV)
    pos = np.dstack((xM, yM))
    
    sumMap = rand_gauss(dim,random_seed = random_seed, nNorms=nNorms, maxCov = maxCov, 
                        zeroToOne = True, minMaxX = [-.5,1.5],minMaxY = [-.5,1.5],
                        minCovScale = minCovScale,minDiagCovScale = minDiagCovScale, maxCrCovScale = maxCrCovScale)

    nSumMap = (sumMap - np.amin(sumMap,axis=(0,1)))
    
    divFac = np.amax(nSumMap,axis=(0,1))
    nSumMap = nSumMap /divFac
    
    scaleRandMin = np.random.uniform(scaleMin,(1+scaleMin)/2,size=(1,1))
    scaleRandMax = np.random.uniform((1+scaleMax)/2,scaleMax,size=(1,1))
    scaleRMinMax = np.concatenate((scaleRandMin,scaleRandMax),axis = 1).flatten()
    nSumMap = np.interp(nSumMap,np.array([0,1],dtype=np.float64),scaleRMinMax)
    
    imHSV = inputIm.convert("HSV")
    imHSV_arr = np.array(imHSV)
    imHSV_arr[:,:,2] = np.minimum(255,np.multiply(imHSV_arr[:,:,2],nSumMap))
    imLumHSV = Image.fromarray(imHSV_arr,"HSV")
    imLum = imLumHSV.convert("RGB")
    return imLum

def apply_artifact(inputImName,artifactType,outputImName = None, outputDir = None,randAdd = 0, ext = "jpeg"):
    artifactType = artifactType.lower()
    # to remove any linkage between the different types of random addition (e.g. marker vs fold)
    typeSeedAdd = {'marker' : 1, 'fold': 2, 'sectioning': 3, 'illumination': 4, 'bubbles': 5}
    
    inputIm = Image.open(inputImName)

    inputImDir,fName = os.path.split(inputImName)
    oPath1, rDir1 = os.path.split(inputImDir)
    _, rDir2 = os.path.split(oPath1)
    fNameNoExt = os.path.splitext(fName)[0]
    fID = os.path.join(rDir2,rDir1,fNameNoExt)

    randMax = (2**32) -1  # max size of the random seed
    # there's potentially some concern about the difference in 32 bit vs 64 bit systems
    random_hash = hash(fID) + randAdd + typeSeedAdd[artifactType]
    random_seed = random_hash % randMax
    
    if artifactType == "marker":
        outputIm = add_marker(inputIm,random_seed = random_seed)
    elif artifactType == "fold":
        outputIm = add_fold(inputIm,random_seed = random_seed)
    elif artifactType == "sectioning":
        outputIm = add_sectioning(inputIm,random_seed = random_seed)
    elif artifactType == "illumination":
        outputIm = add_illumination(inputIm,random_seed = random_seed)
    elif artifactType == "bubbles":
        outputIm = add_bubbles(inputIm,random_seed = random_seed)
    outputSuffix = artifactType[0:4]
    if outputImName is None:
        outputImName = "%s_%s.%s" % (fNameNoExt, outputSuffix, ext)
        if outputDir is not None:
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            outputImName = os.path.join(outputDir,outputImName)
    outputIm.save(outputImName)
    return outputIm

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    apply_artifact(*sys.argv[1:])
