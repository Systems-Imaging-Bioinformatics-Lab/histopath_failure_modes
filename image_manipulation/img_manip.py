import sys
import os
import numpy as np
from scipy import interpolate
from scipy.ndimage import morphology
from scipy.stats import multivariate_normal
from PIL import Image, ImageDraw
from cv2 import GaussianBlur, blur, getPerspectiveTransform, warpPerspective

from deconvolution import Deconvolution
import deconvolution.pixeloperations as po

def rand_spline(dim, inPts = None, nPts = 5, random_seed = None, startEdge = True, endEdge = True):
    # splXY = rand_spline(dim, inPts= None, nPts = 5, random_seed =None, startEdge = True, endEdge = True)
    #     builds a randomized spline from a set of randomized handle points
    #     
    # ###
    # Inputs: Required
    #     dim: a 2 element vector   Width by Height
    # Inputs: Optional
    #     inPts: n x 2 numpy arr    Used to prespecify the handle points of the spline
    #                               note: this is not random
    #     nPts: int                 The number of random handle points in the spline
    #     random_seed: int          The random seed for numpy for consistent generation
    #     startEdge: bool           Whether or not the start of the spline should be on the edge of the image
    #                int(0,1,2,3)   If startEdge is an int, it specifies which edge the spline starts on
    #                               0 = Left, 1 = Top, 2 = Right, 3 = Bottom
    #     endEdge: bool             Whether or not the start of the spline should be on the edge of the image
    #              int(0,1,2,3)     If endEdge is a nonnegative int, it specifies which edge the spline stops on
    #                               0 = Left, 1 = Top, 2 = Right, 3 = Bottom
    #              int(-4,-3,-2,-1) If endEdge is a negative int, it specifies which edge the spline stops on 
    #                               relative to the start
    #                               -4 = Same, -3 = End is 1 step clockwise (e.g. Bottom -> Left)
    #                               -2 = Opposite side, -1 = End is 1 step counterclockwise (e.g. Bottom -> Right)
    # ###
    # Output:
    #     splXY: m x 2 numpy array  Spline array sampled at a 1-pixel interval (distance between m points is ~1px)
    
    np.random.seed(seed=random_seed)

    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    if inPts is None:
        inPts = np.concatenate((np.random.randint((dim[0]-1),size=(nPts,1)),
                                 np.random.randint((dim[1]-1),size=(nPts,1))),
                                axis=1)
        
        startEdgeFlag = (startEdge == True) or (startEdge in range(0,4))
        if startEdgeFlag == True:
            if (startEdge in range(0,4)) and (type(startEdge)!=bool): # allow for manual specification of edge
                edgeNum = startEdge
            else:
                edgeNum = np.random.randint(4)
            LR_v_TB = edgeNum % 2 # left/right vs top/bottom
            LT_V_RB = edgeNum // 2 # left/top vs right/bottom
            
            inPts[0,LR_v_TB] = LT_V_RB * (dim[LR_v_TB]-1) # one edge or the other
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
            
            inPts[nPts-1,LR_v_TB] = LT_V_RB * (dim[LR_v_TB]-1) # one edge or the other
        
    else:
        if isinstance(inPts,list):
            inPts = np.array(inPts)
        nPts = inPts.shape[0]

    distXY = np.sqrt(np.sum(np.diff(inPts,axis=0)**2,axis=1))
    cdXY = np.concatenate((np.zeros((1)),np.cumsum(distXY)),axis=0)
    iDist = np.arange(np.floor(cdXY[-1])+1)
    splXY = interpolate.pchip_interpolate(cdXY,inPts,iDist)
    return splXY

def rand_gauss(dim, nNorms = 25, maxCov = 50,  random_seed = None,centXY = None, zeroToOne = False,
              minMaxX = None, minMaxY = None, minCovScale = .1,minDiagCovScale = .25, maxCrCovScale = .7):
    # sumMap = rand_gauss(dim, nNorms = 25, maxCov = 50, random_seed = None,centXY = None, zeroToOne = False,
    #                  minMaxX = None, minMaxY = None, minCovScale = .1,minDiagCovScale = .25, maxCrCovScale = .7):
    #          Builds a set of randomized Gaussians within a set range of properties, and adds them together
    #
    # ###
    # Inputs: Required
    #     dim: a 2 element vector   Width by Height
    # Inputs: Optional
    #     nNorms: int               The number of Gaussian distributions to generate
    #     maxCov: float (+)         The maximum covariance of each Gaussian
    #                               - Related to the size of each Gaussian distribution in pixels
    #     random_seed: int          The random seed for numpy for consistent generation
    #     centXY: n x 2 float arr   The locations of the Gaussians can be specified manually if desired, instead of randomly
    #     zeroToOne: bool           Whether or not the coordinates are scaled zeroToOne (requires retuning the other sizes)
    #     minMaxX: 2 float vector   The range of where the gaussians are generated in the image in the X dimension
    #                               - Defaults to the range of the image, could be off screen if desired
    #     minMaxY: 2 float vector   The range of where the gaussians are generated in the image in the Y dimension
    #                               - Defaults to the range of the image, could be off screen if desired
    #     minCovScale: float        The minimum on the range of sizes across the set of distributions
    #         Recommend >0 & ≤1
    #     minDiagCovScale: float    Affects the diagonal of the covariance matrix, and the minimum relative size 
    #         Recommend >0 & ≤1     of the two components compared to the scaled max covariance
    #     maxCrCovScale: float      The maximum relative cross covariance (1 = straight line, 0 = uncorrelated)
    #          Recommend >0 & ≤1    Affects the shape of the distributions, a higher number means more eccentricity
    # ###
    # Output:
    #     sumMap: numpy array (dim) Creates a numpy array of the input size, where all the scaled Gaussian 
    #                               distributions have been added together
    
    
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
    else:
        nNorms = centXY.shape[0]
    
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
    # comp_im = add_marker(inputIm,random_seed = None,nPts = 3, sampSpl = None, inPts = None, 
    #                     width = 50, alpha = .75, rgbVal= None,
    #                     rgbRange = np.array([[0,50],[0,50],[0,100]])):
    #           adds a marker line onto the image of a fixed width and color
    #
    # ###
    # Inputs: Required
    #     inputIm: a PIL Image      A 2D RGB image
    # Inputs: Optional
    #     random_seed: int          The random seed for numpy for consistent generation
    #     nPts: int                 The number of random handle points in the spline
    #     sampSpl: n x 2 numpy arr  You can optionally specify the sampled spline (non-random)
    #                               - Note: should be sampled densely enough (i.e. at least every pixel)
    #     inPts: n x 2 numpy arr    Used to prespecify the handle points of the spline
    #                               - Note: this is not random
    #     width: float              The width of the marker line, in pixels
    #     alpha: float (0-1)        The alpha transparency of the marker layer (1 = opaque, 0 = transparent)
    #     rgbVal: 3 uint8 vector    The RGB color of the marker can be optionally specified
    #           >=0 <=255
    #     rgbRange: 3 x 2 uint8 arr The RGB range of the randomized color [[minR,maxR],[minG,maxG],[minB,maxB]]
    #           >=0 <=255           - Leans more blue heavy by default
    # ###
    # Output:
    #     comp_im: a PIL Image      A 2D RGB image with the marker layer on top of the original image
    
    
    np.random.seed(seed=random_seed)
    if rgbVal is None:
        rgbVal = np.zeros((3,1))
        for i in range(3):
            rgbVal[i] = np.random.randint(rgbRange[i,0],rgbRange[i,1])
        rgbVal = rgbVal.flatten()

    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    if sampSpl is None:
        if inPts is None:
            sampSpl = rand_spline(dim, nPts = nPts,random_seed = random_seed)
        else:
            sampSpl = rand_spline(dim, inPts = inPts,random_seed = random_seed)
        
    mask = np.ones(invDim)
    mask[(np.round(sampSpl[:,1])).astype(int),np.round(sampSpl[:,0]).astype(int)] = 0
    # create a distance map to the points on the spline
    bw_dist = morphology.distance_transform_edt(mask)

    # use the distance map to build a fixed width region
    bw_reg = bw_dist <= width
    im_rgba = inputIm.convert("RGBA")
    # build up the semi-transparent colored layer
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
            

    pad_szXY = (max(dim),max(dim),0) # pad x, pad y, no pad z (have to reshape for np.pad, which takes y,x,z)
    sampBlur = (((fold_width//20)*2)+1,((fold_width//20)*2)+1) # has to be odd kernel
    
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
               edgeColorMult = (.75,.75,.75), rgbVal = (225,225,225)):
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
        color_arr[edge_area,i] = np.uint8(meanColor[i] * edgeColorMult[i])

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

def adjust_stain(inputIm,adj_factor = [1,1,1]):
    
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0]) # have to invert the size dim because rows cols is yx vs xy
    iDimRGB = (invDim[0],invDim[1],3)
    stain_dict = {'eosin':[0.91, 0.38, 0.71], 'null': [0.0, 0.0, 0.0],
              'hematoxylin': [0.39, 0.47, 0.85]}
    
    ## https://deconvolution.readthedocs.io/en/latest/readme.html#two-stain-deconvolution
#     dec = Deconvolution(image=inputIm, basis=[[0.91, 0.38, 0.71], [0.39, 0.47, 0.85],[0.0, 0.0, 0.0]])
    dec = Deconvolution(image=inputIm, basis=[stain_dict['eosin'], stain_dict['hematoxylin'],stain_dict['null']])


    # this section is extracted from the deconvolution package, but adjusted to allow for altering the stain levels
    pxO= dec.pixel_operations
    _white255 = np.array([255, 255, 255], dtype=float)
    
    v, u, w = pxO.get_basis()
    vf, uf, wf = np.zeros(iDimRGB), np.zeros(iDimRGB), np.zeros(iDimRGB)
    vf[:], uf[:], wf[:] = v, u, w
    
    # Produce density matrices for both colors + null. Be aware, as Beer's law do not always hold.
    a, b, c = map(po._array_positive, dec.out_scalars())
    af = np.repeat(a, 3).reshape(iDimRGB) * adj_factor[0] # Adjusting the exponential coefficient
    bf = np.repeat(b, 3).reshape(iDimRGB) * adj_factor[1] # For the different stain components
    cf = np.repeat(c, 3).reshape(iDimRGB) * adj_factor[2]

    # exponential map, for changing 
    rgbOut = po._array_to_colour_255(_white255 * (vf ** af) * (uf ** bf) * (wf ** cf))
    rgb_1 = po._array_to_colour_255(_white255 * (vf ** af))
    rgb_2 = po._array_to_colour_255(_white255 * (uf ** bf))
    rgb_3 = po._array_to_colour_255(_white255 * (wf ** cf))
    
    return rgbOut,rgb_1,rgb_2,rgb_3

def add_stain(inputIm,adj_factor = None,scale_max = [3,3,1.5], scale_min = [1.25,1.25,1],random_seed = None):
    if adj_factor is None:
        np.random.seed(seed=random_seed) 
        adj_factor = np.ones((1,3))
        for stI in range(len(scale_max)):
            adj_factor[0,stI] = np.random.uniform(scale_min[stI],scale_max[stI]) ** np.random.choice((-1,1))
        adj_factor = adj_factor.flatten().tolist()
#         print(adj_factor)
    rgbOut,rgb_1,rgb_2,rgb_3 = adjust_stain(inputIm,adj_factor)
    outIm = Image.fromarray(rgbOut,'RGB')
    return outIm


def add_tear(inputIm,sampSpl = None, random_seed = None, nSplPts = 2,
             minSpacing = 20, maxSpacing = 40, minTearStart = 0, maxTearEnd = None,tearStEndFactor = [.2,.8],
             dirMin = 10, dirMax = 30, inLineMax = None, perpMax = None, ptWidth = 2.25, tearAlpha = 1,edgeWidth = 2,
             inLinePercs = np.array([(-.5,-.3,-.2),(.5,.3,.2)]),perpPercs = np.array([(-.5,-.3,-.2),(.5,.3,.2)]),
             t1MinCt = 3, t1MaxCt = 8, minDensity = [.5,.5], maxDensity = [1.5,1.5],
             edgeAlpha = .75, edgeColorMult = [.85,.7,.85],rgbVal = (245,245,245),
             randEdge = True):
    np.random.seed(seed=random_seed)
    dim = inputIm.size # width by height
    invDim = (dim[1],dim[0])
    if sampSpl is None:
        sampSpl = rand_spline(dim, nPts = nSplPts,random_seed = random_seed,endEdge=-2)
    
    # determine where the tears are located
    tearSpacing = np.random.randint(minSpacing,maxSpacing,size=(sampSpl.shape[0],1))
    splLen = sampSpl.shape[0]-1
    if maxTearEnd is None:
        maxTearEnd = splLen
    # randomly trim the start and end
    tearStEnd = np.zeros((2,1))
    tearStEnd[0] = np.random.randint(0,np.floor(splLen*tearStEndFactor[0]),size=(1,1))
    tearStEnd[1] = tearStEnd[0] + np.random.randint(np.floor(splLen*tearStEndFactor[1]),splLen,size=(1,1))
    tearStEnd[0] = 0
    tearStEnd[1] = splLen

    tearStEnd[tearStEnd > maxTearEnd] = maxTearEnd
    tearStEnd[tearStEnd < minTearStart] = minTearStart
    cdTS = np.cumsum(tearSpacing)
    cdTS = cdTS[(cdTS >= tearStEnd[0]) & (cdTS < tearStEnd[1])]

    tearCents = sampSpl[cdTS,:]
    splDer = sampSpl[:-1,:]- sampSpl[1:,:]

    if inLineMax is None:
        inLineMax = np.random.randint(dirMin,dirMax,size=(1,1))
    if perpMax is None:
        perpMax = np.random.randint(dirMin,dirMax,size=(1,1))
    
    splDer = np.concatenate((splDer[[0],:],splDer))
    tearDer = splDer[cdTS,:]
    areaMax = inLineMax * perpMax
    tearDensity = areaMax/ ((ptWidth**2)*np.pi)

    nTears = tearCents.shape[0]
    
    tearCts = np.random.randint(t1MinCt,t1MaxCt,size=(nTears,1))
    for tNo in range(len(minDensity)): # build up the tier matrix
        tearCts = np.append(tearCts, np.random.randint(np.ceil(tearDensity*minDensity[tNo]),
                                                       np.ceil(tearDensity*maxDensity[tNo]),size=(nTears,1)),
                            axis = 1)

    tearCtIdxs = np.concatenate((np.zeros((1)),np.cumsum(np.sum(tearCts,axis=1))),axis=0)

    tearXY = np.zeros((np.sum(tearCts),2))
    tierMats = {}
    # generate tears by using random points
    for tIdx in range(len(cdTS)):
        tierMats[tIdx] = {}
        for tier in range(tearCts.shape[1]): # work in tiers, each tier builds off of the last, gradually filling out the space
            # each tier builds off the last with a uniform distribution
            nPts = tearCts[tIdx,tier]
            if tier == 0:
                centPts = np.repeat(np.reshape(tearCents[tIdx,:],(1,2)),nPts,axis=0)
            else:
                centIdxs = np.random.randint(0,tearCts[tIdx,tier-1],size=(nPts))
                centPts = tierMats[tIdx][tier-1][centIdxs,:]
            inLineFactor = np.random.uniform(inLinePercs[0,tier]*inLineMax,inLinePercs[1,tier]*inLineMax,size=(nPts,1))
            perpFactor = np.random.uniform(perpPercs[0,tier]*perpMax,perpPercs[1,tier]*perpMax,size=(nPts,1))
            cDerIL = tearDer[tIdx,:]
            cDerP = np.array([tearDer[tIdx,1], -tearDer[tIdx,0]])
            totVec = (inLineFactor * cDerIL) + (perpFactor * cDerP)
            newPts = centPts + totVec
            tierMats[tIdx][tier] = newPts.copy()
        idxRng = range(tearCtIdxs[tIdx].astype(int),tearCtIdxs[tIdx+1].astype(int))
        tearXY[idxRng,:] = np.vstack(list(tierMats[tIdx].values()))
    
    # rectify the points so we don't go out of bounds
    tearXY = np.maximum(tearXY,0)
    tearXY[:,0] = np.minimum(tearXY[:,0],dim[0]-1)
    tearXY[:,1] = np.minimum(tearXY[:,1],dim[1]-1)

    # turn these points into a distance mask
    tearMask = np.ones(invDim)
    tearMask[(np.round(tearXY[:,1])).astype(int),np.round(tearXY[:,0]).astype(int)] = 0
    tearDist = morphology.distance_transform_edt(tearMask)
   
    if randEdge == True:
        distRand = np.random.uniform(-int(ptWidth*.5),int(ptWidth*.5),size=invDim)
        tearDist = blur(tearDist+distRand,(5,5))
    tearBW = tearDist <= ptWidth

    alphaArr = (tearBW*tearAlpha*255).astype(np.uint8)
    colorArr = np.zeros((invDim[0],invDim[1],3),dtype=np.uint8)
    edgeArea = np.logical_and(tearDist > ptWidth,tearDist <= ptWidth+edgeWidth)
    
#     blurIm = Image.fromarray(blur(np.array(inputIm),(75,75)),"HSV")
#     imHSV = blurIm.convert("HSV")
#     blurBackground = np.array(imHSV)
#     blurBackground[:,:,1] = np.uint8(np.minimum(blurBackground[:,:,1] * 1.5,255))
#     blurBackground[:,:,2] = np.uint8(np.minimum(blurBackground[:,:,2] * 1.2,255))
#     blurHSV = Image.fromarray(blurBackground,"HSV")
#     blurRGB = np.array(blurHSV.convert("RGB"))
    
    # determine the color of the edge area and the 
    meanColor = np.mean(np.array(inputIm),axis=(0,1))
    for i in range(len(rgbVal)):
        colorArr[:,:,i] = rgbVal[i]
        colorArr[edgeArea,i] = np.uint8(np.minimum(meanColor[i] * edgeColorMult[i],255))

    alphaArr[edgeArea] = edgeAlpha * 255
    alphaMask = Image.fromarray(alphaArr,'L')
    colorLayer = Image.fromarray(colorArr,'RGB')
    comp_im = Image.composite(colorLayer, inputIm, alphaMask)
    return comp_im
    
def apply_artifact(inputImName,artifactType,outputImName = None, outputDir = None,randAdd = 0, ext = "jpeg", perTileRand = None):
    artifactType = artifactType.lower()
    # to remove any linkage between the different types of random addition (e.g. marker vs fold)
    typeSeedAdd = {'marker' : 1, 'fold': 2, 'sectioning': 3, 'illumination': 4, 'bubbles': 5, 'stain' : 6,'tear': 7}
    # to randomize slide/tile based on type of artifact
    typeTileRand = {'marker' : True, 'fold': True, 'sectioning': True, 'illumination': True, 'bubbles': True, 
                    'stain' : False, 'tear': True}
    
    inputIm = Image.open(inputImName)

    inputImDir,fName = os.path.split(inputImName)
    oPath1, rDir1 = os.path.split(inputImDir)
    _, rDir2 = os.path.split(oPath1)
    fNameNoExt = os.path.splitext(fName)[0]
    
    if perTileRand is None:
        perTileRand = typeTileRand[artifactType]
    if perTileRand == True: # take into account the tile name
        fID = os.path.join(rDir2,rDir1,fNameNoExt)
    else: # only take into account the slide name
        fID = os.path.join(rDir2,rDir1)

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
    elif artifactType == "stain":
        outputIm = add_stain(inputIm,random_seed = random_seed)
    elif artifactType == "tear":
        outputIm = add_tear(inputIm,random_seed = random_seed)
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
