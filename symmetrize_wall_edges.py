from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def sigmoid(x,x0,width):
    return 1/(1+np.exp(-(x-x0)/width))


def symmetrize(Im):
    X=np.indices((Im.shape[0],Im.shape[1]))
    print(X.shape)
    w=Im.shape[0]//2
    d=10
    L_0=np.expand_dims(sigmoid(X[0,:,:],w,d),-1)
    L_1=np.expand_dims(sigmoid(X[0,:,:],w+d,1),-1)

    R_0=np.expand_dims(sigmoid(X[0,:,:],Im.shape[0]-w,-d),-1)
    R_1=np.expand_dims(sigmoid(X[0,:,:],Im.shape[0]-w-d,-1),-1)


    Im=np.roll(Im,Im.shape[0]//2,axis=0)

    # plt.imshow((Im).astype(int))
    # plt.show()
    #
    # plt.imshow((Im*(1-4*L_0*R_0)+(2*L_0*R_0*Im)+np.flip(2*L_0*R_0*Im,0)).astype(int))
    # plt.show()

    Im=(Im*(1-4*L_0*R_0)+(2*L_0*R_0*Im)+np.flip(2*L_0*R_0*Im,0))
    Im = np.roll(Im, Im.shape[0]//2, axis=0)
    return Im
type_im='wall'
if type_im=='wall':
    wall_number=43
    Im=load_image(f'./image/wall/wall{wall_number}.png')
    im = Image.fromarray(Im.astype('uint8'), 'RGBA')
    im.save(f'./image/wall/unsym/wall{wall_number}_old.png')

    plt.imshow((Im).astype(int))
    plt.show()
    Im=symmetrize(Im)
    Im=np.transpose(symmetrize(np.transpose(Im,(1,0,2))),(1,0,2))
    Im=np.minimum(np.maximum(Im.astype(int),0),255)
    plt.imshow(Im)
    plt.show()
    Im=Im.astype('uint8')
    im = Image.fromarray(Im, 'RGBA')
    im.save(f'./image/wall/wall{wall_number}.png')

if type_im=='sky':
    wall_number=1
    Im=load_image(f'./image/ciel/ciel{wall_number}.png')
    im = Image.fromarray(Im.astype('uint8'), 'RGBA')
    im.save(f'./image/ciel/unsym/ciel{wall_number}_old.png')

    plt.imshow((Im).astype(int))
    plt.show()
    Im=np.transpose(symmetrize(np.transpose(Im,(1,0,2))),(1,0,2))
    Im=np.minimum(np.maximum(Im.astype(int),0),255)
    plt.imshow(Im)
    plt.show()
    im = Image.fromarray(Im.astype('uint8'), 'RGBA')
    im.save(f'./image/ciel/ciel{wall_number}.png')