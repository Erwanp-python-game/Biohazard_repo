from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

Im=load_image('./image/boss/boss0wlk1a7.png')
Im=np.where(np.expand_dims(np.sum(Im[:,:,:-1],axis=-1),-1)==3,np.array([0,0,0,255]),Im)

