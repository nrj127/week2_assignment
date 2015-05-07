import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
caffe_root = '../'
import sys
sys.path.insert(0,caffe_root+'python')
import scipy
from scipy import misc
import caffe


#MODEL_FILE = '../models/bvlc_alexnet/deploy.prototxt'
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'imagenet/mydata/IMG_1.jpg','imagenet/mydata/IMG_2.jpg','imagenet/mydata/IMG_3.jpg','imagenet/mydata/IMG_4.jpg','imagenet/mydata/IMG_5.jpg','imagenet/mydata/IMG_6.jpg','imagenet/mydata/IMG_7.jpg','imagenet/mydata/IMG_8.jpg'

#IMAGE_FILE1 = 'imagenet/mydata/IMG_1.jpg'
#IMAGE_FILE2 = 'imagenet/mydata/IMG_2.jpg'
#IMAGE_FILE3 = 'imagenet/mydata/IMG_3.jpg'
#IMAGE_FILE4 = 'imagenet/mydata/IMG_4.jpg'
#IMAGE_FILE5 = 'imagenet/mydata/IMG_5.jpg'
#IMAGE_FILE6 = 'imagenet/mydata/IMG_6.jpg'
IMAGE_FILE7 = 'imagenet/mydata/IMG_7.jpg'
#IMAGE_FILE8 = 'imagenet/mydata/IMG_8.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE,PRETRAINED,
						mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


for image in IMAGE_FILE:
	input_image = caffe.io.load_image(image)
	#plt.imshow(input_image)
	#plt.show(input_image)
	prediction = net.predict([input_image])
	print 'prediction shape:', prediction[0].shape
        #plt.plot(prediction[0])
	print 'predicted class:', prediction[0].argmax()
        g= prediction[0].argmax()
        print 'probability.', prediction[0][g]        
        print 'entropy', entropy(prediction[0])
	print ("\n")



#preprocessing crop the image

for image in IMAGE_FILE:
#myimage = misc.imread('imagenet/mydata/IMG_1.jpg')
	myimage = misc.imread(image)
	lx,ly,lz = myimage.shape
	crop_myimage = myimage[lx /4:-lx/4,ly/4:-ly/4]
	crop_myimage = plt.imsave('crop_myimage.jpg',crop_myimage)
	input_image = caffe.io.load_image('crop_myimage.jpg')
	prediction = net.predict([input_image])
	print 'predicted class:',prediction[0].argmax()
	g=prediction[0].argmax()
	print 'probability',prediction[0][g]

print 'end'



#input_image = scipy.misc.imread('imagenet/mydata/IMG_7.jpg')
#input_image = scipy.misc.imresize(input_image,(20,20,3))
#input_image = caffe.io.load_image(input_image)
#prediction = net.predict([input_image])
#print 'predicted class:', prediction[0].argmax()


#prediction = net.predict([input_image], oversample=False)
#print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
#print 'predicted class:', prediction[0].argmax()
