# import the necessary packages
from rootsift import RootSIFT
import cv2
# load the image we are going to extract descriptors from and convert
# it to grayscale
image = cv2.imread("easierSide.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect Difference of Gaussian keypoints in the image
detector = cv2.xfeatures2d.SIFT_create()
(kps, descs) = detector.detectAndCompute(gray, None)
# extract normal SIFT descriptors
extractor = cv2.xfeatures2d.SIFT_create()
(kps, descs) = extractor.detectAndCompute(gray, None)
print ("SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))
# extract RootSIFT descriptors
rs = RootSIFT()
(kps, descs) = rs.compute(gray, kps)
print ("RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))
