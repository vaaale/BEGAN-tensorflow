import cv2
import os

# Get user supplied values
# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
datadir = 'c:/Users/alevaaga/Datasets/img_align_celeba'
for fn in sorted(os.listdir(datadir)):
    print(fn)
    image = cv2.imread(datadir + '/' + fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,5, 5)

    if len(faces) == 0:
        pass
    else:
        x, y, w, h = faces[0]
        image_crop = image[y:y+w, x:x+w, :]
        image_resize = cv2.resize(image_crop, (128, 128))
        cv2.imwrite('128_crop/' + fn[:-4] + '_crop' + fn[-4:], image_resize)

#    for (x, y, w, h) in faces:
#        print x, y, w, h
#       cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)




