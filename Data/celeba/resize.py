import cv2
import os

# Get user supplied values
# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
datadir = 'c:/Users/alevaaga/Datasets'
for fn in sorted(os.listdir(datadir + '/img_align_celeba/')):
    print(fn)
    image = cv2.imread(datadir + '/img_align_celeba/' + fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,5, 5)

    if len(faces) == 0:
        pass
    else:
        x, y, w, h = faces[0]
        image_crop = image[y:y+w, x:x+w, :]
        image_resize = cv2.resize(image_crop, (128, 128))
        output_name = '{}/celeba-128/{}_crop.png'.format(datadir, fn[:-4])
        cv2.imwrite(output_name, image_resize)

#    for (x, y, w, h) in faces:
#        print x, y, w, h
#       cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)




