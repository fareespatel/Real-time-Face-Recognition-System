
# In[1]:

#Loading necessary packages
from IPython.display import YouTubeVideo
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

#Activate plotting in Ipython notebook
get_ipython().magic('matplotlib inline')

# Open a new window for interaction
cv2.startWindowThread()

#Convert the image from BGR to RGB and display the image
def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()
    
#Defining a class for detecting faces from the image
class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |                     cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else                     cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord
    
#Defining the class for capturing multiple frames from the video camera (capture image)
class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print self.video.isOpened()

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

#function for cropping the image and extracting the face region (ROI)
def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

#Normalizing the intensity of the image to remove the effect of white light and high intensity image
def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

#Resizing the image and standardizing all the images to be of the same dimension
def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

#normalizing images using the above defined functions
def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

#Sorround faces with a rectangle when capturing image in real-time
def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (150, 150, 0), 8)



#This function allows user to store their images in the database by annoting their name to their image. It creates a seperate folder for every user and throws a warning if a username exists in the database.
def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 
                                     0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)


# ### Collect image data and train models

# In[3]:

#Capture images in real-time through a videofeed
images, labels, labels_dic = collect_dataset()

#Train models on the captured images
rec_eig = cv2.face.createEigenFaceRecognizer()
rec_eig.train(images, labels)

# needs at least two people 
rec_fisher = cv2.face.createFisherFaceRecognizer()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.createLBPHFaceRecognizer()
rec_lbph.train(images, labels)

print "Models Trained Succesfully"


# In[17]:

#Load the path for the haar cascade calassifiers xml file that comes with the cv2 package. Provide the exact file path for the xml file.
#You can find this file where you have installed the cv2 package

filepath="C:\\Anaconda3\\envs\\py27\\Library\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"
detector = FaceDetector(filepath)
webcam = VideoCamera(0)


# In[18]:

# Make Predictions with the three models in real-time
#Confidence values indicate offset error from the predicted class and the input class of image

cv2.namedWindow("Face Recognition Model", cv2.WINDOW_NORMAL)
while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame, True) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.MinDistancePredictCollector()
            rec_lbph.predict(face, collector)
            conf = collector.getDist()
            pred = collector.getLabel()
            threshold = 140
            print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
            cv2.putText(frame, labels_dic[pred].capitalize(),
                        (faces_coord[i][0], faces_coord[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        clear_output(wait = True)
        draw_rectangle(frame, faces_coord) # rectangle around face
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("Face Recognition Model", frame) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


# In[19]:

#Close the thread
del webcam

