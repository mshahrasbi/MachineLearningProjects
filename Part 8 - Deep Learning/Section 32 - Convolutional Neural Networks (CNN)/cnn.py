# Convolutional Neural Network

# CNN is just a ANN on which you use convolution trick to add some convolutional layers. And why we are using this convolution trick?
# it is to preserve the spatial structure in images and therefore to be able to classify some images.
# So here we are going to solve an image classification problem where our goal will be to classify some images and tell for each image
# the class of the image. And so we are going to work on a very simple image classification problem, we will have some images of cats and
# dogs and we will train a CNN to predict if the image is a dog or a cat.We will have a folder of full of images of cats and dogs, but once
# we build our CNN model you will simply need to change the images of cats and dogs in the folder and replace them by the images you want
# to work with. for example you can replace these cts and dogs images with any other images that you want to work with.Example, if you are 
# working with brain images and you have to predict if this brain image contains a tumor or not well provided you know the answer is of 
# many observations like 1000's of oberservations then you will be able to train CNN to predict if some new brain image contains a tumor yes
# or no. 
# in this project instead of having csv file as dataset we have folder structure as dataset. This folder structure is strutured the way to make our
# life easier as well as CNN model. Lets think about it how can we make a training setand a test set where are the independent variables are now 
# the pixels distributed in 3D arrays and therefore not distributed like in the prevouse data where we have our independent variables and columns 
# next to the final column that contained the dependent variable, here since our dataset no longer has the structure where the rows are the observations
# and the columns are independent variables and the depenedent varaible next to each other Then we cannot add explicitly the dependent variable in our 
# dataset because it wouldn't make much sense to add this dependent variable column along the three arrays representing the images. And you know when we train 
# a machine running model we always need the dependent variable to have the real results that are required to understand correlations between the information
# contained in the independent variables and the real result contained in the dependent variable. But here since we cannot add this dependent variable column 
# in the same table how can we extract the info of this dependent variable? Well we have several solutions:
#
# A classic solution is to only have a dataset containing our images separated into 2 different folders for the training set and folder for the test set, and then
# you know since each of the images has a name, the name of the jpg or png file, well you know what we can do and that is the first solution is to name each
# of these images by the category of the image that is Cat or Dog, then we can write some kind of code to extract the label name Cat or Dog from the name of the 
# image file to specify to the algorithm whether this image belongs to the class Cat or belongs to the class Dog. And in some way we get our dependent variable 
# vector because we can fill in this dependent variable vector with the label names that we manage to extract from the image file names of all our images.
#
# A better solution comes with Karas, it has tools to import images in a very efficient way. To import the images with Karas what we only need to do is 
# to prepare a special structure for our dataset:
# dataset
#   test_set
#       cats
#       dogs
#   training_set
#       cats
#       dogs 

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages

# this Sequential will use to initialize our NN, there were 2 ways of initializing a NN either as a sequence
# of layers or as a graph. And since the CNN is still a sequence of layers so will use Sequentail package
from keras.models import Sequential

# This package is used for the first step of making the CNN, and since our images are 2D we will use Convolution 2D
# package
from keras.layers import Conv2D

# this package is used for step 2 pooling layers
from keras.layers import MaxPooling2D

# this package is used for step 3, flatting we convert all the pooled future maps that we crated through convolution
# and Max pooling into large feature vector then becomes an input of our full connected layers
from keras.layers import Flatten

# This package to add fully connected layers 
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# we have an input image (in our case the input image will be cat or dog) we convert this image into a table with pixel values
# and we got the 0 or 1 pixels. and so this convolution step consists of applying serveral feature detectors on this image. we 
# will pass each feature detectors from right to left and up to down and we will make the feature map from each feature detectors.
# so feature map contains some numbers and the highest numbers of the feature map is where the feature detector could detect a 
# specific feature in the input image.
# so we do this with many feature detectors, we will choose in this first convolution step the number of feature detectors that we
# create and therefore the number of feature maps beacuse we get as many feature maps as feature detectors we use to detect some 
# specific features in the input image and therefore eventually we get our first step convolution done as soon as we create these
# many feature maps that will form our convolutional layer and that is exactly what we are going to build here.

# conv2D, number of fileters: 32, and each filter is 3X3, input shape (shape of input image), making sure all images are going to 
# be 64X64X3 (3 channel of 64X64) it is a 3D array, and activation function is relu so normalize the -ve pixels 
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# poolinf steps is just reducing the size of your feature maps and how do we do that, we get a 2X2 2D array and slid it through 
# feature map and each time we take the maximum of the four cells inside this 2D array. So taking the max is called Max pooling
# and we slide a square table with this trial of two not with a stride of one, we use stride of two and therefore since each time 
# we take the max of a 2X2 table we will end up with a new feature map with reduced size, more precisely the size of the original
# feature map is divided by two when we apply Max pooling on each of our feature maps and then we obtain our next layer composed of
# all these reduced feature maps and that is called the pooling layer.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# taking all our pooled feature maps and put them into one single vector. This is going to be a huge vector because even if we reduced the 
# size of the feature map by divided by two, we still have many pooled feature maps. To summerize we have our feature maps in the pooling
# layer, we apply the flatten step and we get this huge single vector that contains all the different cells of all the different feature maps
# and this single vector is going to be the input layer of a feature ANN that we know from the previous section that is a classic ANN with
# fully connected layers, now we can ask 2 questions:
#  1- why don't we lose the spatial structure by flattening all these feature maps into one same single vector> beacuse by creating our
#       feature maps we extracted the spatial structure informations by getting these high numbers in the feature maps, so these high numbers
#       represent the spatial structure of our images because the high numbers in the feature maps are associated to a specific feature in the
#       input image.
#  2- why didn't we directly take all the pixels of the input image and flatten them into this one same single vector? Because if we directly 
#       flatten the input image pixles into this huge single 1D vector, then each node of this huge vector will represent one pixel of the
#       image independently of the pixles that are around it, so we only get info of the pixel itself and we don't get infos of how this pixel
#       especially connected to the other pixels around it. so basiclly we dont get any info of the spatial structure around this pixel
classifier.add(Flatten())

# Step 4 - Full connection
# It is basically consists of making a classic ANN composed of some fully connected layers, and so why do we need to finish by this, well that is 
# because we managed to convert our input image into this 1D vector that contains some informations of the spatial structure or of some pixel pattern 
# in the image and now what we have to do course is to use this input vector as the layer of classic ANN because ANN as we saw it can be a great 
# classifier for nonlinear problems and since image classifications a nonlinear problem well it will make a perfect job here to classify the images 
# and tell that each image is the image of cat or dog

# here we use the Dense() function for fully connected layer. for unites we choose we pick a power 2, we will pick 128 hidden nodes in the hidden layer.
# activation function we use rectifier function relu 
classifier.add(Dense(units = 128, activation = 'relu'))
# the last layer that we need to add now is the output, and to add this output layer efficiently, we change the activation function beacuse we are not 
# using the rectifier activation function to return the probabilities of each class, now it is going to be the sigmoid function becuase we have binary 
# outcome cat or dog and if had an outcome with more than two categories we will need to use the soft Max activation function. but here we have binary 
# coutcome so it is the sigmoid activation function. 
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images

# we will actually do it in one step because we are going to use a shortcut which is going to be very pratical, it is the Keras lib. Image augmentation
# that basically consists of pre-processing your images to prevent overfitting because what will happen then we will get the results in here but if we 
# don't do this image augmentation well what we might get is a freat accuracy result on the training set but a much lower accuracy under test set. So that
# is the exactly overfitting that corresponds to this particular situation where you get great results on your training sets and your test sets due to a
# new refits on the training set. So before filling our CNN to our images lets proceed to this image augmentation process. refer to Keras Documention online,
# So what is image augmentation and how will it prevent overfitting? 
# well we know that one of the situations that lead to overfitting is when we have new data to train our model in that situation our model find some 
# correlations in the few observations of the training set but fails to generalize these correlations on some new observations and when it comes to 
# images we actually need a lot of images to find and generalize some correltaions because in computer vision or ML model doesn't simply need to find
# some correlations between some independent variables and some dependent varaibles. it needs to find some patterns in the pixels and to do this it
# requires a lot of images. 
# right now we are working with 10000 images, 8000 images on the training sets and that is actually not much to get some great performance results.we either
# need some more images or we can use a trick, and that is where data augmentation comes into play. That is the trick because what it will do, is it will 
# create many batches of our images and in each batch it will apply some random transformations on a random selection of our image just like rotating them,
# flipping them, shifting them, even shearing them. And eventually what we will get during the training is many more diverse images inside these batches and 
# therefore a lot more material to train. 
# So all this image augmentation trick can only reduce overfitting, so in summary image augmentation is a technique that allows us to enrich our data sets
# our training set without adding more images and therefore that allows us to get good performance results with little or no overfitting even with small 
# amount of images 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 2, #25,
                         validation_data = test_set,
                         validation_steps = 2000)

# the training is over. we obtained an accuracy of 84% for the traing set and 75% for the test set.
# this is not too good . So first we obtained this 84% accuracy on the training set That is not bad but
# that is not what we are mostly interested in. we are interested in accuracy of the test set which is 
# 75% and the difference of the training set and the accuracy of the test set to assess whether there is
# overfitting or not. So 75% on Test is not bad, it means we have 3 correct predications out of 4, but 
# then we get quite a large difference between the accuracy on the training set and the accuracy on test set
# So there is a lot of room for improvement. and can achieve an accuracy of more than 80% on the test, without
# doing any parameter tuning. the soluion is to make it deeper deep learning model that is deeper convolutional
# neural network and how can we make it deeper, well we have two options:
#   1 - add another convolutional layer.
#   2 - is to add another fully connected layer
# so the best solution is to add a convolutional layer.
 
# you can still improve this model by adding more convolutional layers will help get an even better accuracy.
# But if we want to get better accuracy well that would be to choose a higher target size here for your images 
# of the training set so that you get more infos of your pixel patterns, because indeed if you increase the 
# size of your images that is the size down to which all your images will be resized, well you will get lots  
# more pixels in the rows and lot more pixels in the columns in your input images and therefore you will have more 
# information totake on the pixels, but better you try this with using GPU.


# Save model
classifier.save('64x3-cnn.model')

from keras.models import model_from_json
classifier_json = classifier.to_json()
with open('64x3-cnn.json', 'w') as json_file:
    json_file.write(classifier_json)

classifier.save_weights('64x3-cnn.h5')
    

##########################################################


# load json and create model
json_file = open('64x3-cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("64x3-cnn.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


import cv2
CATEGORIES = ['Dog', 'Cat']
def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

img = [prepare("Dog.jpg")]

prediction = loaded_model.predict(img)
print(prediction)
print(CATEGORIES[int(prediction[0][0])])











