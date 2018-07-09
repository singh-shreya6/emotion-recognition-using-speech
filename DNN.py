import librosa
import os
import numpy
import scipy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
numpy.seterr(all='warn')

emotion_class_map = {'a' : 0, 'h' : 1, 'n' : 2, 's' : 3}
em = {0 : "ang", 1 : "hap", 2 : "neu", 3 : "sad"}
a_type=["improv","scripted"]                # Type of audios for TRAIN DATA
a_size=["long","medium","short"]            # Length of audios for TRAIN DATA
e_type=["ang","hap","neu","sad"]            # Emotions type for TRAIN DATA

a_type_test=["improv","scripted"]           # Type of audios for TEST DATA
a_size_test=["long","medium","short"]       # Length of audios for TEST DATA
e_type_test=["ang","hap","neu","sad"]       # Emotions type for TEST DATA
num_emotions = len(emotion_class_map)

def getSegment(frames, start, end):
	rows = frames[start : end+1, ]
	return numpy.hstack(rows)
	

def read_audios(folder_path,emotions,audios,aud_type,aud_size,emo_type):
    for i in aud_type:
        for j in aud_size:
            for k in emo_type:
                emotion = k[0]
                f_path=folder_path+i+"/"+j+"/"+k+"/"
                for filename in os.listdir(f_path):
                    print(filename)
                    filepath = f_path + filename
                    if filepath is not None:
                        y, sr =librosa.load(filepath)
                        emotion_class = emotion_class_map[emotion]
                        emotions.append(emotion_class)
                        audios.append(y)						
        
def process_data(audios,emotions,emo,file):
    X = numpy.empty((0, featuresPerSegment))
    Y = numpy.empty((0, num_emotions))
    for i in range(len(audios)):
        print(str(em[emotions[i]]) + " " + str(i))
        audio = audios[i]
        output = emotions[i]
        output_vec = numpy.zeros((1, num_emotions))
        output_vec[0][output] = 1
        frames = numpy.empty((featuresPerFrame, ))
        start = 0
        countf =0
        while True:
            end = start + frameSize
            countf = countf+1
            frame = numpy.zeros((frameSize,))
            if end >= len(audio) :
                for j in range(start,len(audio)):
                    frame[j-start]=audio[j]
                for j in range(len(audio),end):
                    frame[j-start]=0.0
                mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
                frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
                if numpy.isnan(mfcc_coeffs).any() :
                    print("nan\nnan\n")
                    exit()
                frames = numpy.vstack([frames,frame_features])
                break
            for j in range(start,end):
                frame[j-start]=audio[j]
            start = start + hopSize
            mfcc_coeffs = librosa.feature.mfcc(y=frame,sr=sample_rate,n_mfcc=13)
            frame_features = numpy.append(mfcc_coeffs.mean(axis=1), [])
            if numpy.isnan(mfcc_coeffs).any() :
                print("nan\nnan\n")
                exit()
            frames = numpy.vstack([frames,frame_features])
        start_segment = 0
        count = 0
        while True:
            end_segment = start_segment + framesPerSegment - 1 
            if end_segment >= len(frames) :
                break
            count = count +1
            segment = getSegment(frames, start_segment, end_segment)
            start_segment = start_segment + segmentHop -1
            X = numpy.vstack([X, segment])
            emo.append(em[output])
            Y = numpy.vstack([Y, output_vec])
    X_scaled = preprocessing.scale(X)
    scipy.io.savemat('X_'+file+'.mat', {'X' : X})
    scipy.io.savemat('Y_'+file+'.mat', {'Y' : Y})
    scipy.io.savemat('X_'+file+'_scaled.mat', {'X_scaled' : X_scaled})   
    data = scipy.io.loadmat("X_"+file+"_scaled.mat")
    for i in data:
        if '__' not in i and 'readme' not in i:
            numpy.savetxt(("file_"+file+".csv"),data[i],delimiter=',')
        
        
folder_path = "F:/segdata/"
folder_path_test = "F:/segdata/"
emotions=[]
emotions_test=[]
audios=[]
audios_test=[]
emo = []
emo_test = []
read_audios(folder_path,emotions,audios,a_type,a_size,e_type)
read_audios(folder_path_test,emotions_test,audios_test,a_type_test,a_size_test,e_type_test)  


sample_rate = 16000 # in hertz
frameDuration = 0.025 #duration in seconds
hopDuration = 0.010
frameSize = int(sample_rate*frameDuration)
hopSize = int(sample_rate*hopDuration)
featuresPerFrame = 13
framesPerSegment = 25
featuresPerSegment = featuresPerFrame * framesPerSegment
segmentHop = 13


print("Processing train dataset")
file="train"
process_data(audios,emotions,emo,file)
file="test"
process_data(audios_test,emotions_test,emo_test,file)


print("Building NN")

seed = 7
numpy.random.seed(seed)   #run the same code again and again and get the same result.
    
def label_encoder(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
 
def create_model():
    model = Sequential()
    model.add(Dense(15, input_dim=325, activation="relu", kernel_initializer="normal"))
    model.add(Dense(15, activation="relu", kernel_initializer="normal"))
    model.add(Dense(15, activation="relu", kernel_initializer="normal"))
    model.add(Dense(4, activation="softmax", kernel_initializer="normal"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model



filename = 'file_train.csv'
filename_test = 'file_test.csv'
print("reached")
dataframe = pandas.read_csv(filename, header = None)
print()
dataframe[len(dataframe.columns)]=emo

dataset = dataframe.values
    
dataframe_test = pandas.read_csv(filename_test, header = None)

dataframe_test[len(dataframe_test.columns)]=emo_test

dataset_test = dataframe_test.values

n_inputs = len(dataset[0]) -1
n_outputs = len(set([row[-1] for row in dataset]))

n_inputs_test = len(dataset_test[0]) -1
n_outputs_test = len(set([row[-1] for row in dataset_test]))
X = dataset[:,0:n_inputs].astype(float)
Y = dataset[:,n_inputs]
Y = numpy.append(Y,"ang")
Y = numpy.append(Y,"hap")
Y = numpy.append(Y,"neu")
Y = numpy.append(Y,"sad")
X_test = dataset_test[:,0:n_inputs_test].astype(float)
Y_test = dataset_test[:,n_inputs_test]
Y_test = numpy.append(Y_test,"ang")
Y_test = numpy.append(Y_test,"hap")
Y_test = numpy.append(Y_test,"neu")
Y_test = numpy.append(Y_test,"sad")
dummy_y = label_encoder(Y)
dummy_y_test = label_encoder(Y_test)
l = len(dummy_y)-4
dummy_y = dummy_y[0:l]
l = len(dummy_y_test)-4
dummy_y_test = dummy_y_test[0:l]
model = create_model()
model.fit(X, dummy_y, batch_size = 5, epochs=20)
score, acc = model.evaluate(X_test, dummy_y_test,batch_size=5)
print("Accuracy: "+str(acc*100))






