'''
    Author: Mahnoor Anjum
    Reference:
        https://gluon-cv.mxnet.io
'''
import mxnet as mx
import gluoncv
import cv2
import time

### GET THE IMAGE ################################
def get_model(model):
    model_name = model
    # download and load the pre-trained model
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    
    return net 

def get_pred(model, frame):
    # you may modify it to switch to another model. The name is case-insensitive
    
    net = get_model(model)
    frame = mx.nd.array(frame)
    # load image
    img = frame
    # apply default data preprocessing
    transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
    # run forward pass to obtain the predicted score for each class
    pred = net(transformed_img)
    # map predicted values to probability by softmax
    prob = mx.nd.softmax(pred)[0].asnumpy()
    # find the 5 class indices with the highest score
    ind = mx.nd.topk(pred, k=1)[0].astype('int').asnumpy().tolist()
    return net.classes[ind[0]], prob[ind[0]]

y = 100
h = 250
x = 100
w = 500
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
while True:
    time.sleep(1)
    ret, frame = cap.read()
    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    frame = cv2.flip(frame, 1)
    cropped = frame[y:y+h, x:x+w]
    className, prob = get_pred('ResNet50_v1d', frame)
    text = str(className) + " ( Probability: " + str(prob) + " )"
    cv2.putText(frame, text, (10,450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video feed', frame)
    if cv2.waitKey(1)==13: 
        break
cap.release()
cv2.destroyAllWindows()




    
