'''
    Author: Mahnoor Anjum
    Reference:
        https://gluon-cv.mxnet.io
'''
import mxnet as mx
import gluoncv
import cv2

### GET THE IMAGE ################################
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Video feed', frame)
    if cv2.waitKey(1)==13: 
        frame = frame = mx.nd.array(frame)
        break
cap.release()
cv2.destroyAllWindows()

# you may modify it to switch to another model. The name is case-insensitive
model_name = 'ResNet50_v1d'
# download and load the pre-trained model
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# load image
img = frame
# apply default data preprocessing
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
# run forward pass to obtain the predicted score for each class
pred = net(transformed_img)
# map predicted values to probability by softmax
prob = mx.nd.softmax(pred)[0].asnumpy()
# find the 5 class indices with the highest score
ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
# print the class name and predicted probability
print('The input picture is classified to be')
for i in range(5):
    print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))
    
