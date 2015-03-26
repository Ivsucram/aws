import numpy as np
import os
from PIL import Image

def elastic_distortion(input, x):
        image = Image.fromarray(np.array(np.split(np.array(input)*255,28)))
        image = image.rotate(np.random.randint(-30,30))
        x = (x % 10)*2+10
        image = image.transform((x,x), Image.EXTENT, [np.random.randint(0, 4), np.random.randint(0, 4), 28, 28])
        image = image.resize((28,28))
        return np.array(list(image.getdata()))/255
    
def create_distortion_dataset():
    dataset = 'data/mnist_batches.npz'
    epoch = len(os.walk('distort').next()[2]) + 1
    epoch = epoch if epoch <= 3 else np.random.randint(1,3)
    
    mnist = np.load(dataset)
    train_set_x = mnist['train_data']
    mnist = None
    for i in xrange(len(train_set_x)):
        train_set_x[i] = elastic_distortion(input=train_set_x[i], x=np.random.choice(9, 1, p=[0.03, 0.03, 0.04, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])[0])
        #train_set_x[i] = elastic_distortion(input=train_set_x[i], x=epoch)
        print "image" + str(i)
    
    np.save("distort/" + str(epoch), train_set_x)
    print epoch
    epoch = epoch + 1 if epoch < 3 else 1
    #Image.fromarray(np.array(np.split(np.array(train_set_x[np.random.randint(1,len(train_set_x))])*255,28))).show()

if __name__ == '__main__':
	create_distortion_dataset()