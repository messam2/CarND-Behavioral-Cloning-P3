import csv
import cv2
import numpy as np
from docutils.nodes import image


def generate_data(paths, split_str, flip=False):
    global correction
    correction = 0.2  # this is a parameter to tune
    images = []
    measurements = []
    print('Reading Images')

    for path in paths:
        csv_path = path + 'driving_log.csv'
        lines = []

        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for i, line in enumerate(reader):
                if i > 0:
                    lines.append(line)

        for line in lines:
            for i in range(3):
                source_path = line[i]
                file_name = source_path.split(split_str)[-1]
                current_path = path + 'IMG/' + file_name
                image = cv2.imread(current_path)
                images.append(image)
                if i == 0:
                    measurement = float(line[3])
                elif i == 1:
                    measurement = float(line[3]) + correction
                elif i == 2:
                    measurement = float(line[3]) - correction

                measurements.append(measurement)

                if flip:
                    new_image = cv2.flip(image, 1)
                    new_measurement = measurement * (-1)
                    images.append(new_image)
                    measurements.append(new_measurement)
        del lines

    print('image size: ',image.shape)
    print('Converting Images as np array')
    X_train = np.array(images)
    del images
    y_train = np.array(measurements)
    del measurements

    print('Done with Images')
    return X_train, y_train


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Dropout

def simple_net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))

    return model

def le_net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    model.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid", activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(120, activation='elu'))

    model.add(Dropout(0.5))
    model.add(Dense(84, activation='elu'))

    model.add(Dense(10, activation='elu'))

    model.add(Dense(1))

    return model

if __name__ == "__main__":
    paths, split_str = ['../data/'], '/'
    # paths, split_str = ['../track1/3labs_center/', '../track1/1lab_recovery/', '../track1/1lab_smothcurve/', '../track1/2labs_CC/'], '\\'
    # paths, split_str = ['../track2/3labs_center/', '../track2/1lab_recovery/', '../track2/1lab_smothcurve/', '../track2/2labs_CC/'], '\\'
    paths, split_str = ['../track1/3labs_center/', '../track1/1lab_recovery/'], '\\'

    epochs = 10
    batch_size = 64
    # model_name = 'first_try'
    model_name = 'lenet'

    X_train, y_train = generate_data(paths=paths, split_str=split_str, flip=True)
    print("X size: ", len(X_train))

    # model = simple_net()
    model = le_net()

    model.summary()

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=batch_size)

    save_str = 'models/' + model_name + '_e' + str(epochs) + '_b' + str(batch_size) +'.h5'
    model.save(save_str)

    print(save_str)

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    from matplotlib import pyplot as plt
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()