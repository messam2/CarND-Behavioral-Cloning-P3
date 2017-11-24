import csv
import random
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout

def generate_data(paths, split_str, flip=False, use_side_cameras=False, correction=0.2):
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
            if use_side_cameras:
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
                else:
                    source_path = line[0]
                    file_name = source_path.split(split_str)[-1]
                    current_path = path + 'IMG/' + file_name
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])
                    measurements.append(measurement)

                    if flip:
                        new_image = cv2.flip(image, 1)
                        new_measurement = measurement * (-1)
                        images.append(new_image)
                        measurements.append(new_measurement)

        del lines

    print("Testing data number:", len(images))

    print('image size: ', image.shape)
    print('Converting Images as np array')
    X_train = np.array(images)
    del images
    y_train = np.array(measurements)
    del measurements

    print('Done with Images')
    return X_train, y_train


def generate_samples(paths):
    samples = []
    for path in paths:
        csv_path = path + 'driving_log.csv'
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    print("Testing data number:", len(samples))

    return samples


def generator(samples, split_str, flip=False,batch_size_=64, use_side_cameras=False, correction=0.2):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size_):
            batch_samples = samples[offset:offset + batch_size_]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if use_side_cameras:
                    for i in range(0,3):
                        source_path = batch_sample[i]
                        current_path = '..' + source_path.split(split_str)[-1].replace('\\', '/')
                        # current_path = '../data/' + source_path.split(split_str)[-1]
                        image = cv2.imread(current_path)
                        if i == 0:
                            angle = float(batch_sample[3])
                        elif i == 1:
                            angle = float(batch_sample[3]) + correction
                        elif i == 2:
                            angle = float(batch_sample[3]) - correction

                        images.append(image)
                        angles.append(angle)
                else:
                    source_path = batch_sample[0]
                    current_path = '..' + source_path.split(split_str)[-1].replace('\\','/')
                    image = cv2.imread(current_path)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def simple_net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    return model, 'simple'


def le_net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model, 'leNet'


def nvidia_net():
    model = Sequential()
    model.add(Cropping2D(cropping=((40, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model, 'nVidea'


if __name__ == "__main__":
    use_generator = True
    use_side_cameras = True
    flip = False

    correction = 0.2  # this is a parameter to tune
    epochs = 30
    batch_size = 8
    if use_side_cameras:
        real_batch_size = batch_size * 3
    else:
        real_batch_size = batch_size

    # model, model_name = simple_net()
    # model, model_name = le_net()
    model, model_name = nvidia_net()
    model.summary()

    print("Using generator: ", use_generator)
    print("Using side cameras: ", use_side_cameras)
    print("Flipping images: ", flip)
    print("Using", model_name," model")
    print("Epochs: ", epochs)
    print("Batch size: ", real_batch_size)

    if use_generator:
        # paths, split_str = ['../track1/1lab_center/', '../track1/curves_recovery/'] , '07_Behavioral_Cloning'
        paths, split_str = ['../track1/data/'] , '07_Behavioral_Cloning'
        samples = generate_samples(paths)
        train_samples, validation_samples = train_test_split(samples, test_size=0.15)
        if use_side_cameras:
            train_samples_len = len(train_samples) * 3
            validation_samples_len = len(validation_samples) * 3
        else:
            train_samples_len = len(train_samples)
            validation_samples_len = len(validation_samples)

        train_generator = generator(train_samples, split_str=split_str, batch_size_=batch_size,
                                    use_side_cameras=use_side_cameras, correction=correction)
        validation_generator = generator(validation_samples, split_str=split_str, batch_size_=batch_size,
                                         use_side_cameras=use_side_cameras, correction=correction)

        model.compile(loss='mse', optimizer='adam')

        history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=train_samples_len / real_batch_size,
                                         validation_data=validation_generator,
                                         nb_val_samples= validation_samples_len / real_batch_size,
                                         epochs=epochs)
    else:
        # paths, split_str = ['../data/'], '/'
        # paths, split_str = ['../track1/2labs_center/', '../track1/1lab_recovery/', '../track1/1lab_smothcurve/', '../track1/2labs_CC/'], '\\'
        # paths, split_str = ['../track2/2labs_center/', '../track2/1lab_recovery/', '../track2/1lab_smothcurve/', '../track2/2labs_CC/'], '\\'
        paths, split_str = ['../track1/2labs_center/', '../track1/1lab_recovery/'], '\\'
        X_train, y_train = generate_data(paths=paths, split_str=split_str, flip=flip, use_side_cameras=use_side_cameras, correction=correction)

        model.compile(loss='mse', optimizer='adam')

        history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=real_batch_size)


    save_str = 'models/' + model_name + '_e' + str(epochs) + '_b' + str(real_batch_size) + '.h5'
    model.save(save_str)
    print(save_str)

    # plot the training and validation loss for each epoch
    from matplotlib import pyplot as plt

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
