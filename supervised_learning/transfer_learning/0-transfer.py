#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset
using Transfer Learning with Keras Applications.
"""
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR 10 dataset for the ResNet50 model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data.
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels.

    Returns:
        X_p: preprocessed X
        Y_p: preprocessed Y
    """
    # ResNet50 üçün nəzərdə tutulmuş xüsusi giriş emalı (preprocess) funksiyası
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))

    # Etiketləri birbaşa rəqəmdən (0-9) one-hot vektora çeviririk (10 klass üçün)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


def train_model():
    """
    Builds, compiles, and trains the transfer learning model on CIFAR-10,
    then saves it to disk as 'cifar10.h5'.
    """
    # 1. Dataseti yükləyirik
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()

    # 2. Datamızı preprocess edirik
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)

    # 3. Modelimizin giriş formasını təyin edirik (CIFAR-10: 32x32x3)
    inputs = K.Input(shape=(32, 32, 3))

    # --- HINT 2: Lambda Layer ilə ölçünü ResNet50-nin sevdiyi 224x224-ə qaldırırıq
    # Bu addım balaca şəkli modelə verməzdən əvvəl böyüdür (Upscaling)
    resized_inputs = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image,
            height_factor=7,
            width_factor=7,
            data_format='channels_last',
            interpolation='bilinear'
        )
    )(inputs)

    # 4. Əsas hazır modeli (bədəni) yükləyirik. include_top=False başlığı atır.
    base_model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=resized_inputs
    )

    # 5. --- HINT 3: Hazır olan bu nəhəng bədənin qatlarını dondururuq (Freeze)
    # Kompüter hər epoch-da bura boşuna güc sərf etməyəcək
    base_model.trainable = False

    # 6. Modelin çıxışına öz "başlığımızı" (classification head) qoyuruq
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(units=256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)  # Overfitting-in qarşısını almaq üçün

    # Sənin bayaq dediyin o Softmax-lı balaca yekun qat:
    outputs = K.layers.Dense(units=10, activation='softmax')(x)

    # 7. Tam funksional modeli yaradırıq
    model = K.models.Model(inputs=inputs, outputs=outputs)

    # 8. Modeli compile edirik
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 9. Modeli öyrədirik (Məşq)
    # Batch size böyük seçmək və transfer learning dondurulduğu üçün proses sürətli gedəcək
    model.fit(
        X_train_p, Y_train_p,
        batch_size=128,
        epochs=5,  # 5 epoch adətən 87% barierini keçmək üçün tam bəs edir
        validation_data=(X_val_p, Y_val_p),
        verbose=1
    )

    # 10. Modeli cari qovluğa cifar10.h5 adı ilə qeyd edirik
    model.save('cifar10.h5')


if __name__ == '__main__':
    train_model()
