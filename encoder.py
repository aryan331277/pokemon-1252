from tensorflow.keras import layers, Model, Sequential

LATENT_DIM = 100  

encoder = Sequential(name="encoder")
encoder.add(layers.Conv2D(64, (4, 4), strides=2, padding="same", activation="relu", input_shape=(256, 256, 3)))  # 128x128
encoder.add(layers.Conv2D(128, (4, 4), strides=2, padding="same", activation="relu"))  # 64x64
encoder.add(layers.Conv2D(256, (4, 4), strides=2, padding="same", activation="relu"))  # 32x32
encoder.add(layers.Conv2D(512, (4, 4), strides=2, padding="same", activation="relu"))  # 16x16
encoder.add(layers.Flatten())

encoder.add(layers.Dense(LATENT_DIM, name="z_mean"))  
encoder.add(layers.Dense(LATENT_DIM, name="z_log_var"))  

encoder.summary()


