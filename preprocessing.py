from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (256, 256)  # Set a fixed size

train_images = []
for path in tqdm(image_paths):
    img = load_img(path, target_size=IMAGE_SIZE)  # Resize images
    img = img_to_array(img)  # Convert to NumPy array
    train_images.append(img)
    
train_images = np.array(train_images)
train_images=(train_image/255
