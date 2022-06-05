import cv2 as cv
import numpy as np
import random
import os

# Function that crops an image based on the location of the characters
def find_crop(img):
    top_most = img.shape[0] - 1
    bottom_most = 0
    left_most = img.shape[1] - 1
    right_most = 0

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if np.array_equal(img[row][col], [0, 0, 0]):
                if row < top_most:
                    top_most = row
                
                if row > bottom_most:
                    bottom_most = row

                if col < left_most:
                    left_most = col

                if col > right_most:
                    right_most = col

    return img[top_most:bottom_most, left_most:right_most]


# Mapping from integers to Hebrew character names. Used in 
# selecting a random Hebrew character.
index_to_char_map = {0 : 'Alef', 
            1 : 'Ayin', 
            2 : 'Bet', 
            3 : 'Dalet', 
            4 : 'Gimel', 
            5 : 'He', 
            6 : 'Het', 
            7 : 'Kaf', 
            8 : 'Kaf-final', 
            9 : 'Lamed', 
            10 : 'Mem', 
            11 : 'Mem-medial', 
            12 : 'Nun-final', 
            13 : 'Nun-medial', 
            14 : 'Pe', 
            15 : 'Pe-final', 
            16 : 'Qof', 
            17 : 'Resh', 
            18 : 'Samekh', 
            19 : 'Shin', 
            20 : 'Taw', 
            21 : 'Tet', 
            22 : 'Tsadi-final', 
            23 : 'Tsadi-medial', 
            24 : 'Waw', 
            25 : 'Yod', 
            26 : 'Zayin',
            27 : ' '} #space


# Mapping from Hebrew character names to encodings. Used
# in creating labels for each images.
char_map = {'Alef' : 'a', 
            'Ayin' : 'b', 
            'Bet' : 'c', 
            'Dalet' : 'd', 
            'Gimel' : 'e', 
            'He' : 'f', 
            'Het' : 'g', 
            'Kaf' : 'h', 
            'Kaf-final' : 'i', 
            'Lamed' : 'j', 
            'Mem' : 'k', 
            'Mem-medial' : 'l', 
            'Nun-final' : 'm', 
            'Nun-medial' : 'n', 
            'Pe' : 'o', 
            'Pe-final' : 'p', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : 't', 
            'Taw' : 'u', 
            'Tet' : 'v', 
            'Tsadi-final' : 'w', 
            'Tsadi-medial' : 'x', 
            'Waw' : 'y', 
            'Yod' : 'z', 
            'Zayin' : '+',
            ' ' : ' '} #space


main_dir_path = "path/to/image/data"
output_dir_path = "path/to/output/directory"


# Set the initial image dimensions in pixels
background_height = 200
background_width = 1200
channels = 3

# Set the lenght of a space in pixels
space_length = 50

# Set the number of training images to generate
number_of_training_images = 100

# Lists for gathering dimension information of the generated images
img_heights = []
img_widths = []

# Set the maximum label length
max_label_length = 13


for i in range(number_of_training_images):
    # Initialise the image
    print(f"Now creating image {i+1} out of {number_of_training_images}")
    new_training_image = np.zeros([background_height, background_width, channels])
    new_training_image.fill(255.0)
    label = ""
    y_offset = 50
    x_offset = random.randint(20, 70)
    

    # Randomly generate the structure of the label.
    # Randomly choose a number of splits/"words", 
    # between 1 and 3. Randomly choose the location(s)
    # of the splits, which then determines the "word" 
    # lengths. Finally, shuffle the ordering of the lengths.
    letters_left = max_label_length
    letters_per_word = []

    number_of_splits = random.randint(1, 3)
    letters_left -= number_of_splits

    split_points = []
    for i in range(number_of_splits):
        split_point = random.randint(1, letters_left - 1)
        while split_point in split_points:
            split_point = random.randint(1, letters_left - 1)
        
        split_points.append(split_point)

    split_points.sort()
    previous_split = 0
    for point in split_points:
        letters_per_word.append(point - previous_split)
        previous_split = point

    letters_per_word.append(letters_left - split_points[-1])
    print(letters_per_word)


    # One "word" in the "sentence" is only turned into a sequence
    # of spaces when there are at least 3 "words" (in order to not
    # turn first and last words into spaces). Additionally, only
    # for 30% of those "sentences" one "word" is turned into spaces.
    if len(letters_per_word) <= 2 or random.uniform(0, 1) > 0.3:
        for idx, word_length in enumerate(letters_per_word):
            for letter in range(word_length):
                # Select a random Hebrew character
                rand_int = random.randint(0, 26)
                char_name = index_to_char_map[rand_int]
                dir_path = main_dir_path + char_name + "/"
                # Select a random image of that character
                file_names = os.listdir(dir_path)
                char_filename = random.choice(file_names)
                char_img = cv.imread(dir_path + char_filename)

                # Check whether adding this image results in going outside
                # the background image.
                if x_offset + char_img.shape[1] >= new_training_image.shape[1]:
                    print("Ran out of space")
                    break

                # Add the character image, update the label, and update the location for the next character
                new_training_image[y_offset:y_offset+char_img.shape[0], x_offset:x_offset+char_img.shape[1]] = char_img
                label = label + char_map[char_name]
                x_offset += char_img.shape[1]

            # End of the "word" reached, add a space (unless this was the last "word")
            if idx < len(letters_per_word) - 1:
                label = label + " "
                x_offset += space_length


    else:
        # Randomly select the "word" that will be turned into a sequence of spaces
        space_word = random.randint(1, len(letters_per_word) - 2)
        print(f"space word is {space_word}")
        for idx, word_length in enumerate(letters_per_word):
            if idx != space_word:
                for letter in range(word_length):
                    # Same process as above
                    rand_int = random.randint(0, 26)
                    char_name = index_to_char_map[rand_int]
                    dir_path = main_dir_path + char_name + "/"
                    file_names = os.listdir(dir_path)
                    char_filename = random.choice(file_names)
                    char_img = cv.imread(dir_path + char_filename)
                    if x_offset + char_img.shape[1] >= new_training_image.shape[1]:
                        print("Ran out of space")
                        break

                    new_training_image[y_offset:y_offset+char_img.shape[0], x_offset:x_offset+char_img.shape[1]] = char_img
                    label = label + char_map[char_name]
                    x_offset += char_img.shape[1]

                if idx < len(letters_per_word) - 1:
                    label = label + " "
                    x_offset += space_length

            # Place a space for each character in the selected "space word"
            else:
                for letter in range(word_length):
                    label = label + " "
                    x_offset += space_length

                if idx < len(letters_per_word) - 1:
                    label = label + " "
                    x_offset += space_length



    print(label)
    print(len(label))

    # Crop the image, save dimension data, and save the image
    cropped_img = find_crop(new_training_image)
    img_heights.append(cropped_img.shape[0])
    img_widths.append(cropped_img.shape[1])
    cv.imwrite(output_dir_path + f'{label}.png', cropped_img)

# Display the median height and width
median_height = np.median(img_heights)
print(f"Median height: {median_height}")
median_width = np.median(img_widths)
print(f"Median width: {median_width}")

# Resize all generated images to the median height and width
file_names = os.listdir(output_dir_path)
cnt = 0
for name in file_names:
    print(f"Resizing image {cnt+1} out of {number_of_training_images}")
    img = cv.imread(output_dir_path + name)
    img_res = cv.resize(img, (int(median_width), int(median_height)))
    cv.imwrite(output_dir_path + name, img_res)
    cnt += 1