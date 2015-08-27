from PIL import Image, ImageDraw, ImageFont
import numpy as np
import itertools
import os

from system_fonts import FONTS
# Declare font files and encodings

class SamplerError(Exception):
    pass

class ArtificialMnist:
    def __init__(self):
        self.font_list = FONTS
        num_fonts = len(FONTS)
        digits = range(10)
        num_digits = len(digits)
        num_rotations = 20
        rotations = list(np.linspace(-50, 50, num_rotations))
        top_pos = list(itertools.product(range(-3, 6), range(-8, 2)))
        num_top_pos = len(top_pos)
        font_size = range(18, 28)
        num_font_size = len(font_size)

        self.value_list = [FONTS,
                           rotations,
                           digits,
                           font_size,
                           top_pos]

        self.sampling_space = [num_fonts,
                               num_rotations,
                               num_digits,
                               num_font_size,
                               num_top_pos]

        print FONTS
        print("Sampling Space Dimension is",
              reduce(lambda x, y: x*y, self.sampling_space, 1))

    @staticmethod
    def draw_digit(font, rotation, digit_id, font_size, top_left):
        text = str(digit_id)
        # Load font using mapped encoding
        font = ImageFont.truetype('/home/ozan/.fonts/'+font, font_size)

        # Now draw the glyph
        img = Image.new('L', (28, 28), 'black')
        draw = ImageDraw.Draw(img)
        draw.text(top_left, text=text, font=font, fill=255)
        rotated_image = img.rotate(rotation, expand=1)
        return rotated_image.resize((32, 32))

    def get_n_random_samples(self, num_samples, directory):
        if not os.path.exists(directory):
            raise SamplerError(
                "Directory %d does not exist" % (directory))

        ground_truth_labels = []

        # Sample randomly
        for image_id in range(num_samples):
            selected_param = [np.random.random_integers(0, high-1)
                              for high in self.sampling_space]

            selected_values = [self.value_list[i][selected_param[i]]
                               for i in range(len(self.sampling_space))]
            img = self.draw_digit(*selected_values)
            file_name = directory+'image%d.png' % (image_id)
            img.save(file_name)
            ground_truth_labels.append(('image%d.png'%(image_id),
                                        selected_values[2]))

        with open(directory+ 'ground_truth.txt', 'w') as gt_file:
            for label in ground_truth_labels:
                gt_file.write(str(label[0])+':'+str(label[1])+'\n')

