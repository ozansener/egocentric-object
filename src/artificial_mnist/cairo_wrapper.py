from PIL import Image, ImageDraw, ImageFont
import numpy as np


from gen_data.system_fonts import FONTS
# Declare font files and encodings


def draw_digit(digit_id, font_size, font, top_left):
    text = str(digit_id)
    # Load font using mapped encoding
    font = ImageFont.truetype(font, font_size)

    # Now draw the glyph
    img = Image.new('L', (28, 28), 'black')
    draw = ImageDraw.Draw(img)
    draw.text(top_left, text=text, font=font, fill=255)
    return img

# 29 fonts

idd = 0
for digit in range(10):
    for font_ in FONTS:
        for font_size in range(16, 28):
            for top_l_x in range(-3, 15):
                for top_l_y in range(-8, 2):
                    if np.random.random() > 0.998:
                        try:
                            img = draw_digit(digit,
                                         font_size,
                                         font_,
                                         (top_l_x, top_l_y))
                            idd += 1
                            file_name = './out/image%d.png' % (idd)
                            img.save(file_name)
                        except:
                            print "Failed to read", font_
