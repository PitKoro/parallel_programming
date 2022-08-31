from PIL import Image


def bin_to_image(src, dest):
    with open(src, 'rb') as f:
        arr = f.read()

    w = int.from_bytes(arr[:4], byteorder='little')
    h = int.from_bytes(arr[4:8], byteorder='little')

    img = Image.frombytes('RGBA', (w, h), arr[8:], 'raw')
    img.save(dest, 'PNG')


def image_to_bin(src, dest):
    res = b''

    with Image.open(src) as img:
        w, h = map(lambda x: x.to_bytes(4, byteorder='little'), img.size)
        res += w + h

        for pixel in img.convert('RGBA').getdata():
            r, g, b, alpha = map(lambda x: x.to_bytes(1, byteorder='little'), pixel)
            res += r + g + b + alpha

    with open(dest, 'wb') as f:
        f.write(res)