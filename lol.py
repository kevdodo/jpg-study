import ps_lib as ps

if __name__ == '__main__':
    img_path = "target-image.png"
    img = ps.read_image(img_path)
    ps.write_image("cat_jpeg_PIL.jpg", img)
    # ps.write_image("head.png", dog[0:200,100:400])
    print("yuh")