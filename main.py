from color import Color


image = Color("imgs/test5.jpg")
color = image.main_colors()
hex = image.hex(color)
print (hex)

