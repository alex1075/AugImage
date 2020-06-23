import numpy as np

def update_image(self):
    if self.kwargs["how"] == "vertical":
        img = np.flipud(self.img)
    elif self.kwargs["how"] == "horizontal":
        img = np.fliplr(self.img)
    return img

if self.kwargs["how"] == "vertical":
    for box in self.annotations["boxes"]:
    name = box['label']
    # unchanged
    width = box["width"]
    # unchanged
    height = box["height"]
    # check if annotation is on bottom of image
    if box["y"] < img_height/2:
        # new Y position is addition of center and diff of
        # (center and old y)
        nY = img_height/2 + (img_height/2 - box["y"])
    # else annotation is on right side of image
    elif box["y"] >= img_height/2:
        # new Y position is center  diff between old x and center
        nY = img_height/2 - (box["y"] - img_height/2)
    # unchanged
    nX = box["x"]
    # create new boxes for each label
    new_box.append({
    "label": name,
    "x": nX,
    "y": nY,
    "width": width,
    "height": height
})