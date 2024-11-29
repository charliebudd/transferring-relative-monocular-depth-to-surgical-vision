import cv2

def rectify(left_image, right_image, calibration):
    image_size, mapL1, mapL2, mapR1, mapR2 = calibration
    y = (left_image.shape[0] - image_size[1]) // 2
    x = (left_image.shape[1] - image_size[0]) // 2
    left_image = left_image[y:y+image_size[1], x:x+image_size[0]]
    right_image = right_image[y:y+image_size[1], x:x+image_size[0]]
    left_rectified = cv2.remap(left_image, mapL1, mapL2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, mapR1, mapR2, cv2.INTER_LINEAR)
    return left_rectified, right_rectified
