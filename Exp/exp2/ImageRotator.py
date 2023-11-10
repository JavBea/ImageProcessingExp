import cv2


class ImageRotator:
    def __init__(self):
        pass

    def rotate_image(self, input_path, output_path, rotation_angle):
        # 读取输入图像
        image = cv2.imread(input_path)

        # 检查输入的旋转角度，顺时针旋转90度、180度或270度
        if rotation_angle == 90:
            rotated_image = cv2.transpose(image)
            rotated_image = cv2.flip(rotated_image, flipCode=1)
        elif rotation_angle == 180:
            rotated_image = cv2.flip(image, flipCode=-1)
        elif rotation_angle == 270:
            rotated_image = cv2.transpose(image)
            rotated_image = cv2.flip(rotated_image, flipCode=0)
        else:
            print("Invalid rotation angle. Please choose 90, 180, or 270 degrees.")
            return

        # 保存旋转后的图像
        cv2.imwrite(output_path, rotated_image)
        print(f"Image rotated {rotation_angle} degrees and saved to {output_path}")
