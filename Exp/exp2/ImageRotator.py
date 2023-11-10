import cv2


class ImageRotator:
    def __init__(self):
        pass

    def rotate_image(self,input_path, output_path, angle):
        # 读取输入图像
        input_image = cv2.imread(input_path)

        # 检查角度是否合法（支持90度、180度、270度的顺时针旋转）
        if angle not in [90, 180, 270]:
            print("Invalid rotation angle. Supported angles: 90, 180, 270.")
            return

        # 旋转图像
        if angle == 90:
            rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(input_image, cv2.ROTATE_180)
        else:
            rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 保存旋转后的图像
        cv2.imwrite(output_path, rotated_image)
        print(f"Image rotated {angle} degrees clockwise and saved to {output_path}")

