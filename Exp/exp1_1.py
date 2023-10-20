import cv2
import imageio

# 图片路径
image1_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_1\Img1.png'
image2_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_1\Img2.jpg'
image3_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_1\Img3.bmp'
image4_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_1\Img4.gif'

# 读取图片
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)

# 读取gif帧流
image4_frames = imageio.get_reader(image4_path)


# 展示png图片
cv2.imshow("img1", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 展示jpg图片
cv2.imshow("img2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 展示bmp图片
cv2.imshow("img3", image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 读取每一帧
for frame in image4_frames:

    # 转化为openCV格式
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 每一帧一次展示
    cv2.imshow("Gif", frame_bgr)

    # 每帧25ms
    key = cv2.waitKey(25)

    # esc键中断
    if key == 27:
        break

# 关闭窗口
cv2.destroyAllWindows()
