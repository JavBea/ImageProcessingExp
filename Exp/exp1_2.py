from PIL import Image
import cv2

a_path = r"..\resource\exp1\exp1_2\a.png"
bg_path = r"..\resource\exp1\exp1_2\bg.png"

# 读取前景的原图
image = Image.open(a_path)

# 得到前景的alpha通道
alpha_channel = image.getchannel('A')
alpha_save_path = r"E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_2\a_alpha.png"
# 保存到指定路径
alpha_channel.save(alpha_save_path)

# 读取前景，背景及alpha通道图，得到三个整数数组
foreground = cv2.imread(a_path)
background = cv2.imread(bg_path)
alpha = cv2.imread(alpha_save_path)

# 展示alpha通道图，等待1000微秒
cv2.imshow("alpha", alpha)
cv2.waitKey(1000)

# 将数组中的整数类型转化为浮点数，避免精度丢失
foreground = foreground.astype(float)
background = background.astype(float)

# 将alpha通道图代表不透明度，将透明度限制在0-1之间，方便后续合成
alpha = alpha.astype(float) / 255

# I = α F +（1-α）B
# 按照透明度对前景、背景分别进行加权
foreground = cv2.multiply(foreground, alpha)
background = cv2.multiply(1.0 - alpha, background)
# 合成
composition = cv2.add(foreground, background)
# 读出另存
composition_path = r"E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_2\composition.png"
cv2.imwrite(composition_path, composition)

composition = cv2.imread(composition_path)

cv2.destroyAllWindows()
# 展示，等待1000微秒
cv2.imshow("composition", composition)
cv2.waitKey(1000)
cv2.destroyAllWindows()
