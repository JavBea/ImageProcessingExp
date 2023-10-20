from PIL import Image
import cv2

path = r"..\resource\exp1\exp1_2\a.png"
image = Image.open(path)

alpha_channel = image.getchannel('A')
save_path = r"E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp1\exp1_2\a_alpha.png"
alpha_channel.save(save_path)

i = cv2.imread(save_path)
cv2.imshow("AlphaImage", i)
cv2.waitKey(0)
cv2.destroyAllWindows()


