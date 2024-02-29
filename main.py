import cv2

from lib.util import PreProcessing
from lib.model import AlgorithmModel

IMAGE_PATH = './data/test.jpg'

# 数値部分から一番左の数字をトリミングするための座標
START_TOP_LEFT = (820, 340)
START_TOP_RIGHT = (900, 340)
START_BOTTOM_RIGHT = (900, 560)
START_BOTTOM_LEFT = (820, 560)

# 数値部分から真ん中の数字をトリミングするための座標
MIDDLE_TOP_LEFT = (900, 340)
MIDDLE_TOP_RIGHT = (980, 340)
MIDDLE_BOTTOM_RIGHT = (980, 560)
MIDDLE_BOTTOM_LEFT = (900, 560)

# 数値部分から一番右の数字をトリミングするための座標
END_TOP_LEFT = (980, 340)
END_TOP_RIGHT = (1060, 340)
END_BOTTOM_RIGHT = (1060, 560)
END_BOTTOM_LEFT = (980, 560)

def main() -> None:
    img = cv2.imread(IMAGE_PATH)

    # 画像の前処理
    pre_processing = PreProcessing()
    algorithm_model = AlgorithmModel()

    # 一番左の数値をトリミング
    pre_processing.setImage(img)
    pre_processing.trimming(START_TOP_LEFT, START_TOP_RIGHT, START_BOTTOM_RIGHT, START_BOTTOM_LEFT)
    start_number_img = pre_processing.getResult()
    start_number = algorithm_model.predict(start_number_img)

    # 真ん中の数値をトリミング
    pre_processing.setImage(img)
    pre_processing.trimming(MIDDLE_TOP_LEFT, MIDDLE_TOP_RIGHT, MIDDLE_BOTTOM_RIGHT, MIDDLE_BOTTOM_LEFT)
    middle_number_img = pre_processing.getResult()
    middle_number = algorithm_model.predict(middle_number_img)

    # # 一番右の数値をトリミング
    pre_processing.setImage(img)
    pre_processing.trimming(END_TOP_LEFT, END_TOP_RIGHT, END_BOTTOM_RIGHT, END_BOTTOM_LEFT)
    end_number_img = pre_processing.getResult()
    end_number = algorithm_model.predict(end_number_img)

    print('体重： {}{}.{} kg'.format(start_number, middle_number, end_number))

if __name__=='__main__':
    main()
