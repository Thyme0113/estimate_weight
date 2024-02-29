import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

class PreProcessing:
    def __init__(self) -> None:
        # 元画像
        self.__original_img = np.zeros(100)
        # 前処理された画像
        self.__result_img = np.zeros(100)

    def setImage(self, img: NDArray[uint8]) -> None:
        """元画像をセット

        Args:
            img (NDArray[uint8]): 元画像
        """
        self.__original_img = img

    def getResult(self) -> NDArray[uint8]:
        """前処理された画像の取得

        Returns:
            NDArray[uint8]: 前処理された画像
        """
        return self.__result_img

    def trimming(
            self,
            top_left: tuple,
            top_right: tuple,
            bottom_right: tuple, 
            bottom_left: tuple
        ) -> None:
        """４点の座標を元に画像をトリミングする

        Args:
            top_left (tuple): 左上座標
            top_right (tuple): 右上座標
            bottom_right (tuple): 右下座標
            bottom_left (tuple): 左下座標

        Returns:
            NDArray[uint8]: トリミング後の画像
        """
        width = top_right[0]-top_left[0]
        height = bottom_right[1]-top_right[1]

        # 変換前の４点の座標配列
        points = np.array([
            np.array(top_left),
            np.array(top_right),
            np.array(bottom_right),
            np.array(bottom_left),
        ], dtype='float32')

        # 変換後の４点の座標配列
        transformed_points = np.array([
            np.array([0, 0]),
            np.array([width-1, 0]),
            np.array([width-1, height-1]),
            np.array([0, height-1]),
        ], dtype='float32')

        # 透視変換行列の作成
        transformation_matrix = cv2.getPerspectiveTransform(points, transformed_points)
        self.__result_img = cv2.warpPerspective(self.__original_img, transformation_matrix, (int(width), int(height)))

class LineDetector:
    def __init__(self) -> None:
        self.kernel = np.ones((5,5),np.uint8)

    def detectVerticalLine(self, img: NDArray[uint8]) -> list[int]:
        # ビット変換
        reversed_img = cv2.bitwise_not(img)

        # グレースケール変換
        gray_img = cv2.cvtColor(reversed_img, cv2.COLOR_BGR2GRAY)

        # 二値化
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        # 膨張処理
        dilation = cv2.dilate(thresh_img, self.kernel, iterations=1)

        # 高さ、幅、チャネル数
        height, width, channels = img.shape
        half_height, half_width = height//2, width//2

        vertical_lines = []
        # 画像の高さの半分より長い縦線があるなら、縦線ありとみなす
        # 画像の高さの半分より長い縦線がないなら、縦線なしとみなす
        for i in range(width-1):
            for j in range(half_height-1):
                tmp = dilation[j:j+half_height, i:i+1]
                white_pixels = np.count_nonzero(tmp)
                if (white_pixels == tmp.size):
                    vertical_lines.append(i)
                    break
        
        return vertical_lines

    def detectHorizontalLine(self, img: NDArray[uint8]) -> list[int]:
        # ビット変換
        reversed_img = cv2.bitwise_not(img)

        # グレースケール変換
        gray_img = cv2.cvtColor(reversed_img, cv2.COLOR_BGR2GRAY)

        # 二値化
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        # 膨張処理
        dilation = cv2.dilate(thresh_img, self.kernel, iterations=1)

        # 高さ、幅、チャネル数
        height, width, channels = img.shape
        half_height, half_width = height//2, width//2

        horizontal_lines = []
        # 画像の幅の半分より長い横線があるなら、横線ありとみなす
        # 画像の幅の半分より長い横線がないなら、横線なしとみなす
        for i in range(height-1):
            for j in range(half_width-1):
                tmp = dilation[i:i+1, j:j+half_width]
                white_pixels = np.count_nonzero(tmp)
                if (white_pixels == tmp.size):
                    horizontal_lines.append(i)
                    break
        
        return horizontal_lines