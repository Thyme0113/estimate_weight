import ast
from numpy import uint8
from numpy.typing import NDArray
from lib.util import LineDetector

class Model:
    def __init__(self) -> None:
        pass

class ModelException(Exception):
    def __init__(self, message: str = '') -> None:
        self.message = message

class ModelPredictException(ModelException):
    def __str__(self):
        """エラー時の出力

        Returns:
            str: エラーメッセージ
        """
        return (
            f'予測された数値は０～９の数値の特徴と一致しませんでした。\n{self.message}'
        )

class AlgorithmModel(Model):
    def __init__(self) -> None:
        super().__init__()
        # デジタル数字の特徴辞書
        self.number_features_dict = {
            0: {'top': True, 'top_right': True, 'bottom_right': True, 'bottom': True, 'bottom_left':True, 'top_left': True, 'middle': False},
            1: {'top': False,'top_right': True, 'bottom_right': True, 'bottom': False, 'bottom_left': False, 'top_left': False, 'middle': False},
            2: {'top': True, 'top_right': True, 'bottom_right': False, 'bottom': True, 'bottom_left': True, 'top_left': False, 'middle': True},
            3: {'top': True, 'top_right': True, 'bottom_right': True, 'bottom': True, 'bottom_left': False, 'top_left': False, 'middle': True},
            4: {'top': False, 'top_right': True, 'bottom_right': True, 'bottom': False, 'bottom_left':False, 'top_left': True, 'middle': False},
            5: {'top': True,'top_right': False, 'bottom_right': True, 'bottom': True, 'bottom_left': False, 'top_left': True, 'middle': True},
            6: {'top': True, 'top_right': False, 'bottom_right': True, 'bottom': True, 'bottom_left': True, 'top_left': True, 'middle': True},
            7: {'top': True, 'top_right': True, 'bottom_right': True, 'bottom': False, 'bottom_left': False, 'top_left': True, 'middle': False},
            8: {'top': True, 'top_right': True, 'bottom_right': True, 'bottom': True, 'bottom_left': True, 'top_left': True, 'middle': True},
            9: {'top': True, 'top_right': True, 'bottom_right': True, 'bottom': True, 'bottom_left': False, 'top_left': True, 'middle': True},
        }

        # 予測された特徴
        self.number_features = {'top': False, 'top_right': False, 'bottom_right': False, 'bottom': False, 'bottom_left':False, 'top_left': False, 'middle': False}

    def predict(self, img: NDArray[uint8]) -> int:
        """アルゴリズム的に画像の数値を認識する

        Args:
            img (NDArray[uint8]): デジタル数値画像

        Raises:
            ModelPredictException: 予測エラー

        Returns:
            int: 予測された数値
        """
        line_detector = LineDetector()

        height, width, _ = img.shape

        top_img = img[:height//3, :]
        lines = line_detector.detectHorizontalLine(top_img)
        self.number_features['top'] = True if len(lines) > 0 else False

        top_right_img = img[:height//2, width//2:]
        lines = line_detector.detectVerticalLine(top_right_img)
        self.number_features['top_right'] = True if len(lines) > 0 else False

        bottom_right_img = img[height//2:, width//2:]
        lines = line_detector.detectVerticalLine(bottom_right_img)
        self.number_features['bottom_right'] = True if len(lines) > 0 else False

        bottom_img = img[height*2//3:, :]
        lines = line_detector.detectHorizontalLine(bottom_img)
        self.number_features['bottom'] = True if len(lines) > 0 else False

        bottom_left_img = img[height//2:, :width//2]
        lines = line_detector.detectVerticalLine(bottom_left_img)
        self.number_features['bottom_left'] = True if len(lines) > 0 else False

        top_left_img = img[:height//2, :width//2]
        lines = line_detector.detectVerticalLine(top_left_img)
        self.number_features['top_left'] = True if len(lines) > 0 else False

        middle_img = img[height//3:height*2//3, :]
        lines = line_detector.detectHorizontalLine(middle_img)
        self.number_features['middle'] = True if len(lines) > 0 else False

        predict_number = -1
        for number, features in self.number_features_dict.items():
            if self.number_features == features:
                predict_number = number
                break

        if (predict_number < 0):
            message = '特徴配列： {}'.format(ast.literal_eval(str(self.number_features)))
            raise ModelPredictException(message)
        
        return predict_number