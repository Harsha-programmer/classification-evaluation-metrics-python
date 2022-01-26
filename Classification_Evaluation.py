import numpy as np


# https://en.wikipedia.org/wiki/Confusion_matrix
class Evaluation:
    def __init__(self, actual=None, predict=None):
        self.predict = actual
        self.actual = predict
        self.Max = None
        self.pre_validation_for_classification()
        self.classiication_performance_metrics()
        self.verification_for_classification()

    def pre_validation_for_classification(self):
        if self.actual.shape != self.predict.shape:
            raise Exception("Actual and Predicted array shape must be equal")
        if np.prod(np.unique(self.actual) == np.array([0, 1])):
            self.Max = 1
        elif len(np.unique(self.actual)) == 2:  # If Actual and Predicted are Images
            self.Max = np.unique(self.actual)[1]
        elif not np.prod(np.unique(self.actual) == np.array([0, 1])):
            raise Exception("Actual Values are must be 0 and 1")
        if not np.prod(np.unique(self.predict) == np.array([0, 1])):
            raise Exception("Predicted Values are must be 0 and 1")

    def classiication_performance_metrics(self):
        act_one = np.where(self.actual == self.Max)
        act_zero = np.where(self.actual == 0)
        pred_one = np.where(self.predict == self.Max)
        pred_zero = np.where(self.predict == 0)

        '''Find Shape of the Each Dimension for Single Array Conversion'''
        self.array = [self.actual.shape[i] for i in range(len(self.actual.shape))]

        Act_One = np.zeros(shape=act_one[0].shape[0], dtype=np.int16)
        Act_Zero = np.zeros(shape=act_zero[0].shape[0], dtype=np.int16)
        Pred_One = np.zeros(shape=pred_one[0].shape[0], dtype=np.int16)
        Pred_Zero = np.zeros(shape=pred_zero[0].shape[0], dtype=np.int16)

        '''Convert Single Array for Easy Intersection'''
        for iter in range(len(act_one) - 1):
            Act_One += act_one[iter] * np.prod(self.array[iter + 1:])
            Act_Zero += act_zero[iter] * np.prod(self.array[iter + 1:])
            Pred_One += pred_one[iter] * np.prod(self.array[iter + 1:])
            Pred_Zero += pred_zero[iter] * np.prod(self.array[iter + 1:])
        Act_One += act_one[len(act_one) - 1]
        Act_Zero += act_zero[len(act_zero) - 1]
        Pred_One += pred_one[len(pred_one) - 1]
        Pred_Zero += pred_zero[len(pred_zero) - 1]

        '''Find Confusion Matrix'''

        # 1 ---> TP (True Positive) ------> If Actual = 1 and Predicted = 1
        self.TP = len(np.intersect1d(Act_One, Pred_One))
        # 2 ---> TN (True Negative) ------> If Actual = 0 and Predicted = 0
        self.TN = len(np.intersect1d(Act_Zero, Pred_Zero))
        # 3 ---> FP (False Positive) -----> If Actual = 0 and Predicted = 1
        self.FP = len(np.intersect1d(Act_Zero, Pred_One))
        # 4 ---> FN (False Negative) -----> If Actual = 1 and Predicted = 0
        self.FN = len(np.intersect1d(Act_One, Pred_Zero))

        # 5 ---> Overall Accuracy
        self.Accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.Accuracy *= 100  # For Percentage

        # 6 ---> Sensitivity, Hitrate, Recall, or True Positive Rate (TPR) = 1 - FNR
        self.Sensitivity = self.TP / (self.TP + self.FN)
        self.Sensitivity *= 100  # For Percentage

        # 7 ---> Specificity or True Negative Rate (TNR) = 1 - FPR
        self.Specificity = self.TN / (self.TN + self.FP)
        self.Specificity *= 100  # For Percentage

        # 8 ---> Precision or Positive Predictive Value (PPV) = 1 - FDR
        self.Precision = self.TP / (self.TP + self.FP)
        self.Precision *= 100  # For Percentage

        # 9 ---> Fall out or False Positive Rate (FPR) = 1 - TNR
        self.FPR = self.FP / (self.FP + self.TN)
        self.FPR *= 100  # For Percentage

        # 10 ---> False Negative Rate = 1 - TPR
        self.FNR = self.FN / (self.TP + self.FN)
        self.FNR *= 100  # For Percentage

        # 11 ---> Negative Predictive Value (NPV) = 1- FOR
        self.NPV = self.TN / (self.TN + self.FN)
        self.NPV *= 100  # For Percentage

        # 12 ---> False Discovery Rate (FDR) = 1 - PPV
        self.FDR = self.FP / (self.TP + self.FP)
        self.FDR *= 100  # For Percentage

        # 13 ---> F1 score is the harmonic mean of Precision and Sensitivity
        # F1_score = 2 * ((PPV * TPR) / (PPV + TPR))
        self.F1_score = (2 * self.TP) / (2 * self.TP + self.FP + self.FN)
        self.F1_score *= 100  # For Percentage

        # 14 ---> Matthews Correlation Coefficient (MCC)
        self.MCC = ((self.TP * self.TN) - (self.FP * self.FN)) / np.math.sqrt((self.TP + self.FP) *
                                                                              (self.TP + self.FN) * (
                                                                                      self.TN + self.FP) * (
                                                                                      self.TN + self.FN))

        # 15 ---> False Omission Rate (FOR) = 1 - NPV
        self.FOR = self.FN / (self.FN + self.TN)
        self.FOR *= 100  # For Percentage

        # 16 ---> Prevalence Threshold (PT)
        self.PT = np.math.sqrt(self.FPR) / (np.math.sqrt(self.Sensitivity) + np.math.sqrt(self.FPR))
        self.PT *= 100  # For Percentage

        # 17 ---> Threat Score (TS) or Critical Success Index (CSI)
        self.CSI = self.TP / (self.TP + self.FN + self.FP)
        self.CSI *= 100  # For Percentage

        # 18 ---> Balanced Accuracy (BA)
        self.BA = (self.Sensitivity + self.Specificity) / 2

        # 19 ---> Fowlkes–Mallows Index (FM)
        self.FM = np.math.sqrt(self.Sensitivity * self.Precision)

        # 20 ---> Informedness or Bookmaker Informedness (BM)
        self.BM = self.Sensitivity + self.Specificity - 1

        # 21 ---> Markedness (MK) or DeltaP (Δp)
        self.MK = self.Precision + self.NPV - 1

        self.Values = np.array([self.TP, self.TN, self.FP, self.FN, self.Accuracy, self.Sensitivity, self.Specificity,
                                self.Precision, self.FPR, self.FNR, self.NPV, self.FDR, self.F1_score, self.MCC,
                                self.FOR, self.PT, self.CSI, self.BA, self.FM, self.BM, self.MK])

    def verification_for_classification(self):
        Limit_0_100 = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18])
        Limit_minus_1_plus_1 = np.array([13, 19, 20])
        if not (np.prod(self.array) == self.TP + self.TN + self.FP + self.FN):
            raise Exception("Something went wrong - Please check values")
        if (not (0 <= self.Values[Limit_0_100].all() <= 100)) or (
                not (-1 <= self.Values[Limit_minus_1_plus_1].all() <= 1)):
            raise Exception('Something went wrong')


def Test():
    actual = np.random.randint(low=0, high=2, size=(10, 100, 20))
    pred = np.random.randint(low=0, high=2, size=(10, 100, 20))
    Eval = Evaluation(actual=actual, predict=pred)
    print(Eval.Values)


if __name__ == '__main__':
    Test()
