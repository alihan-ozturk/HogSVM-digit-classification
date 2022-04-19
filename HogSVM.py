from cv2 import cv2
import numpy as np

SIZE_IMAGE = 20
NUMBER_CLASSES = 10


def load_digits_and_labels(big_image):
    digits_img = cv2.imread(big_image, 0)

    number_rows = digits_img.shape[1] / SIZE_IMAGE
    rows = np.vsplit(digits_img, digits_img.shape[0] / SIZE_IMAGE)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_rows)
        for digit in row_cells:
            digits.append(digit)

    digits = np.array(digits)
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels


def deskew(img):
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def svm_init(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return model


def svm_train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svm_predict(model, samples):
    return model.predict(samples)[1].ravel()


def svm_evaluate(model, samples, labels):
    predictions = svm_predict(model, samples)
    accuracy = (labels == predictions).mean()
    print("percentage acc", accuracy * 100)


def get_hog():
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    print("get descriptor size:{}".format(hog.getDescriptorSize()))
    return hog


def raw_pixels(img):
    return img.flatten()


digits, labels = load_digits_and_labels("digits.png")

rand = np.random.RandomState(1234)

shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

hog = get_hog()

hog_descriptors = []
for img in digits:
    hog_descriptors.append(hog.compute(deskew(img)))
hog_descriptors = np.squeeze(hog_descriptors)

partition = int(0.5 * len(hog_descriptors))
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])

print("training svm ...")
model = svm_init()
svm_train(model, hog_descriptors_train, labels_train)

print("evaluate svm ...")
svm_evaluate(model, hog_descriptors_test, labels_test)