import cv2
import numpy as np


def get_keypoints_descriptors(image1, image2):
    """
    Extract SIFT keypoints and descriptors using OpenCV functions
    :param image1:
    :param image2:
    :return: SIFT keypoints and descriptors
    """
    def write_kps(image, keypoints, image_name):
        image_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), color=(0, 255, 0), flags=0)
        cv2.imwrite(image_name, image_keypoints)

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Inbuilt SIFT keypoint and descriptors extraction function in OpenCV
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image=img1_gray, mask=None)
    kp2, desc2 = sift.detectAndCompute(image=img2_gray, mask=None)

    write_kps(image1, kp1, 'image1_keypoints.png')
    write_kps(image2, kp2, 'image2_keypoints.png')

    return kp1, kp2, desc1, desc2


def get_matches(desc1, desc2, distance_ratio=0.5, k_nn=2):
    """
    :param desc1: descriptors from image 1 (obtained from query image)
    :param desc2: descriptors from image 2 (obtained from a database image)
    :param distance_ratio: Ratio of distance to reject poor matches.
    :param k_nn: k nearest neighbours to select
    :return: valid keypoints and matches
    """
    # Brute Force matcher in OpenCV.
    matcher = cv2.BFMatcher()
    # Find nearest neighbors using k-NNs.
    # k denoting the number of neighbours to consider
    matches = matcher.knnMatch(desc1, desc2, k=k_nn)
    valid_pts = list([])
    valid_matches = list([])
    for m1, m2 in matches:
        # To understand more about distance ratio,
        # check section 7.1 in https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        if m1.distance < distance_ratio * m2.distance:
            valid_pts.append((m1.trainIdx, m1.queryIdx))
            valid_matches.append([m1])
    return valid_pts, valid_matches


def convert_kps_to_array(keypoints):
    kps_list = [keypoints[idx].pt for idx in range(0, len(keypoints))]
    return np.asarray(kps_list)


def match_images(image1_path, image2_path, distance_ratio, k_nn):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    kp1, kp2, desc1, desc2 = get_keypoints_descriptors(img1, img2)
    val_pts, val_matches = get_matches(desc1, desc2, distance_ratio, k_nn)
    matched_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, val_matches, None, flags=2)
    cv2.imwrite('matched_image.png', matched_image)


if __name__ == '__main__':
    match_images(image1_path='all_souls_000002.jpg',
                 image2_path='all_souls_000006.jpg',
                 distance_ratio=0.8, k_nn=2)
