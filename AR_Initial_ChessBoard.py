import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = './chessboard.mp4'
K = np.array([[1.62154591e+03, 0.00000000e+00, 9.40140768e+02],
              [0.00000000e+00, 1.61449233e+03, 5.42885730e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # Derived from `calibrate_camera.py`
dist_coeff = np.array([-3.83644932e-04, 5.98224468e-01, 2.64193408e-03, -6.65172278e-04, -3.49078962e+00])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the letters on the image
        k_position = (2, 3)  # Coordinates of the cell to draw the letter K
        i_position = (3, 3)  # Coordinates of the cell to draw the letter I
        m_position = (4, 3)  # Coordinates of the cell to draw the letter M

        letters = ['K', 'C', 'S']
        positions = [k_position, i_position, m_position]

        for letter, position in zip(letters, positions):
            cell_center = board_cellsize * np.array([position[0] + 0.5, position[1] + 0.5, 0])
            img_position, _ = cv.projectPoints(cell_center.reshape(1, 3), rvec, tvec, K, dist_coeff)
            img_position = tuple(map(int, img_position.flatten()))
            cv.putText(img, letter, img_position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
