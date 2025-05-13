import cv2
import numpy as np
import pickle
import os

def calibrate_camera(output_file='camera_calibration.pkl', board_size=(9, 6), square_size=25):
    # 체스보드의 내부 코너 수 (행, 열)
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 실제 사각형의 크기(mm)sss

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(1)
    print("Calibration started. Press 's' to capture, 'q' to finish.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret_corners:
            cv2.drawChessboardCorners(frame, board_size, corners, ret_corners)

        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and ret_corners:
            print("Captured frame.")
            objpoints.append(objp)
            imgpoints.append(corners)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Calibrating...")
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs
        }, f)

    print(f"Calibration complete. Saved to '{output_file}'")

if __name__ == "__main__":
    calibrate_camera()
