     # Draw a rectangle around the detected markers
        for corner in corners:
            int_corner = np.int0(corner)
            frame = cv2.polylines(frame, [int_corner], True, (0, 255, 0), 2)