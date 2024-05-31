# Save the frame with the maximum area after the video has been processed
    if max_frame is not None:
        cv2.imwrite(f'output_frames/max_area_frame_{os.path.splitext(filename)[0]}.png', max_frame)