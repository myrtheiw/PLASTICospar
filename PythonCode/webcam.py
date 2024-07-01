import cv2
import time
import os

def capture_photos(save_folder, interval=2):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        i = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Save the frame as a jpg file
            photo_path = os.path.join(save_folder, f"photo_{i+1}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"Saved: {photo_path}")
            i += 1

            # Wait for the specified interval
            time.sleep(interval)

            # Check if the 'q' key is pressed to end the script
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    save_folder = "C:/path/to/your/folder"  # Change this to your desired folder
    capture_photos(save_folder)
