# Perfect Circle 

This project uses OpenCV to detect a hand-drawn circle on a white sheet and evaluate how accurately it resembles a perfect circle. By holding your hand-drawn circle in front of the camera, the system analyzes and displays an accuracy percentage based on contour detection.

## How It Works

1. The program captures video from your webcam.
2. It detects circular shapes using the Hough Circle Transform.
3. It extracts red pixel coordinates from the detected circle.
4. It compares these points with detected white contours to assess accuracy.
5. The accuracy score is displayed on the screen.

## Requirements

Ensure you have the following installed:

- Python 3
- OpenCV (`cv2`)
- NumPy (`numpy`)

You can install dependencies using:

```sh
pip install opencv-python numpy
```

## Usage

Run the script:

```sh
python circle_accuracy.py
```

1. Draw a circle on a white sheet of paper.
2. Hold the paper in front of your webcam.
3. The system will detect your circle and display the accuracy.
4. Press `q` to exit.

## Example Output

- If the detected circle closely matches an ideal circle, you get a high accuracy percentage.
- If the shape deviates significantly, the accuracy is lower.

## Notes

- Make sure the lighting is sufficient for clear detection.
- Use a red pen or marker for better accuracy in detection.

## Contributing

Feel free to submit pull requests or open issues for improvements.

## License

This project is open-source and available under the MIT License.
