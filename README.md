
Connect the dots

This project uses a UR3e robotic arm equipped with an Intel RealSense D405 camera to complete a connect-the-dots image automatically.

The workflow is as follows:

The robot captures an image of the connect-the-dots sheet using the RealSense camera.

The program performs dot detection and applies OCR (Optical Character Recognition) to identify the numbers.

Each detected number is paired with its corresponding dot.

A trajectory is generated for the UR3e robot to connect the dots in ascending numerical order.

The final result is a fully connected image drawn by the robot.

## Setup

To clone this project run

```bash
  git clone https://github.com/TNandi26/ConnectTheDots.git
  cd ConnectTheDots
``` 

Activate a virtual enviroment
```bash
  python -m venv venv
  venv\Scripts\activate (on Windows)
  source venv/bin/activate (on macOS or Linux)
```
Install the dependencies
```bash
  pip install -r requirement.txt
``` 

## Folder structure

The project has a specific folder structure that should be correct from the start. If you wish to change this there is a config.json file where you can do that

```bash
  ConnectTheDots
  │
  ├── src/
  │   └── drawing_robots/
  │       ├── Output_pictures/  # Working directory for every run
  │       ├── Examples/  # Contains example runs
  │       ├── circle_detecion.py/  # Script that detects dots on the image
  │       ├── matchmaker.py/ # Pairs up dots and numbers
  │       ├── number_detecion.py/  # Number detection using OCRs
  │       ├── segment_merge.py/  # Merging together the input image
  │       ├── orchestrator.py/  # Main script that controlls everything
  │       └── config.json  # contains the setting for everything
  │
  ├── Pictures/  #Input images
  ├── README.md  # Project documentation
  └── requirements.txt  # Python dependencies
```
    