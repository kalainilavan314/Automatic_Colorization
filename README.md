# Automatic Colorization of Black and White Images

## Description
This project implements an automatic colorization algorithm for black and white images using a pre-trained deep learning model. The model, based on the Caffe framework, predicts the 'a' and 'b' channels of the Lab color space, given the 'L' channel of an image.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

### Steps
1. Clone the repository:
    ```sh
    git clone https://github.com/kalainilavan314/Automatic_Colorization.git
    cd Automatic_Colorization
    ```
2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Download the pre-trained models and place them in the `models` directory:
    - [colorization_deploy_v2.prototxt](https://github.com/richzhang/colorization/blob/caffe/models/colorization_deploy_v2.prototxt)
    - [colorization_release_v2.caffemodel](https://github.com/richzhang/colorization/blob/caffe/models/colorization_release_v2.caffemodel)
    - [pts_in_hull.npy](https://github.com/richzhang/colorization/blob/caffe/resources/pts_in_hull.npy)

2. Add the path to your black and white image in the script.

3. Run the script:
    ```sh
    python main.py
    ```

## Project Structure

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Richard Zhang](https://richzhang.github.io/) for the colorization model and resources.
- [OpenCV](https://opencv.org/) and [Caffe](http://caffe.berkeleyvision.org/) for providing the framework.


