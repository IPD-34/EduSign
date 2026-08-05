import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.extractor import HolisticExtractor
from src.visualization.drawer import LandmarkDrawer


def main():

    image = cv2.imread(str(Path("test.png")))

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    extractor = HolisticExtractor()

    frame, *_ = extractor.extract(rgb)

    LandmarkDrawer.draw(
        image,
        frame,
    )

    cv2.imshow(
        "Landmarks",
        image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()