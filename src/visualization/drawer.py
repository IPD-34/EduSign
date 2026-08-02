import cv2


class LandmarkDrawer:

    @staticmethod
    def draw_points(image, points, present):

        if not present:
            return

        h, w = image.shape[:2]

        for point in points:

            x = int(point[0] * w)
            y = int(point[1] * h)

            cv2.circle(
                image,
                (x, y),
                2,
                (0, 255, 0),
                -1,
            )

    @classmethod
    def draw(cls, image, frame):

        cls.draw_points(
            image,
            frame.pose,
            frame.pose_present,
        )

        cls.draw_points(
            image,
            frame.face,
            frame.face_present,
        )

        cls.draw_points(
            image,
            frame.left_hand,
            frame.left_hand_present,
        )

        cls.draw_points(
            image,
            frame.right_hand,
            frame.right_hand_present,
        )

        return image