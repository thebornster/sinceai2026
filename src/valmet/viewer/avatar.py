"""A walking worker figure, built from smooth mesh parts so no model file is needed.

The person faces +x in their own frame, with +y on their left and +z up.
Heights are in meters, measured up from the feet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser

SKIN = (224, 172, 130)
VEST = (255, 120, 20)
REFLECTIVE = (225, 230, 225)
SHIRT = (40, 70, 120)
TROUSERS = (45, 50, 65)
BOOTS = (35, 30, 28)
HARD_HAT = (250, 205, 40)
EYES = (30, 30, 35)

HIP_HEIGHT = 0.93
HIP_WIDTH = 0.095  # from the center line to each hip
THIGH_LENGTH = 0.44
SHIN_LENGTH = 0.42
SHOULDER_HEIGHT = 1.40
SHOULDER_WIDTH = 0.195
UPPER_ARM_LENGTH = 0.29
FOREARM_LENGTH = 0.26
HEAD_HEIGHT = 1.60

TORSO_DEPTH = 0.62  # the torso is this much thinner front to back than side to side
# (height, radius) points up the outline of the torso.
TORSO_OUTLINE = [
    (0.96, 0.0),
    (0.96, 0.135),
    (1.08, 0.14),
    (1.25, 0.175),
    (1.38, 0.18),
    (1.45, 0.11),
    (1.47, 0.0),
]

STRIDE_LENGTH = 1.4  # meters walked in one full walking cycle
TURN_RATE = 8.0  # how quickly the person turns to face their direction of travel
SETTLE_RATE = 8.0  # how quickly the arms and legs ease in and out of walking
JUMP_TOLERANCE = 1e-6  # moving further than this beyond the distance walked is a jump


def lathe(outline: list, depth: float = 1.0, segments: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Spin (height, radius) points around the z axis into a mesh.

    Heights must go upward. depth below 1 flattens the shape front to back.
    Returns (vertices, faces).
    """
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices = np.array(
        [
            (radius * np.cos(angle) * depth, radius * np.sin(angle), height)
            for height, radius in outline
            for angle in angles
        ],
        dtype=np.float32,
    )
    faces = []
    for ring in range(len(outline) - 1):
        for i in range(segments):
            a = ring * segments + i
            b = ring * segments + (i + 1) % segments
            c = b + segments
            d = a + segments
            faces += [(a, b, c), (a, c, d)]
    return vertices, np.array(faces, dtype=np.uint32)


def limb(length: float, top_radius: float, bottom_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """A rounded, tapered limb hanging down from its joint at the origin."""
    bottom = [
        (-length + bottom_radius * (1 + np.sin(t)), bottom_radius * np.cos(t))
        for t in np.linspace(-np.pi / 2, 0.0, 6)
    ]
    top = [(top_radius * np.sin(t), top_radius * np.cos(t)) for t in np.linspace(0.0, np.pi / 2, 6)]
    return lathe(bottom + top)


def turn_about_y(angle: float) -> tuple:
    """Quaternion (wxyz). A positive angle swings a hanging limb backward."""
    return (np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0)


def turn_about_z(angle: float) -> tuple:
    """Quaternion (wxyz) that turns the person to face a heading."""
    return (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))


class Person:
    """Draws the person and animates them as they move."""

    def __init__(
        self, server: viser.ViserServer, position: np.ndarray, heading: float = 0.0
    ) -> None:
        self.server = server
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.phase = 0.0  # where the legs are in the walking cycle, in radians
        self.swing = 0.0  # 0 = standing still, 1 = walking fully
        self.root = server.scene.add_frame(
            "/avatar",
            show_axes=False,
            position=self.position,
            wxyz=turn_about_z(heading),
        )
        self.legs = {}  # side -> (hip joint, knee joint)
        self.arms = {}  # side -> (shoulder joint, elbow joint)
        self._add_body()
        self._add_head()
        for side, sign in (("left", 1.0), ("right", -1.0)):
            self.legs[side] = self._add_leg(side, sign)
            self.arms[side] = self._add_arm(side, sign)
        self._pose()

    def _mesh(self, name: str, shape: tuple, color: tuple, **kwargs) -> None:
        vertices, faces = shape
        self.server.scene.add_mesh_simple(name, vertices, faces, color=color, **kwargs)

    def _ball(self, name: str, size: tuple, color: tuple, position: tuple) -> None:
        """A sphere stretched to size (the half-widths along x, y and z)."""
        self.server.scene.add_icosphere(
            name, radius=1.0, color=color, scale=size, position=position
        )

    def _add_body(self) -> None:
        self._mesh("/avatar/torso", lathe(TORSO_OUTLINE, depth=TORSO_DEPTH), VEST)
        self._ball("/avatar/pelvis", (0.105, 0.17, 0.11), TROUSERS, (0.0, 0.0, 0.96))

        # Reflective stripes sit just outside the vest.
        heights = [point[0] for point in TORSO_OUTLINE[1:-1]]
        radii = [point[1] for point in TORSO_OUTLINE[1:-1]]
        for index, height in enumerate((1.10, 1.26)):
            low, high = height, height + 0.045
            outline = [
                (low, 0.0),
                (low, np.interp(low, heights, radii) + 0.006),
                (high, np.interp(high, heights, radii) + 0.006),
                (high, 0.0),
            ]
            self._mesh(
                f"/avatar/stripe_{index}",
                lathe(outline, depth=TORSO_DEPTH),
                REFLECTIVE,
            )

    def _add_head(self) -> None:
        neck = lathe([(1.42, 0.0), (1.42, 0.05), (1.53, 0.045), (1.53, 0.0)])
        self._mesh("/avatar/neck", neck, SKIN)
        self._ball("/avatar/head", (0.1, 0.085, 0.115), SKIN, (0.0, 0.0, HEAD_HEIGHT))
        self._ball("/avatar/nose", (0.022, 0.014, 0.02), SKIN, (0.1, 0.0, 1.585))
        for side, sign in (("left", 1.0), ("right", -1.0)):
            self._ball(
                f"/avatar/{side}_eye",
                (0.009, 0.012, 0.009),
                EYES,
                (0.09, sign * 0.036, 1.615),
            )

        dome = [
            (1.635 + 0.095 * np.sin(t), 0.118 * np.cos(t)) for t in np.linspace(0.0, np.pi / 2, 8)
        ]
        self._mesh("/avatar/hard_hat", lathe([(1.635, 0.0), *dome]), HARD_HAT)
        brim = [(1.625, 0.0), (1.625, 0.145), (1.64, 0.145), (1.64, 0.0)]
        self._mesh("/avatar/hat_brim", lathe(brim), HARD_HAT)

    def _add_leg(self, side: str, sign: float) -> tuple:
        hip_name = f"/avatar/{side}_hip"
        hip = self.server.scene.add_frame(
            hip_name, show_axes=False, position=(0.0, sign * HIP_WIDTH, HIP_HEIGHT)
        )
        self._mesh(f"{hip_name}/thigh", limb(THIGH_LENGTH, 0.085, 0.06), TROUSERS)

        knee_name = f"{hip_name}/knee"
        knee = self.server.scene.add_frame(
            knee_name, show_axes=False, position=(0.0, 0.0, -THIGH_LENGTH)
        )
        self._mesh(f"{knee_name}/shin", limb(SHIN_LENGTH, 0.06, 0.045), TROUSERS)
        self._ball(
            f"{knee_name}/boot",
            (0.13, 0.055, 0.05),
            BOOTS,
            (0.05, 0.0, -SHIN_LENGTH - 0.02),
        )
        return hip, knee

    def _add_arm(self, side: str, sign: float) -> tuple:
        shoulder_name = f"/avatar/{side}_shoulder"
        shoulder = self.server.scene.add_frame(
            shoulder_name,
            show_axes=False,
            position=(0.0, sign * SHOULDER_WIDTH, SHOULDER_HEIGHT),
        )
        self._mesh(f"{shoulder_name}/upper_arm", limb(UPPER_ARM_LENGTH, 0.052, 0.042), SHIRT)

        elbow_name = f"{shoulder_name}/elbow"
        elbow = self.server.scene.add_frame(
            elbow_name, show_axes=False, position=(0.0, 0.0, -UPPER_ARM_LENGTH)
        )
        self._mesh(f"{elbow_name}/forearm", limb(FOREARM_LENGTH, 0.042, 0.033), SHIRT)
        self._ball(
            f"{elbow_name}/hand",
            (0.035, 0.028, 0.05),
            SKIN,
            (0.0, 0.0, -FOREARM_LENGTH - 0.03),
        )
        return shoulder, elbow

    def _pose(self) -> None:
        """Set every joint from the walking cycle. The sides are half a cycle apart."""
        for side, offset in (("left", 0.0), ("right", np.pi)):
            cycle = self.phase + offset
            hip, knee = self.legs[side]
            shoulder, elbow = self.arms[side]
            hip.wxyz = turn_about_y(-0.45 * self.swing * np.sin(cycle))
            # The knee bends while that leg swings forward.
            knee.wxyz = turn_about_y(0.75 * self.swing * max(0.0, np.cos(cycle)))
            # Each arm swings the opposite way to the leg on the same side.
            shoulder.wxyz = turn_about_y(0.4 * self.swing * np.sin(cycle))
            elbow.wxyz = turn_about_y(-0.15 - 0.3 * self.swing)

    def move_to(self, position: np.ndarray, heading: float, walked: float, dt: float) -> None:
        """Move to position and turn toward heading, animating the distance walked.

        Moving further than walked is a jump (Reset, new route), so they face heading at once.
        """
        position = np.array(position, dtype=float)
        moved = float(np.linalg.norm((position - self.position)[:2]))
        self.position = position

        if moved > walked + JUMP_TOLERANCE:
            self.heading = heading
        else:
            # Wrap the difference so they always turn the short way round.
            turn = (heading - self.heading + np.pi) % (2.0 * np.pi) - np.pi
            self.heading += turn * min(1.0, TURN_RATE * dt)
        walking = walked > 0.0
        self.phase += 2.0 * np.pi * walked / STRIDE_LENGTH

        self.root.position = position
        self.root.wxyz = turn_about_z(self.heading)

        # Standing still with the limbs settled: no joints need updating.
        if not walking and self.swing == 0.0:
            return
        self.swing += (float(walking) - self.swing) * min(1.0, SETTLE_RATE * dt)
        if not walking and self.swing < 0.01:
            self.swing = 0.0
        self._pose()
