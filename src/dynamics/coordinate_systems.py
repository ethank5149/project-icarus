import numpy as np


def quat_multiply(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        return q / norm
    return q


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def dcm_to_quat(C):
    trace = np.trace(C)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (C[2, 1] - C[1, 2]) * s
        y = (C[0, 2] - C[2, 0]) * s
        z = (C[1, 0] - C[0, 1]) * s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        w = (C[2, 1] - C[1, 2]) / s
        x = 0.25 * s
        y = (C[0, 1] + C[1, 0]) / s
        z = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        w = (C[0, 2] - C[2, 0]) / s
        x = (C[0, 1] + C[1, 0]) / s
        y = 0.25 * s
        z = (C[1, 2] + C[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        w = (C[1, 0] - C[0, 1]) / s
        x = (C[0, 2] + C[2, 0]) / s
        y = (C[1, 2] + C[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_kinematics(q, omega):
    q = quat_normalize(q)
    w, x, y, z = q
    W = np.array(
        [
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0],
        ]
    )
    return 0.5 * W @ q


def rotate_body_to_inertial(v_body, q):
    return quat_to_dcm(q) @ v_body


def rotate_inertial_to_body(v_inertial, q):
    return quat_to_dcm(q).T @ v_inertial
