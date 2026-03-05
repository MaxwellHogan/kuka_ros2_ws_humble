#!/usr/bin/env python3
import math
import random
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose
from std_msgs.msg import String
from std_srvs.srv import SetBool


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Return quaternion (x,y,z,w) from roll, pitch, yaw (radians)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class RelStep:
    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float


def generate_motion_sequence(
    seed: int,
    n_steps: int = 12,
    total_x_min: float = 2.70,
    total_x_max: float = 3.00,
    lat_limit: float = 0.40,
    first_x_no_neg_y: float = 1.00,
) -> List[RelStep]:
    """
    Generate a list of incremental relative motions (dx, dy, dz, droll, dpitch, dyaw)
    satisfying constraints.

    - Total forward motion in x between [total_x_min, total_x_max]
    - Cumulative y and z position stays within +/- lat_limit
    - For the first first_x_no_neg_y meters of cumulative x, do not command +y (dy <= 0)
    - Cumulative roll/pitch/yaw stays within +/- angle_limit_deg
    """
    rng = random.Random(seed)

    # Choose total forward travel in range
    total_x = rng.uniform(total_x_min, total_x_max)

    # Split into steps with mild randomness but stable sum
    base = total_x / n_steps
    dxs = []
    remaining = total_x
    for i in range(n_steps):
        # small jitter around base while keeping positivity
        jitter = rng.uniform(-0.04, 0.04)  # meters
        dx = max(0.10, base + jitter)
        dxs.append(dx)
    # Renormalize to exact total_x
    s = sum(dxs)
    dxs = [dx * (total_x / s) for dx in dxs]

    # Now create dy while respecting cumulative constraints, and early constraint
    steps: List[RelStep] = []
    cum_x = 0.0
    cum_y = 0.0
    cum_z = 0.0

    # Cumulative rotation constraints
    cum_r = 0.0
    cum_p = 0.0
    cum_yaw = 0.0

    for i, dx in enumerate(dxs):
        cum_x_next = cum_x + dx

        # dy proposal
        # Keep dy small (a few cm) but allow up to bounds. We'll ensure cum_y stays within +/- lat_limit.
        dy_prop = rng.uniform(-0.05, 0.05)  # +/- 5 cm per step typical
        dz_prop = rng.uniform(-0.05, 0.05)  # +/- 5 cm per step typical

        # Early segment: no +y (relative to start => cum_y should not go > 0 due to positive dy)
        if cum_x < first_x_no_neg_y:
            dy_prop = max(dy_prop, 0.0) ## stop going towards robot
            dz_prop = min(dz_prop, 0.0) ## stop going to up

        if cum_x < 0.5:
            angle_lim = math.radians(5) ## keep rotations small at start 
        else:
            angle_lim = math.radians(25) ## keep rotations small at start 



        # Enforce cumulative y bounds by clamping dy so cum_y_next within [-lat_limit, +lat_limit]
        dy_min = -lat_limit - cum_y
        dy_max =  lat_limit - cum_y
        dz_min = -lat_limit - cum_z
        dz_max =  lat_limit - cum_z
        dy = clamp(dy_prop, dy_min, dy_max)
        dz = clamp(dz_prop, dz_min, dz_max)

        cum_y_next = cum_y + dy
        cum_z_next = cum_z + dz

        # Random incremental rotations (small), but keep cumulative within +/- 10 deg
        # Use very small per-step deltas to avoid saturating.
        droll_prop  = rng.uniform(-math.radians(2.0), math.radians(2.0))
        dpitch_prop = rng.uniform(-math.radians(2.0), math.radians(2.0))
        dyaw_prop   = rng.uniform(-math.radians(3.0), math.radians(3.0))

        # Clamp each so cumulative stays within bounds
        droll  = clamp(droll_prop,  -angle_lim - cum_r,   angle_lim - cum_r)
        dpitch = clamp(dpitch_prop, -angle_lim - cum_p,   angle_lim - cum_p)
        dyaw   = clamp(dyaw_prop,   -angle_lim - cum_yaw, angle_lim - cum_yaw)

        # Update cum
        cum_x = cum_x_next
        cum_y = cum_y_next
        cum_y = cum_z_next
        cum_r += droll
        cum_p += dpitch
        cum_yaw += dyaw

        steps.append(RelStep(dx=dx, dy=dy, dz=dz, droll=droll, dpitch=dpitch, dyaw=dyaw))

    # Safety assert-ish checks (won't throw unless you want)
    total_dx = sum(s.dx for s in steps)
    if not (total_x_min - 1e-6 <= total_dx <= total_x_max + 1e-6):
        raise RuntimeError(f"Generated total x {total_dx:.3f} out of bounds")
    if abs(cum_y) > lat_limit + 1e-6:
        raise RuntimeError(f"Final cumulative y {cum_y:.3f} out of bounds")
    if max(abs(cum_r), abs(cum_p), abs(cum_yaw)) > angle_lim + 1e-6:
        raise RuntimeError("Cumulative rotation out of bounds")

    return steps


class BenchmarkRunner(Node):
    def __init__(
        self,
        pose_topic: str,
        done_topic: str,
        benchmark_service: str,
        timeout_s: float,
    ):
        super().__init__("kr50_benchmark_runner")

        self.pose_pub = self.create_publisher(Pose, pose_topic, 10)

        # move_done is transient_local + reliable in your C++ node
        done_qos = QoSProfile(depth=1)
        done_qos.reliability = ReliabilityPolicy.RELIABLE
        done_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.done_sub = self.create_subscription(String, done_topic, self._done_cb, done_qos)

        self.bench_cli = self.create_client(SetBool, benchmark_service)

        self._done_count = 0
        self._last_done: Optional[str] = None
        self._timeout_s = timeout_s

    def _done_cb(self, msg: String):
        self._done_count += 1
        self._last_done = msg.data

    def wait_for_service(self, timeout_s: float = 5.0) -> bool:
        t0 = time.time()
        while not self.bench_cli.wait_for_service(timeout_sec=0.2):
            if time.time() - t0 > timeout_s:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def call_benchmark_enable(self, enable: bool) -> bool:
        req = SetBool.Request()
        req.data = enable
        fut = self.bench_cli.call_async(req)

        t0 = time.time()
        while rclpy.ok() and not fut.done():
            if time.time() - t0 > 5.0:
                self.get_logger().error("Timeout waiting for benchmark_enable service response")
                return False
            rclpy.spin_once(self, timeout_sec=0.1)

        resp = fut.result()
        if resp is None:
            self.get_logger().error("benchmark_enable service returned None")
            return False

        if not resp.success:
            self.get_logger().error(f"benchmark_enable failed: {resp.message}")
            return False

        self.get_logger().info(resp.message)
        return True

    def _wait_for_new_done_after(self, baseline_count: int) -> str:
        """
        Wait for a new move_done message *after* baseline_count.
        Returns the msg.data.
        """
        t0 = time.time()
        while rclpy.ok():
            if self._done_count > baseline_count and self._last_done is not None:
                return self._last_done

            if time.time() - t0 > self._timeout_s:
                return "TIMEOUT"

            rclpy.spin_once(self, timeout_sec=0.05)
        return "INTERRUPTED"

    def run_steps(self, steps: List[RelStep]) -> bool:
        # "Clear" the done topic by letting the latched message arrive, then snapshot count
        # This ensures we don't treat the latched previous COMPLETE as the completion of our next command.
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f"Initial move_done count={self._done_count}, last={self._last_done}")

        for i, s in enumerate(steps):
            # Snapshot count BEFORE publishing
            baseline = self._done_count

            msg = Pose()
            msg.position.x = float(s.dx)
            msg.position.y = float(s.dy)
            msg.position.z = float(s.dz)

            qx, qy, qz, qw = quat_from_rpy(s.droll, s.dpitch, s.dyaw)
            msg.orientation.x = qx
            msg.orientation.y = qy
            msg.orientation.z = qz
            msg.orientation.w = qw

            self.pose_pub.publish(msg)
            self.get_logger().info(
                f"[{i+1}/{len(steps)}] Published rel pose: "
                f"dx={s.dx:.3f} dy={s.dy:.3f} dz={s.dz:.3f} "
                f"dRPY(deg)=({math.degrees(s.droll):.2f},{math.degrees(s.dpitch):.2f},{math.degrees(s.dyaw):.2f})"
            )

            done = self._wait_for_new_done_after(baseline)
            if done == "COMPLETE":
                continue
            elif done == "FAILURE":
                self.get_logger().error(f"Move {i+1} FAILED. Stopping sequence.")
                return False
            elif done == "TIMEOUT":
                self.get_logger().error(f"Move {i+1} TIMEOUT waiting for move_done. Stopping sequence.")
                return False
            else:
                self.get_logger().error(f"Move {i+1} interrupted: {done}")
                return False

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-topic", default="relative_pose_cmd")
    parser.add_argument("--done-topic", default="move_done")
    parser.add_argument("--benchmark-service", default="benchmark_enable")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    rclpy.init()
    node = BenchmarkRunner(
        pose_topic=args.pose_topic,
        done_topic=args.done_topic,
        benchmark_service=args.benchmark_service,
        timeout_s=args.timeout,
    )

    if not node.wait_for_service():
        node.get_logger().error(f"Service '{args.benchmark_service}' not available")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Generate a motion sequence that satisfies your constraints
    seq = generate_motion_sequence(
        seed=args.seed,
        n_steps=args.steps,
    )

    total_dx = sum(s.dx for s in seq)
    node.get_logger().info(f"Generated {len(seq)} steps. Total forward dx={total_dx:.3f} m")

    # Start recording
    if not node.call_benchmark_enable(True):
        node.destroy_node()
        rclpy.shutdown()
        return

    # Execute sequence
    ok = node.run_steps(seq)

    # Stop recording (even if failed)
    node.call_benchmark_enable(False)

    if ok:
        node.get_logger().info("Benchmark sequence completed successfully.")
    else:
        node.get_logger().error("Benchmark sequence ended with failure.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
