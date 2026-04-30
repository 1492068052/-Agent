#!/usr/bin/env python3
"""
Industrial Robot Workstation Simulation Agent

A single-file, dependency-free prototype for an AI/Agent-style industrial robot
workstation simulation workflow.

What it does:
1. Plans a pick-and-place process into robot waypoints.
2. Checks reachability against a simplified industrial robot workspace model.
3. Simulates linear motion between waypoints.
4. Detects collisions between the TCP/tool and workstation obstacles.
5. Estimates cycle time.
6. Generates optimization suggestions.
7. Exports a RAPID-like robot program and a Markdown report.

This is not a replacement for RobotStudio/Process Simulate/ROS-Industrial.
It is a complete, runnable prototype that demonstrates the Agent architecture
and core logic for workstation simulation assistance.

Run:
    python industrial_robot_workstation_agent.py

Outputs:
    output/workstation_report.md
    output/generated_robot_program.mod
    output/simulation_result.json
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass
class Pose:
    """TCP pose in a simplified world frame.

    Position is in millimeters. Rotation is Euler angles in degrees.
    """

    x: float
    y: float
    z: float
    rx: float = 180.0
    ry: float = 0.0
    rz: float = 0.0

    def position(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def distance_to(self, other: "Pose") -> float:
        return euclidean_distance(self.position(), other.position())

    def lerp(self, other: "Pose", t: float) -> "Pose":
        return Pose(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t,
            z=self.z + (other.z - self.z) * t,
            rx=self.rx + (other.rx - self.rx) * t,
            ry=self.ry + (other.ry - self.ry) * t,
            rz=self.rz + (other.rz - self.rz) * t,
        )


@dataclass
class Waypoint:
    name: str
    pose: Pose
    speed_mm_s: float = 600.0
    zone_mm: float = 5.0
    action: Optional[str] = None


@dataclass
class AABBObstacle:
    """Axis-aligned bounding box obstacle."""

    name: str
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    safety_margin_mm: float = 30.0

    def expanded(self) -> "AABBObstacle":
        m = self.safety_margin_mm
        return AABBObstacle(
            name=self.name,
            min_x=self.min_x - m,
            min_y=self.min_y - m,
            min_z=self.min_z - m,
            max_x=self.max_x + m,
            max_y=self.max_y + m,
            max_z=self.max_z + m,
            safety_margin_mm=0.0,
        )

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        x, y, z = point
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )


@dataclass
class ToolModel:
    name: str
    tcp_radius_mm: float = 60.0
    grip_time_s: float = 0.6
    release_time_s: float = 0.4


@dataclass
class RobotModel:
    name: str
    base_pose: Pose
    min_reach_mm: float = 250.0
    max_reach_mm: float = 1650.0
    max_speed_mm_s: float = 1500.0
    payload_kg: float = 10.0

    def distance_from_base(self, pose: Pose) -> float:
        return euclidean_distance(self.base_pose.position(), pose.position())

    def is_pose_reachable(self, pose: Pose) -> Tuple[bool, str]:
        d = self.distance_from_base(pose)
        if d < self.min_reach_mm:
            return False, f"Too close to robot base: {d:.1f} mm < {self.min_reach_mm:.1f} mm"
        if d > self.max_reach_mm:
            return False, f"Outside robot reach: {d:.1f} mm > {self.max_reach_mm:.1f} mm"
        if pose.z < 0:
            return False, f"TCP below floor level: z={pose.z:.1f} mm"
        return True, "reachable"


@dataclass
class PickPlaceTask:
    part_name: str
    pick_pose: Pose
    place_pose: Pose
    approach_height_mm: float = 180.0
    depart_height_mm: float = 180.0
    part_mass_kg: float = 2.0


@dataclass
class WorkstationConfig:
    robot: RobotModel
    tool: ToolModel
    tasks: List[PickPlaceTask]
    obstacles: List[AABBObstacle]
    home_pose: Pose
    sample_step_mm: float = 25.0


@dataclass
class CollisionEvent:
    segment: str
    obstacle: str
    pose: Pose
    distance_along_segment_mm: float


@dataclass
class ReachabilityIssue:
    waypoint: str
    reason: str
    pose: Pose


@dataclass
class SegmentTiming:
    segment: str
    distance_mm: float
    speed_mm_s: float
    time_s: float


@dataclass
class SimulationResult:
    waypoints: List[Waypoint]
    collisions: List[CollisionEvent]
    reachability_issues: List[ReachabilityIssue]
    timing: List[SegmentTiming]
    total_cycle_time_s: float
    suggestions: List[str]
    rapid_program: str

    @property
    def passed(self) -> bool:
        return len(self.collisions) == 0 and len(self.reachability_issues) == 0


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def pose_with_z(pose: Pose, z: float) -> Pose:
    return Pose(x=pose.x, y=pose.y, z=z, rx=pose.rx, ry=pose.ry, rz=pose.rz)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------


class TaskPlannerAgent:
    """Converts process tasks into robot waypoints."""

    def plan(self, config: WorkstationConfig) -> List[Waypoint]:
        waypoints: List[Waypoint] = [
            Waypoint("Home", config.home_pose, speed_mm_s=800.0, zone_mm=10.0)
        ]

        for index, task in enumerate(config.tasks, start=1):
            pick_approach = pose_with_z(
                task.pick_pose, task.pick_pose.z + task.approach_height_mm
            )
            pick_depart = pose_with_z(
                task.pick_pose, task.pick_pose.z + task.depart_height_mm
            )
            place_approach = pose_with_z(
                task.place_pose, task.place_pose.z + task.approach_height_mm
            )
            place_depart = pose_with_z(
                task.place_pose, task.place_pose.z + task.depart_height_mm
            )

            prefix = f"T{index}_{task.part_name}"
            waypoints.extend(
                [
                    Waypoint(f"{prefix}_PickApproach", pick_approach, 700.0, 10.0),
                    Waypoint(f"{prefix}_Pick", task.pick_pose, 250.0, 2.0, action="grip"),
                    Waypoint(f"{prefix}_PickDepart", pick_depart, 500.0, 5.0),
                    Waypoint(f"{prefix}_PlaceApproach", place_approach, 700.0, 10.0),
                    Waypoint(f"{prefix}_Place", task.place_pose, 250.0, 2.0, action="release"),
                    Waypoint(f"{prefix}_PlaceDepart", place_depart, 500.0, 5.0),
                ]
            )

        waypoints.append(Waypoint("Home_End", config.home_pose, speed_mm_s=800.0, zone_mm=10.0))
        return waypoints


class ReachabilityAgent:
    """Checks whether all planned TCP waypoints are inside the robot workspace."""

    def check(self, config: WorkstationConfig, waypoints: Sequence[Waypoint]) -> List[ReachabilityIssue]:
        issues: List[ReachabilityIssue] = []
        for wp in waypoints:
            reachable, reason = config.robot.is_pose_reachable(wp.pose)
            if not reachable:
                issues.append(ReachabilityIssue(wp.name, reason, wp.pose))
        return issues


class CollisionSimulationAgent:
    """Samples TCP path and checks collisions against AABB obstacles.

    The TCP/tool is approximated as a sphere. Obstacles are expanded by the tool
    radius, so point-in-box testing becomes a conservative collision check.
    """

    def simulate(
        self, config: WorkstationConfig, waypoints: Sequence[Waypoint]
    ) -> List[CollisionEvent]:
        collisions: List[CollisionEvent] = []
        expanded_obstacles = self._expanded_obstacles(config)

        for start, end in zip(waypoints[:-1], waypoints[1:]):
            segment_name = f"{start.name} -> {end.name}"
            segment_distance = start.pose.distance_to(end.pose)
            samples = max(2, int(math.ceil(segment_distance / config.sample_step_mm)))

            for i in range(samples + 1):
                t = i / samples
                pose = start.pose.lerp(end.pose, t)
                point = pose.position()

                for obstacle in expanded_obstacles:
                    if obstacle.contains_point(point):
                        collisions.append(
                            CollisionEvent(
                                segment=segment_name,
                                obstacle=obstacle.name,
                                pose=pose,
                                distance_along_segment_mm=segment_distance * t,
                            )
                        )
                        break

                # Avoid generating hundreds of repeated collision points for the same segment.
                if collisions and collisions[-1].segment == segment_name:
                    break

        return collisions

    def _expanded_obstacles(self, config: WorkstationConfig) -> List[AABBObstacle]:
        expanded: List[AABBObstacle] = []
        for obstacle in config.obstacles:
            o = obstacle.expanded()
            radius = config.tool.tcp_radius_mm
            expanded.append(
                AABBObstacle(
                    name=o.name,
                    min_x=o.min_x - radius,
                    min_y=o.min_y - radius,
                    min_z=o.min_z - radius,
                    max_x=o.max_x + radius,
                    max_y=o.max_y + radius,
                    max_z=o.max_z + radius,
                    safety_margin_mm=0.0,
                )
            )
        return expanded


class CycleTimeAgent:
    """Estimates cycle time from path length, speed, and gripper actions."""

    def estimate(self, config: WorkstationConfig, waypoints: Sequence[Waypoint]) -> Tuple[List[SegmentTiming], float]:
        timing: List[SegmentTiming] = []
        total = 0.0

        for start, end in zip(waypoints[:-1], waypoints[1:]):
            speed = clamp(end.speed_mm_s, 1.0, config.robot.max_speed_mm_s)
            distance = start.pose.distance_to(end.pose)
            move_time = distance / speed
            timing.append(
                SegmentTiming(
                    segment=f"{start.name} -> {end.name}",
                    distance_mm=distance,
                    speed_mm_s=speed,
                    time_s=move_time,
                )
            )
            total += move_time

            if end.action == "grip":
                total += config.tool.grip_time_s
            elif end.action == "release":
                total += config.tool.release_time_s

        return timing, total


class OptimizationAgent:
    """Creates practical engineering suggestions from simulation results."""

    def suggest(
        self,
        config: WorkstationConfig,
        waypoints: Sequence[Waypoint],
        collisions: Sequence[CollisionEvent],
        reachability_issues: Sequence[ReachabilityIssue],
        timing: Sequence[SegmentTiming],
    ) -> List[str]:
        suggestions: List[str] = []

        if reachability_issues:
            suggestions.append(
                "存在不可达点位：优先检查机器人底座位置、工件/料框布局，以及是否需要提高/降低作业台高度。"
            )
            for issue in reachability_issues[:3]:
                suggestions.append(
                    f"点位 {issue.waypoint} 不可达，原因：{issue.reason}。建议将该点向机器人基座方向移动，或重新布置机器人安装位置。"
                )

        if collisions:
            suggestions.append(
                "存在碰撞风险：建议提高 Approach/Depart 高度，或在障碍物上方增加中间过渡点。"
            )
            for collision in collisions[:3]:
                suggestions.append(
                    f"路径 {collision.segment} 与 {collision.obstacle} 存在碰撞风险；可在该段加入避障 waypoint 或提高安全高度。"
                )

        if timing:
            slowest = sorted(timing, key=lambda x: x.time_s, reverse=True)[:3]
            for item in slowest:
                if item.time_s > 1.0:
                    suggestions.append(
                        f"节拍瓶颈：{item.segment} 用时 {item.time_s:.2f}s，距离 {item.distance_mm:.0f}mm。可优化点位顺序、缩短空行程或提高非接触段速度。"
                    )

        payload_total = sum(task.part_mass_kg for task in config.tasks)
        if payload_total > config.robot.payload_kg:
            suggestions.append(
                f"任务总负载 {payload_total:.1f}kg 高于机器人额定负载 {config.robot.payload_kg:.1f}kg。请检查末端夹具和单次搬运策略。"
            )

        if not suggestions:
            suggestions.append("当前方案未发现明显不可达或碰撞问题，可进入更高精度仿真软件进行关节级验证。")

        suggestions.append(
            "注意：本程序使用 TCP 球体 + AABB 的简化碰撞模型，适合早期方案评估；正式上线前仍需在 RobotStudio、Process Simulate 或真实控制器中做完整验证。"
        )
        return suggestions


class RapidCodeAgent:
    """Generates ABB RAPID-like pseudo program from waypoints."""

    def generate(self, config: WorkstationConfig, waypoints: Sequence[Waypoint]) -> str:
        lines: List[str] = []
        lines.append("MODULE WorkstationAgentProgram")
        lines.append("  ! Auto-generated by Industrial Robot Workstation Simulation Agent")
        lines.append("  ! Units: mm, degrees")
        lines.append("")

        for wp in waypoints:
            lines.append(f"  CONST robtarget {wp.name}:={self._rapid_target(wp.pose)};")

        lines.append("")
        lines.append("  PROC main()")
        lines.append("    ConfL\\Off;")
        lines.append("    ConfJ\\Off;")
        lines.append("")

        for wp in waypoints:
            speed_name = self._speed_name(wp.speed_mm_s)
            zone_name = self._zone_name(wp.zone_mm)
            lines.append(f"    MoveL {wp.name},{speed_name},{zone_name},tool0;")
            if wp.action == "grip":
                lines.append("    SetDO do_gripper_close, 1;")
                lines.append(f"    WaitTime {config.tool.grip_time_s:.2f};")
            elif wp.action == "release":
                lines.append("    SetDO do_gripper_close, 0;")
                lines.append(f"    WaitTime {config.tool.release_time_s:.2f};")

        lines.append("")
        lines.append("  ENDPROC")
        lines.append("ENDMODULE")
        return "\n".join(lines)

    def _rapid_target(self, pose: Pose) -> str:
        # Quaternion is simplified as [1,0,0,0]. For production RAPID generation,
        # convert Euler orientation to a real quaternion.
        return (
            f"[[{pose.x:.1f},{pose.y:.1f},{pose.z:.1f}],"
            f"[1,0,0,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]]"
        )

    def _speed_name(self, speed_mm_s: float) -> str:
        common = [50, 100, 200, 300, 500, 700, 1000, 1500]
        nearest = min(common, key=lambda x: abs(x - speed_mm_s))
        return f"v{nearest}"

    def _zone_name(self, zone_mm: float) -> str:
        if zone_mm <= 2:
            return "fine"
        common = [5, 10, 20, 50, 100]
        nearest = min(common, key=lambda x: abs(x - zone_mm))
        return f"z{nearest}"


class ReportAgent:
    """Creates JSON and Markdown reports."""

    def to_json(self, result: SimulationResult) -> str:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)

    def to_markdown(self, config: WorkstationConfig, result: SimulationResult) -> str:
        lines: List[str] = []
        lines.append("# 工业机器人工作站仿真 Agent 报告")
        lines.append("")
        lines.append("## 1. 总体结论")
        status = "通过" if result.passed else "需要优化"
        lines.append(f"- 仿真状态：**{status}**")
        lines.append(f"- 机器人型号：{config.robot.name}")
        lines.append(f"- 工具模型：{config.tool.name}")
        lines.append(f"- 点位数量：{len(result.waypoints)}")
        lines.append(f"- 碰撞数量：{len(result.collisions)}")
        lines.append(f"- 不可达点数量：{len(result.reachability_issues)}")
        lines.append(f"- 估算节拍：{result.total_cycle_time_s:.2f} s")
        lines.append("")

        lines.append("## 2. 工艺点位")
        lines.append("| # | 点位 | X | Y | Z | 动作 |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for i, wp in enumerate(result.waypoints, start=1):
            lines.append(
                f"| {i} | {wp.name} | {wp.pose.x:.1f} | {wp.pose.y:.1f} | {wp.pose.z:.1f} | {wp.action or '-'} |"
            )
        lines.append("")

        lines.append("## 3. 可达性检查")
        if result.reachability_issues:
            lines.append("| 点位 | 原因 | X | Y | Z |")
            lines.append("|---|---|---:|---:|---:|")
            for issue in result.reachability_issues:
                lines.append(
                    f"| {issue.waypoint} | {issue.reason} | {issue.pose.x:.1f} | {issue.pose.y:.1f} | {issue.pose.z:.1f} |"
                )
        else:
            lines.append("未发现不可达点位。")
        lines.append("")

        lines.append("## 4. 碰撞检查")
        if result.collisions:
            lines.append("| 路径段 | 障碍物 | 路径距离/mm | X | Y | Z |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for c in result.collisions:
                lines.append(
                    f"| {c.segment} | {c.obstacle} | {c.distance_along_segment_mm:.1f} | {c.pose.x:.1f} | {c.pose.y:.1f} | {c.pose.z:.1f} |"
                )
        else:
            lines.append("未发现 TCP/工具与障碍物的碰撞风险。")
        lines.append("")

        lines.append("## 5. 节拍估算")
        lines.append("| 路径段 | 距离/mm | 速度/mm/s | 用时/s |")
        lines.append("|---|---:|---:|---:|")
        for t in result.timing:
            lines.append(
                f"| {t.segment} | {t.distance_mm:.1f} | {t.speed_mm_s:.1f} | {t.time_s:.2f} |"
            )
        lines.append("")

        lines.append("## 6. 优化建议")
        for suggestion in result.suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")

        lines.append("## 7. 自动生成的 RAPID-like 程序")
        lines.append("```rapid")
        lines.append(result.rapid_program)
        lines.append("```")
        lines.append("")
        return "\n".join(lines)


class WorkstationSimulationCoordinator:
    """Orchestrates all agents."""

    def __init__(self) -> None:
        self.planner = TaskPlannerAgent()
        self.reachability = ReachabilityAgent()
        self.collision = CollisionSimulationAgent()
        self.cycle_time = CycleTimeAgent()
        self.optimizer = OptimizationAgent()
        self.code = RapidCodeAgent()
        self.report = ReportAgent()

    def run(self, config: WorkstationConfig) -> SimulationResult:
        waypoints = self.planner.plan(config)
        reachability_issues = self.reachability.check(config, waypoints)
        collisions = self.collision.simulate(config, waypoints)
        timing, total_cycle_time_s = self.cycle_time.estimate(config, waypoints)
        suggestions = self.optimizer.suggest(
            config=config,
            waypoints=waypoints,
            collisions=collisions,
            reachability_issues=reachability_issues,
            timing=timing,
        )
        rapid_program = self.code.generate(config, waypoints)

        return SimulationResult(
            waypoints=list(waypoints),
            collisions=collisions,
            reachability_issues=reachability_issues,
            timing=timing,
            total_cycle_time_s=total_cycle_time_s,
            suggestions=suggestions,
            rapid_program=rapid_program,
        )

    def export(self, config: WorkstationConfig, result: SimulationResult, output_dir: str = "output") -> Dict[str, str]:
        mkdir(output_dir)
        report_md = self.report.to_markdown(config, result)
        result_json = self.report.to_json(result)

        report_path = os.path.join(output_dir, "workstation_report.md")
        json_path = os.path.join(output_dir, "simulation_result.json")
        rapid_path = os.path.join(output_dir, "generated_robot_program.mod")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(result_json)
        with open(rapid_path, "w", encoding="utf-8") as f:
            f.write(result.rapid_program)

        return {
            "report": report_path,
            "json": json_path,
            "rapid": rapid_path,
        }


# -----------------------------------------------------------------------------
# Example workstation
# -----------------------------------------------------------------------------


def build_demo_config() -> WorkstationConfig:
    """Creates a demo cell similar to a small pick-and-place workstation."""

    robot = RobotModel(
        name="ABB IRB 1600 simplified",
        base_pose=Pose(0, 0, 0),
        min_reach_mm=250,
        max_reach_mm=1650,
        max_speed_mm_s=1500,
        payload_kg=10,
    )

    tool = ToolModel(
        name="Pneumatic two-finger gripper",
        tcp_radius_mm=65,
        grip_time_s=0.6,
        release_time_s=0.4,
    )

    tasks = [
        PickPlaceTask(
            part_name="PartA",
            pick_pose=Pose(720, -420, 260),
            place_pose=Pose(760, 520, 280),
            approach_height_mm=220,
            depart_height_mm=220,
            part_mass_kg=2.5,
        ),
        PickPlaceTask(
            part_name="PartB",
            pick_pose=Pose(900, -360, 260),
            place_pose=Pose(960, 500, 280),
            approach_height_mm=220,
            depart_height_mm=220,
            part_mass_kg=2.8,
        ),
    ]

    obstacles = [
        AABBObstacle(
            name="Input conveyor guard",
            min_x=600,
            min_y=-650,
            min_z=0,
            max_x=1050,
            max_y=-540,
            max_z=620,
            safety_margin_mm=40,
        ),
        AABBObstacle(
            name="Fixture column",
            min_x=800,
            min_y=80,
            min_z=0,
            max_x=940,
            max_y=220,
            max_z=760,
            safety_margin_mm=40,
        ),
        AABBObstacle(
            name="Output tray wall",
            min_x=620,
            min_y=680,
            min_z=0,
            max_x=1160,
            max_y=800,
            max_z=520,
            safety_margin_mm=40,
        ),
    ]

    return WorkstationConfig(
        robot=robot,
        tool=tool,
        tasks=tasks,
        obstacles=obstacles,
        home_pose=Pose(600, 0, 900),
        sample_step_mm=25,
    )


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    config = build_demo_config()
    coordinator = WorkstationSimulationCoordinator()
    result = coordinator.run(config)
    paths = coordinator.export(config, result)

    print("Industrial Robot Workstation Simulation Agent")
    print("=" * 54)
    print(f"Status: {'PASS' if result.passed else 'NEEDS OPTIMIZATION'}")
    print(f"Waypoints: {len(result.waypoints)}")
    print(f"Collisions: {len(result.collisions)}")
    print(f"Reachability issues: {len(result.reachability_issues)}")
    print(f"Estimated cycle time: {result.total_cycle_time_s:.2f} s")
    print("")
    print("Suggestions:")
    for i, suggestion in enumerate(result.suggestions, start=1):
        print(f"  {i}. {suggestion}")
    print("")
    print("Generated files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
