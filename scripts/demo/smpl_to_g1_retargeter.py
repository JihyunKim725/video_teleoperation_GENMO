# SPDX-License-Identifier: Apache-2.0
"""
SMPL → G1 리타겟팅 모듈 (SONIC 원본 pico_manager 기반 수정)

GENMO 출력 SMPL 파라미터를 G1 29-DOF 관절각 + SONIC ZMQ v3 형식으로 변환.

두 가지 모드:
    1. sonic_v3 : 손목 6DOF (elbow swing + wrist 결합), 나머지 SMPL 데이터 패스스루
    2. full      : 전체 29-DOF Euler 변환

수정 이력:
    v2 (2026-04) — pico_manager_thread_server.py 원본 기반 전면 수정
        - smpl_joints: transl 제거, smpl_root_ytoz_up + remove_smpl_base_rot + quat_apply 적용
        - wrist 인덱스: IsaacLab 인터리브 방식 [23,25,27] / [24,26,28]
        - wrist 값: elbow swing + wrist 결합 공식 (decompose_rotation_aa)
        - body_quat: ytoz_up + remove_smpl_base_rot 적용 후 quaternion

출처:
    - pico_manager_thread_server.py: process_smpl_joints(), PoseStreamer.run_once()
    - g1_29dof.urdf: 관절 이름/순서/축/범위 (URDF 검증 완료)
    - observation_config.yaml: SONIC 인코더 모드 "smpl" (mode_id=2)
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# ============================================================================
# gear_sonic 유틸 임포트
# 출처: pico_manager_thread_server.py line 39-46, 67-72
# ============================================================================

try:
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        compute_human_joints,
        quat_apply,
        quat_inv,
        quaternion_to_angle_axis,
        quaternion_to_rotation_matrix,
    )
    from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up

    _GEAR_SONIC_AVAILABLE = True
except ImportError:
    _GEAR_SONIC_AVAILABLE = False
    logger.warning(
        "[Retargeter] gear_sonic 유틸 임포트 실패 — fallback FK 사용 (정확도 낮음). "
        "gear_sonic_sim 환경에서 실행하세요."
    )


# ============================================================================
# G1 29-DOF 정의 (g1_29dof.urdf 검증 완료)
# ============================================================================

# fmt: off
G1_JOINT_NAMES = [
    "left_hip_pitch",       # 0   Y
    "left_hip_roll",        # 1   X
    "left_hip_yaw",         # 2   Z
    "left_knee",            # 3   Y
    "left_ankle_pitch",     # 4   Y
    "left_ankle_roll",      # 5   X
    "right_hip_pitch",      # 6   Y
    "right_hip_roll",       # 7   X
    "right_hip_yaw",        # 8   Z
    "right_knee",           # 9   Y
    "right_ankle_pitch",    # 10  Y
    "right_ankle_roll",     # 11  X
    "waist_yaw",            # 12  Z
    "waist_roll",           # 13  X
    "waist_pitch",          # 14  Y
    "left_shoulder_pitch",  # 15  Y
    "left_shoulder_roll",   # 16  X
    "left_shoulder_yaw",    # 17  Z
    "left_elbow",           # 18  Y
    "left_wrist_roll",      # 19  X  (URDF 순서)
    "left_wrist_pitch",     # 20  Y
    "left_wrist_yaw",       # 21  Z
    "right_shoulder_pitch", # 22  Y
    "right_shoulder_roll",  # 23  X  ← IsaacLab ZMQ: L_wrist_roll
    "right_shoulder_yaw",   # 24  Z  ← IsaacLab ZMQ: R_wrist_roll
    "right_elbow",          # 25  Y  ← IsaacLab ZMQ: L_wrist_pitch
    "right_wrist_roll",     # 26  X  ← IsaacLab ZMQ: R_wrist_pitch
    "right_wrist_pitch",    # 27  Y  ← IsaacLab ZMQ: L_wrist_yaw
    "right_wrist_yaw",      # 28  Z  ← IsaacLab ZMQ: R_wrist_yaw
]

# IsaacLab ZMQ v3 wrist 인덱스 (pico_manager line 1354-1362)
# URDF 순서와 다른 인터리브 배치 — ZMQ 송수신 시 이 인덱스 사용
G1_L_WRIST_ROLL_IDX  = 23
G1_L_WRIST_PITCH_IDX = 25
G1_L_WRIST_YAW_IDX   = 27
G1_R_WRIST_ROLL_IDX  = 24
G1_R_WRIST_PITCH_IDX = 26
G1_R_WRIST_YAW_IDX   = 28

G1_JOINT_LIMITS_DEFAULT = torch.tensor([
    [-2.5307, +2.8798], [-0.5236, +2.9671], [-2.7576, +2.7576],
    [-0.0873, +2.8798], [-0.8727, +0.5236], [-0.2618, +0.2618],
    [-2.5307, +2.8798], [-2.9671, +0.5236], [-2.7576, +2.7576],
    [-0.0873, +2.8798], [-0.8727, +0.5236], [-0.2618, +0.2618],
    [-2.6180, +2.6180], [-0.5200, +0.5200], [-0.5200, +0.5200],
    [-3.0892, +2.6704], [-1.5882, +2.2515], [-2.6180, +2.6180],
    [-1.0472, +2.0944], [-1.9722, +1.9722], [-1.6144, +1.6144],
    [-1.6144, +1.6144], [-3.0892, +2.6704], [-2.2515, +1.5882],
    [-2.6180, +2.6180], [-1.0472, +2.0944], [-1.9722, +1.9722],
    [-1.6144, +1.6144], [-1.6144, +1.6144],
])
# fmt: on

# SMPL body_pose 관절 인덱스 (body_pose[i] = body_pose_flat[i*3:(i+1)*3])
SMPL_BP = {
    "left_hip": 0,   "right_hip": 1,   "spine1": 2,
    "left_knee": 3,  "right_knee": 4,  "spine2": 5,
    "left_ankle": 6, "right_ankle": 7, "spine3": 8,
    "left_foot": 9,  "right_foot": 10, "neck": 11,
    "left_collar": 12,   "right_collar": 13, "head": 14,
    "left_shoulder": 15, "right_shoulder": 16,
    "left_elbow": 17,    "right_elbow": 18,
    "left_wrist": 19,    "right_wrist": 20,
}

# T-pose 골격 (human_joints_info.pkl 검증)
T_POSE_22 = torch.tensor([
    [+0.003, -0.351, +0.012], [+0.061, -0.444, -0.014],
    [-0.060, -0.455, -0.009], [+0.000, -0.242, -0.016],
    [+0.116, -0.823, -0.023], [-0.104, -0.818, -0.026],
    [+0.010, -0.110, -0.022], [+0.073, -1.226, -0.055],
    [-0.089, -1.228, -0.046], [-0.002, -0.057, +0.007],
    [+0.120, -1.284, +0.063], [-0.128, -1.287, +0.073],
    [-0.014, +0.108, -0.025], [+0.045, +0.028, +0.000],
    [-0.049, +0.027, -0.007], [+0.011, +0.268, -0.004],
    [+0.164, +0.085, -0.016], [-0.152, +0.080, -0.019],
    [+0.418, +0.013, -0.058], [-0.423, +0.044, -0.046],
    [+0.670, +0.036, -0.061], [-0.672, +0.039, -0.061],
])
SMPL_PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


# ============================================================================
# 회전 변환 유틸 (fallback용)
# ============================================================================


def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (*, 3) → rotation matrix (*, 3, 3). Rodrigues 공식."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / theta
    cos_t = torch.cos(theta)[..., None]
    sin_t = torch.sin(theta)[..., None]
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zero = torch.zeros_like(x)
    K = torch.cat([
        torch.cat([zero,  -z,   y], dim=-1).unsqueeze(-2),
        torch.cat([z,   zero,  -x], dim=-1).unsqueeze(-2),
        torch.cat([-y,     x, zero], dim=-1).unsqueeze(-2),
    ], dim=-2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype)
    return I + sin_t * K + (1 - cos_t) * (K @ K)


def _safe_euler_from_rotmat(R_mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → Euler [pitch_Y, roll_X, yaw_Z]. 짐벌락 방지."""
    sy      = torch.sqrt(R_mat[..., 0, 0] ** 2 + R_mat[..., 1, 0] ** 2)
    singular = sy < 1e-6
    pitch   = torch.atan2(-R_mat[..., 2, 0], sy)
    roll    = torch.atan2( R_mat[..., 2, 1], R_mat[..., 2, 2])
    yaw     = torch.atan2( R_mat[..., 1, 0], R_mat[..., 0, 0])
    pitch_s = torch.atan2(-R_mat[..., 2, 0], sy)
    roll_s  = torch.atan2(-R_mat[..., 0, 2], R_mat[..., 1, 1])
    yaw_s   = torch.zeros_like(yaw)
    pitch = torch.where(singular, pitch_s, pitch)
    roll  = torch.where(singular, roll_s,  roll)
    yaw   = torch.where(singular, yaw_s,   yaw)
    return torch.stack([pitch, roll, yaw], dim=-1)


def smpl_aa_to_euler(aa: torch.Tensor) -> torch.Tensor:
    """SMPL axis-angle → Euler [pitch, roll, yaw]."""
    return _safe_euler_from_rotmat(aa_to_rotmat(aa))


def compose_rotations(aa1: torch.Tensor, aa2: torch.Tensor) -> torch.Tensor:
    """두 axis-angle을 결합한 rotation matrix 반환."""
    return aa_to_rotmat(aa1) @ aa_to_rotmat(aa2)


# ============================================================================
# fallback SMPL FK (gear_sonic 없을 때만 사용)
# ============================================================================


def smpl_fk_22joints(body_pose, global_orient, transl):
    """SMPL FK → 22관절 위치. (gear_sonic 없을 때 fallback)"""
    single = body_pose.ndim == 1
    if single:
        body_pose     = body_pose.unsqueeze(0)
        global_orient = global_orient.unsqueeze(0)
        transl        = transl.unsqueeze(0)

    N       = body_pose.shape[0]
    device  = body_pose.device
    J_tpose = T_POSE_22.to(device)
    parents = SMPL_PARENTS_22

    full_aa  = torch.cat([global_orient, body_pose], dim=-1).reshape(N, 22, 3)
    rot_mats = aa_to_rotmat(full_aa)

    rel_joints = J_tpose.unsqueeze(0).expand(N, -1, -1).clone()
    for i in range(1, 22):
        rel_joints[:, i] = J_tpose[i] - J_tpose[parents[i]]

    transforms = torch.zeros(N, 22, 4, 4, device=device)
    transforms[:, :, 3, 3] = 1.0
    for i in range(22):
        transforms[:, i, :3, :3] = rot_mats[:, i]
        transforms[:, i, :3,  3] = rel_joints[:, i]

    global_T = [transforms[:, 0]]
    for i in range(1, 22):
        global_T.append(global_T[parents[i]] @ transforms[:, i])
    global_T = torch.stack(global_T, dim=1)
    joints   = global_T[:, :, :3, 3] + transl.unsqueeze(1)

    return joints.squeeze(0) if single else joints


def smpl_fk_24joints(body_pose, global_orient, transl):
    """SMPL FK → 24관절. fallback용."""
    single = body_pose.ndim == 1
    if single:
        body_pose     = body_pose.unsqueeze(0)
        global_orient = global_orient.unsqueeze(0)
        transl        = transl.unsqueeze(0)

    j22  = smpl_fk_22joints(body_pose, global_orient, transl)
    head = j22[:, 15:16]
    nose = head + torch.tensor([0, 0.05, 0.02], device=j22.device)
    leye = head + torch.tensor([0.03, 0.06, 0.04], device=j22.device)
    j24  = torch.cat([j22, nose, leye], dim=1)

    return j24.squeeze(0) if single else j24


# ============================================================================
# 설정
# ============================================================================


@dataclass
class RetargetConfig:
    g1_urdf_path: Optional[str]  = None
    joint_limit_margin: float    = 0.95
    velocity_dt: float           = 1.0 / 30.0
    max_joint_velocity: float    = 5.0        # rad/s
    smooth_window: int           = 3
    translation_scale: float     = 0.65       # 인간→G1 root 스케일
    floor_z_offset: float        = 0.0


# ============================================================================
# 메인 리타겟터
# ============================================================================


class SmplToG1Retargeter:
    """SMPL → G1 29-DOF 리타겟팅.

    사용법:
        retargeter = SmplToG1Retargeter(RetargetConfig(g1_urdf_path="path/to/g1.urdf"))
        result     = retargeter.retarget(smpl_params, mode="sonic_v3")
    """

    def __init__(self, config: Optional[RetargetConfig] = None):
        self._config = config or RetargetConfig()
        if self._config.g1_urdf_path and Path(self._config.g1_urdf_path).exists():
            self._joint_limits = self._load_limits_from_urdf(self._config.g1_urdf_path)
        else:
            self._joint_limits = G1_JOINT_LIMITS_DEFAULT.clone()

        margin = self._config.joint_limit_margin
        center = self._joint_limits.mean(dim=1, keepdim=True)
        half   = (self._joint_limits[:, 1:2] - self._joint_limits[:, 0:1]) / 2
        self._safe_limits = torch.cat(
            [center - half * margin, center + half * margin], dim=1
        )
        self._stats = {"total_frames": 0, "limit_violations": 0}

    @staticmethod
    def _load_limits_from_urdf(urdf_path: str) -> torch.Tensor:
        tree   = ET.parse(urdf_path)
        limits = []
        for j in tree.getroot().findall(".//joint"):
            if j.get("type") == "fixed":
                continue
            lim = j.find("limit")
            if lim is not None:
                limits.append([float(lim.get("lower", 0)), float(lim.get("upper", 0))])
        return torch.tensor(limits[:29])

    # ------------------------------------------------------------------
    # 핵심 API
    # ------------------------------------------------------------------

    def retarget(self, smpl_params: dict, mode: str = "sonic_v3") -> dict:
        """SMPL → G1 변환.

        입력: body_pose(F,63), global_orient(F,3), transl(F,3), betas(F,10)
        출력: joint_pos(F,29), joint_vel(F,29), smpl_joints(F,24,3), smpl_pose(F,21,3)

        Note: transl은 smpl_joints 계산에 사용하지 않음 (pico_manager 원본 동일).
        """
        bp = smpl_params["body_pose"]      # (Nf, 63)
        go = smpl_params["global_orient"]  # (Nf, 3)
        Nf = bp.shape[0]
        self._stats["total_frames"] += Nf

        # ── smpl_joints / smpl_pose: pico_manager process_smpl_joints() 동일
        smpl_joints, smpl_pose = self._compute_smpl_joints_local(bp, go)

        if mode == "sonic_v3":
            joint_pos = self._retarget_sonic_v3(bp)
        else:
            joint_pos = self._retarget_full(bp)

        joint_pos, nv = self._apply_joint_limits(joint_pos)
        self._stats["limit_violations"] += nv
        joint_vel = self._compute_joint_velocity(joint_pos)
        joint_vel = self._clamp_velocity(joint_vel)

        return {
            "joint_pos":   joint_pos,    # (Nf, 29)
            "joint_vel":   joint_vel,    # (Nf, 29)
            "smpl_joints": smpl_joints,  # (Nf, 24, 3)
            "smpl_pose":   smpl_pose,    # (Nf, 21, 3)
        }

    # ------------------------------------------------------------------
    # smpl_joints_local 계산 (pico_manager 원본 동일)
    # ------------------------------------------------------------------

    def _compute_smpl_joints_local(
        self,
        body_pose: torch.Tensor,      # (Nf, 63) axis-angle
        global_orient: torch.Tensor,  # (Nf, 3)  axis-angle
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SONIC 원본과 동일한 smpl_joints_local 계산.

        출처: pico_manager_thread_server.py process_smpl_joints() (line 461-477)
             compute_human_joints 시그니처 확인 완료:
               (body_pose, global_orient, human_joints_info_path, use_thumb_joints)
               global_orient = axis-angle 입력

        처리 순서:
            1. global_orient aa  → quaternion
            2. smpl_root_ytoz_up : Y-up → Z-up 좌표계 변환
            3. quaternion        → aa   (compute_human_joints 입력용)
            4. compute_human_joints(body_pose, go_aa_z) : transl 없이 FK
            5. remove_smpl_base_rot : SMPL 기저 회전 제거
            6. quat_apply(quat_inv(go_final), joints) : local pelvis frame 변환

        검증 완료 (2026-04):
            Δz(head - ankle) = 0.992m > 0.8  →  Z-up 정상
            joints 최대값    = 0.966m < 1.5  →  인체 스케일 정상
        """
        Nf        = body_pose.shape[0]
        smpl_pose = body_pose.reshape(Nf, 21, 3)   # 변환 없이 그대로 반환

        if not _GEAR_SONIC_AVAILABLE:
            # fallback: transl=0으로 FK (좌표계 변환 없음, 정확도 낮음)
            logger.warning("[Retargeter] gear_sonic 없음 — fallback FK (transl=0)")
            j = smpl_fk_24joints(
                body_pose, global_orient, torch.zeros_like(global_orient)
            )
            return j, smpl_pose

        # Step 1: axis-angle → quaternion (w-first)
        # 출처: pico_manager line 461
        go_quat = angle_axis_to_quaternion(global_orient)    # (Nf, 4)

        # Step 2: Y-up → Z-up 변환
        # 출처: pico_manager line 462-463
        go_quat_z = smpl_root_ytoz_up(go_quat)               # (Nf, 4)

        # Step 3: quaternion → axis-angle (compute_human_joints는 aa 입력)
        # 출처: pico_manager line 464
        go_aa_z = quaternion_to_angle_axis(go_quat_z)        # (Nf, 3)

        # Step 4: FK — transl 없이 관절 위치 계산
        # 출처: pico_manager line 467-470
        # 시그니처: (body_pose, global_orient, human_joints_info_path, use_thumb_joints)
        joints = compute_human_joints(
            body_pose=body_pose,     # (Nf, 63) axis-angle
            global_orient=go_aa_z,  # (Nf, 3)  Z-up 변환 후 axis-angle
        )                            # (Nf, J, 3)  J ≥ 24

        # ZMQ v3: smpl_joints [N, 24, 3] — 앞 24관절만 사용
        joints = joints[:, :24, :]   # (Nf, 24, 3)

        # Step 5: SMPL 기저 회전 제거
        # 출처: pico_manager line 473-474
        go_quat_final = remove_smpl_base_rot(go_quat_z, w_last=False)   # (Nf, 4)

        # Step 6: local pelvis frame 변환
        # 출처: pico_manager line 476-477
        go_inv     = quat_inv(go_quat_final)                             # (Nf, 4)
        go_inv_exp = go_inv.unsqueeze(1).expand(-1, 24, -1)              # (Nf, 24, 4)
        smpl_joints_local = quat_apply(go_inv_exp, joints)               # (Nf, 24, 3)

        return smpl_joints_local, smpl_pose

    # ------------------------------------------------------------------
    # 모드별 변환
    # ------------------------------------------------------------------

    def _retarget_sonic_v3(self, body_pose: torch.Tensor) -> torch.Tensor:
        """SONIC v3: 손목 6DOF (elbow swing + wrist 결합 공식).

        출처: pico_manager_thread_server.py PoseStreamer.run_once() (line 1345-1403)

        IsaacLab 인터리브 인덱스 (pico_manager line 1354-1362):
            joint_pos[23] = L_wrist_roll    joint_pos[24] = R_wrist_roll
            joint_pos[25] = L_wrist_pitch   joint_pos[26] = R_wrist_pitch
            joint_pos[27] = L_wrist_yaw     joint_pos[28] = R_wrist_yaw

        wrist 계산 공식 (pico_manager line 1389-1395):
            L_roll  =  elbow_swing_X + wrist_X
            L_pitch = -wrist_Y
            L_yaw   =  elbow_swing_Z + wrist_Z
            R_roll  = -(elbow_swing_X + wrist_X)   ← 부호 반전
            R_pitch = -wrist_Y
            R_yaw   =  elbow_swing_Z + wrist_Z
        """
        Nf    = body_pose.shape[0]
        jp_np = np.zeros((Nf, 29), dtype=np.float32)
        bp_np = body_pose.detach().cpu().numpy().reshape(Nf, 21, 3)

        # SMPL 관절 인덱스 (출처: pico_manager line 1349-1353)
        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        smpl_l_elbow_aa = bp_np[:, SMPL_L_ELBOW_IDX]   # (Nf, 3)
        smpl_l_wrist_aa = bp_np[:, SMPL_L_WRIST_IDX]   # (Nf, 3)
        smpl_r_elbow_aa = bp_np[:, SMPL_R_ELBOW_IDX]   # (Nf, 3)
        smpl_r_wrist_aa = bp_np[:, SMPL_R_WRIST_IDX]   # (Nf, 3)

        # elbow twist-swing 분해 (G1 elbow 축: Y)
        # 출처: pico_manager line 1368-1376
        g1_elbow_axis = np.array([0.0, 1.0, 0.0])

        if _GEAR_SONIC_AVAILABLE:
            _, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_elbow_axis)
            _, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_elbow_axis)

            # swing quaternion → Euler XYZ
            # decompose_rotation_aa 반환: (Nf, 4) w-first → scipy: [x,y,z,w]
            # 출처: pico_manager line 1379-1388
            l_swing_euler = R.from_quat(
                g1_l_elbow_q_swing[:, [1, 2, 3, 0]]
            ).as_euler("XYZ", degrees=False)   # (Nf, 3)
            r_swing_euler = R.from_quat(
                g1_r_elbow_q_swing[:, [1, 2, 3, 0]]
            ).as_euler("XYZ", degrees=False)
        else:
            # fallback: elbow swing 기여 없이 wrist만
            l_swing_euler = np.zeros((Nf, 3), dtype=np.float32)
            r_swing_euler = np.zeros((Nf, 3), dtype=np.float32)
            logger.warning("[Retargeter] gear_sonic 없음 — elbow swing 기여 생략")

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        # wrist 값 계산 (출처: pico_manager line 1389-1395)
        g1_l_wrist_roll  =  (l_swing_euler[:, 0] + l_wrist_euler[:, 0])
        g1_l_wrist_pitch =  (-l_wrist_euler[:, 1])
        g1_l_wrist_yaw   =  (l_swing_euler[:, 2] + l_wrist_euler[:, 2])

        g1_r_wrist_roll  = -(r_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch =  (-r_wrist_euler[:, 1])
        g1_r_wrist_yaw   =  (r_swing_euler[:, 2] + r_wrist_euler[:, 2])

        # IsaacLab 인터리브 인덱스로 할당 (출처: pico_manager line 1397-1403)
        jp_np[:, G1_L_WRIST_ROLL_IDX]  = g1_l_wrist_roll
        jp_np[:, G1_L_WRIST_PITCH_IDX] = g1_l_wrist_pitch
        jp_np[:, G1_L_WRIST_YAW_IDX]   = g1_l_wrist_yaw
        jp_np[:, G1_R_WRIST_ROLL_IDX]  = g1_r_wrist_roll
        jp_np[:, G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch
        jp_np[:, G1_R_WRIST_YAW_IDX]   = g1_r_wrist_yaw

        return torch.from_numpy(jp_np).to(body_pose.device)

    def _retarget_full(self, body_pose: torch.Tensor) -> torch.Tensor:
        """전체 29-DOF: collar+shoulder 결합, 다축 분해 포함."""
        Nf = body_pose.shape[0]
        bp = body_pose.reshape(Nf, 21, 3)
        jp = torch.zeros(Nf, 29, device=body_pose.device)

        def _bp(name):
            return bp[:, SMPL_BP[name]]

        def _e(aa):
            return smpl_aa_to_euler(aa)

        # 왼쪽 다리
        e = _e(_bp("left_hip"))
        jp[:, 0], jp[:, 1], jp[:, 2] = e[:, 0], e[:, 1], e[:, 2]
        jp[:, 3] = _e(_bp("left_knee"))[:, 0]
        e = _e(_bp("left_ankle"))
        jp[:, 4], jp[:, 5] = e[:, 0], e[:, 1]

        # 오른쪽 다리
        e = _e(_bp("right_hip"))
        jp[:, 6], jp[:, 7], jp[:, 8] = e[:, 0], e[:, 1], e[:, 2]
        jp[:, 9] = _e(_bp("right_knee"))[:, 0]
        e = _e(_bp("right_ankle"))
        jp[:, 10], jp[:, 11] = e[:, 0], e[:, 1]

        # 허리: spine1 + spine2*0.5 결합
        R_waist = compose_rotations(_bp("spine1"), _bp("spine2") * 0.5)
        e = _safe_euler_from_rotmat(R_waist)
        jp[:, 12], jp[:, 13], jp[:, 14] = e[:, 2], e[:, 1], e[:, 0]

        # 왼팔: collar+shoulder 결합
        R_ls = compose_rotations(_bp("left_collar"), _bp("left_shoulder"))
        e = _safe_euler_from_rotmat(R_ls)
        jp[:, 15], jp[:, 16], jp[:, 17] = e[:, 0], e[:, 1], e[:, 2]
        jp[:, 18] = _e(_bp("left_elbow"))[:, 0]
        e = _e(_bp("left_wrist"))
        jp[:, 19], jp[:, 20], jp[:, 21] = e[:, 1], e[:, 0], e[:, 2]

        # 오른팔
        R_rs = compose_rotations(_bp("right_collar"), _bp("right_shoulder"))
        e = _safe_euler_from_rotmat(R_rs)
        jp[:, 22], jp[:, 23], jp[:, 24] = e[:, 0], e[:, 1], e[:, 2]
        jp[:, 25] = _e(_bp("right_elbow"))[:, 0]
        e = _e(_bp("right_wrist"))
        jp[:, 26], jp[:, 27], jp[:, 28] = e[:, 1], e[:, 0], e[:, 2]

        return jp

    # ------------------------------------------------------------------
    # 안전 / 속도
    # ------------------------------------------------------------------

    def _apply_joint_limits(self, jp: torch.Tensor) -> tuple[torch.Tensor, int]:
        lo = self._safe_limits[:, 0].to(jp.device)
        hi = self._safe_limits[:, 1].to(jp.device)
        nv = int(((jp < lo) | (jp > hi)).sum().item())
        if nv > 0:
            logger.debug(f"[Retargeter] 관절 한계 위반 {nv}개 → 클램핑")
        return torch.clamp(jp, lo, hi), nv

    def _compute_joint_velocity(self, jp: torch.Tensor) -> torch.Tensor:
        dt  = self._config.velocity_dt
        vel = torch.zeros_like(jp)
        if jp.shape[0] > 1:
            vel[1:] = (jp[1:] - jp[:-1]) / dt
        return vel

    def _clamp_velocity(self, vel: torch.Tensor) -> torch.Tensor:
        mv = self._config.max_joint_velocity
        n  = int((vel.abs() > mv).sum().item())
        if n > 0:
            logger.debug(f"[Retargeter] 속도 클램핑 {n}개 (>{mv:.1f} rad/s)")
        return torch.clamp(vel, -mv, mv)

    def smooth(self, jp: torch.Tensor) -> torch.Tensor:
        """이동평균 스무딩."""
        w = self._config.smooth_window
        if w <= 1 or jp.shape[0] < w:
            return jp
        kernel = torch.ones(1, 1, w, device=jp.device) / w
        x = jp.T.unsqueeze(1)
        x = F.pad(x, (w // 2, w // 2), mode="replicate")
        return F.conv1d(x, kernel).squeeze(1).T

    @property
    def stats(self):
        return self._stats.copy()


# ============================================================================
# CLI 테스트
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SMPL→G1 리타겟팅 테스트")
    parser.add_argument("--mode",   default="both", choices=["sonic_v3", "full", "both"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--urdf",   default=None)
    parser.add_argument(
        "--smpl", default=None,
        help="smpl_params.pt 경로 (실제 GENMO 출력으로 검증)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config     = RetargetConfig(g1_urdf_path=args.urdf)
    retargeter = SmplToG1Retargeter(config)

    if args.smpl:
        data   = torch.load(args.smpl, map_location="cpu")
        params = data["body_params_global"]
        Nf     = params["body_pose"].shape[0]
        smpl_params = {
            "body_pose":     params["body_pose"],
            "global_orient": params["global_orient"],
            "transl":        params["transl"],
            "betas":         params.get("betas", torch.zeros(Nf, 10)),
        }
        print(f"실제 GENMO 출력 검증: {Nf}프레임")
    else:
        Nf = args.frames
        smpl_params = {
            "body_pose":     torch.randn(Nf, 63) * 0.3,
            "global_orient": torch.randn(Nf, 3)  * 0.1,
            "transl":        torch.randn(Nf, 3)  * 0.5,
            "betas":         torch.zeros(Nf, 10),
        }

    errors = []

    def _test(mode):
        r = retargeter.retarget(smpl_params, mode=mode)
        for k, s in [
            ("joint_pos",   (Nf, 29)),
            ("joint_vel",   (Nf, 29)),
            ("smpl_joints", (Nf, 24, 3)),
            ("smpl_pose",   (Nf, 21, 3)),
        ]:
            ok = tuple(r[k].shape) == s
            print(f"  {'✓' if ok else '✗'} {k}: {r[k].shape} == {s}")
            if not ok:
                errors.append(f"{mode}: {k} shape 오류")

        sj   = r["smpl_joints"].numpy()
        jmax = np.abs(sj).max()
        dz   = float(sj[:, 15, 2].mean() - sj[:, 7, 2].mean())
        print(f"  smpl_joints 최대값: {jmax:.3f}m  (기대: < 1.5)")
        print(f"  Δz head-ankle:      {dz:.3f}m   (기대: > 0.8 → Z-up)")
        if jmax >= 1.5:
            errors.append(f"{mode}: smpl_joints 범위 초과 {jmax:.2f}m")
        if dz < 0.8:
            errors.append(f"{mode}: Z-up 이상 Δz={dz:.3f}")
        return r

    if args.mode in ("sonic_v3", "both"):
        print("\n=== SONIC v3 모드 테스트 ===")
        r = _test("sonic_v3")

        # 인터리브 wrist 인덱스만 non-zero
        wrist_idx = [G1_L_WRIST_ROLL_IDX, G1_L_WRIST_PITCH_IDX, G1_L_WRIST_YAW_IDX,
                     G1_R_WRIST_ROLL_IDX, G1_R_WRIST_PITCH_IDX, G1_R_WRIST_YAW_IDX]
        other_idx = [i for i in range(29) if i not in wrist_idx]
        z_other   = r["joint_pos"][:, other_idx].abs().sum().item()
        print(f"  {'✓' if z_other == 0 else '✗'} 비손목 관절 합 = {z_other:.6f}  (기대: 0)")
        if z_other != 0:
            errors.append("sonic_v3: 비손목 관절 non-zero")
        print(f"  관절 한계 위반: {retargeter.stats['limit_violations']}개")

    if args.mode in ("full", "both"):
        print("\n=== 전체 29-DOF 모드 테스트 ===")
        _test("full")

    print(f"\n{'='*50}")
    print(f"결과: {'FAIL — ' + str(len(errors)) + '개 오류' if errors else 'ALL PASSED'}")
    if errors:
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()