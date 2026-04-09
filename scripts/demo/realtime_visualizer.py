# SPDX-License-Identifier: Apache-2.0
"""
실시간 SMPL 메시 시각화 모듈

GENMO 추론 결과(InferenceResult)를 받아 incam/global 뷰를 실시간 렌더링.
별도 프로세스에서 실행하여 추론 파이프라인을 블로킹하지 않음.

사용법:
    from realtime_visualizer import RealtimeVisualizer, VisualizerConfig

    vis = RealtimeVisualizer(VisualizerConfig(record=True))
    vis.start()

    while running:
        result = inference.run(data_dict)
        vis.update(result, camera_frame=frame_rgb)

    vis.stop()  # → 1_incam.mp4, 2_global.mp4, smpl_params.pt 저장

재사용 코드:
    - gem/utils/vis/pyrender_incam.py: Renderer — incam 메시 오버레이
    - gem/utils/vis/o3d_render.py: create_meshes, get_ground, Settings
    - gem/utils/vis/renderer.py: get_global_cameras_static_v2, get_ground_params_from_points
    - gem/utils/smplx_utils.py: make_smplx("supermotion") — SMPL 바디 모델
    - demo_utils.py: render_incam_frames, render_global_frames, normalize_global_verts
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 설정
# ============================================================================


@dataclass
class VisualizerConfig:
    """시각화 설정."""

    # 디스플레이
    show_window: bool = True         # cv2.imshow 윈도우 표시
    window_name: str = "GEM Realtime"
    layout: str = "side_by_side"     # "side_by_side" | "incam_only" | "global_only"

    # 렌더링
    incam_color: tuple = (0.8, 0.8, 0.9)   # incam 메시 색상 (R, G, B)
    global_color: tuple = (0.69, 0.39, 0.96)  # global 메시 색상 (보라색, demo_utils와 동일)
    render_width: int = 640
    render_height: int = 480

    # 녹화
    record: bool = False
    output_dir: str = "outputs/realtime"
    video_fps: int = 30

    # 성능
    queue_maxsize: int = 3           # 렌더링 큐 (추론→시각화)
    target_render_fps: int = 30


# ============================================================================
# 렌더링 워커 (별도 프로세스)
# ============================================================================


def _egl_render_worker(
    config: VisualizerConfig,
    render_queue: mp.Queue,
    frame_queue: mp.Queue,
    stats_queue: mp.Queue,
    status_queue: mp.Queue,
    stop_event: mp.Event,
):
    """EGL 전용 렌더링 프로세스 (spawn 방식으로 시작).

    PYOPENGL_PLATFORM=egl을 이 프로세스 내부에만 격리.
    pyrender로 렌더링한 numpy frame을 frame_queue로 전달.
    cv2.imshow는 일절 사용하지 않음 — _display_worker가 담당.

    출처:
        - pyrender_incam.py: Renderer — incam 오버레이
        - o3d_render.py: create_meshes, Settings — global 뷰
        - renderer.py: get_global_cameras_static_v2 — 카메라 위치 계산
    """
    import traceback

    # 최상위 예외 캐치 — 프로세스 crash 원인을 stderr에 기록
    # (부모 프로세스와 logger를 공유하지 않으므로 stderr 직접 사용)
    try:
        _egl_render_worker_main(
            config, render_queue, frame_queue, stats_queue, status_queue, stop_event
        )
    except Exception:
        print("[EGLRenderProcess] 치명적 오류 — 프로세스 종료:", flush=True)
        traceback.print_exc()
        raise


def _egl_render_worker_main(
    config: VisualizerConfig,
    render_queue: mp.Queue,
    frame_queue: mp.Queue,
    stats_queue: mp.Queue,
    status_queue: mp.Queue,
    stop_event: mp.Event,
):
    """_egl_render_worker의 실제 구현체 (예외 캐치 분리용)."""
    import os

    # EGL을 이 프로세스(fork)에만 격리 설정.
    # cv2.imshow는 _display_worker(별도 프로세스)에서만 실행되므로 충돌 없음.
    # fork 컨텍스트이므로 부모의 sys.path를 그대로 상속 — 별도 경로 조작 불필요.
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    def _send_status(msg: str, step: int, total: int):
        """status_queue에 비블로킹으로 로딩 상태 전송."""
        try:
            status_queue.put_nowait({"msg": msg, "step": step, "total": total})
        except queue.Full:
            pass

    _send_status("Loading SMPL model... (1/3)", 0, 3)

    from gem.utils.smplx_utils import make_smplx

    # SMPL 바디 모델 로드 (렌더링용 — full vertices 필요)
    # 출처: demo_utils.py line 136 — make_smplx("supermotion")
    body_model = make_smplx("supermotion")
    body_model.eval()
    smpl_faces = body_model.faces.astype(np.int32)

    _send_status("Initializing incam renderer... (2/3)", 1, 3)

    # incam 렌더러 초기화 (pyrender 기반)
    # 출처: gem/utils/vis/pyrender_incam.py
    from gem.utils.vis.pyrender_incam import Renderer as PyRenderIncam
    incam_renderer = None  # focal_length 필요 → 첫 프레임에서 초기화

    _send_status("Initializing global renderer... (3/3)", 2, 3)

    # GlobalRenderer 초기화 — OffscreenRenderer 1회 생성 (EGL 컨텍스트 캐시)
    # layout이 global을 포함할 때만 생성
    if config.layout in ("side_by_side", "global_only"):
        global_renderer = GlobalRenderer(
            config.render_width, config.render_height,
            config.global_color, smpl_faces,
        )
    else:
        global_renderer = None

    _send_status("Ready — waiting for inference...", 3, 3)

    # 성능 측정
    frame_count = 0
    fps_timer = time.monotonic()
    current_fps = 0.0

    while not stop_event.is_set():
        try:
            item = render_queue.get(timeout=0.03)
        except queue.Empty:
            continue

        if item is None:  # 종료 신호
            break

        t0 = time.monotonic()

        camera_frame = item["camera_frame"]    # (H, W, 3) RGB uint8 또는 None
        body_params_incam = item["body_params_incam"]
        body_params_global = item["body_params_global"]
        K_fullimg = item["K_fullimg"]           # (3, 3) 또는 (L, 3, 3)
        perf_info = item.get("perf_info", {})
        bbx_xys = item.get("bbx_xys")

        # SMPL forward → vertices (incam + global 동시 처리)
        # 출처: demo_utils.py lines 146-152 — body_model forward
        bp_incam = _get_last_frame(body_params_incam)
        bp_global = _get_last_frame(body_params_global)

        incam_betas = (
            bp_incam["betas"].unsqueeze(0)
            if bp_incam.get("betas") is not None and bp_incam["betas"].ndim == 1
            else bp_incam.get("betas", torch.zeros(1, 10))
        )
        global_betas = bp_global.get("betas", torch.zeros(10))
        if global_betas.ndim == 1:
            global_betas = global_betas.unsqueeze(0)

        with torch.no_grad():
            smpl_out_incam = body_model(
                body_pose=bp_incam["body_pose"].unsqueeze(0),
                global_orient=bp_incam["global_orient"].unsqueeze(0),
                transl=bp_incam["transl"].unsqueeze(0),
                betas=incam_betas,
            )
            verts_incam = smpl_out_incam.vertices[0].numpy()  # (V, 3)

            if global_renderer is not None:
                smpl_out_global = body_model(
                    body_pose=bp_global["body_pose"].unsqueeze(0),
                    global_orient=bp_global["global_orient"].unsqueeze(0),
                    transl=bp_global["transl"].unsqueeze(0),
                    betas=global_betas,
                )
                verts_global = smpl_out_global.vertices[0].numpy()  # (V, 3)

        # --- Incam 렌더링 ---
        incam_frame = None

        # YOLO 바운딩 박스 그리기 (3D 메쉬 렌더링 전)
        if camera_frame is not None and bbx_xys is not None:
            import cv2 as _cv2
            cx, cy, s = bbx_xys
            half = s / 2.0
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)
            _cv2.rectangle(camera_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            _cv2.putText(camera_frame, "YOLO Person", (x1, y1 - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if camera_frame is not None and config.layout in ("side_by_side", "incam_only"):
            # focal length 추출
            K = K_fullimg
            if K.ndim == 3:
                K = K[0]  # 첫 프레임의 K
            if isinstance(K, torch.Tensor):
                K = K.numpy()
            focal = float(K[0, 0])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            # pyrender incam 렌더러 (지연 초기화)
            if incam_renderer is None or incam_renderer.focal_length != focal:
                incam_renderer = PyRenderIncam(focal_length=focal, faces=smpl_faces)

            # cam_t = transl (카메라 좌표계에서의 위치)
            cam_t = bp_incam["transl"].numpy()

            incam_frame = incam_renderer(
                vertices=verts_incam,
                cam_t=cam_t,
                image=camera_frame,
                mesh_base_color=config.incam_color,
                camera_center=[cx, cy],
            )

        # --- Global 렌더링 (GlobalRenderer 재사용 — EGL 컨텍스트 유지) ---
        global_frame = None
        if global_renderer is not None:
            try:
                global_frame = global_renderer.render(verts_global)
            except Exception as e:
                logger.debug(f"[GlobalRenderer] 렌더 실패: {e}")

        # --- 합성 ---
        display_frame = _compose_display(
            incam_frame, global_frame, config.layout,
            config.render_width, config.render_height,
        )

        # FPS / 레이턴시 오버레이
        render_ms = (time.monotonic() - t0) * 1000
        frame_count += 1
        elapsed = time.monotonic() - fps_timer
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.monotonic()

        display_frame = _draw_stats_overlay(
            display_frame, current_fps, render_ms, perf_info
        )

        # --- frame_queue로 전달 (_display_worker가 cv2.imshow 처리) ---
        frame_item = {
            "display_frame": display_frame,
            "incam_frame": incam_frame,
            "global_frame": global_frame,
        }
        try:
            frame_queue.put_nowait(frame_item)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(frame_item)
            except queue.Full:
                pass

        # 통계 전송 (비블로킹)
        try:
            stats_queue.put_nowait({
                "render_fps": current_fps,
                "render_ms": render_ms,
            })
        except queue.Full:
            pass

    # 종료 시 EGL 컨텍스트 명시적 해제
    if global_renderer is not None:
        global_renderer.delete()


def _display_worker(
    config: VisualizerConfig,
    frame_queue: mp.Queue,
    status_queue: mp.Queue,
    stop_event: mp.Event,
):
    """cv2.imshow + VideoWriter 전용 프로세스.

    PYOPENGL_PLATFORM 환경변수를 설정하지 않음으로써
    Qt5 기반 cv2의 X11 OpenGL 컨텍스트와 EGL 충돌을 원천 차단.

    frame_queue에서 _egl_render_worker가 렌더링한 numpy frame을 받아
    화면 표시 및 녹화만 담당.

    status_queue에서 _egl_render_worker의 로딩 진행 상태를 수신하여
    frame이 없는 동안 로딩 화면에 진행 바와 텍스트를 갱신.
    """
    import cv2
    import os

    # 초기 로딩 대기 화면 (EGL 렌더러 시작 전)
    last_display_frame = _make_loading_frame(
        config, {"msg": "Starting EGL renderer...", "step": 0, "total": 3}
    )

    if config.show_window:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(config.window_name, cv2.cvtColor(last_display_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    # 비디오 녹화기
    incam_writer = None
    global_writer = None
    if config.record:
        os.makedirs(config.output_dir, exist_ok=True)

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.03)
        except queue.Empty:
            # status_queue에서 최신 로딩 상태 수신 → 로딩 화면 갱신
            try:
                status = status_queue.get_nowait()
                last_display_frame = _make_loading_frame(config, status)
            except queue.Empty:
                pass

            # 마지막 프레임 재표시 (윈도우 응답 유지)
            if config.show_window:
                cv2.imshow(config.window_name, cv2.cvtColor(last_display_frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
            continue

        if item is None:  # 종료 신호
            break

        display_frame = item["display_frame"]
        incam_frame = item.get("incam_frame")
        global_frame = item.get("global_frame")
        last_display_frame = display_frame

        # --- 화면 표시 ---
        if config.show_window:
            cv2.imshow(config.window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

        # --- 녹화 ---
        if config.record:
            if incam_frame is not None and incam_writer is None:
                incam_path = os.path.join(config.output_dir, "1_incam.mp4")
                ih, iw = incam_frame.shape[:2]
                incam_writer = cv2.VideoWriter(
                    incam_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    config.video_fps, (iw, ih),
                )
            if global_frame is not None and global_writer is None:
                global_path = os.path.join(config.output_dir, "2_global.mp4")
                gh, gw = global_frame.shape[:2]
                global_writer = cv2.VideoWriter(
                    global_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    config.video_fps, (gw, gh),
                )
            if incam_writer is not None and incam_frame is not None:
                incam_writer.write(cv2.cvtColor(incam_frame, cv2.COLOR_RGB2BGR))
            if global_writer is not None and global_frame is not None:
                global_writer.write(cv2.cvtColor(global_frame, cv2.COLOR_RGB2BGR))

    # 정리
    if config.show_window:
        cv2.destroyAllWindows()
    if incam_writer is not None:
        incam_writer.release()
        logger.info(f"[Visualizer] incam 녹화 저장: {config.output_dir}/1_incam.mp4")
    if global_writer is not None:
        global_writer.release()
        logger.info(f"[Visualizer] global 녹화 저장: {config.output_dir}/2_global.mp4")


# ============================================================================
# 렌더링 헬퍼
# ============================================================================


def _make_loading_frame(config: VisualizerConfig, status: dict) -> np.ndarray:
    """EGL 렌더러 로딩 진행 상태를 시각화한 프레임 생성.

    Args:
        status: {"msg": str, "step": int, "total": int}
    Returns:
        (H, W, 3) uint8 RGB numpy 배열
    """
    import cv2 as _cv2

    h, w = config.render_height, config.render_width
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    msg = status.get("msg", "Loading...")
    step = status.get("step", 0)
    total = max(status.get("total", 1), 1)

    # 메시지 텍스트
    _cv2.putText(
        frame, msg,
        (20, h // 2 - 24),
        _cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, _cv2.LINE_AA,
    )

    # 진행 바 배경 (회색)
    bar_x0, bar_y0 = 20, h // 2
    bar_x1, bar_y1 = w - 20, h // 2 + 18
    _cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)

    # 진행 바 전경 (초록)
    filled_w = int((bar_x1 - bar_x0) * step / total)
    if filled_w > 0:
        _cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + filled_w, bar_y1), (0, 200, 100), -1)

    # 단계 텍스트 (n / total)
    step_txt = f"{step} / {total}"
    _cv2.putText(
        frame, step_txt,
        (20, h // 2 + 44),
        _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, _cv2.LINE_AA,
    )

    return frame


def _get_last_frame(params: dict) -> dict:
    """파라미터 dict에서 마지막 프레임만 추출."""
    result = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] > 1:
            result[k] = v[-1]
        else:
            result[k] = v
    return result


class GlobalRenderer:
    """Global 뷰 렌더러 — OffscreenRenderer와 Scene을 재사용하여 EGL 컨텍스트 생성 비용 제거.

    기존 _render_global_simple은 매 프레임마다
      OffscreenRenderer 생성/파괴 → EGL 컨텍스트 생성/파괴 (수십 ms)
      Scene / Material / Camera / Light 재생성 (매 프레임)
    하는 문제가 있었음.

    이 클래스는 프레임마다 변하는 것(메시 정점)과
    고정인 것(Renderer, Scene, Camera, Light, Material)을 분리.

    사용법:
        gr = GlobalRenderer(640, 480, (0.69, 0.39, 0.96), smpl_faces)
        rendered = gr.render(verts_numpy)   # 매 프레임 — 메시 교체만 수행
        gr.delete()                          # 종료 시 EGL 해제
    """

    def __init__(self, width: int, height: int, color: tuple, faces: np.ndarray):
        import pyrender

        self._width = width
        self._height = height
        self._faces = faces

        # OffscreenRenderer 1회 생성 — EGL 컨텍스트 캐시
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height
        )

        # Scene 1회 생성
        self._scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=(0.3, 0.3, 0.3),
        )

        # 카메라 고정 (정면 5m 뒤)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 5.0
        camera = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=width / height
        )
        self._scene.add(camera, pose=cam_pose)

        # 조명 고정
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self._scene.add(light, pose=cam_pose)

        # 재질 고정 (색상은 config에서 1회 결정)
        self._material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(color[0], color[1], color[2], 1.0),
        )

        # 메시 노드 — 첫 render() 호출 시 추가, 이후 교체
        self._mesh_node = None

    def render(self, verts: np.ndarray) -> np.ndarray:
        """메시 정점만 교체 후 렌더링.

        Args:
            verts: (V, 3) float32 numpy — SMPL forward 결과
        Returns:
            (H, W, 3) uint8 RGB numpy
        """
        import pyrender
        import trimesh

        # trimesh 생성 + 180도 회전 (좌표계 보정)
        mesh = trimesh.Trimesh(verts, self._faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=self._material)

        # 이전 메시 노드 제거 → 새 노드 추가 (Scene 재사용)
        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
        self._mesh_node = self._scene.add(py_mesh)

        rendered, _ = self._renderer.render(self._scene)
        return rendered  # (H, W, 3) uint8 RGB

    def delete(self):
        """EGL 컨텍스트 명시적 해제."""
        try:
            self._renderer.delete()
        except Exception:
            pass


def _compose_display(
    incam: Optional[np.ndarray],
    global_view: Optional[np.ndarray],
    layout: str,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """incam과 global 뷰를 합성."""
    if layout == "incam_only" and incam is not None:
        return incam
    if layout == "global_only" and global_view is not None:
        return global_view

    if incam is not None and global_view is not None:
        # side-by-side: 동일 높이로 맞춤
        h1, w1 = incam.shape[:2]
        h2, w2 = global_view.shape[:2]
        target_h = max(h1, h2)
        if h1 != target_h:
            scale = target_h / h1
            incam = cv2.resize(incam, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            global_view = cv2.resize(global_view, (int(w2 * scale), target_h))
        return np.concatenate([incam, global_view], axis=1)

    if incam is not None:
        return incam
    if global_view is not None:
        return global_view

    # 둘 다 없으면 빈 프레임
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def _draw_stats_overlay(
    frame: np.ndarray,
    render_fps: float,
    render_ms: float,
    perf_info: dict,
) -> np.ndarray:
    """FPS 및 레이턴시 정보를 프레임 위에 표시."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    lines = [
        f"Render: {render_fps:.0f} FPS ({render_ms:.0f}ms)",
    ]
    if "preprocess_ms" in perf_info:
        lines.append(f"Preproc: {perf_info['preprocess_ms']:.0f}ms")
    if "inference_ms" in perf_info:
        lines.append(f"Infer: {perf_info['inference_ms']:.0f}ms")
    if "total_ms" in perf_info:
        lines.append(f"Total: {perf_info['total_ms']:.0f}ms")

    # 반투명 배경
    overlay_h = len(lines) * 25 + 10
    frame[:overlay_h, :220] = (frame[:overlay_h, :220].astype(np.float32) * 0.4).astype(np.uint8)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 22 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )

    return frame


# ============================================================================
# 메인 시각화 클래스
# ============================================================================


class RealtimeVisualizer:
    """실시간 SMPL 메시 시각화 매니저.

    별도 프로세스에서 렌더링을 수행하여 추론 파이프라인을 블로킹하지 않음.

    스레드/프로세스 구조:
        Main Process:  Preprocessor → Inference → vis.update(result, frame)
        Render Process: render_queue → SMPL forward → pyrender → cv2.imshow / VideoWriter

    출처:
        - demo_utils.py: render_incam_frames (lines 455-501), render_global_frames (504-575)
        - pyrender_incam.py: Renderer 클래스 — incam 오버레이
        - o3d_render.py: create_meshes, Settings — global 뷰 (대신 pyrender 사용)
    """

    def __init__(self, config: VisualizerConfig):
        self._config = config
        self._render_queue: Optional[mp.Queue] = None   # update() → _egl_render_worker
        self._frame_queue: Optional[mp.Queue] = None    # _egl_render_worker → _display_worker
        self._stats_queue: Optional[mp.Queue] = None    # _egl_render_worker → main
        self._status_queue: Optional[mp.Queue] = None   # _egl_render_worker → _display_worker (로딩 상태)
        self._stop_event: Optional[mp.Event] = None
        self._render_process: Optional[mp.Process] = None  # EGL 렌더링 전용 (spawn)
        self._display_process: Optional[mp.Process] = None  # cv2.imshow 전용
        self._last_stats: dict = {}

    @property
    def is_running(self) -> bool:
        render_alive = self._render_process is not None and self._render_process.is_alive()
        display_alive = self._display_process is not None and self._display_process.is_alive()
        return render_alive and display_alive

    @property
    def render_stats(self) -> dict:
        """최신 렌더링 통계."""
        # 큐에서 가장 최근 통계를 꺼냄 (비블로킹)
        while True:
            try:
                self._last_stats = self._stats_queue.get_nowait()
            except (queue.Empty, AttributeError):
                break
        return self._last_stats

    def start(self):
        """렌더링 프로세스 시작.

        _egl_render_worker: fork 컨텍스트로 시작.
            → 부모의 sys.path를 그대로 상속하므로 모듈 탐색 실패 없음.
            → cv2는 이미 import되어 있지만, cv2.namedWindow/imshow를 호출하기
              전까지는 Qt5 OpenGL 컨텍스트가 생성되지 않음.
              따라서 자식이 PYOPENGL_PLATFORM=egl을 설정한 뒤 pyrender를
              import해도 X11 / EGL 충돌이 발생하지 않음.
        _display_worker: fork 컨텍스트.
            → PYOPENGL_PLATFORM 미설정 상태로 cv2.imshow(X11) 전용 실행.
        """
        logger.info("[Visualizer] 시작 ...")

        ctx = mp.get_context("spawn")
        self._render_queue = ctx.Queue(maxsize=self._config.queue_maxsize)
        self._frame_queue = ctx.Queue(maxsize=self._config.queue_maxsize)
        self._stats_queue = ctx.Queue(maxsize=5)
        self._status_queue = ctx.Queue(maxsize=10)  # 로딩 상태 메시지 (소량)
        self._stop_event = ctx.Event()

        # EGL 렌더링 전용 프로세스 — fork 컨텍스트 방지: spawn 사용
        # (spawn 대신 fork를 사용: sys.path 상속, 모듈 재import 불필요) -> deadlock 수정 위해 spawn 적용
        self._render_process = ctx.Process(
            target=_egl_render_worker,
            args=(self._config, self._render_queue, self._frame_queue,
                  self._stats_queue, self._status_queue, self._stop_event),
            name="EGLRenderProcess",
            daemon=True,
        )
        self._render_process.start()
        logger.info("[Visualizer] EGL 렌더링 프로세스 시작됨")

        # cv2.imshow 전용 프로세스 — spawn 사용
        self._display_process = ctx.Process(
            target=_display_worker,
            args=(self._config, self._frame_queue, self._status_queue, self._stop_event),
            name="DisplayProcess",
            daemon=True,
        )
        self._display_process.start()
        logger.info("[Visualizer] Display 프로세스 시작됨")

    def stop(self):
        """렌더링/디스플레이 프로세스 정상 종료."""
        logger.info("[Visualizer] 종료 중 ...")

        # render_queue → EGL 워커 종료 신호
        if self._render_queue is not None:
            try:
                self._render_queue.put(None, timeout=2.0)
            except (queue.Full, BrokenPipeError):
                pass

        # frame_queue → display 워커 종료 신호
        if self._frame_queue is not None:
            try:
                self._frame_queue.put(None, timeout=2.0)
            except (queue.Full, BrokenPipeError):
                pass

        if self._stop_event is not None:
            self._stop_event.set()

        # EGL 렌더링 프로세스 종료
        if self._render_process is not None:
            self._render_process.join(timeout=5.0)
            if self._render_process.is_alive():
                self._render_process.terminate()

        # Display 프로세스 종료
        if self._display_process is not None:
            self._display_process.join(timeout=5.0)
            if self._display_process.is_alive():
                self._display_process.terminate()

        # 큐 정리 — atexit에서 feeder thread join hang 방지
        for q in (self._render_queue, self._frame_queue, self._stats_queue, self._status_queue):
            if q is not None:
                q.cancel_join_thread()
                try:
                    while not q.empty():
                        q.get_nowait()
                except (EOFError, OSError):
                    pass

        logger.info("[Visualizer] 종료 완료")

    def update(
        self,
        result,  # InferenceResult (from realtime_gem_inference.py)
        camera_frame: Optional[np.ndarray] = None,
        perf_info: Optional[dict] = None,
        bbx_xys: Optional[np.ndarray] = None,
    ):
        """새 추론 결과를 렌더링 큐에 전송.

        Args:
            result: InferenceResult — body_params_incam, body_params_global, K_fullimg
            camera_frame: (H, W, 3) RGB uint8 — incam 오버레이 배경
            perf_info: {"preprocess_ms": ..., "inference_ms": ..., "total_ms": ...}
        """
        if not self.is_running:
            return

        # 유효 프레임 파라미터 사용 (있으면), 없으면 전체
        body_params_incam = (
            result.valid_body_params_incam
            if result.valid_body_params_incam
            else result.body_params_incam
        )
        body_params_global = (
            result.valid_body_params_global
            if result.valid_body_params_global
            else result.body_params_global
        )

        item = {
            "body_params_incam": body_params_incam,
            "body_params_global": body_params_global,
            "K_fullimg": result.K_fullimg,
            "camera_frame": camera_frame,
            "perf_info": perf_info or {},
            "bbx_xys": bbx_xys,
        }

        # 큐가 가득 차면 오래된 것 드롭 (실시간 우선)
        try:
            self._render_queue.put_nowait(item)
        except queue.Full:
            try:
                self._render_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._render_queue.put_nowait(item)
            except queue.Full:
                pass  # 포기


# ============================================================================
# CLI 테스트
# ============================================================================


def main():
    """독립 테스트: Mock 데이터로 시각화 검증."""
    import argparse

    parser = argparse.ArgumentParser(description="실시간 시각화 테스트")
    parser.add_argument("--mock", action="store_true", help="Mock 데이터 사용")
    parser.add_argument("--record", action="store_true", help="녹화 활성화")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--no-window", action="store_true", help="윈도우 비활성화")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = VisualizerConfig(
        show_window=not args.no_window,
        record=args.record,
        output_dir="outputs/vis_test",
    )

    if args.mock:
        _run_mock_vis_test(config, args.duration)
    else:
        _run_real_vis_test(config, args.duration)


def _run_mock_vis_test(config: VisualizerConfig, duration: float):
    """Mock InferenceResult로 시각화 테스트."""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        body_params_incam: dict
        body_params_global: dict
        K_fullimg: torch.Tensor
        valid_body_params_incam: dict = None
        valid_body_params_global: dict = None

    logger.info("=== Mock 시각화 테스트 ===")

    L = 30
    mock_result = MockResult(
        body_params_incam={
            "body_pose": torch.zeros(L, 63),
            "global_orient": torch.zeros(L, 3),
            "transl": torch.tensor([[0, 0, 3.0]] * L),
            "betas": torch.zeros(L, 10),
        },
        body_params_global={
            "body_pose": torch.zeros(L, 63),
            "global_orient": torch.zeros(L, 3),
            "transl": torch.zeros(L, 3),
            "betas": torch.zeros(L, 10),
        },
        K_fullimg=torch.tensor([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1],
        ], dtype=torch.float32),
    )

    # 더미 카메라 프레임
    camera_frame = np.ones((480, 640, 3), dtype=np.uint8) * 180

    vis = RealtimeVisualizer(config)
    vis.start()

    t0 = time.time()
    count = 0
    try:
        while time.time() - t0 < duration and vis.is_running:
            vis.update(
                mock_result,
                camera_frame=camera_frame,
                perf_info={"preprocess_ms": 5.0, "inference_ms": 40.0, "total_ms": 45.0},
            )
            count += 1
            time.sleep(1.0 / 30)  # 30fps 시뮬레이션
    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()

    logger.info(f"{count}프레임 전송, {time.time() - t0:.1f}초")
    stats = vis.render_stats
    logger.info(f"렌더링 통계: {stats}")


def _run_real_vis_test(config: VisualizerConfig, duration: float):
    """실제 SMPL 모델로 시각화 테스트 (GPU 필요)."""
    logger.info("=== 실제 시각화 테스트 ===")
    logger.info("실제 모델 테스트는 추론 파이프라인과 함께 실행해야 합니다.")
    logger.info("python create_test_and_run.py --mock-e2e 로 통합 테스트를 권장합니다.")


if __name__ == "__main__":
    main()
