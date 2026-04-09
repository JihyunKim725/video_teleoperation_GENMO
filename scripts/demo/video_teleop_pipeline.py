# SPDX-License-Identifier: Apache-2.0
"""
Video Teleoperation Pipeline — 엔드투엔드 통합 스크립트

카메라 영상에서 인간 동작을 추정하여 G1 로봇에 실시간 전송.

아키텍처:
    Camera ──▶ Preprocessor ──▶ GEM Inference ──▶ Retargeter ──▶ ZMQ Publisher
       (Thread 1)           (Thread 2)         (Thread 3)      (Thread 3)
                                                    │
                                               Visualizer (Process 2, optional)

사용법:
    # 시뮬레이션 (웹캠 + MuJoCo)
    python video_teleop_pipeline.py --camera webcam --mode sim

    # 녹화 영상 오프라인 (검증용)
    python video_teleop_pipeline.py --camera video:path/to/file.mp4 --no-zmq --save

    # RealSense + 실제 G1
    python video_teleop_pipeline.py --camera realsense --mode real --zmq-host 192.168.123.161

출처:
    - realtime_preprocessor.py: CameraCapture + PerFrameProcessor + SlidingWindowBuffer
    - realtime_gem_inference.py: RealtimeGEMInference + SessionAccumulator
    - smpl_to_g1_retargeter.py: SmplToG1Retargeter
    - zmq_motion_publisher.py: ZmqMotionPublisher
    - realtime_visualizer.py: RealtimeVisualizer
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# 파이프라인 설정
# ============================================================================


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정. YAML 파일에서 로드 가능."""

    # 카메라
    camera_type: str = "webcam"        # "realsense" | "webcam" | "video:/path"
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    camera_device: int = 0

    # GENMO 추론
    ckpt_path: str = "inputs/pretrained/gem_smpl.ckpt"
    inference_interval: int = 30       # 매 N프레임마다 추론
    window_size: int = 120
    static_cam: bool = True
    postprocess: bool = True

    # 리타겟팅
    g1_urdf_path: Optional[str] = None
    retarget_mode: str = "sonic_v3"    # "sonic_v3" | "full"
    joint_limit_margin: float = 0.95

    # ZMQ
    zmq_enabled: bool = True
    zmq_host: str = "localhost"
    zmq_port: int = 5556
    zmq_topic: str = "pose"
    zmq_socket_mode: str = "bind"

    # 시각화
    vis_enabled: bool = True
    vis_record: bool = False
    vis_output_dir: str = "outputs/teleop"

    # 저장
    save_smpl: bool = False            # smpl_params.pt 저장

    # 모드
    mode: str = "sim"                  # "sim" | "real"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """YAML에서 설정 로드."""
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        config = cls()
        flat = {}
        for section in cfg.values():
            if isinstance(section, dict):
                flat.update(section)
            else:
                continue

        for key, val in flat.items():
            if hasattr(config, key):
                setattr(config, key, val)
        return config


# ============================================================================
# 파이프라인 단계별 타이밍
# ============================================================================


class PerfMonitor:
    """단계별 처리 시간 추적."""

    def __init__(self):
        self._timers: dict[str, list[float]] = {}
        self._last_report = time.monotonic()
        self._lock = threading.Lock()

    def record(self, stage: str, ms: float):
        with self._lock:
            if stage not in self._timers:
                self._timers[stage] = []
            self._timers[stage].append(ms)

    def report(self, interval: float = 5.0) -> Optional[str]:
        """interval초마다 통계 문자열 반환."""
        now = time.monotonic()
        if now - self._last_report < interval:
            return None
        self._last_report = now

        with self._lock:
            lines = []
            for stage, times in self._timers.items():
                if not times:
                    continue
                recent = times[-30:]
                avg = np.mean(recent)
                p95 = np.percentile(recent, 95) if len(recent) >= 5 else np.max(recent)
                lines.append(f"  {stage}: {avg:.0f}ms avg, {p95:.0f}ms p95")
            return "\n".join(lines) if lines else None

    def get_perf_info(self) -> dict:
        """시각화 오버레이용 최신 타이밍."""
        with self._lock:
            info = {}
            for stage, times in self._timers.items():
                if times:
                    info[f"{stage}_ms"] = times[-1]
            return info


# ============================================================================
# 메인 파이프라인
# ============================================================================


class VideoTeleopPipeline:
    """카메라 → GENMO → 리타겟 → SONIC 전체 파이프라인.

    스레드 구조:
        preprocessor_thread: 카메라 + YOLO + ViTPose + HMR2 → data_queue
        inference_thread:    data_queue → GEM predict → result_queue
        retarget_thread:     result_queue → retarget → ZMQ publish + vis
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._perf = PerfMonitor()
        self._stop_event = threading.Event()

        # 모듈 (lazy init)
        self._preprocessor = None
        self._inference = None
        self._retargeter = None
        self._zmq_pub = None
        self._visualizer = None
        self._accumulator = None

        # 스레드 간 큐
        self._data_queue: queue.Queue = queue.Queue(maxsize=3)
        self._result_queue: queue.Queue = queue.Queue(maxsize=3)

        # 스레드
        self._threads: list[threading.Thread] = []

    def setup(self):
        """모든 모듈 초기화."""
        logger.info("=" * 60)
        logger.info("Video Teleoperation Pipeline — 초기화")
        logger.info("=" * 60)
        cfg = self._config

        # 1. 전처리기
        logger.info("[1/5] 전처리기 ...")
        from realtime_preprocessor import (
            PreprocessorConfig,
            RealtimePreprocessor,
        )
        # camera_type 파싱: "video:/path/to/file.mp4" → type="video", path="/path/..."
        cam_type = cfg.camera_type.split(":")[0]
        video_path = cfg.camera_type.split(":", 1)[1] if ":" in cfg.camera_type else ""
        pp_config = PreprocessorConfig(
            camera_type=cam_type,
            camera_device_id=cfg.camera_device,
            video_path=video_path,
            camera_resolution=cfg.camera_resolution,
            camera_fps=cfg.camera_fps,
            window_size=cfg.window_size,
            inference_interval=cfg.inference_interval,
        )
        self._preprocessor = RealtimePreprocessor(pp_config)

        # 2. GEM 추론
        logger.info("[2/5] GEM 모델 ...")
        from realtime_gem_inference import (
            GEMInferenceConfig,
            RealtimeGEMInference,
            SessionAccumulator,
        )
        gem_config = GEMInferenceConfig(
            ckpt_path=cfg.ckpt_path,
            window_size=cfg.window_size,
            inference_interval=cfg.inference_interval,
            static_cam=cfg.static_cam,
            postprocess=cfg.postprocess,
        )
        self._inference = RealtimeGEMInference(gem_config)
        self._inference.load_model()
        self._accumulator = SessionAccumulator() if cfg.save_smpl else None

        # 3. 리타겟터
        logger.info("[3/5] 리타겟터 ...")
        from smpl_to_g1_retargeter import RetargetConfig, SmplToG1Retargeter
        rt_config = RetargetConfig(
            g1_urdf_path=cfg.g1_urdf_path,
            joint_limit_margin=cfg.joint_limit_margin,
        )
        self._retargeter = SmplToG1Retargeter(rt_config)

        # 4. ZMQ 퍼블리셔
        if cfg.zmq_enabled:
            logger.info("[4/5] ZMQ 퍼블리셔 ...")
            from zmq_motion_publisher import ZmqMotionPublisher, ZmqPublisherConfig
            zmq_config = ZmqPublisherConfig(
                host=cfg.zmq_host,
                port=cfg.zmq_port,
                topic=cfg.zmq_topic,
                socket_mode=cfg.zmq_socket_mode,
            )
            self._zmq_pub = ZmqMotionPublisher(zmq_config)
        else:
            logger.info("[4/5] ZMQ 비활성화")

        # 5. 시각화
        if cfg.vis_enabled:
            logger.info("[5/5] 시각화 ...")
            from realtime_visualizer import RealtimeVisualizer, VisualizerConfig
            vis_config = VisualizerConfig(
                show_window=True,
                record=cfg.vis_record,
                output_dir=cfg.vis_output_dir,
            )
            self._visualizer = RealtimeVisualizer(vis_config)
        else:
            logger.info("[5/5] 시각화 비활성화")

        logger.info("초기화 완료")

    def run(self):
        """파이프라인 실행 (블로킹)."""
        logger.info("파이프라인 시작 ...")

        # 모듈 시작
        self._preprocessor.start()
        if self._zmq_pub:
            self._zmq_pub.start()
            self._zmq_pub.send_command(start=True)
        if self._visualizer:
            self._visualizer.start()

        # 워커 스레드 시작
        t_infer = threading.Thread(target=self._inference_loop, name="InferenceThread", daemon=True)
        t_retarget = threading.Thread(target=self._retarget_loop, name="RetargetThread", daemon=True)
        self._threads = [t_infer, t_retarget]
        for t in self._threads:
            t.start()

        logger.info("모든 스레드 시작됨 — Ctrl+C로 종료")

        # 메인 루프: 전처리 → data_queue
        frame_count = 0
        _health_timer = time.monotonic()
        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                # timeout=0.05: 메인 루프가 50ms마다 순환
                # (기존 1.0초 블로킹 → stop_event 체크/헬스 모니터 반응성 향상)
                payload = self._preprocessor.get_inference_data(timeout=0.05)

                if payload is not None:
                    package = {"data_dict": payload["data_dict"], "camera_frame": payload["camera_frame"]}

                    try:
                        self._data_queue.put(package, timeout=0.5)
                    except queue.Full:
                        try:
                            self._data_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._data_queue.put_nowait(package)

                    dt = (time.monotonic() - t0) * 1000
                    self._perf.record("preprocess", dt)
                    frame_count += 1

                # 성능 리포트 (5초 간격)
                report = self._perf.report(interval=5.0)
                if report:
                    logger.info(f"성능 통계:\n{report}")

                # 파이프라인 헬스 모니터 (5초 간격)
                if time.monotonic() - _health_timer >= 5.0:
                    _health_timer = time.monotonic()
                    self._log_pipeline_health(t_infer, t_retarget)

                # 짧은 대기 (CPU 100% 방지)
                time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("\nCtrl+C 감지 — 종료 중 ...")

        self.shutdown()

    def _inference_loop(self):
        """추론 스레드: data_queue → GEM predict → result_queue."""
        _first_success = True

        while not self._stop_event.is_set():
            try:
                package = self._data_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 유효 키포인트 없으면 스킵 — 사람 미검출 구간 전체가 버퍼에 쌓인 경우
            has_2d = package["data_dict"]["mask"]["has_2d_mask"]
            if not has_2d.any():
                logger.debug("[InferenceThread] 유효 키포인트 없음 — 추론 스킵")
                continue

            t0 = time.monotonic()
            try:
                result = self._inference.run(package["data_dict"])
                dt = (time.monotonic() - t0) * 1000
                self._perf.record("inference", dt)

                output = {
                    "result": result,
                    "camera_frame": package.get("camera_frame"),
                    "bbx_xys": package["data_dict"]["bbx_xys"][-1].numpy()
                }
                try:
                    self._result_queue.put_nowait(output)
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._result_queue.put_nowait(output)

                if self._accumulator:
                    self._accumulator.add(result)

                if _first_success:
                    logger.info(f"[InferenceThread] 첫 번째 추론 성공 ({dt:.0f}ms) — 시각화 데이터 흐름 시작")
                    _first_success = False

            except Exception as e:
                logger.error(f"[InferenceThread] 에러: {e}", exc_info=True)

    def _retarget_loop(self):
        """리타겟 스레드: result_queue → retarget → ZMQ + vis."""
        _first_vis_update = True
        _last_vis_warn = 0.0

        while not self._stop_event.is_set():
            try:
                output = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.monotonic()
            try:
                result = output["result"]

                # SMPL 파라미터 → G1 관절각
                smpl_for_sonic = self._inference.get_latest_smpl_for_sonic(result)
                retarget_input = {
                    "body_pose": smpl_for_sonic["body_pose"],
                    "global_orient": smpl_for_sonic["global_orient"],
                    "transl": smpl_for_sonic["transl"],
                    "betas": smpl_for_sonic["betas"],
                }
                retarget_output = self._retargeter.retarget(
                    retarget_input, mode=self._config.retarget_mode,
                )

                # body_quat 계산: global_orient (axis-angle) → quaternion (w,x,y,z)
                # 출처: video_teleop_genmo.py — v3 필수 필드
                go = smpl_for_sonic["global_orient"]  # (N, 3)
                if go.ndim == 1:
                    go = go.unsqueeze(0)
                # 마지막 프레임의 global_orient → quaternion
                go_last = go[-1]  # (3,)
                angle = go_last.norm().item()
                if angle > 1e-8:
                    axis = go_last / angle
                    w = np.cos(angle / 2)
                    xyz = (axis * np.sin(angle / 2)).numpy()
                    body_quat = np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)
                else:
                    body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                retarget_output["body_quat"] = body_quat.reshape(1, 1, 4)
                dt_rt = (time.monotonic() - t0) * 1000
                self._perf.record("retarget", dt_rt)

                # ZMQ 발행
                if self._zmq_pub and self._zmq_pub.is_running:
                    self._zmq_pub.publish_retarget(retarget_output)

                # 시각화
                if self._visualizer:
                    if self._visualizer.is_running:
                        self._visualizer.update(
                            result,
                            camera_frame=output.get("camera_frame"),
                            perf_info=self._perf.get_perf_info(),
                            bbx_xys=output.get("bbx_xys"),
                        )
                        if _first_vis_update:
                            logger.info("[RetargetThread] 첫 번째 시각화 업데이트 전송 완료")
                            _first_vis_update = False
                    else:
                        # EGL/Display 프로세스 사망 시 — 5초 쿨다운으로 경고
                        _now = time.monotonic()
                        if _now - _last_vis_warn >= 5.0:
                            logger.warning(
                                "[RetargetThread] Visualizer 프로세스가 응답 없음 "
                                "— EGL 렌더러 또는 Display 프로세스 종료 여부 확인 필요"
                            )
                            _last_vis_warn = _now

                dt_total = (time.monotonic() - t0) * 1000
                self._perf.record("retarget+pub", dt_total)

            except Exception as e:
                logger.error(f"[RetargetThread] 에러: {e}", exc_info=True)

    def _log_pipeline_health(self, t_infer: threading.Thread, t_retarget: threading.Thread):
        """파이프라인 각 구성요소의 상태를 로그로 출력 (5초 간격 호출)."""
        lines = [
            f"  data_queue:   {self._data_queue.qsize()}/{self._data_queue.maxsize}",
            f"  result_queue: {self._result_queue.qsize()}/{self._result_queue.maxsize}",
            f"  InferThread:  {'alive' if t_infer.is_alive() else '⚠ DEAD'}",
            f"  RetargThread: {'alive' if t_retarget.is_alive() else '⚠ DEAD'}",
        ]
        if self._visualizer:
            vis_state = "running" if self._visualizer.is_running else "⚠ STOPPED"
            lines.append(f"  Visualizer:   {vis_state}")
        logger.info("파이프라인 헬스:\n" + "\n".join(lines))

    def shutdown(self):
        """Graceful shutdown — 모든 리소스 해제."""
        logger.info("파이프라인 종료 중 ...")
        self._stop_event.set()

        # 스레드 종료 대기
        for t in self._threads:
            t.join(timeout=3.0)

        # 모듈 종료
        if self._zmq_pub:
            self._zmq_pub.send_command(stop=True)
            self._zmq_pub.stop()

        if self._visualizer:
            self._visualizer.stop()

        if self._preprocessor:
            self._preprocessor.stop()

        # 결과 저장
        if self._accumulator:
            out_dir = self._config.vis_output_dir
            os.makedirs(out_dir, exist_ok=True)
            self._accumulator.save(os.path.join(out_dir, "smpl_params.pt"))

        # 통계 출력
        logger.info("=" * 60)
        logger.info("최종 통계:")
        report = self._perf.report(interval=0)
        if report:
            logger.info(report)
        if self._zmq_pub:
            logger.info(f"  ZMQ: {self._zmq_pub.stats}")
        if self._retargeter:
            logger.info(f"  리타겟: {self._retargeter.stats}")
        logger.info("=" * 60)


# ============================================================================
# 기본 설정 YAML 생성
# ============================================================================


DEFAULT_CONFIG_YAML = """\
# Video Teleoperation Pipeline 설정
# 출처: g1_29dof_gear_wbc.yaml, observation_config.yaml

camera:
  camera_type: webcam          # realsense | webcam | video:/path/to/file.mp4
  camera_resolution: [640, 480]
  camera_fps: 30
  camera_device: 0

genmo:
  ckpt_path: inputs/pretrained/gem_smpl.ckpt
  inference_interval: 60       # 매 30프레임 (=1초)마다 추론
  window_size: 60
  static_cam: true
  postprocess: true

retarget:
  g1_urdf_path: null           # null이면 내장 기본값 사용
  retarget_mode: sonic_v3      # sonic_v3 | full
  joint_limit_margin: 0.95

zmq:
  zmq_enabled: true
  zmq_host: localhost
  zmq_port: 5556
  zmq_topic: pose
  zmq_socket_mode: bind        # bind (sim) | connect (real)

visualize:
  vis_enabled: true
  vis_record: false
  vis_output_dir: outputs/teleop

save:
  save_smpl: false
"""


def write_default_config(path: str):
    """기본 설정 파일 생성."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(DEFAULT_CONFIG_YAML)
    logger.info(f"기본 설정 파일 생성: {path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Video Teleoperation Pipeline — GENMO → SONIC → G1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
시나리오:
  A) 녹화 영상 오프라인:  --camera video:path.mp4 --no-zmq --save
  B) 웹캠 + MuJoCo sim:   --camera webcam --mode sim
  C) RealSense + 실제 G1:  --camera realsense --mode real --zmq-host 192.168.123.161
""",
    )
    parser.add_argument("--config", default=None, help="YAML 설정 파일")
    parser.add_argument("--gen-config", default=None, help="기본 설정 파일 생성 경로")

    # 카메라
    parser.add_argument("--camera", default=None,
                        help="webcam | realsense | video:/path (설정 파일 오버라이드)")
    parser.add_argument("--camera-device", type=int, default=None)

    # 모드
    parser.add_argument("--mode", choices=["sim", "real"], default=None)

    # GENMO
    parser.add_argument("--ckpt", default=None, help="GEM 체크포인트 경로")

    # ZMQ
    parser.add_argument("--zmq-host", default=None)
    parser.add_argument("--zmq-port", type=int, default=None)
    parser.add_argument("--no-zmq", action="store_true", help="ZMQ 비활성화")

    # 시각화
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--record", action="store_true")

    # 저장
    parser.add_argument("--save", action="store_true", help="smpl_params.pt 저장")

    # 리타겟
    parser.add_argument("--retarget-mode", choices=["sonic_v3", "full"], default=None)
    parser.add_argument("--urdf", default=None, help="G1 URDF 경로")

    # 디버깅
    parser.add_argument("--dry-run", action="store_true", help="모듈 초기화만 (실행 안 함)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 기본 설정 생성
    if args.gen_config:
        write_default_config(args.gen_config)
        return

    # 설정 로드
    if args.config and Path(args.config).exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # CLI 오버라이드
    if args.camera:
        config.camera_type = args.camera
    if args.camera_device is not None:
        config.camera_device = args.camera_device
    if args.mode:
        config.mode = args.mode
        if args.mode == "real":
            config.zmq_socket_mode = "bind"
    if args.ckpt:
        config.ckpt_path = args.ckpt
    if args.zmq_host:
        config.zmq_host = args.zmq_host
    if args.zmq_port:
        config.zmq_port = args.zmq_port
    if args.no_zmq:
        config.zmq_enabled = False
    if args.no_vis:
        config.vis_enabled = False
    if args.record:
        config.vis_record = True
    if args.save:
        config.save_smpl = True
    if args.retarget_mode:
        config.retarget_mode = args.retarget_mode
    if args.urdf:
        config.g1_urdf_path = args.urdf

    # 설정 출력
    logger.info(f"카메라: {config.camera_type}")
    logger.info(f"모드: {config.mode}")
    logger.info(f"ZMQ: {'ON' if config.zmq_enabled else 'OFF'} ({config.zmq_host}:{config.zmq_port})")
    logger.info(f"시각화: {'ON' if config.vis_enabled else 'OFF'}")
    logger.info(f"리타겟: {config.retarget_mode}")

    # 파이프라인 실행
    pipeline = VideoTeleopPipeline(config)

    # Ctrl+C 시그널 핸들러
    def signal_handler(sig, frame):
        logger.info("\nSIGINT — 종료 요청")
        pipeline._stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    pipeline.setup()

    if args.dry_run:
        logger.info("dry-run 모드 — 초기화 성공, 실행 건너뜀")
        pipeline.shutdown()
        return

    pipeline.run()


if __name__ == "__main__":
    main()
