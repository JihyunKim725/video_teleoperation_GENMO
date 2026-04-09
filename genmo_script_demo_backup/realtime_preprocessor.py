# SPDX-License-Identifier: Apache-2.0
"""
실시간 비디오 → GENMO 전처리 파이프라인

오프라인 demo_smpl_hpe.py의 Stage 1을 프레임 단위로 변환.
RealSense D435 / 웹캠 → YOLO → ViTPose → HMR2 → 120프레임 슬라이딩 윈도우 버퍼

사용법:
    from realtime_preprocessor import RealtimePreprocessor

    pp = RealtimePreprocessor(config)
    pp.start()
    while True:
        data_dict = pp.get_inference_data()  # 추론 트리거 시에만 반환
        if data_dict is not None:
            pred = gem_model.predict(data_dict)

출처:
    - 오프라인 파이프라인: GENMO/scripts/demo/demo_smpl_hpe.py (lines 243-276)
    - YOLO: ultralytics 패키지 (detect_and_track → demo_utils.py)
    - ViTPose: CocoPoseExtractor (demo_utils.py)
    - HMR2: get_image_features (demo_utils.py → gem/network/hmr2/)
    - 데이터 dict: assemble_data() (demo_smpl_hpe.py lines 44-81)
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 설정 (Configuration)
# ============================================================================


@dataclass
class PreprocessorConfig:
    """전처리 파이프라인 설정.

    기본값은 GENMO demo_smpl_hpe.py의 기본 동작과 동일하게 설정.
    """

    # --- 카메라 ---
    camera_type: str = "realsense"       # "realsense" | "webcam" | "video"
    camera_resolution: tuple[int, int] = (640, 480)  # (width, height)
    camera_fps: int = 30
    camera_device_id: int = 0            # 웹캠 장치 번호
    video_path: str = ""                 # camera_type="video"일 때 사용

    # --- YOLO ---
    # 출처: GENMO 레포 루트 yolov8x.pt
    yolo_model_path: str = "yolov8x.pt"
    yolo_conf_threshold: float = 0.5
    yolo_person_class_id: int = 0        # COCO person 클래스

    # --- ViTPose ---
    # 출처: GENMO/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
    vitpose_ckpt: str = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"

    # --- HMR2 ---
    # 출처: GENMO/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
    hmr2_ckpt: str = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"

    # --- 슬라이딩 윈도우 ---
    # 출처: GENMO 분석 문서 - "120프레임 청크 + overlap"
    window_size: int = 30               # GEM 기대 시퀀스 길이
    inference_interval: int = 10         # 매 N프레임마다 추론 (30 = 1초 @ 30fps)

    # --- 스레드 ---
    frame_queue_maxsize: int = 5         # 카메라 → 전처리 큐
    inference_queue_maxsize: int = 2     # 전처리 → 추론 큐

    # --- 기타 ---
    device: str = "cuda"
    static_cam: bool = True              # 정적 카메라 가정


# ============================================================================
# 카메라 캡처 (Thread 1)
# ============================================================================


class CameraCapture:
    """카메라 프레임 캡처 스레드.

    RealSense D435 또는 OpenCV 웹캠에서 RGB 프레임을 캡처하여
    frame_queue에 넣는다.

    출처:
        - pyrealsense2 SDK: https://github.com/IntelRealSense/librealsense
        - RealSense D435 스펙: 최대 1920x1080 @ 30fps RGB
    """

    def __init__(self, config: PreprocessorConfig, frame_queue: queue.Queue):
        self._config = config
        self._frame_queue = frame_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._K: Optional[np.ndarray] = None  # 카메라 내부 파라미터 (3, 3)

    @property
    def intrinsics_K(self) -> np.ndarray:
        """카메라 내부 파라미터 행렬 (3, 3).

        RealSense: 실제 calibration 데이터에서 자동 추출.
        웹캠: 해상도 기반 추정값 (fx=fy=focal_length, cx=W/2, cy=H/2).

        출처: pyrealsense2 intrinsics 문서
              https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20
        """
        if self._K is None:
            raise RuntimeError("카메라 미초기화. start()를 먼저 호출하세요.")
        return self._K

    def _init_realsense(self):
        """RealSense D435 파이프라인 초기화 및 intrinsics 추출."""
        import pyrealsense2 as rs

        self._rs_pipeline = rs.pipeline()
        rs_config = rs.config()

        w, h = self._config.camera_resolution
        rs_config.enable_stream(
            rs.stream.color, w, h, rs.format.bgr8, self._config.camera_fps
        )
        profile = self._rs_pipeline.start(rs_config)

        # 카메라 내부 파라미터 자동 추출
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self._K = np.array([
            [intr.fx,  0,       intr.ppx],
            [0,        intr.fy, intr.ppy],
            [0,        0,       1       ],
        ], dtype=np.float32)

        logger.info(f"[CameraCapture] RealSense 초기화: {w}x{h}@{self._config.camera_fps}fps")

    def _init_opencv(self):
        """OpenCV VideoCapture 초기화 (웹캠/비디오 파일)."""
        if self._config.camera_type == "video":
            if not os.path.exists(self._config.video_path):
                raise FileNotFoundError(
                    f"비디오 파일을 찾을 수 없습니다: {self._config.video_path}"
                )
            self._cap = cv2.VideoCapture(self._config.video_path)
        else:
            self._cap = cv2.VideoCapture(self._config.camera_device_id)

        if not self._cap.isOpened():
            device = (
                self._config.video_path
                if self._config.camera_type == "video"
                else f"카메라 인덱스 {self._config.camera_device_id}"
            )
            raise RuntimeError(
                f"카메라/비디오를 열 수 없습니다: {device}\n"
                f"  웹캠: 장치가 연결되어 있는지 확인 (ls /dev/video*)\n"
                f"  비디오: 파일 경로가 올바른지 확인"
            )

        w, h = self._config.camera_resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.camera_fps)

        # 실제 해상도 확인 (요청값과 다를 수 있음)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == 0 or actual_h == 0:
            raise RuntimeError(
                f"카메라 해상도가 0x0입니다. 장치가 제대로 연결되어 있는지 확인하세요."
            )

        # 웹캠은 calibration 없으므로 추정값 사용
        # 출처: 일반적 웹캠 focal length 추정 — f ≈ max(W, H)
        focal = float(max(actual_w, actual_h))

        self._K = np.array([
            [focal, 0,     actual_w / 2.0],
            [0,     focal, actual_h / 2.0],
            [0,     0,     1             ],
        ], dtype=np.float32)

        logger.info(f"[CameraCapture] OpenCV 초기화: {actual_w}x{actual_h}")

    def _enqueue_frame(self, frame_bgr: np.ndarray):
        """BGR 프레임을 RGB 변환 후 큐에 넣기. 가득 차면 가장 오래된 프레임 드롭."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp = time.monotonic()

        try:
            self._frame_queue.put_nowait((frame_rgb, timestamp))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()  # 오래된 프레임 버림
            except queue.Empty:
                pass
            self._frame_queue.put_nowait((frame_rgb, timestamp))

    def _capture_loop_realsense(self):
        """RealSense 캡처 루프."""
        import pyrealsense2 as rs

        while self._running:
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            self._enqueue_frame(np.asanyarray(color_frame.get_data()))

    def _capture_loop_opencv(self):
        """OpenCV 캡처 루프."""
        while self._running:
            ret, frame_bgr = self._cap.read()
            if not ret:
                if self._config.camera_type == "video":
                    logger.info("[CameraCapture] 비디오 파일 종료")
                    self._running = False
                    break
                continue
            self._enqueue_frame(frame_bgr)

            # 비디오 파일 모드: fps 맞추기
            if self._config.camera_type == "video":
                time.sleep(1.0 / self._config.camera_fps)

    def start(self):
        """캡처 스레드 시작."""
        if self._config.camera_type == "realsense":
            self._init_realsense()
            target = self._capture_loop_realsense
        else:
            self._init_opencv()
            target = self._capture_loop_opencv

        self._running = True
        self._thread = threading.Thread(target=target, name="CameraCapture", daemon=True)
        self._thread.start()

    def stop(self):
        """캡처 스레드 정지 및 리소스 해제."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._config.camera_type == "realsense" and hasattr(self, "_rs_pipeline"):
            self._rs_pipeline.stop()
        elif hasattr(self, "_cap"):
            self._cap.release()


# ============================================================================
# 프레임별 전처리 (Thread 2 핵심 로직)
# ============================================================================


class PerFrameProcessor:
    """단일 프레임에 대해 YOLO + ViTPose + HMR2를 순차 실행.

    오프라인 코드 대응:
        detect_and_track()       → self.detect_person()     [line 252]
        CocoPoseExtractor        → self.extract_keypoints() [line 263]
        get_image_features()     → self.extract_features()  [line 270]

    GPU 메모리 추정 (FP32, RTX 3090 24GB):
        YOLO v8x:    ~400 MB
        ViTPose-H:   ~350 MB
        HMR2 ViT:    ~500 MB
        합계:        ~1,250 MB (동시 로드 시)
    """

    def __init__(self, config: PreprocessorConfig):
        self._config = config
        self._device = torch.device(config.device)
        self._models_loaded = False

        # YOLO 추적 상태 — 단일 인물 ID 유지
        self._last_bbox: Optional[np.ndarray] = None
        self._miss_count: int = 0       # 연속 미검출 횟수
        self._max_miss: int = 3         # 최대 허용 미검출 프레임 수

    def load_models(self):
        """모든 전처리 모델을 GPU에 사전 로드.

        병목 2 해결: demo_smpl_hpe.py에서는 매번 모델을 로드했으나,
        여기서는 한 번만 로드하고 재사용.
        """
        logger.info("[PerFrameProcessor] 모델 로딩 시작 ...")
        t0 = time.time()

        # --- YOLO v8x ---
        # 출처: ultralytics 패키지, GENMO 레포 루트 yolov8x.pt
        from ultralytics import YOLO
        self._yolo = YOLO(self._config.yolo_model_path)
        logger.info(f"  YOLO v8x 로드 완료")

        # --- ViTPose-H ---
        # 출처: GENMO/scripts/demo/demo_utils.py → CocoPoseExtractor
        from demo_utils import CocoPoseExtractor
        self._pose_extractor = CocoPoseExtractor(device=self._config.device)
        logger.info(f"  ViTPose-H 로드 완료")

        # --- HMR2 ViT ---
        # 출처: demo_utils.py line 304 → gem.utils.hmr2_extractor.HMR2FeatureExtractor
        # get_image_features()가 내부적으로 사용하는 동일한 클래스
        from gem.utils.hmr2_extractor import HMR2FeatureExtractor
        self._hmr2_extractor = HMR2FeatureExtractor(
            self._config.hmr2_ckpt, device=self._config.device
        )
        logger.info(f"  HMR2 ViT 로드 완료")

        self._models_loaded = True
        logger.info(f"[PerFrameProcessor] 전체 로딩 완료 ({time.time() - t0:.1f}s)")

    def detect_person(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """YOLO v8x로 단일 인물 bbox 검출.

        오프라인 대응: detect_and_track(video_path, preprocess_dir) [line 252]

        단일 인물 추적 전략:
            1. 첫 프레임: confidence 최대인 person 선택
            2. 이후 프레임: 이전 bbox와 IoU 최대인 person 선택
            3. 미검출 시: 이전 bbox 반환 (최대 3프레임)
            4. 3프레임 연속 미검출: None 반환

        Returns:
            bbx_xys: (3,) ndarray [center_x, center_y, scale] 또는 None
        """
        results = self._yolo(
            frame_rgb,
            conf=self._config.yolo_conf_threshold,
            classes=[self._config.yolo_person_class_id],
            verbose=False,
        )

        # 결과 파싱: xyxy 배열에서 직접 검출 수 판정
        # (ultralytics Boxes 객체의 len()보다 xyxy shape이 더 신뢰성 있음)
        try:
            boxes = results[0].boxes
            if boxes is None:
                raise ValueError("boxes is None")
            xyxy = boxes.xyxy.cpu().numpy()   # (N, 4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            if len(xyxy) == 0:
                raise ValueError("empty detections")
        except (IndexError, ValueError, AttributeError):
            # 미검출 — fallback
            self._miss_count += 1
            if self._last_bbox is not None and self._miss_count <= self._max_miss:
                logger.debug(f"[YOLO] 미검출 ({self._miss_count}/{self._max_miss}) — 이전 bbox 사용")
                return self._last_bbox.copy()
            return None

        self._miss_count = 0

        if self._last_bbox is None:
            best_idx = int(np.argmax(confs))
        else:
            best_idx = self._find_best_match(xyxy)

        # xyxy → bbx_xys 변환
        # 출처: demo_utils.py line 89-91 — get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)
        # bbx_xys = [center_x, center_y, bbox_size] (bbox_size는 픽셀 단위)
        from gem.utils.geo_transform import get_bbx_xys_from_xyxy

        best_xyxy = torch.tensor(xyxy[best_idx:best_idx+1], dtype=torch.float32)  # (1, 4)
        bbx_xys_t = get_bbx_xys_from_xyxy(best_xyxy, base_enlarge=1.2)  # (1, 3)
        bbx_xys = bbx_xys_t.squeeze(0).numpy()  # (3,)

        self._last_bbox = bbx_xys
        return bbx_xys

    def _find_best_match(self, xyxy: np.ndarray) -> int:
        """이전 bbox와 IoU 최대인 검출 인덱스 반환."""
        prev_cx, prev_cy, prev_s = self._last_bbox
        # bbx_xys의 s는 bbox 크기(픽셀 단위)
        # 출처: demo_utils.py line 156 — hs = s / 2
        half = prev_s / 2.0
        prev_box = np.array([prev_cx - half, prev_cy - half, prev_cx + half, prev_cy + half])

        best_iou, best_idx = -1.0, 0
        for i, box in enumerate(xyxy):
            iou = self._iou(prev_box, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """두 bbox [x1,y1,x2,y2]의 IoU."""
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0

    def extract_keypoints(self, frame_rgb: np.ndarray, bbx_xys: np.ndarray) -> torch.Tensor:
        """ViTPose-H 2D keypoint 추출 (단일 프레임).

        오프라인 대응: CocoPoseExtractor.extract(frames, bbx_xys, batch_size=32) [line 263]
        병목 3 해결: batch_size=32 → 단일 프레임 처리로 변경

        Returns: kp2d (17, 3) — COCO-17 [x, y, confidence]
        """
        frames_batch = frame_rgb[np.newaxis, ...]           # (1, H, W, 3)
        bbx_batch = torch.from_numpy(bbx_xys[np.newaxis])  # (1, 3)
        kp2d = self._pose_extractor.extract(frames_batch, bbx_batch, batch_size=1)
        return kp2d.squeeze(0)  # (17, 3)

    def extract_features(self, frame_rgb: np.ndarray, bbx_xys: np.ndarray) -> tuple[torch.Tensor, bool]:
        """HMR2 ViT 이미지 feature 추출 (단일 프레임).

        오프라인 대응: get_image_features() [demo_utils.py line 269-314]

        HMR2FeatureExtractor.extract_frame_features()를 직접 호출.
        파일 I/O(임시 MP4 생성/삭제) 및 비디오 코덱 오버헤드 없이
        numpy 배열을 직접 전처리하여 HMR2 ViT 모델에 전달.

        Returns: (f_img (1024,), has_img bool)
        """
        try:
            bbx_tensor = torch.from_numpy(bbx_xys).float()  # (3,)
            f_img = self._hmr2_extractor.extract_frame_features(frame_rgb, bbx_tensor)
            return f_img, True

        except Exception as e:
            logger.warning(f"[HMR2] Feature 추출 실패: {e}")
            return torch.zeros(1024), False

    @torch.no_grad()
    def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """단일 프레임 전체 전처리: YOLO → ViTPose → HMR2.

        Returns:
            dict: bbx_xys(3,), kp2d(17,3), f_img(1024,), has_img, timestamp
            None: 사람 미검출 시
        """
        if not self._models_loaded:
            raise RuntimeError("load_models()를 먼저 호출하세요.")

        bbx_xys = self.detect_person(frame_rgb)
        if bbx_xys is None:
            return None

        kp2d = self.extract_keypoints(frame_rgb, bbx_xys)
        f_img, has_img = self.extract_features(frame_rgb, bbx_xys)

        return {
            "bbx_xys": torch.from_numpy(bbx_xys),  # (3,)
            "kp2d": kp2d,                           # (17, 3)
            "f_img": f_img,                         # (1024,)
            "has_img": has_img,
            "timestamp": time.monotonic(),
        }


# ============================================================================
# 슬라이딩 윈도우 버퍼
# ============================================================================


class SlidingWindowBuffer:
    """120프레임 FIFO 슬라이딩 윈도우 버퍼.

    오프라인에서는 전체 영상을 한 번에 처리 (demo_smpl_hpe.py line 261),
    실시간에서는 ring buffer로 프레임을 하나씩 축적.

    zero-padding 전략 (버퍼 미충족 시):
        - 유효 프레임: 버퍼 오른쪽에 정렬
        - 부족분: 왼쪽 zero-padding + mask=False

    출처: GENMO 분석 문서 — "없는 정보는 0으로 채우고 마스크로 표시"
    """

    def __init__(self, config: PreprocessorConfig):
        self._window_size = config.window_size
        self._inference_interval = config.inference_interval
        self._static_cam = config.static_cam
        self._buffer: deque[dict] = deque(maxlen=self._window_size)
        self._frame_count = 0
        self._total_frames = 0

    @property
    def current_length(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) > 0

    def add_frame(self, frame_data: dict) -> bool:
        """프레임 추가. Returns True if 추론 트리거."""
        self._buffer.append(frame_data)
        self._frame_count += 1
        self._total_frames += 1

        if self._frame_count >= self._inference_interval:
            self._frame_count = 0
            return True
        return False

    def get_data_dict(self, K_fullimg: np.ndarray) -> dict:
        """현재 버퍼 → GEM.predict() 입력 형식 변환.

        오프라인 대응: assemble_data() [demo_smpl_hpe.py lines 44-81]

        Args:
            K_fullimg: (3, 3) 카메라 내부 파라미터

        Returns:
            data dict matching assemble_data() output structure
            모든 텐서 shape은 (120, *) — window_size 고정
        """
        L = self._window_size  # 항상 120
        actual = len(self._buffer)

        # zero-padding 초기화
        kp2d = torch.zeros(L, 17, 3)
        bbx_xys = torch.zeros(L, 3)
        f_imgseq = torch.zeros(L, 1024)
        has_img_mask = torch.zeros(L, dtype=torch.bool)
        has_2d_mask = torch.zeros(L, dtype=torch.bool)

        # 유효 프레임 → 오른쪽 정렬
        offset = L - actual
        for i, fd in enumerate(self._buffer):
            idx = offset + i
            kp2d[idx] = fd["kp2d"]
            bbx_xys[idx] = fd["bbx_xys"]
            f_imgseq[idx] = fd["f_img"]
            has_img_mask[idx] = fd["has_img"]
            has_2d_mask[idx] = True

        # 카메라 파라미터 (정적 카메라)
        # 출처: get_camera_static(L, W, H) [demo_smpl_hpe.py line 274]
        K_rep = torch.from_numpy(K_fullimg).float().unsqueeze(0).expand(L, -1, -1)
        R_w2c = torch.eye(3).unsqueeze(0).expand(L, -1, -1)
        cam_angvel = torch.zeros(L, 6)
        cam_tvel = torch.zeros(L, 3)

        # 마스크 — 출처: assemble_data() [demo_smpl_hpe.py lines 71-77]
        from gem.utils.net_utils import get_valid_mask

        cam_valid_len = L if not self._static_cam else 0

        return {
            "kp2d": kp2d,                                      # (120, 17, 3)
            "bbx_xys": bbx_xys,                                # (120, 3)
            "K_fullimg": K_rep,                                 # (120, 3, 3)
            "cam_angvel": cam_angvel,                           # (120, 6)
            "cam_tvel": cam_tvel,                               # (120, 3)
            "R_w2c": R_w2c,                                     # (120, 3, 3)
            "f_imgseq": f_imgseq,                               # (120, 1024)
            "has_text": torch.tensor([False]),
            "mask": {
                "has_img_mask": has_img_mask,                   # (120,)
                "has_2d_mask": has_2d_mask,                     # (120,)
                "has_cam_mask": get_valid_mask(L, cam_valid_len),  # (120,)
                "has_audio_mask": get_valid_mask(L, 0),         # (120,) 전부 False
                "has_music_mask": get_valid_mask(L, 0),         # (120,) 전부 False
            },
            "length": torch.tensor(L),
            "meta": [{"mode": "default"}],
        }


# ============================================================================
# 통합 전처리 파이프라인
# ============================================================================


class RealtimePreprocessor:
    """실시간 비디오 → GEM 입력 전처리 통합 파이프라인.

    스레드 구조:
        Thread 1 (CameraCapture):    카메라 → frame_queue (30fps)
        Thread 2 (_process_loop):    frame_queue → YOLO+ViTPose+HMR2 → 버퍼
        Thread 3 (외부에서 소비):    inference_queue → GEM 추론

    GPU 메모리 총 사용량 추정 (FP32):
        ┌──────────────┬────────────┬───────────────────────────────┐
        │ 컴포넌트     │ VRAM (MB)  │ 출처                          │
        ├──────────────┼────────────┼───────────────────────────────┤
        │ YOLO v8x     │ ~400       │ ultralytics 벤치마크          │
        │ ViTPose-H    │ ~350       │ ViTPose-H 632M params         │
        │ HMR2 ViT     │ ~500       │ ViT-H/16 backbone             │
        │ GEM (추론)   │ ~2,500     │ Transformer denoiser + enc/dec│
        ├──────────────┼────────────┼───────────────────────────────┤
        │ 합계         │ ~3,750     │ RTX 3090(24GB) / 4090(24GB) OK│
        └──────────────┴────────────┴───────────────────────────────┘
    """

    def __init__(self, config: PreprocessorConfig):
        self._config = config
        self._frame_queue: queue.Queue = queue.Queue(maxsize=config.frame_queue_maxsize)
        self._inference_queue: queue.Queue = queue.Queue(maxsize=config.inference_queue_maxsize)
        self._camera = CameraCapture(config, self._frame_queue)
        self._processor = PerFrameProcessor(config)
        self._buffer = SlidingWindowBuffer(config)
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

        # 성능 모니터링
        self._perf = {
            "total_ms": deque(maxlen=100),
            "fps": 0.0,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def perf_stats(self) -> dict:
        vals = list(self._perf["total_ms"])
        return {
            "total_ms_mean": np.mean(vals) if vals else 0,
            "total_ms_p95": np.percentile(vals, 95) if len(vals) >= 5 else 0,
            "fps": self._perf["fps"],
            "buffer_fill": self._buffer.current_length,
            "buffer_capacity": self._config.window_size,
        }

    def start(self):
        """전체 파이프라인 시작."""
        logger.info("[RealtimePreprocessor] 시작 ...")
        self._processor.load_models()
        self._camera.start()

        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop, name="FrameProcessor", daemon=True
        )
        self._process_thread.start()
        logger.info("[RealtimePreprocessor] 실행 중")

    def stop(self):
        """전체 파이프라인 정상 종료."""
        logger.info("[RealtimePreprocessor] 종료 중 ...")
        self._running = False
        self._camera.stop()
        if self._process_thread is not None:
            self._process_thread.join(timeout=5.0)
        logger.info("[RealtimePreprocessor] 종료 완료")

    def get_inference_data(self, timeout: float = 1.0) -> Optional[dict]:
        """추론 스레드에서 호출: data dict 반환 (트리거 시) 또는 None."""
        try:
            return self._inference_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _process_loop(self):
        """전처리 메인 루프 (Thread 2)."""
        fps_counter = 0
        fps_timer = time.monotonic()

        while self._running:
            try:
                frame_rgb, ts = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                frame_data = self._processor.process_frame(frame_rgb)
                dt = (time.monotonic() - t0) * 1000
                self._perf["total_ms"].append(dt)

                if frame_data is None:
                    continue

                should_trigger = self._buffer.add_frame(frame_data)

                if should_trigger and self._buffer.is_ready:
                    K = self._camera.intrinsics_K
                    data_dict = self._buffer.get_data_dict(K)
                    payload = {"data_dict": data_dict, "camera_frame": frame_rgb.copy()}
                    try:
                        self._inference_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            self._inference_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._inference_queue.put_nowait(payload)

                    logger.info(
                        f"[추론 트리거] 버퍼 {self._buffer.current_length}/120 | "
                        f"전처리 {dt:.0f}ms"
                    )

            except Exception as e:
                logger.error(f"[ProcessLoop] 예외: {e}", exc_info=True)
                continue  # 예외 발생해도 루프 계속

            fps_counter += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                self._perf["fps"] = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.monotonic()


# ============================================================================
# CLI 테스트
# ============================================================================


def main():
    """독립 실행 테스트."""
    import argparse

    parser = argparse.ArgumentParser(description="실시간 전처리 파이프라인 테스트")
    parser.add_argument("--camera", default="webcam", choices=["realsense", "webcam", "video"])
    parser.add_argument("--video", default="")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = PreprocessorConfig(
        camera_type=args.camera,
        video_path=args.video,
        inference_interval=args.interval,
    )

    pp = RealtimePreprocessor(config)
    pp.start()

    t0, trigger_count = time.time(), 0

    try:
        while time.time() - t0 < args.duration and pp.is_running:
            data = pp.get_inference_data(timeout=1.0)
            if data is not None:
                trigger_count += 1
                # shape 검증
                assert data["kp2d"].shape == (120, 17, 3)
                assert data["bbx_xys"].shape == (120, 3)
                assert data["f_imgseq"].shape == (120, 1024)
                assert data["K_fullimg"].shape == (120, 3, 3)
                assert data["mask"]["has_audio_mask"].sum() == 0
                logger.info(f"  트리거 #{trigger_count} — {pp.perf_stats}")
    except KeyboardInterrupt:
        pass
    finally:
        pp.stop()

    logger.info(f"\n{time.time() - t0:.1f}초 동안 {trigger_count}회 트리거")


if __name__ == "__main__":
    main()
