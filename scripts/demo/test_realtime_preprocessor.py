# SPDX-License-Identifier: Apache-2.0
"""
realtime_preprocessor.py 단위 테스트

실행:
    pytest test_realtime_preprocessor.py -v -k "not integration"   # Mock (GPU 불필요)
    pytest test_realtime_preprocessor.py -v -k "integration"       # GPU + 체크포인트 필요
"""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from realtime_preprocessor import (
    CameraCapture,
    PerFrameProcessor,
    PreprocessorConfig,
    SlidingWindowBuffer,
)


# ============================================================================
# 픽스처
# ============================================================================

@pytest.fixture
def config():
    return PreprocessorConfig(
        camera_type="webcam", camera_resolution=(640, 480), camera_fps=30,
        window_size=120, inference_interval=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

@pytest.fixture
def frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def frame_data():
    return {
        "bbx_xys": torch.tensor([320.0, 240.0, 1.5]),
        "kp2d": torch.randn(17, 3),
        "f_img": torch.randn(1024),
        "has_img": True,
        "timestamp": time.monotonic(),
    }

@pytest.fixture
def K():
    return np.array([[640,0,320],[0,640,240],[0,0,1]], dtype=np.float32)


# ============================================================================
# 1. SlidingWindowBuffer
# ============================================================================

class TestSlidingWindowBuffer:

    def test_initial_empty(self, config):
        buf = SlidingWindowBuffer(config)
        assert buf.current_length == 0
        assert buf.is_ready is False

    def test_fifo_maxlen(self, config, frame_data):
        buf = SlidingWindowBuffer(config)
        for _ in range(150):
            buf.add_frame(frame_data)
        assert buf.current_length == 120

    def test_trigger_interval(self, config, frame_data):
        """매 30프레임마다 트리거 발생."""
        buf = SlidingWindowBuffer(config)
        triggers = [i+1 for i in range(120) if buf.add_frame(frame_data)]
        assert triggers == [30, 60, 90, 120]

    def test_zero_padding_shape(self, config, frame_data, K):
        """10프레임만 있을 때 shape이 (120, *)인지."""
        buf = SlidingWindowBuffer(config)
        for _ in range(10):
            buf.add_frame(frame_data)

        mock_mask = torch.zeros(120, dtype=torch.bool)
        with patch("gem.utils.net_utils.get_valid_mask", return_value=mock_mask):
            data = buf.get_data_dict(K)

        assert data["kp2d"].shape == (120, 17, 3)
        assert data["bbx_xys"].shape == (120, 3)
        assert data["f_imgseq"].shape == (120, 1024)
        assert data["K_fullimg"].shape == (120, 3, 3)
        assert data["cam_angvel"].shape == (120, 6)
        assert data["R_w2c"].shape == (120, 3, 3)

    def test_mask_padding(self, config, frame_data, K):
        """10프레임: 앞 110 = False, 뒤 10 = True."""
        buf = SlidingWindowBuffer(config)
        for _ in range(10):
            buf.add_frame(frame_data)

        mock_mask = torch.zeros(120, dtype=torch.bool)
        with patch("gem.utils.net_utils.get_valid_mask", return_value=mock_mask):
            data = buf.get_data_dict(K)

        assert data["mask"]["has_img_mask"][:110].sum() == 0
        assert data["mask"]["has_img_mask"][110:].sum() == 10

    def test_right_alignment(self, config, K):
        """유효 프레임이 오른쪽 정렬."""
        buf = SlidingWindowBuffer(config)
        for i in range(3):
            buf.add_frame({
                "bbx_xys": torch.tensor([float(i), 0, 0]),
                "kp2d": torch.zeros(17, 3),
                "f_img": torch.zeros(1024),
                "has_img": True, "timestamp": 0,
            })

        mock_mask = torch.zeros(120, dtype=torch.bool)
        with patch("gem.utils.net_utils.get_valid_mask", return_value=mock_mask):
            data = buf.get_data_dict(K)

        assert data["bbx_xys"][117, 0].item() == 0.0
        assert data["bbx_xys"][118, 0].item() == 1.0
        assert data["bbx_xys"][119, 0].item() == 2.0
        assert data["bbx_xys"][:117].abs().sum().item() == 0.0

    def test_unused_modalities(self, config, frame_data, K):
        """audio, music, text는 0/False."""
        buf = SlidingWindowBuffer(config)
        for _ in range(30):
            buf.add_frame(frame_data)

        mock_mask = torch.zeros(120, dtype=torch.bool)
        with patch("gem.utils.net_utils.get_valid_mask", return_value=mock_mask):
            data = buf.get_data_dict(K)

        assert data["has_text"].item() is False
        assert data["mask"]["has_audio_mask"].sum() == 0
        assert data["mask"]["has_music_mask"].sum() == 0


# ============================================================================
# 2. PerFrameProcessor (IoU + detect 로직)
# ============================================================================

class TestPerFrameProcessor:

    def test_iou_identical(self):
        box = np.array([10, 10, 50, 50], dtype=np.float32)
        assert PerFrameProcessor._iou(box, box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        b = np.array([20, 20, 30, 30], dtype=np.float32)
        assert PerFrameProcessor._iou(a, b) == pytest.approx(0.0)

    def test_iou_partial(self):
        a = np.array([0, 0, 20, 20], dtype=np.float32)
        b = np.array([10, 10, 30, 30], dtype=np.float32)
        assert 0.0 < PerFrameProcessor._iou(a, b) < 1.0

    def test_detect_first_frame(self, config, frame):
        proc = PerFrameProcessor(config)
        proc._models_loaded = True

        mock_result = MagicMock()
        mock_result.boxes.xyxy = torch.tensor([[100.0, 100.0, 300.0, 400.0]])
        mock_result.boxes.conf = torch.tensor([0.9])
        proc._yolo = MagicMock(return_value=[mock_result])

        # get_bbx_xys_from_xyxy mock — demo_utils.py line 89-91과 동일 포맷
        # 반환값: (1, 3) tensor [center_x, center_y, bbox_size]
        mock_bbx_xys = torch.tensor([[200.0, 250.0, 360.0]])  # cx, cy, size
        with patch("realtime_preprocessor.get_bbx_xys_from_xyxy", create=True):
            with patch(
                "gem.utils.geo_transform.get_bbx_xys_from_xyxy",
                return_value=mock_bbx_xys,
            ):
                bbx = proc.detect_person(frame)

        assert bbx is not None
        assert bbx.shape == (3,)
        assert bbx[0] == pytest.approx(200.0)   # center_x
        assert bbx[1] == pytest.approx(250.0)   # center_y

    def test_miss_fallback_and_timeout(self, config, frame):
        proc = PerFrameProcessor(config)
        proc._models_loaded = True
        proc._last_bbox = np.array([320.0, 240.0, 1.5], dtype=np.float32)

        mock_result = MagicMock()
        mock_result.boxes = None
        proc._yolo = MagicMock(return_value=[mock_result])

        # 1~3번 미검출: fallback
        for i in range(3):
            assert proc.detect_person(frame) is not None, f"miss {i+1}"

        # 4번째: None
        assert proc.detect_person(frame) is None


# ============================================================================
# 3. CameraCapture (OpenCV mock)
# ============================================================================

class TestCameraCapture:

    def test_opencv_intrinsics(self, config):
        fq = queue.Queue(maxsize=5)
        cam = CameraCapture(config, fq)

        with patch("cv2.VideoCapture") as mock_cap:
            inst = MagicMock()
            inst.isOpened.return_value = True
            inst.get.side_effect = lambda p: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
            }.get(p, 0)
            mock_cap.return_value = inst
            cam._init_opencv()

        K = cam.intrinsics_K
        assert K.shape == (3, 3)
        assert K[0, 0] == pytest.approx(640.0)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)

    def test_opencv_open_failure(self, config):
        """카메라/비디오 열기 실패 시 RuntimeError 발생."""
        fq = queue.Queue(maxsize=5)
        cam = CameraCapture(config, fq)

        with patch("cv2.VideoCapture") as mock_cap:
            inst = MagicMock()
            inst.isOpened.return_value = False  # 열기 실패
            mock_cap.return_value = inst

            with pytest.raises(RuntimeError, match="카메라/비디오를 열 수 없습니다"):
                cam._init_opencv()

    def test_video_file_not_found(self):
        """존재하지 않는 비디오 경로 시 FileNotFoundError."""
        cfg = PreprocessorConfig(
            camera_type="video", video_path="/nonexistent/video.mp4"
        )
        fq = queue.Queue(maxsize=5)
        cam = CameraCapture(cfg, fq)

        with pytest.raises(FileNotFoundError, match="비디오 파일을 찾을 수 없습니다"):
            cam._init_opencv()


# ============================================================================
# 4. 통합 테스트 (GPU + 체크포인트)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 필요")
class TestIntegration:

    @pytest.mark.integration
    def test_process_frame_real(self, config, frame):
        """실제 모델로 단일 프레임 출력 shape 검증."""
        proc = PerFrameProcessor(config)
        proc.load_models()
        result = proc.process_frame(frame)
        if result is not None:
            assert result["bbx_xys"].shape == (3,)
            assert result["kp2d"].shape == (17, 3)
            assert result["f_img"].shape == (1024,)
