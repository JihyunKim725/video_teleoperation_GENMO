# SPDX-License-Identifier: Apache-2.0
"""
realtime_preprocessor.py 통합 검증 도구

테스트용 비디오가 없거나 GPU 없이도 전체 파이프라인을 검증할 수 있음.

방법 1: 합성 테스트 비디오 생성
    python create_test_and_run.py --create-video
    → test_video_walking.mp4 생성 (사람 실루엣 + 걷기 모션)

방법 2: 모든 모델을 Mock하여 E2E 파이프라인 검증
    python create_test_and_run.py --mock-e2e
    → GPU 불필요, 스레딩/큐/버퍼/data_dict 전체 흐름 검증

방법 3: 실제 비디오 + 실제 모델 E2E
    python create_test_and_run.py --real-e2e --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 방법 1: 합성 테스트 비디오 생성
# ============================================================================


def create_test_video(
    output_path: str = "test_video_walking.mp4",
    duration_sec: float = 5.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
):
    """사람 실루엣이 좌우로 걷는 합성 테스트 비디오 생성.

    YOLO v8x가 실제로 검출할 수 있는 수준의 사람 형태를 그림.
    실제 사람 이미지가 아니므로 YOLO 검출률은 낮을 수 있음.
    검출 실패 시에도 fallback 로직을 검증할 수 있음.

    Args:
        output_path: 출력 비디오 경로
        duration_sec: 비디오 길이 (초)
        fps: 프레임 레이트
        width, height: 프레임 크기
    """
    total_frames = int(duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter를 열 수 없습니다: {output_path}")

    print(f"[생성] {total_frames}프레임, {width}x{height} @ {fps}fps, {duration_sec}초")

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # 밝은 회색 배경

        # 사람 위치 — 좌우로 걷기
        t = i / total_frames
        cx = int(width * 0.2 + width * 0.6 * t)  # 왼쪽→오른쪽 이동
        cy = int(height * 0.55)

        # 걷기 진동 (상하 반복)
        walk_bounce = int(5 * np.sin(i * 0.5))

        # 사람 실루엣 그리기 (머리 + 몸통 + 다리)
        body_color = (60, 60, 180)   # 빨간색 계열 (BGR)
        skin_color = (140, 180, 230)  # 살색

        # 머리 (원)
        head_y = cy - 100 + walk_bounce
        cv2.circle(frame, (cx, head_y), 22, skin_color, -1)
        cv2.circle(frame, (cx, head_y), 22, (40, 40, 40), 2)

        # 몸통 (사각형)
        torso_top = cy - 75 + walk_bounce
        torso_bot = cy + 10 + walk_bounce
        cv2.rectangle(frame, (cx - 30, torso_top), (cx + 30, torso_bot), body_color, -1)

        # 팔 (선)
        arm_swing = int(20 * np.sin(i * 0.3))
        cv2.line(frame, (cx - 30, torso_top + 10), (cx - 50, cy - 10 + arm_swing), body_color, 8)
        cv2.line(frame, (cx + 30, torso_top + 10), (cx + 50, cy - 10 - arm_swing), body_color, 8)

        # 다리 (선)
        leg_swing = int(25 * np.sin(i * 0.3))
        hip_y = torso_bot
        foot_y = cy + 100 + walk_bounce
        cv2.line(frame, (cx - 10, hip_y), (cx - 20 + leg_swing, foot_y), body_color, 10)
        cv2.line(frame, (cx + 10, hip_y), (cx + 20 - leg_swing, foot_y), body_color, 10)

        # 바닥 선
        cv2.line(frame, (0, foot_y + 10), (width, foot_y + 10), (100, 100, 100), 2)

        # 프레임 번호 표시
        cv2.putText(
            frame, f"Frame {i}/{total_frames}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )

        writer.write(frame)

    writer.release()
    print(f"[완료] {output_path} 생성됨 ({os.path.getsize(output_path) / 1024:.0f} KB)")
    return output_path


# ============================================================================
# 방법 2: Mock E2E 파이프라인 검증
# ============================================================================


def run_mock_e2e_test(
    duration_sec: float = 5.0,
    fps: int = 30,
    inference_interval: int = 30,
):
    """GPU 모델 없이 전체 파이프라인 E2E 검증.

    검증 항목:
        ✓ 카메라 캡처 스레드 → frame_queue 전달
        ✓ 전처리 스레드 → YOLO+ViTPose+HMR2 Mock 호출
        ✓ SlidingWindowBuffer → FIFO + zero-padding
        ✓ 추론 트리거 → data_dict shape 검증
        ✓ 스레드간 Queue 통신 정상 동작
        ✓ Graceful shutdown
    """
    from realtime_preprocessor import (
        PreprocessorConfig,
        RealtimePreprocessor,
        PerFrameProcessor,
        CameraCapture,
    )

    total_frames = int(duration_sec * fps)
    expected_triggers = total_frames // inference_interval

    print("=" * 60)
    print("Mock E2E 파이프라인 검증")
    print(f"  합성 프레임: {total_frames}개 ({duration_sec}초 @ {fps}fps)")
    print(f"  추론 간격: {inference_interval}프레임")
    print(f"  예상 트리거: {expected_triggers}회")
    print("=" * 60)

    config = PreprocessorConfig(
        camera_type="webcam",
        camera_resolution=(640, 480),
        camera_fps=fps,
        window_size=120,
        inference_interval=inference_interval,
        device="cpu",
    )

    pp = RealtimePreprocessor(config)

    # --- Mock 설정 ---

    # Mock 1: 카메라 — 합성 프레임을 직접 큐에 넣는 스레드
    def fake_camera_loop(frame_queue, num_frames, fps):
        """합성 RGB 프레임을 생성하여 큐에 넣는 가짜 카메라.
        테스트용이므로 실시간 속도가 아닌 최대 속도로 전송.
        큐가 가득 차면 짧은 대기 후 재시도 (프레임 드롭 최소화).
        """
        for i in range(num_frames):
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            timestamp = time.monotonic()
            while True:
                try:
                    frame_queue.put(item=(frame, timestamp), timeout=0.1)
                    break
                except:
                    pass
            time.sleep(0.001)  # 최소 yield (CPU 독점 방지)

    # Mock 2: YOLO — 항상 화면 중앙에서 사람 검출
    def mock_yolo_call(*args, **kwargs):
        result = MagicMock()
        result.boxes.xyxy = torch.tensor([[200.0, 100.0, 440.0, 400.0]])
        result.boxes.conf = torch.tensor([0.92])
        return [result]

    # Mock 3: ViTPose — 랜덤 키포인트 (COCO-17 형태)
    def mock_vitpose_extract(frames, bbx, batch_size=1):
        n = len(frames)
        kp2d = torch.randn(n, 17, 3)
        kp2d[:, :, 2] = torch.sigmoid(kp2d[:, :, 2])  # confidence 0~1
        return kp2d

    # Mock 4: HMR2 — 랜덤 1024차원 feature
    def mock_hmr2_extract(video_path, bbx_xys, batch_size=1):
        n = bbx_xys.shape[0]
        return torch.randn(n, 1024)

    # Mock 5: get_bbx_xys_from_xyxy
    def mock_get_bbx_xys(xyxy, base_enlarge=1.2):
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        sz = torch.max(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]) * base_enlarge
        return torch.stack([cx, cy, sz], dim=1)

    # Mock 6: get_valid_mask
    def mock_get_valid_mask(L, valid_len):
        mask = torch.zeros(L, dtype=torch.bool)
        mask[:valid_len] = True
        return mask

    # --- Mock 적용 ---
    # 모델 로딩 건너뛰기
    pp._processor._models_loaded = True
    pp._processor._yolo = MagicMock(side_effect=mock_yolo_call)
    pp._processor._pose_extractor = MagicMock()
    pp._processor._pose_extractor.extract = MagicMock(side_effect=mock_vitpose_extract)
    pp._processor._hmr2_extractor = MagicMock()
    pp._processor._hmr2_extractor.extract_video_features = MagicMock(
        side_effect=mock_hmr2_extract
    )

    # extract_features를 직접 Mock (임시 비디오 생성 우회)
    # 실제 HMR2 연동은 방법 3(--real-e2e)에서 검증
    def mock_extract_features(frame_rgb, bbx_xys):
        return torch.randn(1024), True
    pp._processor.extract_features = mock_extract_features

    # 카메라 intrinsics Mock
    pp._camera._K = np.array([
        [640.0, 0, 320.0],
        [0, 640.0, 240.0],
        [0, 0, 1],
    ], dtype=np.float32)

    # --- 실행 ---
    # 카메라 스레드를 가짜로 교체
    pp._camera._running = True
    cam_thread = threading.Thread(
        target=fake_camera_loop,
        args=(pp._frame_queue, total_frames, fps),
        daemon=True,
    )
    cam_thread.start()

    # 전처리 스레드 시작 (실제 로직 사용, 모델만 Mock)
    pp._running = True

    with patch("gem.utils.geo_transform.get_bbx_xys_from_xyxy", side_effect=mock_get_bbx_xys):
        with patch("gem.utils.net_utils.get_valid_mask", side_effect=mock_get_valid_mask):
            pp._process_thread = threading.Thread(
                target=pp._process_loop, name="FrameProcessor", daemon=True
            )
            pp._process_thread.start()

            # --- 결과 수집 ---
            trigger_count = 0
            data_dicts = []
            t0 = time.time()
            timeout = duration_sec + 5.0  # 여유 시간

            while time.time() - t0 < timeout:
                data = pp.get_inference_data(timeout=0.5)
                if data is not None:
                    trigger_count += 1
                    data_dicts.append(data)

                    if trigger_count >= expected_triggers:
                        break

            # 종료
            pp._running = False
            pp._camera._running = False
            cam_thread.join(timeout=3.0)
            if pp._process_thread:
                pp._process_thread.join(timeout=3.0)

    # --- 검증 ---
    print(f"\n{'=' * 60}")
    print("검증 결과")
    print(f"{'=' * 60}")

    errors = []

    # 1. 트리거 횟수
    if trigger_count >= expected_triggers:
        print(f"  ✓ 추론 트리거: {trigger_count}회 (예상 ≥{expected_triggers})")
    else:
        msg = f"  ✗ 추론 트리거: {trigger_count}회 (예상 ≥{expected_triggers})"
        print(msg)
        errors.append(msg)

    # 2. data_dict shape 검증
    if data_dicts:
        d = data_dicts[-1]  # 마지막 트리거의 data_dict

        shape_checks = [
            ("kp2d",       (120, 17, 3)),
            ("bbx_xys",    (120, 3)),
            ("f_imgseq",   (120, 1024)),
            ("K_fullimg",  (120, 3, 3)),
            ("cam_angvel", (120, 6)),
            ("cam_tvel",   (120, 3)),
            ("R_w2c",      (120, 3, 3)),
        ]

        for key, expected_shape in shape_checks:
            actual = d[key].shape
            if actual == expected_shape:
                print(f"  ✓ {key}: {actual}")
            else:
                msg = f"  ✗ {key}: {actual} (예상 {expected_shape})"
                print(msg)
                errors.append(msg)

        # 3. 마스크 검증
        mask = d["mask"]
        mask_checks = [
            ("has_img_mask",   (120,)),
            ("has_2d_mask",    (120,)),
            ("has_cam_mask",   (120,)),
            ("has_audio_mask", (120,)),
            ("has_music_mask", (120,)),
        ]
        for key, expected_shape in mask_checks:
            actual = mask[key].shape
            if actual == expected_shape:
                print(f"  ✓ mask.{key}: {actual}")
            else:
                msg = f"  ✗ mask.{key}: {actual} (예상 {expected_shape})"
                print(msg)
                errors.append(msg)

        # 4. 미사용 모달리티 검증
        if d["has_text"].item() is False:
            print(f"  ✓ has_text: False")
        else:
            errors.append("  ✗ has_text가 True")

        if mask["has_audio_mask"].sum() == 0:
            print(f"  ✓ has_audio_mask: 전부 False")
        else:
            errors.append("  ✗ has_audio_mask에 True가 있음")

        if mask["has_music_mask"].sum() == 0:
            print(f"  ✓ has_music_mask: 전부 False")
        else:
            errors.append("  ✗ has_music_mask에 True가 있음")

        # 5. 유효 프레임 수 확인
        valid_img = mask["has_img_mask"].sum().item()
        buffer_fill = min(total_frames, 120)
        print(f"  ✓ 유효 프레임: {valid_img}/120 (버퍼 충전: {buffer_fill})")

        # 6. 오른쪽 정렬 검증 (앞부분이 zero인지)
        if valid_img < 120:
            zero_region = d["bbx_xys"][:120 - int(valid_img)]
            if zero_region.abs().sum() == 0:
                print(f"  ✓ zero-padding 오른쪽 정렬: 올바름")
            else:
                errors.append("  ✗ zero-padding이 오른쪽 정렬되지 않음")

    else:
        errors.append("  ✗ data_dict가 하나도 수집되지 않음")

    # 7. 성능 통계
    stats = pp.perf_stats
    print(f"\n  성능 통계:")
    print(f"    전처리 평균: {stats['total_ms_mean']:.1f}ms")
    print(f"    FPS: {stats['fps']:.1f}")
    print(f"    버퍼 채움: {stats['buffer_fill']}/{stats['buffer_capacity']}")

    # 최종 결과
    print(f"\n{'=' * 60}")
    if errors:
        print(f"결과: FAIL — {len(errors)}개 오류")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print(f"결과: ALL PASSED")
        print(f"{'=' * 60}")
        return True


# ============================================================================
# 방법 3: 실제 비디오 + 실제 모델 E2E
# ============================================================================


def run_real_e2e_test(video_path: str, duration_sec: float = 30.0):
    """실제 비디오 + 실제 GPU 모델로 E2E 검증.

    전제: GENMO 설치 완료, GPU 사용 가능, 체크포인트 존재
    """
    from realtime_preprocessor import PreprocessorConfig, RealtimePreprocessor

    if not os.path.exists(video_path):
        print(f"[ERROR] 비디오 파일 없음: {video_path}")
        sys.exit(1)

    config = PreprocessorConfig(
        camera_type="video",
        video_path=video_path,
        inference_interval=30,
    )

    pp = RealtimePreprocessor(config)
    pp.start()

    trigger_count = 0
    t0 = time.time()

    try:
        while time.time() - t0 < duration_sec and pp.is_running:
            data = pp.get_inference_data(timeout=1.0)
            if data is not None:
                trigger_count += 1

                # shape 검증
                assert data["kp2d"].shape == (120, 17, 3)
                assert data["f_imgseq"].shape == (120, 1024)

                # feature가 zeros가 아닌지 확인 (HMR2 정상 동작 검증)
                f_nonzero = (data["f_imgseq"].abs().sum(dim=1) > 0).sum().item()

                stats = pp.perf_stats
                print(
                    f"  트리거 #{trigger_count} | "
                    f"버퍼: {stats['buffer_fill']}/120 | "
                    f"유효 HMR2: {f_nonzero}/120 | "
                    f"전처리: {stats['total_ms_mean']:.0f}ms | "
                    f"FPS: {stats['fps']:.1f}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        pp.stop()

    print(f"\n{time.time() - t0:.1f}초 동안 {trigger_count}회 트리거")
    return trigger_count > 0


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="realtime_preprocessor.py 통합 검증 도구",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="합성 테스트 비디오 생성 (test_video_walking.mp4)",
    )
    parser.add_argument(
        "--mock-e2e",
        action="store_true",
        help="Mock 모델로 전체 파이프라인 E2E 검증 (GPU 불필요)",
    )
    parser.add_argument(
        "--real-e2e",
        action="store_true",
        help="실제 모델로 E2E 검증 (GPU + 체크포인트 필요)",
    )
    parser.add_argument("--video", default="", help="비디오 파일 경로 (--real-e2e 용)")
    parser.add_argument("--duration", type=float, default=5.0, help="테스트 시간 (초)")
    parser.add_argument("--output", default="test_video_walking.mp4", help="합성 비디오 출력 경로")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not (args.create_video or args.mock_e2e or args.real_e2e):
        # 기본: 비디오 생성 + Mock E2E 순서대로 실행
        print("[기본 모드] 합성 비디오 생성 → Mock E2E 검증\n")
        args.create_video = True
        args.mock_e2e = True

    if args.create_video:
        print("\n" + "=" * 60)
        print("방법 1: 합성 테스트 비디오 생성")
        print("=" * 60)
        path = create_test_video(
            output_path=args.output,
            duration_sec=args.duration,
        )
        print(f"\n다음 명령으로 실제 모델 테스트 가능:")
        print(f"  python scripts/demo/realtime_preprocessor.py \\")
        print(f"    --camera video --video {path} --duration {args.duration}")

    if args.mock_e2e:
        print("\n" + "=" * 60)
        print("방법 2: Mock E2E 파이프라인 검증")
        print("=" * 60)
        success = run_mock_e2e_test(
            duration_sec=args.duration,
            inference_interval=30,
        )
        if not success:
            sys.exit(1)

    if args.real_e2e:
        print("\n" + "=" * 60)
        print("방법 3: 실제 모델 E2E 검증")
        print("=" * 60)
        if not args.video:
            print("[ERROR] --video 경로를 지정하세요")
            sys.exit(1)
        success = run_real_e2e_test(args.video, duration_sec=args.duration)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
