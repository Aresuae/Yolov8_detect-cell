import os
import sys
import time
import types
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Disable Ultralytics auto pip-install to avoid runtime interruption.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def ensure_ultralytics_compat_modules() -> None:
    """
    Provide compatibility aliases for checkpoints trained with newer ultralytics
    package layout (e.g. ultralytics.nn.modules.conv), while current project
    uses legacy single-file layout ultralytics.nn.modules.
    """
    def alias_module(new_name: str, old_name: str) -> bool:
        if new_name in sys.modules:
            return True
        try:
            sys.modules[new_name] = importlib.import_module(old_name)
            return True
        except Exception:
            return False

    def alias_package_submodules(new_pkg: str, old_pkg: str) -> None:
        if not alias_module(new_pkg, old_pkg):
            return
        try:
            pkg = importlib.import_module(old_pkg)
            pkg_dir = os.path.dirname(pkg.__file__) if getattr(pkg, "__file__", None) else None
        except Exception:
            pkg_dir = None
        if not pkg_dir or not os.path.isdir(pkg_dir):
            return

        for name in os.listdir(pkg_dir):
            abs_path = os.path.join(pkg_dir, name)
            mod_name = ""
            if name.endswith(".py") and name != "__init__.py":
                mod_name = name[:-3]
            elif os.path.isdir(abs_path) and os.path.exists(os.path.join(abs_path, "__init__.py")):
                mod_name = name
            if mod_name:
                alias_module(f"{new_pkg}.{mod_name}", f"{old_pkg}.{mod_name}")

    # New ultralytics package paths -> legacy yolo.* paths in this repo.
    alias_package_submodules("ultralytics.utils", "ultralytics.yolo.utils")
    alias_package_submodules("ultralytics.data", "ultralytics.yolo.data")
    alias_package_submodules("ultralytics.engine", "ultralytics.yolo.engine")
    alias_module("ultralytics.cfg", "ultralytics.yolo.cfg")

    try:
        base_module = importlib.import_module("ultralytics.nn.modules")
    except Exception:
        return

    # Let Python treat this module like a namespace package for submodules.
    if not hasattr(base_module, "__path__"):
        base_module.__path__ = []  # type: ignore[attr-defined]

    for sub_name in ("conv", "block", "head", "transformer"):
        full_name = f"ultralytics.nn.modules.{sub_name}"
        if full_name in sys.modules:
            continue
        alias_module = types.ModuleType(full_name)
        alias_module.__dict__.update(base_module.__dict__)
        sys.modules[full_name] = alias_module

    # Some checkpoints serialize loss class names not present in this legacy repo.
    try:
        import torch
        import torch.nn as nn

        loss_module = importlib.import_module("ultralytics.yolo.utils.loss")
        if "ultralytics.utils.loss" in sys.modules:
            loss_module = sys.modules["ultralytics.utils.loss"]

        class _CompatLoss(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                # Loss objects are not needed for inference-only loading.
                return torch.tensor(0.0)

        for loss_name in (
            "v8DetectionLoss",
            "v8SegmentationLoss",
            "v8PoseLoss",
            "v8ClassificationLoss",
            "E2EDetectLoss",
            "TVPDetectLoss",
        ):
            if not hasattr(loss_module, loss_name):
                setattr(loss_module, loss_name, _CompatLoss)
    except Exception:
        # Compatibility patch is best-effort and should not break app startup.
        pass


@dataclass
class DetectionResult:
    model_name: str
    annotated: np.ndarray
    count: int
    infer_time_ms: float
    raw_boxes: List[Tuple[int, int, int, int, str, float]]


class ModelAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = False
        self.weight_path: Optional[str] = None

    def load(self, weight_path: str) -> None:
        raise NotImplementedError

    def infer(
        self,
        image_bgr: np.ndarray,
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ) -> DetectionResult:
        raise NotImplementedError

    @staticmethod
    def draw_boxes(
        image_bgr: np.ndarray,
        boxes: List[Tuple[int, int, int, int, str, float]],
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ) -> np.ndarray:
        canvas = image_bgr.copy()
        if not (show_box or show_label or show_conf):
            return canvas

        for x1, y1, x2, y2, cls_name, conf in boxes:
            color = (0, 200, 0)
            if show_box:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label_parts = []
            if show_label:
                label_parts.append(cls_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")
            text = " ".join(label_parts)
            if text:
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                y_text = max(20, y1)
                cv2.rectangle(canvas, (x1, y_text - h - 8), (x1 + w + 6, y_text), color, -1)
                cv2.putText(
                    canvas,
                    text,
                    (x1 + 3, y_text - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
        return canvas


class YOLOv8Adapter(ModelAdapter):
    def __init__(self):
        super().__init__("YOLOv8")
        self.model = None

    def load(self, weight_path: str) -> None:
        from ultralytics import YOLO
        ensure_ultralytics_compat_modules()
        try:
            self.model = YOLO(weight_path)
            self.weight_path = weight_path
            self.loaded = True
        except Exception as e:
            raise RuntimeError(
                "YOLOv8 权重加载失败。请确认该 .pt 与当前 ultralytics 版本匹配，"
                "并在对应模型类型下加载（YOLOv5/v7/v8）。\n"
                f"原始错误: {e}"
            ) from e

    def infer(
        self,
        image_bgr: np.ndarray,
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ) -> DetectionResult:
        if not self.loaded or self.model is None:
            raise RuntimeError("YOLOv8 model is not loaded.")

        start = time.perf_counter()
        results = self.model.predict(source=image_bgr, conf=conf_thres, verbose=False)
        infer_time_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]

        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy().astype(float)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                cls_id = cls_ids[i]
                cls_name = result.names.get(cls_id, str(cls_id))
                boxes.append((x1, y1, x2, y2, cls_name, confs[i]))

        annotated = self.draw_boxes(image_bgr, boxes, show_box, show_label, show_conf)
        return DetectionResult(self.model_name, annotated, len(boxes), infer_time_ms, boxes)


class YOLOv5OrV7Adapter(ModelAdapter):
    def __init__(self, model_name: str, repo: str):
        super().__init__(model_name)
        self.repo = repo
        self.model = None

    def load(self, weight_path: str) -> None:
        import torch

        self.model = torch.hub.load(
            self.repo,
            "custom",
            path=weight_path,
            trust_repo=True,
            force_reload=False,
        )
        self.weight_path = weight_path
        self.loaded = True

    def infer(
        self,
        image_bgr: np.ndarray,
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ) -> DetectionResult:
        if not self.loaded or self.model is None:
            raise RuntimeError(f"{self.model_name} model is not loaded.")

        self.model.conf = conf_thres
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        results = self.model(image_rgb)
        infer_time_ms = (time.perf_counter() - start) * 1000.0

        names = self.model.names
        pred = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))
        boxes = []
        for row in pred:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            cls_id_i = int(cls_id)
            cls_name = names[cls_id_i] if isinstance(names, (list, tuple)) else names.get(cls_id_i, str(cls_id_i))
            boxes.append((int(x1), int(y1), int(x2), int(y2), cls_name, float(conf)))

        annotated = self.draw_boxes(image_bgr, boxes, show_box, show_label, show_conf)
        return DetectionResult(self.model_name, annotated, len(boxes), infer_time_ms, boxes)


class MultiModelDetector:
    def __init__(self):
        self.adapters: Dict[str, ModelAdapter] = {
            "YOLOv5": YOLOv5OrV7Adapter("YOLOv5", "ultralytics/yolov5"),
            "YOLOv7": YOLOv5OrV7Adapter("YOLOv7", "WongKinYiu/yolov7"),
            "YOLOv8": YOLOv8Adapter(),
        }

    def load_model(self, model_name: str, weight_path: str) -> None:
        if model_name not in self.adapters:
            raise ValueError(f"Unknown model: {model_name}")
        self.adapters[model_name].load(weight_path)

    def infer_single(
        self,
        model_names: List[str],
        image_bgr: np.ndarray,
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ) -> List[DetectionResult]:
        outputs = []
        for name in model_names:
            adapter = self.adapters[name]
            if not adapter.loaded:
                continue
            outputs.append(adapter.infer(image_bgr, conf_thres, show_box, show_label, show_conf))
        return outputs

    def loaded_models(self) -> List[str]:
        return [name for name, adapter in self.adapters.items() if adapter.loaded]


class ImageDetectThread(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        detector: MultiModelDetector,
        image_path: str,
        model_names: List[str],
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ):
        super().__init__()
        self.detector = detector
        self.image_path = image_path
        self.model_names = model_names
        self.conf_thres = conf_thres
        self.show_box = show_box
        self.show_label = show_label
        self.show_conf = show_conf

    def run(self) -> None:
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise RuntimeError(f"Cannot read image: {self.image_path}")
            results = self.detector.infer_single(
                self.model_names,
                image,
                self.conf_thres,
                self.show_box,
                self.show_label,
                self.show_conf,
            )
            self.result_ready.emit(image, results)
        except Exception as e:
            self.error.emit(str(e))


class FolderDetectThread(QtCore.QThread):
    file_done = QtCore.pyqtSignal(str, object)
    finished_summary = QtCore.pyqtSignal(int)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        detector: MultiModelDetector,
        folder_path: str,
        output_dir: Optional[str],
        model_names: List[str],
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ):
        super().__init__()
        self.detector = detector
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.model_names = model_names
        self.conf_thres = conf_thres
        self.show_box = show_box
        self.show_label = show_label
        self.show_conf = show_conf

    def run(self) -> None:
        try:
            image_files = []
            for name in os.listdir(self.folder_path):
                ext = os.path.splitext(name.lower())[1]
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(self.folder_path, name))

            if not image_files:
                raise RuntimeError("No image files found in selected folder.")

            os.makedirs(self.output_dir, exist_ok=True) if self.output_dir else None

            processed = 0
            for image_path in image_files:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                results = self.detector.infer_single(
                    self.model_names,
                    image,
                    self.conf_thres,
                    self.show_box,
                    self.show_label,
                    self.show_conf,
                )
                self.file_done.emit(image_path, results)

                if self.output_dir:
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    for res in results:
                        out_path = os.path.join(self.output_dir, f"{base}_{res.model_name}.jpg")
                        cv2.imwrite(out_path, res.annotated)
                processed += 1

            self.finished_summary.emit(processed)
        except Exception as e:
            self.error.emit(str(e))


class CameraDetectThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        detector: MultiModelDetector,
        camera_id: int,
        model_names: List[str],
        conf_thres: float,
        show_box: bool,
        show_label: bool,
        show_conf: bool,
    ):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.model_names = model_names
        self.conf_thres = conf_thres
        self.show_box = show_box
        self.show_label = show_label
        self.show_conf = show_conf
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error.emit("Cannot open camera.")
            return

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Failed to read camera frame.")
                    break
                results = self.detector.infer_single(
                    self.model_names,
                    frame,
                    self.conf_thres,
                    self.show_box,
                    self.show_label,
                    self.show_conf,
                )
                self.frame_ready.emit(frame, results)
                self.msleep(10)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            cap.release()


def cvimg_to_qpix(image_bgr: np.ndarray, target_size: QtCore.QSize) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, w * c, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg.copy())
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模型目标检测系统 (YOLOv5 / YOLOv7 / YOLOv8)")
        self.setMinimumSize(1080, 680)
        self.detector = MultiModelDetector()

        self.current_original: Optional[np.ndarray] = None
        self.current_results: Dict[str, DetectionResult] = {}
        self.current_image_path: Optional[str] = None

        self.image_thread: Optional[ImageDetectThread] = None
        self.folder_thread: Optional[FolderDetectThread] = None
        self.camera_thread: Optional[CameraDetectThread] = None

        self._build_ui()
        self._apply_styles()
        self._init_window_geometry()

    def _init_window_geometry(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1360, 860)
            return
        available = screen.availableGeometry()
        target_w = min(1680, max(1080, int(available.width() * 0.9)))
        target_h = min(980, max(680, int(available.height() * 0.9)))
        x = available.x() + max(0, (available.width() - target_w) // 2)
        y = available.y() + max(0, (available.height() - target_h) // 2)
        self.setGeometry(x, y, target_w, target_h)

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1f232a;
                color: #e6edf3;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #39414f;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
                color: #7cc5ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
            QPushButton {
                background-color: #2f6feb;
                border: none;
                border-radius: 6px;
                color: #ffffff;
                padding: 8px 12px;
                min-height: 34px;
            }
            QPushButton:hover { background-color: #3d7dff; }
            QPushButton:pressed { background-color: #245dd4; }
            QPushButton:disabled {
                background-color: #3a3f4b;
                color: #8b949e;
            }
            QLineEdit {
                background-color: #11161d;
                border: 1px solid #39414f;
                border-radius: 6px;
                padding: 6px;
                color: #e6edf3;
            }
            QTableWidget {
                background-color: #11161d;
                border: 1px solid #39414f;
                border-radius: 6px;
                gridline-color: #39414f;
                alternate-background-color: #161b22;
            }
            QHeaderView::section {
                background-color: #262c36;
                color: #9ecbff;
                border: none;
                border-right: 1px solid #39414f;
                padding: 6px;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #39414f;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background: #2b313c;
                color: #c9d1d9;
                padding: 8px 14px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #2f6feb; color: #ffffff; }
            QSlider::groove:horizontal {
                border-radius: 4px;
                height: 8px;
                background: #39414f;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            """
        )

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main_layout = QtWidgets.QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(320)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(10)

        model_group = QtWidgets.QGroupBox("模型加载")
        model_layout = QtWidgets.QGridLayout(model_group)
        self.model_rows = {}
        for row, model_name in enumerate(["YOLOv5", "YOLOv7", "YOLOv8"]):
            chk = QtWidgets.QCheckBox(f"{model_name} 参与检测")
            chk.setChecked(True)
            path_edit = QtWidgets.QLineEdit()
            path_edit.setReadOnly(True)
            btn = QtWidgets.QPushButton(f"加载 {model_name} 权重")
            btn.clicked.connect(lambda _, n=model_name: self.load_weight(n))
            model_layout.addWidget(chk, row * 2, 0, 1, 2)
            model_layout.addWidget(path_edit, row * 2 + 1, 0)
            model_layout.addWidget(btn, row * 2 + 1, 1)
            self.model_rows[model_name] = (chk, path_edit, btn)
        left_layout.addWidget(model_group)

        param_group = QtWidgets.QGroupBox("检测参数")
        param_layout = QtWidgets.QGridLayout(param_group)
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_value = QtWidgets.QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v / 100.0:.2f}")
        )
        self.show_box_chk = QtWidgets.QCheckBox("显示检测框")
        self.show_box_chk.setChecked(True)
        self.show_label_chk = QtWidgets.QCheckBox("显示类别标签")
        self.show_label_chk.setChecked(True)
        self.show_conf_chk = QtWidgets.QCheckBox("显示置信度")
        self.show_conf_chk.setChecked(True)

        param_layout.addWidget(QtWidgets.QLabel("置信度阈值"), 0, 0)
        param_layout.addWidget(self.conf_slider, 0, 1)
        param_layout.addWidget(self.conf_value, 0, 2)
        param_layout.addWidget(self.show_box_chk, 1, 0, 1, 3)
        param_layout.addWidget(self.show_label_chk, 2, 0, 1, 3)
        param_layout.addWidget(self.show_conf_chk, 3, 0, 1, 3)
        left_layout.addWidget(param_group)

        input_group = QtWidgets.QGroupBox("输入与运行")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        self.btn_image = QtWidgets.QPushButton("单张图片检测")
        self.btn_folder = QtWidgets.QPushButton("文件夹批量检测")
        self.btn_camera_start = QtWidgets.QPushButton("启动摄像头检测")
        self.btn_camera_stop = QtWidgets.QPushButton("停止摄像头检测")
        self.btn_camera_stop.setEnabled(False)
        self.btn_save = QtWidgets.QPushButton("保存当前检测结果")
        self.btn_save_batch = QtWidgets.QPushButton("批量保存当前图片多模型结果")
        self.btn_save_batch.setEnabled(False)

        self.btn_image.clicked.connect(self.run_single_image)
        self.btn_folder.clicked.connect(self.run_folder_batch)
        self.btn_camera_start.clicked.connect(self.start_camera)
        self.btn_camera_stop.clicked.connect(self.stop_camera)
        self.btn_save.clicked.connect(self.save_current_result)
        self.btn_save_batch.clicked.connect(self.save_current_all_results)

        for btn in [
            self.btn_image,
            self.btn_folder,
            self.btn_camera_start,
            self.btn_camera_stop,
            self.btn_save,
            self.btn_save_batch,
        ]:
            input_layout.addWidget(btn)
        left_layout.addWidget(input_group)

        stat_group = QtWidgets.QGroupBox("检测统计")
        stat_layout = QtWidgets.QVBoxLayout(stat_group)
        self.status_label = QtWidgets.QLabel("状态：就绪")
        self.count_label = QtWidgets.QLabel("目标数量：0")
        self.time_label = QtWidgets.QLabel("推理耗时：0 ms")
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["模型", "目标数", "推理耗时(ms)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        stat_layout.addWidget(self.status_label)
        stat_layout.addWidget(self.count_label)
        stat_layout.addWidget(self.time_label)
        stat_layout.addWidget(self.table)
        left_layout.addWidget(stat_group)
        left_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(10)
        self.original_label = QtWidgets.QLabel("原图显示区域")
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setMinimumHeight(260)
        self.original_label.setStyleSheet("border: 1px solid #808080;")

        self.result_tabs = QtWidgets.QTabWidget()
        self.result_labels = {}
        for name in ["YOLOv5", "YOLOv7", "YOLOv8"]:
            lbl = QtWidgets.QLabel(f"{name} 检测结果")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid #808080;")
            self.result_tabs.addTab(lbl, name)
            self.result_labels[name] = lbl
        right_layout.addWidget(self.original_label, 1)
        right_layout.addWidget(self.result_tabs, 1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])

        main_layout.addWidget(splitter, 1)

    def _selected_models(self) -> List[str]:
        names = []
        for name, (chk, _, _) in self.model_rows.items():
            if chk.isChecked():
                names.append(name)
        return names

    def _conf_value(self) -> float:
        return self.conf_slider.value() / 100.0

    def _display_images(self):
        if self.current_original is not None:
            self.original_label.setPixmap(cvimg_to_qpix(self.current_original, self.original_label.size()))
        for model_name, label in self.result_labels.items():
            if model_name in self.current_results:
                label.setPixmap(
                    cvimg_to_qpix(self.current_results[model_name].annotated, label.size())
                )
            else:
                label.setText(f"{model_name} 无结果")
        self.btn_save_batch.setEnabled(bool(self.current_results))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._display_images()

    def _update_stats(self, results: List[DetectionResult]):
        self.table.setRowCount(len(results))
        total_count = 0
        total_time = 0.0
        for i, res in enumerate(results):
            total_count += res.count
            total_time += res.infer_time_ms
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(res.model_name))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(res.count)))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{res.infer_time_ms:.2f}"))
        self.count_label.setText(f"目标数量：{total_count}")
        self.time_label.setText(f"推理耗时：{total_time:.2f} ms")

    def _show_error(self, message: str):
        self.status_label.setText(f"状态：错误 - {message}")
        QtWidgets.QMessageBox.critical(self, "Error", message)

    def load_weight(self, model_name: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"加载 {model_name} 权重", "", "PyTorch Weights (*.pt)"
        )
        if not path:
            return
        self.status_label.setText(f"状态：正在加载 {model_name}...")
        QtWidgets.QApplication.processEvents()
        try:
            self.detector.load_model(model_name, path)
            _, path_edit, _ = self.model_rows[model_name]
            path_edit.setText(path)
            self.status_label.setText(f"状态：{model_name} 加载成功")
        except Exception as e:
            self._show_error(str(e))

    def run_single_image(self):
        if self.image_thread and self.image_thread.isRunning():
            return
        model_names = self._selected_models()
        if not model_names:
            self._show_error("请至少选择一个模型。")
            return
        loaded = self.detector.loaded_models()
        model_names = [m for m in model_names if m in loaded]
        if not model_names:
            self._show_error("所选模型均未加载权重。")
            return

        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not image_path:
            return
        self.current_image_path = image_path

        self.status_label.setText("状态：单图检测中...")
        self.image_thread = ImageDetectThread(
            self.detector,
            image_path,
            model_names,
            self._conf_value(),
            self.show_box_chk.isChecked(),
            self.show_label_chk.isChecked(),
            self.show_conf_chk.isChecked(),
        )
        self.image_thread.result_ready.connect(self.on_image_result)
        self.image_thread.error.connect(self._show_error)
        self.image_thread.start()

    def on_image_result(self, original: np.ndarray, results: List[DetectionResult]):
        self.current_original = original
        self.current_results = {r.model_name: r for r in results}
        self._display_images()
        self._update_stats(results)
        self.status_label.setText("状态：单图检测完成")

    def run_folder_batch(self):
        if self.folder_thread and self.folder_thread.isRunning():
            return
        model_names = self._selected_models()
        if not model_names:
            self._show_error("请至少选择一个模型。")
            return
        loaded = self.detector.loaded_models()
        model_names = [m for m in model_names if m in loaded]
        if not model_names:
            self._show_error("所选模型均未加载权重。")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择批量保存输出文件夹（可取消）")
        save_dir = save_dir if save_dir else None

        self.status_label.setText("状态：批量检测中...")
        self.folder_thread = FolderDetectThread(
            self.detector,
            folder,
            save_dir,
            model_names,
            self._conf_value(),
            self.show_box_chk.isChecked(),
            self.show_label_chk.isChecked(),
            self.show_conf_chk.isChecked(),
        )
        self.folder_thread.file_done.connect(self.on_folder_file_done)
        self.folder_thread.finished_summary.connect(self.on_folder_done)
        self.folder_thread.error.connect(self._show_error)
        self.folder_thread.start()

    def on_folder_file_done(self, image_path: str, results: List[DetectionResult]):
        image = cv2.imread(image_path)
        if image is None:
            return
        self.current_image_path = image_path
        self.current_original = image
        self.current_results = {r.model_name: r for r in results}
        self._display_images()
        self._update_stats(results)
        self.status_label.setText(f"状态：正在处理 {os.path.basename(image_path)}")

    def on_folder_done(self, count: int):
        self.status_label.setText(f"状态：批量检测完成，共处理 {count} 张图片")

    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        model_names = self._selected_models()
        if not model_names:
            self._show_error("请至少选择一个模型。")
            return
        loaded = self.detector.loaded_models()
        model_names = [m for m in model_names if m in loaded]
        if not model_names:
            self._show_error("所选模型均未加载权重。")
            return

        self.camera_thread = CameraDetectThread(
            self.detector,
            camera_id=0,
            model_names=model_names,
            conf_thres=self._conf_value(),
            show_box=self.show_box_chk.isChecked(),
            show_label=self.show_label_chk.isChecked(),
            show_conf=self.show_conf_chk.isChecked(),
        )
        self.camera_thread.frame_ready.connect(self.on_camera_frame)
        self.camera_thread.error.connect(self._show_error)
        self.camera_thread.start()

        self.btn_camera_start.setEnabled(False)
        self.btn_camera_stop.setEnabled(True)
        self.status_label.setText("状态：摄像头检测中...")

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait(1000)
        self.btn_camera_start.setEnabled(True)
        self.btn_camera_stop.setEnabled(False)
        self.status_label.setText("状态：摄像头检测已停止")

    def on_camera_frame(self, frame: np.ndarray, results: List[DetectionResult]):
        self.current_original = frame
        self.current_results = {r.model_name: r for r in results}
        self._display_images()
        self._update_stats(results)

    def save_current_result(self):
        if not self.current_results:
            self._show_error("当前没有可保存的检测结果。")
            return
        model_name = self.result_tabs.tabText(self.result_tabs.currentIndex())
        if model_name not in self.current_results:
            self._show_error(f"{model_name} 当前无检测结果。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存检测结果", f"{model_name}_result.jpg", "Image Files (*.jpg *.png *.bmp)"
        )
        if not path:
            return
        cv2.imwrite(path, self.current_results[model_name].annotated)
        self.status_label.setText(f"状态：已保存 {model_name} 结果")

    def save_current_all_results(self):
        if not self.current_results:
            self._show_error("当前没有可保存的检测结果。")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not out_dir:
            return
        stem = "result"
        if self.current_image_path:
            stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        for model_name, res in self.current_results.items():
            out_path = os.path.join(out_dir, f"{stem}_{model_name}.jpg")
            cv2.imwrite(out_path, res.annotated)
        self.status_label.setText("状态：多模型结果已批量保存")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_camera()
        super().closeEvent(event)


def main():
    if hasattr(QtWidgets.QApplication, "setAttribute"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    if hasattr(app, "setHighDpiScaleFactorRoundingPolicy"):
        app.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
