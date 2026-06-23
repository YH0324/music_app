import numpy as np

def hello():
    return f"Python OK, numpy {np.__version__}"

def recognize_image(image_path, out_dir, models_dir, yolo_runner):
    """完整辨識:一張譜 → MIDI。純 YOLO + OpenCV,無 CNN。"""
    import omr_core
    import yolo_bridge
    bridge = yolo_bridge.YoloBridge(yolo_runner, imgsz=1024)
    r = omr_core.recognize(image_path, out_dir,
                           yolo_runner=bridge, models_dir=models_dir)
    return (f"辨識完成!\n"
            f"音符數: {r['total_notes']}\n"
            f"事件數: {r['total_events']}\n"
            f"MIDI: {'已產生' if r['midi_exists'] else '失敗'}\n"
            f"路徑: {r['midi_path']}")