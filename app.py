"""
Gradio web interface for Satellite-Guided DDColor.
"""

import gradio as gr
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime
import copy
import subprocess
import sys
import yaml

from src.dataset import CVUSADataset
from src.model import SatelliteGuidedDDColor
from src.utils import load_config
from src.polar_transform import create_polar_from_satellite
from inference import colorize_panorama, prepare_panorama


# Global model
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
config = None
training_process = None
loaded_checkpoint_path = None


def load_model(checkpoint_path=None):
    """Load the model."""
    global model, config, loaded_checkpoint_path

    if model is None:
        config = load_config("config.yaml")

        model = SatelliteGuidedDDColor(
            ddcolor_weights_path=config['ddcolor']['weights_path'],
            ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
            context_dim=config['model']['context_dim'],
            polar_encoder_pretrained=config['model']['polar_encoder_pretrained'],
            correction_type=config['model']['correction_type'],
            residual_scale=config['model']['residual_scale'],
        ).to(device)

        model.eval()

    if checkpoint_path and checkpoint_path != loaded_checkpoint_path:
        print(f"[Gradio] Loading checkpoint: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        loaded_checkpoint_path = checkpoint_path
        print("[Gradio] Checkpoint loaded.", flush=True)

    return model


def rgb_to_gray(rgb):
    """Convert RGB to grayscale."""
    if len(rgb.shape) == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None].repeat(3, axis=2)
    return rgb


@torch.no_grad()
def colorize(
    street_view,
    context_image=None,
    context_type="Polar image",
    checkpoint=None,
    show_base=False,
):
    """
    Colorize street view with satellite guidance.

    Args:
        street_view: RGB or grayscale street view image
        context_image: Polar image or original satellite/overhead image
        context_type: How to interpret the context image
        checkpoint: Path to model checkpoint
        show_base: Whether to show DDColor base output

    Returns:
        Tuple of (gray_input, ddcolor_base, satellite_guided, comparison)
    """
    if street_view is None:
        return None, None, None, None, None

    try:
        print("[Gradio] Colorize request received.", flush=True)
        # Load model
        load_model(checkpoint)

        # Convert to RGB
        if isinstance(street_view, np.ndarray):
            if street_view.shape[-1] == 4:  # RGBA
                street_view = street_view[:, :, :3]
        else:
            street_view = np.array(street_view)

        if context_image is None:
            raise ValueError("Please provide a polar image or an overhead satellite image for context.")
        if isinstance(context_image, np.ndarray) and context_image.shape[-1] == 4:
            context_image = context_image[:, :, :3]
        elif not isinstance(context_image, np.ndarray):
            context_image = np.array(context_image)

        polar_size = tuple(config['model']['polar_input_size'])
        if context_type == "Satellite image":
            result = colorize_panorama(
                model,
                street_view.astype(np.uint8),
                context_image.astype(np.uint8),
                torch.device(device),
                polar_size,
            )
        else:
            gray_input, gray_patches = prepare_panorama(street_view.astype(np.uint8))
            polar_rgb = cv2.resize(
                context_image.astype(np.uint8),
                (polar_size[1], polar_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            polar = torch.from_numpy(polar_rgb).permute(2, 0, 1).float().div(255.0)
            polar = polar.unsqueeze(0).repeat(4, 1, 1, 1).to(device)
            output = model(gray_patches.to(device), polar)

            def merge(name):
                patches = output[name].detach().cpu().permute(0, 2, 3, 1).numpy()
                return np.clip(np.concatenate(list(patches), axis=1), 0, 1)

            result = {
                "gray": gray_input,
                "polar": polar_rgb,
                "base": merge("base_rgb"),
                "final": merge("final_rgb"),
            }
        gray_input = result['gray']
        base_rgb = (result['base'] * 255).round().astype(np.uint8)
        final_rgb = (result['final'] * 255).round().astype(np.uint8)

        # Create comparison
        comparison = np.concatenate([gray_input, base_rgb, final_rgb], axis=1)
        polar_rgb = result['polar']

        if not show_base:
            base_rgb = None

        print("[Gradio] Colorize request finished.", flush=True)
        return gray_input, polar_rgb, base_rgb, final_rgb, comparison

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return street_view, None, None, None, None


def browse_dataset(split="train", sample_idx=0):
    """Browse dataset samples."""
    dataset_root = config['dataset']['root'] if config else "C:/Users/31133/Desktop/dataset1/CVUSA_processed_split"

    try:
        dataset = CVUSADataset(dataset_root, split=split, load_polar=True)

        if sample_idx >= len(dataset):
            sample_idx = 0

        sample = dataset[sample_idx]

        # Convert to uint8
        rgb = (sample['rgb'] * 255).astype(np.uint8)
        gray = (sample['gray'] * 255).astype(np.uint8)

        if sample['polar'] is not None:
            polar = (sample['polar'] * 255).astype(np.uint8)
        else:
            polar = None

        info = f"File ID: {sample['file_id']}<br>Panorama ID: {sample['panorama_id']}<br>Patch: {sample['patch_idx']}"

        return rgb, gray, polar, info, len(dataset)

    except Exception as e:
        return None, None, None, f"Error: {e}", 0


def get_available_checkpoints():
    """Get list of available checkpoints."""
    checkpoint_dir = Path("checkpoints")
    checkpoints = []

    if checkpoint_dir.exists():
        for exp_dir in checkpoint_dir.iterdir():
            if exp_dir.is_dir():
                for ckpt in exp_dir.glob("*.pth"):
                    checkpoints.append(str(ckpt))

    checkpoints = sorted(checkpoints)
    return [None] + checkpoints


def create_inference_tab():
    """Create a minimal inference-only panel."""
    best_checkpoint = "checkpoints\\film_ddcolor_cu130_20260527\\best.pth"

    def simple_colorize(street_view, polar_image):
        gray, polar, base, final, comparison = colorize(
            street_view=street_view,
            context_image=polar_image,
            context_type="Polar image",
            checkpoint=best_checkpoint,
            show_base=True,
        )
        return final, base, comparison

    with gr.Row():
        with gr.Column():
            street_view_input = gr.Image(label="输入街景图", type="numpy")
            context_input = gr.Image(label="输入对应 Polar 图", type="numpy")
            colorize_btn = gr.Button("开始上色", variant="primary")

        with gr.Column():
            final_output = gr.Image(label="最终上色结果")
            base_output = gr.Image(label="DDColor 基础结果")
            comparison_output = gr.Image(label="对比图：灰度 | DDColor | 改进结果")

    colorize_btn.click(
        fn=simple_colorize,
        inputs=[street_view_input, context_input],
        outputs=[final_output, base_output, comparison_output],
    )

    return None


def create_dataset_browser_tab():
    """Create the dataset browser tab."""
    with gr.Row():
        with gr.Column():
            split_select = gr.Dropdown(
                label="Split",
                choices=["train", "val", "test"],
                value="train",
            )
            sample_slider = gr.Slider(
                label="Sample Index",
                minimum=0,
                maximum=1000,
                value=0,
                step=1,
            )
            sample_count = gr.Number(label="Total Samples", value=0, interactive=False)
            info_text = gr.Markdown()

        with gr.Column():
            rgb_view = gr.Image(label="RGB Ground Truth")
            gray_view = gr.Image(label="Grayscale")
            polar_view = gr.Image(label="Polar Satellite View")

    def update_sample_count(split):
        dataset_root = config['dataset']['root'] if config else "C:/Users/31133/Desktop/dataset1/CVUSA_processed_split"
        try:
            dataset = CVUSADataset(dataset_root, split=split, load_polar=True)
            return len(dataset), gr.Slider(maximum=len(dataset)-1)
        except:
            return 0, gr.Slider(maximum=1000)

    def browse(split, idx):
        return browse_dataset(split, int(idx))

    split_select.change(
        fn=update_sample_count,
        inputs=[split_select],
        outputs=[sample_count, sample_slider],
    )

    sample_slider.change(
        fn=browse,
        inputs=[split_select, sample_slider],
        outputs=[rgb_view, gray_view, polar_view, info_text],
    )

    # Initial load
    rgb_view.change(
        fn=lambda x: x,
        inputs=[split_select],
        outputs=[sample_count],
    )

    return gr.Tab("Dataset Browser")


def create_training_tab():
    """Create training monitoring tab."""
    def start_training(batch, epoch_count, learning_rate):
        global training_process
        if training_process is not None and training_process.poll() is None:
            return read_training_log(), "A UI-launched training process is already running."
        base_config = load_config("config.yaml")
        run_config = copy.deepcopy(base_config)
        run_config["training"]["batch_size"] = int(batch)
        run_config["training"]["epochs"] = int(epoch_count)
        run_config["training"]["lr"] = float(learning_rate)
        exp_name = "ui_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        config_path = log_dir / f"{exp_name}.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(run_config, file, allow_unicode=True, sort_keys=False)
        log_path = log_dir / f"{exp_name}.stdout.log"
        error_path = log_dir / f"{exp_name}.stderr.log"
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        training_process = subprocess.Popen(
            [sys.executable, "-u", "train.py", "--config", str(config_path), "--exp_name", exp_name],
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=error_path.open("w", encoding="utf-8"),
            creationflags=creationflags,
        )
        return "", f"Training launched: `{exp_name}` (PID `{training_process.pid}`)"

    def read_training_log():
        logs = sorted(Path("outputs/logs").glob("*.stdout.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return "No training log found."
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
        return text[-6000:]

    with gr.Row():
        with gr.Column():
            batch_size = gr.Number(label="Batch Size", value=4)
            epochs = gr.Number(label="Epochs", value=30)
            lr = gr.Number(label="Learning Rate", value=0.0001)
            start_train_btn = gr.Button("Start Training", variant="primary")
            refresh_btn = gr.Button("Refresh Latest Log")

        with gr.Column():
            log_output = gr.Textbox(label="Training Log", lines=20, placeholder="Training logs will appear here...")
            status_output = gr.Markdown()

    start_train_btn.click(
        fn=start_training,
        inputs=[batch_size, epochs, lr],
        outputs=[log_output, status_output],
    )
    refresh_btn.click(
        fn=read_training_log,
        outputs=[log_output],
    )

    return gr.Tab("Training")


def create_evaluation_tab():
    """Create evaluation tab."""
    checkpoints = get_available_checkpoints()

    with gr.Row():
        with gr.Column():
            checkpoint_select = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoints,
                value=checkpoints[0] if checkpoints else None,
            )
            split_select = gr.Dropdown(
                label="Split",
                choices=["val", "test"],
                value="val",
            )
            evaluate_btn = gr.Button("Run Evaluation", variant="primary")

        with gr.Column():
            results_output = gr.Markdown()

    evaluate_btn.click(
        fn=lambda ckpt, split: f"To run evaluation, use: python evaluate.py --checkpoint {ckpt} --split {split}",
        inputs=[checkpoint_select, split_select],
        outputs=[results_output],
    )

    return gr.Tab("Evaluation")


def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="Satellite-Guided DDColor") as demo:
        gr.Markdown(
            """
            # 街景灰度图上色
            上传街景图和对应的 Polar 图，点击开始上色。
            """
        )
        create_inference_tab()

    return demo


if __name__ == "__main__":
    demo = create_interface()

    # Load config for browser
    try:
        config = load_config("config.yaml")
    except:
        config = None

    # Get Gradio config
    gradio_config = config.get('gradio', {}) if config else {}

    demo.launch(
        server_name=gradio_config.get('server_name', '0.0.0.0'),
        server_port=gradio_config.get('server_port', 7860),
        share=gradio_config.get('share', False),
        show_error=True,
    )
