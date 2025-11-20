import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import math
import json

# --- Helper Classes & Functions ---

class BaseAudioProcessor:
    def __init__(self, audio, num_frames, height, width, frame_rate):
        """
        Base class to process audio data.
        """
        # Convert waveform tensor to mono numpy array
        if isinstance(audio['waveform'], torch.Tensor):
            self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        else:
            self.audio = audio['waveform'].squeeze(0).mean(axis=0)
            
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate

        self.audio_duration = len(self.audio) / self.sample_rate
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else self.audio_duration / self.num_frames

        self.spectrum = None

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        # Handle out of bounds
        if start_sample >= len(self.audio):
            return np.zeros(1)
        return self.audio[start_sample:end_sample]

    def compute_spectrum(self, frame_index, fft_size, min_frequency, max_frequency):
        audio_frame = self._get_audio_frame(frame_index)
        if len(audio_frame) < fft_size:
            audio_frame = np.pad(audio_frame, (0, max(0, fft_size - len(audio_frame))), mode='constant')

        # Apply window function
        window = np.hanning(len(audio_frame))
        audio_frame = audio_frame * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio_frame, n=fft_size))

        # Extract desired frequency range
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        freq_indices = np.where((freqs >= min_frequency) & (freqs <= max_frequency))[0]
        
        if len(freq_indices) > 0:
            spectrum = spectrum[freq_indices]
        else:
            return np.zeros(1)

        # Check if spectrum is not empty
        if spectrum.size > 0:
            # Apply logarithmic scaling
            spectrum = np.log1p(spectrum)

            # Fixed scaling to preserve dynamics
            # Typical log1p spectrum values are in 0-15 range
            # We divide by 12.0 to map to roughly 0-1 without destroying relative volume
            spectrum = spectrum / 12.0
            spectrum = np.clip(spectrum, 0.0, 1.0)
            
            # Removed hard noise gate to prevent cutting off quiet sections/fade-outs.
            # User can use sensitivity to adjust response.
        else:
            spectrum = np.zeros(1)

        return spectrum

    def update_spectrum(self, new_spectrum, smoothing):
        if self.spectrum is None or len(self.spectrum) != len(new_spectrum):
            self.spectrum = np.zeros(len(new_spectrum))

        # Apply smoothing
        self.spectrum = smoothing * self.spectrum + (1 - smoothing) * new_spectrum

def pil2tensor(image):
    """Convert PIL Image to tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def parse_color(color):
    """Parse color string to RGB tuple"""
    if isinstance(color, str):
        if ',' in color:
            return tuple(int(c.strip()) for c in color.split(','))
        else:
            from PIL import ImageColor
            try:
                return ImageColor.getrgb(color)
            except:
                return (255, 0, 0) # Default red
    return color

# --- Main Node Class ---

class AudioReactivePathAnimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "screen_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "screen_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "paths_data": ("STRING", {"default": '{"paths": [], "canvas_size": {"width": 512, "height": 512}}', "multiline": True}),
                "shape": ([
                    'circle',
                    'square',
                    'triangle',
                    'hexagon',
                    'star',
                ], {"default": 'circle'}),
                "shape_size": ("INT", {"default": 20, "min": 2, "max": 500, "step": 1}),
                "shape_color": ("STRING", {"default": 'red'}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "movement_mode": (["amplitude", "accumulate"], {"default": "amplitude"}),
                "flip_on_beat": ("BOOLEAN", {"default": True}),
                "beat_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "motion_blur": ("BOOLEAN", {"default": False}),
                "release": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "amplitude_curve": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "opt_feature": ("FEATURE",),
                "bg_color": ("STRING", {"default": 'black'}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "trail_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Audio processing params (hidden/advanced or just standard defaults if not exposed)
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "duration_frames": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Duration in frames. 0 uses audio length."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "coordinates")
    FUNCTION = "animate"
    CATEGORY = "AudioReactive"

    def draw_shape(self, draw, shape, center_x, center_y, size, rotation, fill_color):
        """Draw a shape at the specified location"""
        half_size = size / 2

        if shape == 'circle':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            draw.ellipse(bbox, fill=fill_color)

        elif shape == 'square':
            bbox = [center_x - half_size, center_y - half_size,
                   center_x + half_size, center_y + half_size]
            draw.rectangle(bbox, fill=fill_color)

        elif shape == 'triangle':
            points = [
                (center_x, center_y - half_size),
                (center_x - half_size, center_y + half_size),
                (center_x + half_size, center_y + half_size),
            ]
            if rotation != 0:
                points = self.rotate_points(points, center_x, center_y, rotation)
            draw.polygon(points, fill=fill_color)

        elif shape == 'hexagon':
            points = []
            for i in range(6):
                angle = math.radians(60 * i + rotation)
                x = center_x + half_size * math.cos(angle)
                y = center_y + half_size * math.sin(angle)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)

        elif shape == 'star':
            points = []
            for i in range(10):
                angle = math.radians(36 * i + rotation)
                r = half_size if i % 2 == 0 else half_size * 0.4
                x = center_x + r * math.cos(angle - math.pi / 2)
                y = center_y + r * math.sin(angle - math.pi / 2)
                points.append((x, y))
            draw.polygon(points, fill=fill_color)

    def rotate_points(self, points, cx, cy, angle):
        """Rotate points around a center"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rotated = []
        for x, y in points:
            x -= cx
            y -= cy
            new_x = x * cos_a - y * sin_a + cx
            new_y = x * sin_a + y * cos_a + cy
            rotated.append((new_x, new_y))
        return rotated

    def interpolate_path(self, points, t):
        """
        Interpolate position along a path at time t (0.0 to 1.0)
        """
        if len(points) == 0:
            return (0, 0)
        if len(points) == 1:
            return (points[0]['x'], points[0]['y'])

        # Calculate total path length
        total_length = 0
        segment_lengths = []
        for i in range(len(points) - 1):
            dx = points[i + 1]['x'] - points[i]['x']
            dy = points[i + 1]['y'] - points[i]['y']
            length = math.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)
            total_length += length

        if total_length == 0:
            return (points[0]['x'], points[0]['y'])

        # Find target distance along path
        target_distance = t * total_length

        # Find which segment contains target distance
        current_distance = 0
        for i, seg_length in enumerate(segment_lengths):
            if current_distance + seg_length >= target_distance:
                # Interpolate within this segment
                segment_t = (target_distance - current_distance) / seg_length if seg_length > 0 else 0
                x = points[i]['x'] + (points[i + 1]['x'] - points[i]['x']) * segment_t
                y = points[i]['y'] + (points[i + 1]['y'] - points[i]['y']) * segment_t
                return (x, y)
            current_distance += seg_length

        # Return last point if we've gone past the end
        return (points[-1]['x'], points[-1]['y'])

    def animate(self, audio, frame_rate, screen_width, screen_height, paths_data, shape, shape_size, shape_color, sensitivity, smoothing, movement_mode="amplitude", flip_on_beat=True, beat_threshold=0.05, motion_blur=False, release=0.2, amplitude_curve=1.0, opt_feature=None, bg_color='black', blur_radius=0.0, trail_length=0.0, fft_size=2048, min_frequency=20.0, max_frequency=8000.0, duration_frames=0):
        
        # Parse colors
        shape_color_rgb = parse_color(shape_color)
        bg_color_rgb = parse_color(bg_color)

        # Parse paths
        try:
            paths_obj = json.loads(paths_data)
            paths = paths_obj.get('paths', [])
            canvas_size = paths_obj.get('canvas_size', {'width': screen_width, 'height': screen_height})
        except json.JSONDecodeError:
            print("AudioReactivePathAnimator: Invalid JSON in paths_data")
            paths = []
            canvas_size = {'width': screen_width, 'height': screen_height}

        # Calculate scaling
        canvas_width = canvas_size.get('width', screen_width)
        canvas_height = canvas_size.get('height', screen_height)
        scale_x = screen_width / canvas_width if canvas_width > 0 else 1.0
        scale_y = screen_height / canvas_height if canvas_height > 0 else 1.0

        # Scale paths
        scaled_paths = []
        for path in paths:
            scaled_path = path.copy()
            scaled_points = []
            for point in path.get('points', []):
                scaled_points.append({
                    'x': point['x'] * scale_x,
                    'y': point['y'] * scale_y
                })
            scaled_path['points'] = scaled_points
            scaled_paths.append(scaled_path)

        # Initialize Audio Processor
        # Calculate num_frames based on audio duration
        if isinstance(audio['waveform'], torch.Tensor):
             waveform = audio['waveform']
        else:
             waveform = torch.tensor(audio['waveform'])
             
        audio_duration = len(waveform.squeeze(0).mean(axis=0)) / audio['sample_rate']
        
        # Determine animation duration and effective frame rate
        print(f"DEBUG: duration_frames input: {duration_frames}")
        print(f"DEBUG: audio_duration: {audio_duration}")
        
        if duration_frames > 0:
            num_frames = duration_frames
            # Stretch/squash audio to fit exactly into num_frames
            if audio_duration > 0:
                effective_frame_rate = num_frames / audio_duration
            else:
                effective_frame_rate = frame_rate
        else:
            num_frames = int(audio_duration * frame_rate)
            effective_frame_rate = frame_rate
        
        processor = BaseAudioProcessor(audio, num_frames, screen_height, screen_width, effective_frame_rate)
        
        images_list = []
        masks_list = []
        previous_output = None
        
        # Initialize tracks for animated coordinates
        animated_tracks = [[] for _ in scaled_paths]
        
        # State for accumulate mode (per path)
        path_states = [{'t': 0.0, 'direction': 1.0} for _ in scaled_paths]

        # Animation Loop
        for i in range(num_frames):
            # Compute spectrum/feature
            spectrum = processor.compute_spectrum(i, fft_size, min_frequency, max_frequency)
            
            # Update spectrum with smoothing
            processor.update_spectrum(spectrum, smoothing)
            
            # Calculate feature value (mean of current smoothed spectrum)
            feature_value = np.mean(processor.spectrum)
            
            # Apply Amplitude Curve (Expander/Noise Gate effect)
            # Values > 1.0 suppress low values and accentuate peaks
            if amplitude_curve != 1.0:
                feature_value = pow(feature_value, amplitude_curve)
            
            # Draw
            image = Image.new("RGB", (screen_width, screen_height), bg_color_rgb)
            draw = ImageDraw.Draw(image)
            
            for p_idx, path in enumerate(scaled_paths):
                points = path.get('points', [])
                if not points:
                    continue
                
                # Calculate t based on mode
                if movement_mode == "amplitude":
                    # Amplitude mode: position = volume
                    target_t = feature_value * sensitivity
                    target_t = max(0.0, min(1.0, target_t))
                    
                    # Apply release (envelope follower)
                    # If new value is lower than current, decay slowly
                    # If new value is higher, attack instantly (or use smoothing)
                    if 'val' not in path_states[p_idx]:
                        path_states[p_idx]['val'] = 0.0
                        
                    current_val = path_states[p_idx]['val']
                    
                    if target_t > current_val:
                        # Attack
                        current_val = target_t
                    else:
                        # Release
                        current_val = current_val * (1.0 - release) + target_t * release
                        
                    path_states[p_idx]['val'] = current_val
                    t = current_val
                    
                elif movement_mode == "accumulate":
                    # Accumulate mode: move based on volume, bounce at ends
                    state = path_states[p_idx]
                    
                    # Beat detection for rhythmic flip using Spectral Flux
                    if flip_on_beat:
                        # Initialize state variables if missing
                        if 'prev_vol' not in state:
                            state['prev_vol'] = feature_value
                            state['cooldown'] = 0
                            
                        # Calculate Flux (positive change in volume)
                        flux = feature_value - state['prev_vol']
                        state['prev_vol'] = feature_value
                        
                        # Check for beat
                        if state['cooldown'] <= 0:
                            if flux > beat_threshold:
                                # Beat detected! Flip direction
                                state['direction'] *= -1.0
                                state['cooldown'] = 5 # Short cooldown (e.g. ~160ms at 30fps)
                        else:
                            state['cooldown'] -= 1

                    # Speed factor
                    speed = feature_value * sensitivity * 0.05
                    
                    state['t'] += speed * state['direction']
                    
                    # Bounce logic at ends (always active)
                    if state['t'] >= 1.0:
                        state['t'] = 1.0
                        state['direction'] = -1.0
                    elif state['t'] <= 0.0:
                        state['t'] = 0.0
                        state['direction'] = 1.0
                        
                    t = state['t']
                else:
                    t = 0.0

                # Interpolate position
                x, y = self.interpolate_path(points, t)
                
                # Draw shape
                # Motion Blur: Draw line from previous position to current position
                if motion_blur and i > 0:
                    # We need the previous position for this specific path
                    # We can store it in path_states
                    if 'prev_x' in path_states[p_idx]:
                        prev_x = path_states[p_idx]['prev_x']
                        prev_y = path_states[p_idx]['prev_y']
                        
                        # Draw line
                        draw.line([(prev_x, prev_y), (x, y)], fill=shape_color_rgb, width=shape_size)
                        
                    # Update previous position
                    path_states[p_idx]['prev_x'] = x
                    path_states[p_idx]['prev_y'] = y
                else:
                    # Initialize prev pos for first frame
                    path_states[p_idx]['prev_x'] = x
                    path_states[p_idx]['prev_y'] = y

                self.draw_shape(draw, shape, x, y, shape_size, 0, shape_color_rgb)
                
                # Record coordinate for this frame
                animated_tracks[p_idx].append({
                    "x": float(x),
                    "y": float(y)
                })

            # Blur
            if blur_radius > 0:
                image = image.filter(ImageFilter.GaussianBlur(blur_radius))
                
            # Tensor conversion
            image_tensor = pil2tensor(image)
            
            # Trails
            if trail_length > 0 and previous_output is not None:
                image_tensor = image_tensor + trail_length * previous_output
                # REMOVED: image_tensor = image_tensor / image_tensor.max() ... 
                # This was causing trails to brighten indefinitely and fill the screen.
                
            previous_output = image_tensor.clone()
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
            
            # Mask
            mask = image_tensor[:, :, :, 0] # Red channel as mask
            
            images_list.append(image_tensor)
            masks_list.append(mask)
            
        # Stack
        if not images_list:
             # Return empty/black if no frames
             empty_img = torch.zeros((1, screen_height, screen_width, 3))
             empty_mask = torch.zeros((1, screen_height, screen_width))
             return (empty_img, empty_mask, "[]")

        out_images = torch.cat(images_list, dim=0)
        out_masks = torch.cat(masks_list, dim=0)
        
        # Serialize animated coordinates
        # WAN ATI / WanTrackToVideo expects exactly 121 points covering the full duration.
        
        final_tracks = []
        target_length = 121
        
        for track in animated_tracks:
            if not track:
                final_tracks.append([])
                continue
                
            resampled_track = []
            src_len = len(track)
            
            if src_len == 1:
                # Static point
                resampled_track = [track[0] for _ in range(target_length)]
            else:
                for i in range(target_length):
                    # Map target index i (0..120) to source index (0..src_len-1)
                    t = i / (target_length - 1)
                    src_idx_float = t * (src_len - 1)
                    
                    idx0 = int(src_idx_float)
                    idx1 = min(idx0 + 1, src_len - 1)
                    fraction = src_idx_float - idx0
                    
                    p0 = track[idx0]
                    p1 = track[idx1]
                    
                    # Linear interpolation
                    x = p0['x'] + (p1['x'] - p0['x']) * fraction
                    y = p0['y'] + (p1['y'] - p0['y']) * fraction
                    
                    # Use int to match FL_PathAnimator exactly
                    resampled_track.append({
                        "x": int(round(x)),
                        "y": int(round(y))
                    })
            
            final_tracks.append(resampled_track)

        coord_string = json.dumps(final_tracks)
        
        print(f"DEBUG: Resampled tracks to {target_length} points for WAN ATI compatibility.")
        
        return (out_images, out_masks, coord_string)

# Mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AudioReactivePathAnimator": AudioReactivePathAnimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioReactivePathAnimator": "Audio Reactive Path Animator"
}
